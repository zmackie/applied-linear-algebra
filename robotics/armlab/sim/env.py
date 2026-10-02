"""MuJoCo Franka Panda tabletop environment with end-effector-waypoint control.

Control runs at 25 Hz (0.04 s per control step, matching the report's step length).
A policy emits waypoints; each is tracked with a straight-line Cartesian interpolation
and damped-least-squares IK onto the arm's position servos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from ..assets import panda_dir
from ..policy.types import Observation, Waypoint
from .geometry import lookat_xyaxes, rot_z, rotvec_from_mat, wrap_angle
from .tasks import TASKS, BinSpec, BlockSpec, Layout, Task, TaskStatus

CONTROL_DT = 0.04
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
READY_XYZ = np.array([0.28, 0.0, 0.42])
WORKSPACE = {"x": (0.2, 0.8), "y": (-0.5, 0.5), "z": (0.008, 0.6)}

CAMERAS = {
    # name: (pos, lookat, fovy, description shown to the policy)
    "head": ((0.5, 0.0, 0.82), (0.5, 0.0, 0.0), 45,
             "Fixed camera 0.82 m above the table looking straight down. Image up = away from the robot base (+x), image left = robot's left (+y). The robot base is just below the bottom edge."),
    "front": ((1.35, 0.75, 0.85), (0.45, 0.0, 0.08), 45, "Third-person viewing camera (video only)."),
    "closeup": ((1.05, -0.62, 0.38), (0.45, -0.05, 0.14), 50, "Close third-person camera (video only)."),
}


@dataclass
class FrameInfo:
    step: int
    sim_time: float
    phase: str  # "move" | "grip" | "think" | "settle"


class ArmEnv:
    def __init__(self, task: str | Task, seed: int = 0, obs_size=(384, 512), obs_mode: str = "vision"):
        self.task = TASKS[task] if isinstance(task, str) else task
        self.obs_h, self.obs_w = obs_size
        self.obs_mode = obs_mode
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._depth_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.reset(seed)

    # ------------------------------------------------------------------ build / reset

    def _build(self, layout: Layout) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(panda_dir() / "panda.xml"))
        spec.option.timestep = 0.002
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
        spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
        wb = spec.worldbody
        wb.add_light(pos=[0.5, 0.0, 2.0], dir=[0, 0, -1], diffuse=[0.45, 0.45, 0.45], castshadow=True)
        wb.add_light(pos=[1.5, 1.0, 1.5], dir=[-1, -0.6, -1], diffuse=[0.15, 0.15, 0.15], castshadow=False)
        tex = spec.add_texture(name="tabletex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                               rgb1=[0.46, 0.37, 0.29], rgb2=[0.43, 0.35, 0.27], width=256, height=256)
        mat = spec.add_material(name="table", texrepeat=[6, 6], texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], material="table",
                    friction=[1.0, 0.005, 0.0001])

        # Stiffer gripper servo (~10 N squeeze on a 4 cm block instead of ~2 N) so carried
        # blocks do not slip out during fast moves.
        grip = spec.actuator("actuator8")
        grip.gainprm[0] *= 5
        grip.biasprm[1] *= 5
        grip.biasprm[2] *= 5
        hand = spec.body("hand")
        hand.add_site(name="eef", pos=[0, 0, 0.1034], size=[0.005, 0, 0], rgba=[1, 0, 1, 0])
        # Wrist camera: looks along the fingers (+z of the hand), mounted slightly off-axis.
        hand.add_camera(name="wrist", pos=[0.06, 0.0, 0.02], quat=_wrist_cam_quat(), fovy=75)
        for name, (pos, target, fovy, _) in CAMERAS.items():
            up = (1.0, 0.0, 0.0) if name == "head" else (0.0, 0.0, 1.0)
            wb.add_camera(name=name, pos=list(pos), xyaxes=lookat_xyaxes(pos, target, up), fovy=fovy)

        for b in layout.bins:
            self._add_bin(wb, b)
        for blk in layout.blocks:
            body = wb.add_body(name=blk.name, pos=[blk.xy[0], blk.xy[1], blk.half + 0.0005],
                               quat=[math.cos(blk.yaw / 2), 0, 0, math.sin(blk.yaw / 2)])
            body.add_freejoint()
            spec.add_material(name=f"mat_{blk.name}", rgba=list(blk.rgba), specular=0.05, shininess=0.1)
            body.add_geom(name=blk.name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[blk.half] * 3, material=f"mat_{blk.name}",
                          mass=0.05 * (blk.half / 0.02) ** 3, friction=[1.5, 0.01, 0.0002], condim=4)
        if layout.conveyor is not None:
            c = layout.conveyor
            # A heavy plate on a frictionless slide joint whose velocity is pinned each control step,
            # so friction (not teleporting) carries the block along.
            plate = wb.add_body(name="conveyor", pos=[c.start_xy[0], c.start_xy[1], c.half[2]])
            direction = np.array([*c.velocity_xy, 0.0])
            plate.add_joint(name="conveyor_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(direction / np.linalg.norm(direction)))
            plate.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(c.half), rgba=[0.2, 0.2, 0.22, 1], mass=20.0,
                           friction=[2.0, 0.01, 0.0002])
            # Lift the riding block onto the plate.
            spec.body(layout.conveyor_block).pos[2] += 2 * c.half[2]
        return spec.compile()

    @staticmethod
    def _add_bin(wb, b: BinSpec):
        body = wb.add_body(name=b.name, pos=[b.xy[0], b.xy[1], 0])
        hx, hy, h, t = b.half_xy[0], b.half_xy[1], b.height, 0.006
        box = mujoco.mjtGeom.mjGEOM_BOX
        body.add_geom(type=box, size=[hx, hy, t / 2], pos=[0, 0, t / 2], rgba=list(b.rgba))
        for sx in (-1, 1):
            body.add_geom(type=box, size=[t / 2, hy + t, h / 2], pos=[sx * (hx + t / 2), 0, h / 2], rgba=list(b.rgba))
        for sy in (-1, 1):
            body.add_geom(type=box, size=[hx, t / 2, h / 2], pos=[0, sy * (hy + t / 2), h / 2], rgba=list(b.rgba))

    def reset(self, seed: int = 0) -> Observation:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.layout = self.task.sample(self.rng)
        self.model = self._build(self.layout)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        for r in (*self._renderers.values(), *self._depth_renderers.values()):
            r.close()
        self._renderers.clear()
        self._depth_renderers.clear()

        m, d = self.model, self.data
        self._arm_qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self._arm_dadr = np.array([m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self._qlo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self._qhi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        self._eef = m.site("eef").id
        self._finger_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger")}
        self._block_geom = {b.name: m.geom(b.name).id for b in self.layout.blocks}
        self._block_body = {b.name: m.body(b.name).id for b in self.layout.blocks}
        self._conveyor_dof = m.joint("conveyor_slide").dofadr[0] if self.layout.conveyor else None

        d.qpos[self._arm_qadr] = HOME_Q
        d.qpos[self._arm_qadr[-1] + 1: self._arm_qadr[-1] + 3] = 0.04
        d.ctrl[:7] = HOME_Q
        d.ctrl[7] = GRIP_OPEN
        mujoco.mj_forward(m, d)
        self._R_home = d.site_xmat[self._eef].reshape(3, 3).copy()
        # Start from a retracted "ready" pose so the overhead camera sees the table.
        q_ready = self.solve_ik(READY_XYZ, 0.0, HOME_Q, iters=200)
        d.qpos[self._arm_qadr] = q_ready
        d.ctrl[:7] = q_ready
        mujoco.mj_forward(m, d)
        self.target_xyz = d.site_xpos[self._eef].copy()
        self.target_yaw = 0.0
        self.grip_cmd = GRIP_OPEN
        self._q_cmd = q_ready.copy()
        self.step_count = 0
        self.frame_callbacks: list[Callable[["ArmEnv", FrameInfo], None]] = []
        self._physics(int(0.3 / CONTROL_DT), "settle", record=False)
        self.step_count = 0
        return self.observe()

    # ------------------------------------------------------------------ state queries

    @property
    def sim_time(self) -> float:
        return self.step_count * CONTROL_DT

    def eef_xyz(self) -> np.ndarray:
        return self.data.site_xpos[self._eef].copy()

    def eef_yaw(self) -> float:
        R = self.data.site_xmat[self._eef].reshape(3, 3) @ self._R_home.T
        return math.atan2(R[1, 0], R[0, 0])

    def gripper_width(self) -> float:
        a = self._arm_qadr[-1]
        return float(self.data.qpos[a + 1] + self.data.qpos[a + 2])

    def object_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self._block_body[name]].copy()

    def object_vel(self, name: str) -> np.ndarray:
        """Linear velocity (world frame) of a block."""
        adr = self.model.jnt_dofadr[self.model.body_jntadr[self._block_body[name]]]
        return self.data.qvel[adr:adr + 3].copy()

    def object_yaw(self, name: str) -> float:
        R = self.data.xmat[self._block_body[name]].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def layout_block(self, name: str) -> BlockSpec:
        return next(b for b in self.layout.blocks if b.name == name)

    def layout_bin(self, name: str) -> BinSpec:
        return next(b for b in self.layout.bins if b.name == name)

    def held_objects(self) -> list[str]:
        """Blocks touched by both fingers."""
        touching: dict[str, set] = {}
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self._finger_geoms:
                    for name, gid in self._block_geom.items():
                        if b == gid:
                            touching.setdefault(name, set()).add(m.geom_bodyid[a])
        return [n for n, bodies in touching.items() if len(bodies) >= 2]

    def in_bin(self, obj: str, bin_name: str) -> bool:
        b = self.layout_bin(bin_name)
        p = self.object_pos(obj)
        dx, dy = abs(p[0] - b.xy[0]), abs(p[1] - b.xy[1])
        return bool(dx < b.half_xy[0] and dy < b.half_xy[1] and p[2] < b.height + 0.01 and obj not in self.held_objects())

    def status(self) -> TaskStatus:
        return self.task.check(self)

    # ------------------------------------------------------------------ rendering

    def _renderer(self, h: int, w: int, depth: bool = False) -> mujoco.Renderer:
        pool = self._depth_renderers if depth else self._renderers
        if (h, w) not in pool:
            r = mujoco.Renderer(self.model, h, w)
            if depth:
                r.enable_depth_rendering()
            pool[(h, w)] = r
        return pool[(h, w)]

    def render(self, camera: str, h: int | None = None, w: int | None = None, depth: bool = False,
               segmentation: bool = False, shadows: bool = True) -> np.ndarray:
        h, w = h or self.obs_h, w or self.obs_w
        r = self._renderer(h, w, depth)
        if segmentation:
            r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=camera)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
        out = r.render()
        if segmentation:
            r.disable_segmentation_rendering()
        return out

    def camera_pose(self, camera: str) -> tuple[np.ndarray, np.ndarray, float]:
        cid = self.model.camera(camera).id
        return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy(), float(self.model.cam_fovy[cid])

    def pixel_to_xyz(self, camera: str, u: int, v: int) -> np.ndarray | None:
        """World point seen at pixel (u, v) of the policy-sized image from `camera`, via the depth buffer."""
        h, w = self.obs_h, self.obs_w
        if not (0 <= u < w and 0 <= v < h):
            return None
        depth = self.render(camera, h, w, depth=True)
        z = float(depth[int(v), int(u)])
        if not np.isfinite(z) or z <= 0 or z > 10:
            return None
        pos, mat, fovy = self.camera_pose(camera)
        f = (h / 2) / math.tan(math.radians(fovy) / 2)
        x_cam = np.array([(u + 0.5 - w / 2) / f * z, -(v + 0.5 - h / 2) / f * z, -z])
        return pos + mat @ x_cam

    def xyz_to_pixel(self, camera: str, xyz) -> tuple[int, int] | None:
        h, w = self.obs_h, self.obs_w
        pos, mat, fovy = self.camera_pose(camera)
        p = mat.T @ (np.asarray(xyz) - pos)
        if p[2] >= 0:
            return None
        f = (h / 2) / math.tan(math.radians(fovy) / 2)
        u = w / 2 + f * p[0] / -p[2]
        v = h / 2 - f * p[1] / -p[2]
        return int(u), int(v)

    # ------------------------------------------------------------------ observation

    def observe(self) -> Observation:
        st = self.status()
        objects = None
        if self.obs_mode == "state":
            objects = {}
            for b in self.layout.blocks:
                p = self.object_pos(b.name)
                objects[b.name] = {"xyz": [round(float(v), 3) for v in p], "yaw_deg": round(math.degrees(self.object_yaw(b.name)), 1),
                                   "size_m": round(2 * b.half, 3)}
            for b in self.layout.bins:
                objects[b.name] = {"xyz": [b.xy[0], b.xy[1], 0.0], "inner_size_m": [2 * b.half_xy[0], 2 * b.half_xy[1]],
                                   "rim_height_m": b.height}
        return Observation(
            instruction=self.task.instruction,
            images={"head": self.render("head"), "wrist": self.render("wrist")},
            eef_xyz=self.eef_xyz(),
            eef_yaw_deg=math.degrees(self.eef_yaw()),
            gripper_width=self.gripper_width(),
            step=self.step_count,
            max_steps=self.task.max_steps,
            sim_time=self.sim_time,
            task_done=st.success,
            objects=objects,
            pixel_to_xyz=self.pixel_to_xyz,
            xyz_to_pixel=self.xyz_to_pixel,
            camera_info={"head": CAMERAS["head"][3],
                         "wrist": "Camera on the gripper looking along the fingers (down when the gripper points down)."},
        )

    # ------------------------------------------------------------------ control

    def _target_rot(self, yaw: float) -> np.ndarray:
        return rot_z(yaw) @ self._R_home

    def solve_ik(self, xyz, yaw, q_init, iters: int = 60, tol: float = 1e-4) -> np.ndarray:
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        q = np.array(q_init, float)
        R_t = self._target_rot(yaw)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self._arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self._eef]
            R = d.site_xmat[self._eef].reshape(3, 3)
            err = np.concatenate([np.asarray(xyz) - p, 0.5 * rotvec_from_mat(R_t @ R.T)])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self._eef)
            J = np.vstack([jacp, jacr])[:, self._arm_dadr]
            lam = 0.02
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
            # Null-space pull toward the home posture keeps the elbow sensible.
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (HOME_Q - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self._qlo + 1e-3, self._qhi - 1e-3)
        return q

    def _physics(self, n_control_steps: int, phase: str, record: bool = True):
        substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        for _ in range(n_control_steps):
            if self._conveyor_dof is not None:
                self.data.qvel[self._conveyor_dof] = float(np.linalg.norm(self.layout.conveyor.velocity_xy))
            mujoco.mj_step(self.model, self.data, nstep=substeps)
            if record:
                self.step_count += 1
                for cb in self.frame_callbacks:
                    cb(self, FrameInfo(self.step_count, self.sim_time, phase))

    def budget_left(self) -> int:
        return self.task.max_steps - self.step_count

    def clamp(self, xyz) -> np.ndarray:
        return np.array([np.clip(xyz[0], *WORKSPACE["x"]), np.clip(xyz[1], *WORKSPACE["y"]), np.clip(xyz[2], *WORKSPACE["z"])])

    def execute(self, waypoints: list[Waypoint], max_lin_speed: float = 0.25, max_yaw_speed: float = math.radians(120),
                phase_label: str = "move") -> dict:
        """Track waypoints in order. Returns per-segment execution info."""
        info = {"executed": 0, "steps": 0, "clamped": False, "final_error_m": None}
        start_step = self.step_count
        for wp in waypoints:
            if self.budget_left() <= 0:
                break
            goal = self.clamp(wp.xyz)
            info["clamped"] |= bool(np.linalg.norm(goal - np.asarray(wp.xyz)) > 1e-6)
            goal_yaw = math.radians(wp.yaw_deg)
            dyaw = wrap_angle(goal_yaw - self.target_yaw)
            start, start_yaw = self.target_xyz.copy(), self.target_yaw
            n = max(1, math.ceil(max(np.linalg.norm(goal - start) / (max_lin_speed * CONTROL_DT),
                                     abs(dyaw) / (max_yaw_speed * CONTROL_DT))))
            for i in range(1, n + 1):
                if self.budget_left() <= 0:
                    break
                a = i / n
                self.target_xyz = start + a * (goal - start)
                self.target_yaw = start_yaw + a * dyaw
                self._q_cmd = self.solve_ik(self.target_xyz, self.target_yaw, self._q_cmd)
                self.data.ctrl[:7] = self._q_cmd
                self._physics(1, phase_label)
            for _ in range(15):  # settle
                if np.linalg.norm(self.eef_xyz() - self.target_xyz) < 0.004 or self.budget_left() <= 0:
                    break
                self._physics(1, phase_label)
            if wp.gripper is not None:
                want = GRIP_OPEN if wp.gripper == "open" else GRIP_CLOSED
                if want != self.grip_cmd:
                    self.grip_cmd = want
                    self.data.ctrl[7] = want
                    self._physics(min(14, max(0, self.budget_left())), "grip")
            info["executed"] += 1
        info["steps"] = self.step_count - start_step
        info["final_error_m"] = round(float(np.linalg.norm(self.eef_xyz() - self.target_xyz)), 4)
        return info

    def idle(self, seconds: float, phase: str = "think"):
        """Advance the world while the arm holds its last command (used for wall-clock latency)."""
        n = min(int(round(seconds / CONTROL_DT)), max(0, self.budget_left()))
        self._physics(n, phase)

    def close(self):
        for r in (*self._renderers.values(), *self._depth_renderers.values()):
            r.close()
        self._renderers.clear()
        self._depth_renderers.clear()


def _wrist_cam_quat():
    # Camera looks along its -z; rotate 180 deg about x so it looks along the hand's +z.
    return [0.0, 1.0, 0.0, 0.0]
