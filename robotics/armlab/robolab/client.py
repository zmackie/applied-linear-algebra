"""RoboLab (NVIDIA Isaac Lab benchmark) inference client that runs armlab's VLM policies.

Uses RoboLab's absolute end-effector IK action space (DroidIKActionCfg): each action is
[x, y, z, qw, qx, qy, qz, gripper] for the Robotiq flange ("base_link") in the robot-root frame,
gripper 1 = close. The policy speaks fingertip waypoints + yaw (like the MuJoCo sim); this client
turns them into a per-step action chunk at the env's 15 Hz control rate.

Only importable inside a RoboLab / Isaac Lab Python environment (see modal_apps/robolab_eval.py).
"""
from __future__ import annotations

import math
import time

import numpy as np
from robolab.eval.base_client import InferenceClient

from ..policy.types import Decision, Observation, Usage, Waypoint
from ..sim.geometry import mat_to_quat, rot_z, wrap_angle


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class VLMRoboLabClient(InferenceClient):
    open_loop_horizon = 1  # chunks have variable length; see _needs_refresh

    def __init__(self, policy, control_hz: float = 15.0, max_speed: float = 0.2, grip_hold_s: float = 0.7,
                 use_gt_state: bool = True, log: list | None = None):
        super().__init__()
        self.policy = policy
        self.dt = 1.0 / control_hz
        self.max_speed = max_speed
        self.grip_hold_steps = max(1, round(grip_hold_s * control_hz))
        self.use_gt_state = use_gt_state
        self.log = log if log is not None else []
        self._state: dict[int, dict] = {}
        self.usage = Usage()

    # ------------------------------------------------------------------ episode bookkeeping

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._state.clear()
        else:
            self._state.pop(env_id, None)

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= len(self._chunks[env_id])

    # ------------------------------------------------------------------ observation

    def _image_keys(self, raw_obs) -> tuple[str | None, str | None]:
        keys = list(raw_obs.get("image_obs", {}).keys())
        wrist = next((k for k in keys if "wrist" in k), None)
        head = next((k for k in keys if k != wrist), None)
        return head, wrist

    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        head_k, wrist_k = self._image_keys(raw_obs)
        p = raw_obs["proprio_obs"]
        ex = {
            "head": self._to_numpy(raw_obs["image_obs"][head_k], env_id).astype(np.uint8) if head_k else None,
            "wrist": self._to_numpy(raw_obs["image_obs"][wrist_k], env_id).astype(np.uint8) if wrist_k else None,
            "ee_pos": self._to_numpy(p["ee_pos"], env_id).astype(float),
            "ee_quat": self._to_numpy(p["ee_quat"], env_id).astype(float),
            "eef_pos": self._to_numpy(p["eef_pos"], env_id).astype(float) if "eef_pos" in p else None,
            "gripper": float(self._to_numpy(p["gripper_pos"], env_id).reshape(-1)[0]),
            "env_id": env_id,
        }
        gt = self._get_env_gt_state(raw_obs, env_id) if self.use_gt_state else None
        ex["gt"] = gt
        return ex

    def _episode_state(self, ex: dict) -> dict:
        env_id = ex["env_id"]
        if env_id not in self._state:
            R0 = quat_to_mat(ex["ee_quat"])
            tip = ex["eef_pos"] if ex["eef_pos"] is not None else ex["ee_pos"] + R0 @ np.array([0, 0, 0.16])
            self._state[env_id] = {
                "R0": R0,
                "tip_offset_local": R0.T @ (tip - ex["ee_pos"]),  # flange -> fingertip, in flange frame
                "cmd_tip": tip.copy(),
                "cmd_yaw": 0.0,
                "cmd_grip": 0.0,
                "decisions": 0,
            }
        return self._state[env_id]

    def _tip(self, ex: dict, st: dict) -> np.ndarray:
        return ex["ee_pos"] + quat_to_mat(ex["ee_quat"]) @ st["tip_offset_local"]

    def _objects_in_root(self, ex: dict) -> dict | None:
        gt = ex.get("gt")
        if not gt or "objects" not in gt:
            return None
        robot = gt.get("robot", {})
        # Env-local -> robot-root transform from the end-effector pose seen in both frames.
        R_env_ee = quat_to_mat(robot["ee_quat"])
        R_root_ee = quat_to_mat(ex["ee_quat"])
        R_env_root = R_env_ee @ R_root_ee.T
        t = np.asarray(robot["ee_pos"], float) - R_env_root @ ex["ee_pos"]
        out = {}
        for name, o in gt["objects"].items():
            p_root = R_env_root.T @ (np.asarray(o["pos"], float) - t)
            out[name] = {"xyz": [round(float(v), 3) for v in p_root]}
        return out

    def _pack_request(self, ex: dict, instruction: str) -> dict:
        st = self._episode_state(ex)
        R = quat_to_mat(ex["ee_quat"]) @ st["R0"].T
        obs = Observation(
            instruction=instruction,
            images={"head": ex["head"], "wrist": ex["wrist"] if ex["wrist"] is not None else ex["head"]},
            eef_xyz=self._tip(ex, st),
            eef_yaw_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])),
            gripper_width=0.085 * (1.0 - ex["gripper"]),  # Robotiq 2F-85: 0 = open, 1 = closed
            step=st["decisions"], max_steps=0, sim_time=0.0,
            objects=self._objects_in_root(ex),
            camera_info={"head": "Exterior camera over the robot's shoulder looking at the table.",
                         "wrist": "Camera on the gripper."},
        )
        if st["decisions"] and self.log:
            obs.feedback = f"fingertip now {np.round(obs.eef_xyz, 3).tolist()}, gripper closedness {ex['gripper']:.2f}"
        return {"obs": obs, "ex": ex, "st": st}

    # ------------------------------------------------------------------ policy call

    def _query_server(self, request: dict) -> Decision:
        t0 = time.time()
        decision = self.policy.act(request["obs"])
        decision.usage.latency_s = decision.usage.latency_s or (time.time() - t0)
        self.usage.add(decision.usage)
        request["st"]["decisions"] += 1
        if hasattr(self.policy, "record"):
            self.policy.record(decision, request["obs"].feedback)
        self.log.append({"source": decision.source, "rationale": decision.rationale,
                         "waypoints": [w.to_dict() for w in decision.waypoints], "think_s": decision.usage.latency_s})
        request["decision"] = decision
        return request

    def _unpack_response(self, response: dict) -> np.ndarray:
        decision: Decision = response["decision"]
        st = response["st"]
        rows = []
        waypoints = decision.waypoints or [Waypoint(tuple(st["cmd_tip"]), math.degrees(st["cmd_yaw"]), None)]
        for wp in waypoints:
            goal, goal_yaw = np.asarray(wp.xyz, float), math.radians(wp.yaw_deg)
            start, start_yaw = st["cmd_tip"].copy(), st["cmd_yaw"]
            dyaw = wrap_angle(goal_yaw - start_yaw)
            n = max(1, math.ceil(max(np.linalg.norm(goal - start) / (self.max_speed * self.dt), abs(dyaw) / (math.radians(90) * self.dt))))
            for i in range(1, n + 1):
                a = i / n
                rows.append(self._action(st, start + a * (goal - start), start_yaw + a * dyaw, st["cmd_grip"]))
            st["cmd_tip"], st["cmd_yaw"] = goal, start_yaw + dyaw
            if wp.gripper is not None:
                st["cmd_grip"] = 1.0 if wp.gripper == "close" else 0.0
                rows += [self._action(st, goal, st["cmd_yaw"], st["cmd_grip"])] * self.grip_hold_steps
        return np.asarray(rows, dtype=np.float32)

    def _action(self, st: dict, tip, yaw: float, grip: float) -> np.ndarray:
        R = rot_z(yaw) @ st["R0"]
        flange = np.asarray(tip) - R @ st["tip_offset_local"]
        return np.concatenate([flange, mat_to_quat(R), [grip]])

    def _build_visualization(self, ex: dict):
        return ex["head"]
