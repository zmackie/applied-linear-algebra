"""RoboLab client math, tested against a stub of robolab's InferenceClient (Isaac Lab is not needed)."""
import math
import sys
import types

import numpy as np

import armlab  # noqa: F401


def _stub_robolab():
    if "robolab.eval.base_client" in sys.modules:
        return
    from abc import ABC

    class InferenceClient(ABC):
        def __init__(self):
            self._chunks, self._counters = {}, {}

        def infer(self, obs, instruction, *, env_id=0):
            ex = self._extract_observation(obs, env_id=env_id)
            if self._needs_refresh(env_id):
                chunk = self._unpack_response(self._query_server(self._pack_request(ex, instruction)))
                self._chunks[env_id], self._counters[env_id] = chunk, 0
            a = self._chunks[env_id][self._counters[env_id]]
            self._counters[env_id] += 1
            return {"action": a, "viz": None}

        def reset(self, *, env_id=None):
            self._chunks.clear()
            self._counters.clear()

        @staticmethod
        def _get_env_gt_state(raw_obs, env_id):
            return raw_obs.get("gt_state", {}).get(env_id)

        @staticmethod
        def _to_numpy(v, env_id=0):
            return np.asarray(v)[env_id]

    for name in ("robolab", "robolab.eval"):
        sys.modules[name] = types.ModuleType(name)
    mod = types.ModuleType("robolab.eval.base_client")
    mod.InferenceClient = InferenceClient
    sys.modules["robolab.eval.base_client"] = mod


def _raw_obs(ee_pos, ee_quat, eef_pos, objects=None, env_offset=(1.0, 2.0, 0.0)):
    img = np.zeros((1, 48, 64, 3), np.uint8)
    obs = {"image_obs": {"over_shoulder_left_camera": img, "wrist_cam": img},
           "proprio_obs": {"ee_pos": np.array([ee_pos]), "ee_quat": np.array([ee_quat]),
                           "eef_pos": np.array([eef_pos]), "gripper_pos": np.array([[0.0]])}}
    if objects is not None:
        off = np.asarray(env_offset)
        obs["gt_state"] = {0: {"robot": {"ee_pos": (np.asarray(ee_pos) + off).tolist(), "ee_quat": list(ee_quat)},
                               "objects": {k: {"pos": (np.asarray(v) + off).tolist()} for k, v in objects.items()}}}
    return obs


def test_client_converts_waypoints_to_flange_actions():
    _stub_robolab()
    from armlab.policy.types import Decision, Waypoint
    from armlab.robolab.client import CLOSE_AXIS_LOCAL, TIP_OFFSET_M, VLMRoboLabClient, quat_to_mat

    seen = {}

    class FakePolicy:
        def act(self, obs):
            seen["obs"] = obs
            return Decision([Waypoint((0.5, 0.1, 0.05), 90.0, "close")], "llm", "grab")

    # Gripper pointing down: flange quat = 180 deg about x; fingertip TIP_OFFSET_M below the flange.
    # Real RoboLab reports eef_pos == ee_pos (eef_frame has no position offset), so the client must not use it.
    q_down = [0.0, 1.0, 0.0, 0.0]
    ee = [0.4, 0.0, 0.4]
    client = VLMRoboLabClient(FakePolicy(), control_hz=15, max_speed=0.2, use_gt_state=True)
    raw = _raw_obs(ee, q_down, ee, objects={"cube": [0.5, 0.1, 0.02]})
    a0 = client.infer(raw, "pick the cube")["action"]
    obs = seen["obs"]
    assert np.allclose(obs.eef_xyz, [0.4, 0.0, 0.4 - TIP_OFFSET_M], atol=1e-6)
    assert np.allclose(obs.objects["cube"]["xyz"], [0.5, 0.1, 0.02], atol=1e-3)  # env-local -> robot root
    chunk = client._chunks[0]
    assert chunk.shape[1] == 8 and len(chunk) > 10
    last = chunk[-1]
    R = quat_to_mat(last[3:7])
    tip = last[:3] + R @ np.array([0, 0, TIP_OFFSET_M])
    assert np.allclose(tip, [0.5, 0.1, 0.05], atol=1e-6)
    assert last[7] == 1.0 and a0[7] == 0.0
    # Policy convention: yaw 90 = fingers close along y. Robotiq closes along its base_link y axis.
    close_dir = R @ CLOSE_AXIS_LOCAL
    assert np.allclose(np.abs(close_dir), [0, 1, 0], atol=1e-6)
    # Settle at the goal before the gripper closes: the last open-gripper rows already sit at the goal.
    first_close = int(np.argmax(chunk[:, 7] == 1.0))
    assert first_close >= client.settle_steps
    assert np.allclose(chunk[first_close - client.settle_steps, :3], chunk[first_close, :3])
    # Steps never exceed max_speed * dt.
    tips = np.array([c[:3] + quat_to_mat(c[3:7]) @ [0, 0, TIP_OFFSET_M] for c in chunk])
    assert np.max(np.linalg.norm(np.diff(tips, axis=0), axis=1)) <= 0.2 / 15 + 1e-6


def test_fingertip_axis_is_whichever_flange_axis_points_down():
    _stub_robolab()
    from armlab.robolab.client import fingertip_offset_local, quat_to_mat

    # Flange z pointing down (180 deg about x) -> +z local.
    assert np.allclose(fingertip_offset_local(quat_to_mat([0, 1, 0, 0]), 0.15), [0, 0, 0.15])
    # Flange x pointing down (-90 deg about y): local +x maps to world -z.
    q = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
    R = quat_to_mat(q)
    off = fingertip_offset_local(R, 0.15)
    assert np.allclose(R @ off, [0, 0, -0.15], atol=1e-9)


def test_yaw_zero_closes_along_x_at_robolab_reset_pose():
    """RoboLab's DROID reset pose (seen on Modal): flange quat ~ (0.707, 0, 0.707, 0), i.e. base_link x points
    down and its y (the closing axis) along root +y. yaw 0 must still mean "fingers close along x"."""
    _stub_robolab()
    from armlab.policy.types import Decision, Waypoint
    from armlab.robolab.client import CLOSE_AXIS_LOCAL, VLMRoboLabClient, quat_to_mat

    seen = {}
    yaw = {"v": 0.0}

    class P:
        def act(self, obs):
            seen["yaw"] = obs.eef_yaw_deg
            seen["tip"] = obs.eef_xyz
            return Decision([Waypoint((0.5, 0.0, 0.1), yaw["v"], None)], "llm", "")

    q = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
    for target, expect in ((0.0, [1, 0, 0]), (90.0, [0, 1, 0]), (45.0, [math.sqrt(0.5), math.sqrt(0.5), 0])):
        yaw["v"] = target
        c = VLMRoboLabClient(P(), use_gt_state=False)
        c.infer(_raw_obs([0.36, 0.0, 0.47], q, [0.36, 0.0, 0.47]), "x")
        assert math.isclose(abs(seen["yaw"]), 90.0, abs_tol=1e-4)  # reset pose closes along y
        assert np.allclose(seen["tip"], [0.36 - 0.0, 0.0, 0.47 - 0.155], atol=1e-6)  # local x points down
        R = quat_to_mat(c._chunks[0][-1][3:7])
        assert np.allclose(np.abs(R @ CLOSE_AXIS_LOCAL), np.abs(expect), atol=1e-6)
        assert np.allclose(R[:, 0], [0, 0, -1], atol=1e-6)  # still pointing straight down
