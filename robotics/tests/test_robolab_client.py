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
    from armlab.robolab.client import VLMRoboLabClient, quat_to_mat

    seen = {}

    class FakePolicy:
        def act(self, obs):
            seen["obs"] = obs
            return Decision([Waypoint((0.5, 0.1, 0.05), 90.0, "close")], "llm", "grab")

    # Gripper pointing down: flange quat = 180 deg about x; fingertip 0.16 m below the flange.
    q_down = [0.0, 1.0, 0.0, 0.0]
    ee = [0.4, 0.0, 0.4]
    client = VLMRoboLabClient(FakePolicy(), control_hz=15, max_speed=0.2, use_gt_state=True)
    raw = _raw_obs(ee, q_down, [0.4, 0.0, 0.24], objects={"cube": [0.5, 0.1, 0.02]})
    a0 = client.infer(raw, "pick the cube")["action"]
    obs = seen["obs"]
    assert np.allclose(obs.eef_xyz, [0.4, 0.0, 0.24], atol=1e-6)
    assert np.allclose(obs.objects["cube"]["xyz"], [0.5, 0.1, 0.02], atol=1e-3)  # env-local -> robot root
    chunk = client._chunks[0]
    assert chunk.shape[1] == 8 and len(chunk) > 10
    last = chunk[-1]
    R = quat_to_mat(last[3:7])
    tip = last[:3] + R @ np.array([0, 0, 0.16])
    assert np.allclose(tip, [0.5, 0.1, 0.05], atol=1e-6)
    assert last[7] == 1.0 and a0[7] == 0.0
    # Yaw of 90 deg about world z relative to the start orientation.
    R_rel = R @ quat_to_mat(q_down).T
    assert math.isclose(math.degrees(math.atan2(R_rel[1, 0], R_rel[0, 0])), 90.0, abs_tol=1e-4)
    # Steps never exceed max_speed * dt.
    tips = np.array([c[:3] + quat_to_mat(c[3:7]) @ [0, 0, 0.16] for c in chunk])
    assert np.max(np.linalg.norm(np.diff(tips, axis=0), axis=1)) <= 0.2 / 15 + 1e-6
