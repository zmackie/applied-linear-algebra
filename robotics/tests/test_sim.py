import json

import numpy as np
import pytest

import armlab  # noqa: F401  (sets MUJOCO_GL)
from armlab.eval.runner import run_episode
from armlab.policy.llm import DirectVLMPolicy, HybridPolicy, annotate_head, decision_schema
from armlab.policy.scripted import ScriptedPolicy
from armlab.policy.types import Waypoint
from armlab.policy.vlm import ScriptedVLM, extract_json
from armlab.sim.env import ArmEnv
from armlab.sim.tasks import TASKS


@pytest.mark.parametrize("task", list(TASKS))
def test_oracle_solves_every_task(task, tmp_path):
    res = run_episode(task, seed=0, policy_kind="oracle", out_dir=tmp_path, verbose=False)
    assert res.success, res
    assert res.score == 100.0


def test_reset_is_deterministic():
    a, b = ArmEnv("sort_warm_cool", seed=3), ArmEnv("sort_warm_cool", seed=3)
    for blk in a.layout.blocks:
        assert np.allclose(a.object_pos(blk.name), b.object_pos(blk.name))
    assert [x.name for x in a.layout.blocks] == [x.name for x in b.layout.blocks]


def test_ik_reaches_target():
    env = ArmEnv("blocks_into_bin", seed=0)
    env.execute([Waypoint((0.5, 0.15, 0.2), 30.0, None)])
    assert np.linalg.norm(env.eef_xyz() - [0.5, 0.15, 0.2]) < 0.01
    assert abs(np.degrees(env.eef_yaw()) - 30.0) < 3.0


def test_pixel_roundtrip():
    env = ArmEnv("blocks_into_bin", seed=0)
    blk = env.layout.blocks[0].name
    p = env.object_pos(blk)
    top = p + [0, 0, 0.02]
    u, v = env.xyz_to_pixel("head", top)
    q = env.pixel_to_xyz("head", u, v)
    assert np.linalg.norm(q[:2] - top[:2]) < 0.01 and abs(q[2] - 0.04) < 0.006


def test_extract_json_tolerates_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}


def _oracle_as_model(env, use_pixels=True):
    """A fake VLM that answers like a model would, using the oracle's plan (exercises parsing + pixel grounding)."""
    oracle = ScriptedPolicy(env)

    def respond(system, parts):
        d = oracle.act(None)
        wps = []
        for i, w in enumerate(d.waypoints):
            pix = env.xyz_to_pixel("head", (w.xyz[0], w.xyz[1], 0.0)) if use_pixels and i == 1 else None
            if pix and w.xyz[2] < 0.05:
                # Point at the surface seen at the grasp point (the block's top face), offset down to the grasp height.
                surface = env.pixel_to_xyz("head", *pix)
                wps.append({"target_type": "pixel", "xyz": [0, 0, 0], "pixel_u": pix[0], "pixel_v": pix[1],
                            "z_offset": w.xyz[2] - surface[2], "yaw_deg": w.yaw_deg, "gripper": w.gripper or "none"})
            else:
                wps.append({"target_type": "xyz", "xyz": list(w.xyz), "pixel_u": 0, "pixel_v": 0, "z_offset": 0,
                            "yaw_deg": w.yaw_deg, "gripper": w.gripper or "none"})
        return {"observation": "fake", "plan": d.rationale, "locate": [], "waypoints": wps,
                "task_complete": d.done, "notes": "", "choice": "accept", "accept_count": len(wps)}
    return respond


def test_direct_policy_loop_with_fake_model(tmp_path):
    env_holder = {}

    def respond(system, parts):
        return env_holder["fn"](system, parts)

    vlm = ScriptedVLM(respond)
    pol = DirectVLMPolicy(vlm=vlm)
    # run_episode builds its own env; hook the fake to it through the policy reset.
    orig_reset = pol.reset

    def reset(instr):
        orig_reset(instr)
    pol.reset = reset

    from armlab.eval import runner
    real_env_cls = runner.ArmEnv

    class HookedEnv(real_env_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            env_holder["fn"] = _oracle_as_model(self)
    runner.ArmEnv = HookedEnv
    try:
        res = run_episode("blocks_into_bin", 0, "direct", policy=pol, out_dir=tmp_path, video=False, verbose=False)
    finally:
        runner.ArmEnv = real_env_cls
    assert res.success, res
    assert res.model_calls >= 6
    # Images and state text were sent each call.
    system, parts = vlm.calls[0]
    assert any(isinstance(p, np.ndarray) for p in parts)
    assert any(isinstance(p, str) and "fingertip xyz" in p for p in parts)


def test_hybrid_accepts_and_corrects(tmp_path):
    env = ArmEnv("blocks_into_bin", seed=1)
    calls = {"n": 0}

    def respond(system, parts):
        calls["n"] += 1
        if calls["n"] == 1:  # correct the first proposal with a harmless move
            return {"observation": "", "plan": "reposition", "locate": [], "task_complete": False, "notes": "n1",
                    "choice": "correct", "accept_count": 0,
                    "waypoints": [{"target_type": "xyz", "xyz": [0.4, 0.0, 0.2], "pixel_u": 0, "pixel_v": 0,
                                   "z_offset": 0, "yaw_deg": 0, "gripper": "open"}]}
        return {"observation": "", "plan": "ok", "locate": [], "waypoints": [], "task_complete": False,
                "notes": "", "choice": "accept", "accept_count": 99}

    pol = HybridPolicy(vlm=ScriptedVLM(respond), system1=ScriptedPolicy(env, source="system1"))
    pol.reset(env.task.instruction)
    d1 = pol.act(env.observe())
    assert d1.source == "llm-correction" and d1.waypoints[0].xyz == (0.4, 0.0, 0.2)
    d2 = pol.act(env.observe())
    assert d2.source == "system1" and d2.waypoints == d2.candidate


def test_locate_round_trip():
    env = ArmEnv("blocks_into_bin", seed=0)
    seen = []

    def respond(system, parts):
        txt = [p for p in parts if isinstance(p, str) and p.startswith("LOCATE RESULTS")]
        if not txt:
            return {"observation": "", "plan": "look", "locate": [{"u": 256, "v": 192}], "waypoints": [],
                    "task_complete": False, "notes": ""}
        seen.append(json.loads(txt[0].split(": ", 1)[1].rsplit(". Now", 1)[0]))
        return {"observation": "", "plan": "go", "locate": [], "task_complete": False, "notes": "",
                "waypoints": [{"target_type": "xyz", "xyz": [0.5, 0, 0.2], "pixel_u": 0, "pixel_v": 0,
                               "z_offset": 0, "yaw_deg": 0, "gripper": "none"}]}
    pol = DirectVLMPolicy(vlm=ScriptedVLM(respond))
    pol.reset("x")
    d = pol.act(env.observe())
    assert len(d.waypoints) == 1 and seen and seen[0][0]["xyz"] is not None
    assert d.usage.calls == 2


def test_schema_is_strict_object():
    for hybrid in (False, True):
        s = decision_schema(hybrid)
        assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])


def test_annotation_draws(tmp_path):
    env = ArmEnv("blocks_into_bin", seed=0)
    obs = env.observe()
    img = annotate_head(obs, [Waypoint((0.5, 0.1, 0.1))])
    assert img.shape == obs.images["head"].shape and not np.array_equal(img, obs.images["head"])


def test_realtime_clock_lets_world_move(tmp_path):
    env = ArmEnv("conveyor_pick", seed=0)
    y0 = env.object_pos("orange_block")[1]
    env.idle(2.0)
    assert env.object_pos("orange_block")[1] < y0 - 0.05
