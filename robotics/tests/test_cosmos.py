import json

import numpy as np
import pytest

import armlab  # noqa: F401
from armlab.cosmos import physics_filter, safety
from armlab.cosmos.reason import VideoReasoner, split_answer, write_video
from armlab.policy.vlm import ScriptedVLM


@pytest.fixture
def clip(tmp_path):
    frames = [np.full((64, 80, 3), i * 10, np.uint8) for i in range(20)]
    return write_video(frames, tmp_path / "clip.mp4", fps=10)


def test_split_answer_cosmos_format():
    r, a = split_answer("<think>\nthe cup is tipping\n</think>\n\n{\"x\": 1}")
    assert r == "the cup is tipping" and a == '{"x": 1}'
    assert split_answer("plain") == ("", "plain")


def test_reasoner_frames_mode_sends_timestamped_frames(clip):
    vlm = ScriptedVLM(lambda s, p: {"ok": True})
    out = VideoReasoner(vlm, num_frames=4).ask(clip, "what?")
    assert out["json"] == {"ok": True}
    parts = vlm.calls[0][1]
    assert sum(isinstance(p, np.ndarray) for p in parts) == 4
    assert any(isinstance(p, str) and p.startswith("Frame at t=") for p in parts)


def test_safety_report(clip):
    def respond(s, p):
        return "<think>forklift near worker</think>" + json.dumps({
            "summary": "A forklift passes a worker without a hard hat.", "people": 1,
            "hazards": [{"category": "missing PPE", "severity": "high", "time_s": 1.2, "description": "no hard hat"},
                        {"category": "vehicle", "severity": "WEIRD", "description": "forklift close"}],
            "recommended_actions": ["Enforce hard hats"]})
    r = safety.analyze_clip(VideoReasoner(ScriptedVLM(respond)), clip, "Dock 4")
    assert r["hazards"][1]["severity"] == "low" and r["reasoning"] == "forklift near worker"
    md = safety.daily_report([r], "Dock 4")
    assert "Needs attention today" in md and "missing PPE" in md and "Hazards flagged:** 2" in md


def test_api_analyze(clip, monkeypatch):
    from fastapi.testclient import TestClient

    from armlab.cosmos import api

    fake = VideoReasoner(ScriptedVLM(lambda s, p: {"what_is_happening": "a hand reaches for a mug",
                                                   "objects": ["mug: upright"], "next_action": "grasp the mug handle",
                                                   "next_action_steps": ["open hand", "close on handle"],
                                                   "hazards": [], "task_progress": "in progress"}))
    monkeypatch.setattr(api, "get_reasoner", lambda: fake)
    client = TestClient(api.app)
    assert client.get("/health").json()["ok"]
    r = client.post("/v1/analyze", files={"video": ("c.mp4", clip.read_bytes(), "video/mp4")}, data={"task": "make tea"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["next_action"] == "grasp the mug handle" and body["task_progress"] == "in progress"
    assert "make tea" in "".join(p for p in fake.vlm.calls[0][1] if isinstance(p, str))


def test_physics_scenario_renders_and_breaks_physics():
    frames = physics_filter.render_scenario("passthrough", seed=100)
    assert len(frames) > 40 and frames[0].shape == (*physics_filter.CLIP_SIZE, 3)


def test_physics_judge_and_score(clip):
    r = VideoReasoner(ScriptedVLM(lambda s, p: {"plausible": "false", "confidence": 0.9, "violations": ["floats"]}))
    v = physics_filter.judge(r, clip)
    assert v["plausible"] is False and v["violations"] == ["floats"]
    m = physics_filter.score([
        {"scenario": "drop", "plausible": True, "pred_plausible": True},
        {"scenario": "teleport", "plausible": False, "pred_plausible": False},
        {"scenario": "vanish", "plausible": False, "pred_plausible": True},
        {"scenario": "place", "plausible": True, "pred_plausible": False},
    ])
    assert m["accuracy"] == 0.5 and m["reject_precision"] == 0.5 and m["reject_recall"] == 0.5


def test_physics_evaluate_retries_and_skips_failed_clips(clip, tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(physics_filter.time, "sleep", lambda s: None)
    ds = tmp_path / "ds"
    ds.mkdir()
    rows = [("drop_0.mp4", "drop", True), ("vanish_0.mp4", "vanish", False), ("broken_0.mp4", "teleport", False)]
    for name, _, _ in rows:
        shutil.copy(clip, ds / name)
    (ds / "manifest.jsonl").write_text("".join(json.dumps({"clip": c, "scenario": s, "plausible": p}) + "\n"
                                               for c, s, p in rows))

    def respond(system, parts):
        return {"plausible": True, "confidence": 0.9, "violations": []}

    r = VideoReasoner(ScriptedVLM(respond))
    real_judge = physics_filter.judge
    flaky = {"vanish_0.mp4": 1}

    def judge(reasoner, path):
        if path.name == "broken_0.mp4":
            raise RuntimeError("endpoint down")
        if flaky.get(path.name):
            flaky[path.name] -= 1
            raise RuntimeError("cold start")
        return real_judge(reasoner, path)

    monkeypatch.setattr(physics_filter, "judge", judge)
    out = tmp_path / "eval.json"
    m = physics_filter.evaluate(ds, r, out)
    doc = json.loads(out.read_text())
    assert m["n"] == 2 and m["errors"] == 1 and m["mode"] == "frames"
    assert [x["pred_plausible"] for x in doc["results"]] == [True, True, None]
    assert m["reject_recall"] == 0.0 and m["false_reject_rate"] == 0.0


def test_transfer_export(tmp_path):
    from armlab.cosmos import transfer_export

    specs = transfer_export.export("blocks_into_bin", 0, tmp_path, variations=2, seconds=0.5)
    assert len(specs) == 2
    spec = json.loads(specs[0].read_text())
    assert spec["video_path"] == "input_rgb.mp4" and spec["seg"]["control_path"] == "control_seg.mp4"
    for n in ("input_rgb.mp4", "control_depth.mp4", "control_seg.mp4"):
        assert (tmp_path / n).stat().st_size > 0
