"""RoboLab episode rows -> results-page layout (modal_apps/robolab_eval.py publish helpers)."""
import importlib.util
import json
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")

from armlab.web.results import load_run  # noqa: E402

_APP = Path(__file__).resolve().parents[1] / "modal_apps" / "robolab_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("robolab_eval", _APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROWS = [
    {"task_name": "RubiksCubeTask", "run": 0, "success": True, "score": 1.0, "reason": "done",
     "timing": {"policy_inference_s": 20.0}},
    {"task_name": "RubiksCubeTask", "run": 1, "success": False, "score": 0.5,
     "reason": "Condition not satisfied: object_grabbed(object=cube) (step 1/2)", "timing": {}},
]


def test_episode_result_and_results_page_roundtrip(tmp_path):
    m = _load()
    run = tmp_path / "robolab-test"
    for r in ROWS:
        ep = m.episode_result(r)
        d = run / ep["task"] / f"seed{ep['seed']}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(ep))
    (run / "summary.txt").write_text(m.summary_table(ROWS, ["RubiksCubeTask"]))
    loaded = load_run(tmp_path, "robolab-test")
    assert [(e.task, e.seed, e.success) for e in loaded.episodes] == [("RubiksCubeTask", 0, True),
                                                                     ("RubiksCubeTask", 1, False)]
    assert loaded.episodes[0].score == 100 and loaded.episodes[0].error == ""
    assert "object_grabbed" in loaded.episodes[1].error
    assert "OVERALL" in loaded.summary and "1/2" in loaded.summary


def test_driver_supported():
    m = _load()
    assert m.driver_supported("580.95.05")
    assert not m.driver_supported("610.57.04")
    assert m.driver_supported("garbage")
