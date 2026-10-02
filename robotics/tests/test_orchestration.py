"""Run-from-anywhere pieces: doctor, results page, safety inbox processing. No network, no Modal token."""
import datetime as dt
import json

import numpy as np
import pytest

from armlab import doctor
from armlab.cosmos import safety
from armlab.cosmos.reason import VideoReasoner, write_video
from armlab.policy.vlm import ScriptedVLM, make_vlm
from armlab.web import results


# ------------------------------------------------------------------------------------------------ doctor

def _no_net(name, url):
    return doctor.Check(f"reach {name}", doctor.FAIL, "unreachable: offline test")


def test_doctor_without_token_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    calls = []
    checks = doctor.run_checks(env={}, network=True, modal_cli=lambda *a: calls.append(a) or [], host_check=_no_net)
    by = {c.name: c for c in checks}
    assert by["modal token"].status == doctor.FAIL
    assert by["modal secret armlab-llm"].status == doctor.SKIP and "no Modal token" in by["modal secret armlab-llm"].detail
    assert calls == []  # never shells out to modal without a token
    assert "required check(s) failing" in doctor.format_report(checks)
    assert doctor.main(["--offline"]) == 0  # informational by default


def test_doctor_never_prints_secret_values(tmp_path, monkeypatch, capsys):
    secret = "as-SUPERSECRETVALUE123"
    env = {"MODAL_TOKEN_ID": "ak-TOKENIDVALUE", "MODAL_TOKEN_SECRET": secret, "ANTHROPIC_API_KEY": "sk-ant-XYZSECRET"}
    rows = {"secret": [{"Name": "armlab-llm"}], "volume": [{"Name": "armlab-runs"}],
            "app": [{"Description": "armlab-results", "State": "deployed"}]}
    ok = lambda n, u: doctor.Check(f"reach {n}", doctor.OK, "HTTPS ok")
    checks = doctor.run_checks(env=env, modal_cli=lambda kind, _: rows[kind], host_check=ok)
    out = doctor.format_report(checks) + json.dumps([c.__dict__ for c in checks])
    assert secret not in out and "TOKENIDVALUE" not in out and "XYZSECRET" not in out
    by = {c.name: c for c in checks}
    assert by["modal token"].status == doctor.OK and "env" in by["modal token"].detail
    assert by["modal secret armlab-llm"].status == doctor.OK
    assert by["modal secret huggingface"].status == doctor.FAIL
    assert by["modal app armlab-safety-cron"].status == doctor.WARN and not by["modal app armlab-safety-cron"].required
    assert by["model keys in this shell"].status == doctor.WARN and "ANTHROPIC_API_KEY" in by["model keys in this shell"].detail


def test_doctor_reads_modal_toml_profile(tmp_path, monkeypatch):
    cfg = tmp_path / "modal.toml"
    cfg.write_text('[me]\ntoken_id = "ak-x"\ntoken_secret = "as-y"\nactive = true\n')
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(cfg))
    assert doctor.modal_token_source({}) == "config:me"
    cfg.write_text('[me]\ntoken_id = "ak-x"\n')
    assert doctor.modal_token_source({}) is None


# ------------------------------------------------------------------------------------------------ results page

@pytest.fixture
def runs_root(tmp_path):
    run = tmp_path / "20261002-120000-oracle-modal"
    for task, seed, ok in [("blocks_into_bin", 0, True), ("conveyor_pick", 0, False)]:
        ep = run / task / f"seed{seed}"
        ep.mkdir(parents=True)
        (ep / "result.json").write_text(json.dumps({"task": task, "seed": seed, "success": ok, "score": 100.0 if ok else 40.0,
                                                    "decisions": 4, "input_tokens": 0, "output_tokens": 0, "error": ""}))
        write_video([np.zeros((32, 48, 3), np.uint8)] * 5, ep / "video.mp4", fps=5)
    (run / "summary.txt").write_text("task  success\nOVERALL   1/2   70.0\n")
    (run / "config.json").write_text(json.dumps({"policy": "oracle", "tasks": "all", "seeds": "0"}))
    csv_only = tmp_path / "older-run"
    csv_only.mkdir()
    (csv_only / "results.csv").write_text("task,seed,success,score,decisions,input_tokens,output_tokens,error\n"
                                          "stack_in_order,3,True,100.0,5,10,20,\n")
    rep = tmp_path / results.SAFETY_REPORTS
    rep.mkdir(parents=True)
    (rep / "safety-2026-10-01.md").write_text(safety.daily_report([], "Cell 3", dt.date(2026, 10, 1)))
    (tmp_path / "secret.txt").write_text("outside")  # not part of any run; still must not escape root
    return tmp_path


def test_results_pages_render(runs_root):
    from fastapi.testclient import TestClient

    c = TestClient(results.make_app(runs_root))
    idx = c.get("/")
    assert idx.status_code == 200 and "20261002-120000-oracle-modal" in idx.text and "older-run" in idx.text
    assert "safety-" not in idx.text or "2026-10-01" in idx.text
    assert "width=device-width" in idx.text
    page = c.get("/runs/20261002-120000-oracle-modal").text
    assert "OVERALL" in page and "<video" in page and "conveyor_pick" in page and "1/2 episodes succeeded" in page
    assert "<video" not in c.get("/runs/older-run").text and "stack_in_order" in c.get("/runs/older-run").text
    assert c.get("/runs/nope").status_code == 404
    assert c.get("/api/runs").json()[0]["name"]
    s = c.get("/safety")
    assert "2026-10-01" in s.text
    assert "Daily safety report: Cell 3" in c.get("/safety/2026-10-01").text


def test_results_video_supports_range_and_blocks_traversal(runs_root):
    from fastapi.testclient import TestClient

    c = TestClient(results.make_app(runs_root))
    url = "/files/20261002-120000-oracle-modal/blocks_into_bin/seed0/video.mp4"
    full = c.get(url)
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
    part = c.get(url, headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and len(part.content) == 10  # iOS Safari needs this to play video
    assert c.get("/files/../../etc/passwd").status_code == 404
    assert c.get("/files/%2e%2e/%2e%2e/etc/passwd").status_code == 404
    assert results.resolve_file(runs_root / "older-run", "../secret.txt") is None
    assert c.get("/runs/%2e%2e").status_code == 404 and c.get("/runs/_safety").status_code == 404


def test_results_access_key(runs_root):
    from fastapi.testclient import TestClient

    c = TestClient(results.make_app(runs_root, access_key="k123"))
    assert c.get("/").status_code == 401
    assert c.get("/health").status_code == 200
    r = c.get("/?key=k123", follow_redirects=False)
    assert r.status_code == 303 and "armlab_key" in r.headers["set-cookie"]
    assert TestClient(results.make_app(runs_root, access_key="k123"), cookies={"armlab_key": "k123"}).get("/").status_code == 200


def test_markdown_to_html():
    h = results.markdown_to_html("# T\n**a** `b`\n\n- x\n- y\n\n| A | B |\n|---|---|\n| 1 | <2> |\n_end_")
    assert "<h1>T</h1>" in h and "<b>a</b>" in h and "<code>b</code>" in h and "<li>y</li>" in h
    assert "<td>&lt;2&gt;</td>" in h and "<i>end</i>" in h


# ------------------------------------------------------------------------------------------------ safety cron logic

def test_safety_process_inbox(tmp_path):
    inbox = tmp_path / results.SAFETY_INBOX
    inbox.mkdir(parents=True)
    write_video([np.full((32, 48, 3), 9, np.uint8)] * 8, inbox / "cam1.mp4", fps=4)
    resp = {"summary": "worker near robot", "people": 1, "hazards": [
        {"category": "robot proximity", "severity": "high", "time_s": 1, "description": "inside the cell"}],
        "recommended_actions": ["stop the robot"]}
    day = dt.date(2026, 10, 2)
    md = safety.process_inbox(tmp_path, lambda: VideoReasoner(ScriptedVLM(lambda s, p: resp)), "Cell 3", day)
    assert md == tmp_path / results.SAFETY_REPORTS / "safety-2026-10-02.md"
    assert "robot proximity" in md.read_text() and not list(inbox.glob("*.mp4"))
    assert (tmp_path / results.SAFETY_PROCESSED / "2026-10-02" / "cam1.mp4").is_file()
    page = results.render_safety_report(tmp_path, "2026-10-02")
    assert "Needs attention today" in page and "<video" in page


def test_safety_empty_inbox_never_builds_reasoner(tmp_path):
    def boom():
        raise AssertionError("should not wake the endpoint")

    md = safety.process_inbox(tmp_path, boom, "Cell 3", dt.date(2026, 10, 3))
    assert "Clips reviewed:** 0" in md.read_text()


# ------------------------------------------------------------------------------------------------ cosmos spec

def test_cosmos_spec_is_self_hosted(monkeypatch):
    monkeypatch.setenv("ARMLAB_COSMOS_URL", "https://example.invalid/v1")
    monkeypatch.delenv("ARMLAB_COSMOS_MODEL", raising=False)
    v = make_vlm("cosmos")
    assert v.model == "nvidia/Cosmos3-Nano" and str(v.client.base_url).startswith("https://example.invalid/v1")
    monkeypatch.setenv("ARMLAB_COSMOS_MODEL", "nvidia/Cosmos-Reason2-8B")
    assert make_vlm("cosmos").model == "nvidia/Cosmos-Reason2-8B"
    assert make_vlm("cosmos:other/model@https://x.invalid/v1").model == "other/model"
    with pytest.raises(ValueError, match="gone"):
        make_vlm("nvidia:nvidia/cosmos-reason2-8b")
