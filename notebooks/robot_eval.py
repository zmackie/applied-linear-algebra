import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo

    ROBOTICS = Path(__file__).resolve().parent.parent / "robotics"
    sys.path.insert(0, str(ROBOTICS))
    RUNS = ROBOTICS / "runs"
    return RUNS, json, mo


@app.cell
def _(mo):
    mo.md(
        r"""
        # Robot-arm evaluations

        Browse runs produced by `armlab-eval` (local) or `modal_apps/armlab_eval.py` (Modal).
        Each run holds one folder per task and seed with `video.mp4`, `decisions.jsonl` and `result.json`.

        Quick start (from `robotics/`):

        ```sh
        armlab-eval --policy oracle --tasks all --seeds 0 --video --run-name oracle-demo
        armlab-eval --policy direct --vlm anthropic:claude-opus-5-5:medium --tasks blocks_into_bin --seeds 0-2 --video
        ```
        """
    )
    return


@app.cell
def _(RUNS, mo):
    runs = sorted([p.name for p in RUNS.glob("*") if p.is_dir()], reverse=True) if RUNS.exists() else []
    run_pick = mo.ui.dropdown(runs, value=runs[0] if runs else None, label="Run")
    run_pick
    return (run_pick,)


@app.cell
def _(RUNS, json, mo, run_pick):
    mo.stop(run_pick.value is None, mo.md("No runs yet. Run `armlab-eval` first."))
    run_dir = RUNS / run_pick.value
    episodes = []
    for res in sorted(run_dir.glob("*/seed*/result.json")):
        r = json.loads(res.read_text())
        r["dir"] = str(res.parent)
        episodes.append(r)
    summary = (run_dir / "summary.txt").read_text() if (run_dir / "summary.txt").exists() else ""
    rows = [{"task": e["task"], "seed": e["seed"], "success": e["success"], "score": e["score"],
             "decisions": e["decisions"], "model s": e["model_latency_s"],
             "tokens": e["input_tokens"] + e["output_tokens"], "corr %": round(100 * e.get("correction_fraction", 0), 1),
             "error": e["error"]} for e in episodes]
    mo.vstack([mo.md(f"```\n{summary}\n```") if summary else mo.md(""), mo.ui.table(rows, selection=None)])
    return (episodes,)


@app.cell
def _(episodes, mo):
    labels = [f"{e['task']} / seed {e['seed']}" for e in episodes]
    ep_pick = mo.ui.dropdown(labels, value=labels[0] if labels else None, label="Episode")
    ep_pick
    return ep_pick, labels


@app.cell
def _(episodes, ep_pick, json, labels, mo):
    from pathlib import Path as _P

    mo.stop(ep_pick.value is None)
    ep = episodes[labels.index(ep_pick.value)]
    video = _P(ep["dir"]) / "video.mp4"
    log = [json.loads(line) for line in (_P(ep["dir"]) / "decisions.jsonl").read_text().splitlines() if line.strip()]
    steps = [{"#": d["decision"], "source": d["source"], "think s": d["think_s"], "score": d["score"],
              "rationale": d["rationale"][:140]} for d in log]
    mo.vstack([
        mo.video(src=str(video), controls=True, width=900) if video.exists() else mo.md("_No video for this episode (run with `--video`)._"),
        mo.ui.table(steps, selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
