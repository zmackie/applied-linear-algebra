"""Run RoboLab (NVIDIA Isaac Lab benchmark) with an armlab VLM policy on a Modal L40S.

UNTESTED end to end: this environment had no GPU or Modal account. It follows RoboLab's own
Dockerfile (nvcr.io/nvidia/isaac-lab:2.2.0 base, `pip install -e .` with Isaac's python) and its
documented run flags. Expect to iterate on the first build.

Needs a Modal secret `armlab-llm` with ANTHROPIC_API_KEY and/or OPENAI_API_KEY.

  modal run modal_apps/robolab_eval.py --tasks BananaInBowlTask,RubiksCubeAndBananaTask --vlm anthropic:claude-opus-5-5:medium
  modal run modal_apps/robolab_eval.py --tasks paper10 --num-runs 5     # the report's 10-task subset
  modal volume get robolab-runs <run-name> runs/robolab/
"""
from __future__ import annotations

import modal

ISAAC_PY = "/workspace/isaaclab/_isaac_sim/python.sh"

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-lab:2.2.0", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "git-lfs", "ffmpeg")
    .run_commands(
        "git lfs install",
        "git clone --depth 1 https://github.com/NVlabs/RoboLab /workspace/robolab",
        "cd /workspace/robolab && git lfs pull",  # ~7 GB of USD assets
        f"cd /workspace/robolab && {ISAAC_PY} -m pip install --no-cache-dir -e .",
        f"{ISAAC_PY} -m pip install --no-cache-dir 'anthropic>=0.60' 'openai>=1.60' pillow imageio",
    )
    # Kit preloads the typing_extensions bundled with its extensions (pip_prebundle), which is too old for current
    # anthropic SDKs (TypedDict(extra_items=...)), so `import anthropic` fails inside Isaac Sim. typing_extensions is
    # backwards compatible: overwrite the bundled copies with the one pip just installed.
    .run_commands(
        f"TE=$(realpath $({ISAAC_PY} -c 'import typing_extensions as t; print(t.__file__)' | tail -1))"
        " && echo using $TE && find $(realpath /isaac-sim) -name typing_extensions.py -type f -not -path \"$TE\" -print"
        " -exec cp \"$TE\" {} \\;",
    )
    .env({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PYTHONPATH": "/workspace/armlab_src",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir("armlab", "/workspace/armlab_src/armlab", ignore=["**/__pycache__"])
)

app = modal.App("armlab-robolab", image=image)
runs = modal.Volume.from_name("robolab-runs", create_if_missing=True)


@app.function(gpu="L40S", secrets=[modal.Secret.from_name("armlab-llm")], volumes={"/runs": runs},
              timeout=6 * 3600, memory=32768)
def evaluate(tasks: list[str], vlm: str, num_runs: int, run: str, extra: list[str]) -> str:
    import shutil
    import subprocess
    from pathlib import Path

    cmd = [ISAAC_PY, "-m", "armlab.robolab.run_llm_eval", "--headless", "--enable-gt-state", "--vlm", vlm,
           "--num-runs", str(num_runs), "--output-folder-name", run, "--log-file", f"/runs/{run}_decisions.jsonl",
           "--task", *tasks, *extra]
    print(" ".join(cmd), flush=True)
    log_path = Path("/runs") / f"{run}.log"
    # Isaac Sim is chatty (thousands of deprecation warnings); keep everything in the log file on the volume and
    # echo only the interesting lines here.
    keep = ("[RoboLab]", "[armlab]", "Traceback", "Error", "error:", "Exception", "  File ", "success", "Success")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd="/workspace/robolab", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            if any(k in line for k in keep) and "[Warning]" not in line:
                print(line.rstrip()[:400], flush=True)
        proc.wait()
    out = Path("/workspace/robolab/output")
    for d in out.glob(f"*{run}*"):
        shutil.copytree(d, Path("/runs") / d.name, dirs_exist_ok=True)
    kit_logs = Path("/isaac-sim/kit/logs")
    if kit_logs.exists():
        shutil.copytree(kit_logs, Path("/runs") / f"{run}_kit_logs", dirs_exist_ok=True)
    runs.commit()
    return f"exit code {proc.returncode}; outputs copied to volume robolab-runs ({run}/, {run}.log)"


# The 10 RoboLab tasks GPT-6 Astra was evaluated on in the report (Table 2), mapped to RoboLab task classes.
PAPER_TASKS = {
    "blocks into bin": "BlocksInBinTask",
    "pumpkins in clutter": "ClutterPumpkinTask",
    "butter on raisin box": "ButterAboveRaisinTask",
    "stack blocks in order": "BlockStackingSpecifiedOrderTask",
    "reorient red mug": "ReorientRedMugTask",
    "larger raisin box into bin": "LargerObjectRaisinBoxInBinTask",
    "sauce bottle into crate": "SauceBottlesCrateTask",
    "canned food into bin": "CannedFoodInBinTask",
    "yogurt into bowl": "YogurtInBowlTask",
    "rubik's cube into bowl": "RubiksCubeTask",
}


@app.function(volumes={"/runs": runs}, timeout=600)
def collect(run_names: list[str]) -> list[dict]:
    """Episode rows (RoboLab's episode_results.jsonl) for the given run folders on the volume."""
    import json
    from pathlib import Path

    runs.reload()
    rows = []
    for name in run_names:
        f = Path("/runs") / name / "episode_results.jsonl"
        if f.exists():
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def summary_table(rows: list[dict], task_order: list[str]) -> str:
    names = {v: k for k, v in PAPER_TASKS.items()}
    lines = [f"{'task':<34} {'success':>8} {'mean score':>10} {'policy s/ep':>11}"]
    tot_s = tot_n = 0
    for t in task_order:
        rs = [r for r in rows if r.get("task_name") == t or r.get("env_name") == t]
        if not rs:
            lines.append(f"{t:<34} {'no episodes':>8}")
            continue
        k = sum(bool(r["success"]) for r in rs)
        tot_s, tot_n = tot_s + k, tot_n + len(rs)
        score = sum(float(r.get("score") or 0) for r in rs) / len(rs)
        pol = sum(r.get("timing", {}).get("policy_inference_s", 0) for r in rs) / len(rs)
        label = f"{t} ({names[t]})" if t in names else t
        lines.append(f"{label[:34]:<34} {f'{k}/{len(rs)}':>8} {score:>10.2f} {pol:>11.0f}")
    lines.append(f"{'OVERALL':<34} {f'{tot_s}/{tot_n}':>8}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(tasks: str = "BananaInBowlTask", vlm: str = "anthropic:claude-opus-5-5:medium", num_runs: int = 5,
         run: str = "", extra: str = "", parallel: bool = True):
    """--tasks paper10 runs the report's 10-task subset. With --parallel (default) each task gets its own L40S
    container and output folder <run>-<task>; otherwise all tasks run sequentially in one container."""
    import time

    run = run or time.strftime("%Y%m%d-%H%M%S-armlab")
    task_list = list(PAPER_TASKS.values()) if tasks == "paper10" else tasks.split(",")
    xs = extra.split() if extra else []
    if parallel and len(task_list) > 1:
        names = [f"{run}-{t}" for t in task_list]
        for name, msg in zip(names, evaluate.starmap([([t], vlm, num_runs, n, xs) for t, n in zip(task_list, names)],
                                                     return_exceptions=True)):
            print(name, msg)
    else:
        names = [run]
        print(evaluate.remote(task_list, vlm, num_runs, run, xs))
    print("\n" + summary_table(collect.remote(names), task_list))
    print(f"\nDownload: modal volume get robolab-runs <name> runs/robolab/   (names: {', '.join(names)})")
