"""Run RoboLab (NVIDIA Isaac Lab benchmark) with an armlab VLM policy on a Modal L40S.

UNTESTED end to end: this environment had no GPU or Modal account. It follows RoboLab's own
Dockerfile (nvcr.io/nvidia/isaac-lab:2.2.0 base, `pip install -e .` with Isaac's python) and its
documented run flags. Expect to iterate on the first build.

Needs a Modal secret `armlab-llm` with ANTHROPIC_API_KEY and/or OPENAI_API_KEY.

  modal run modal_apps/robolab_eval.py --tasks BananaInBowlTask,RubiksCubeAndBananaTask --vlm anthropic:claude-opus-5-5:medium
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
    .env({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PYTHONPATH": "/workspace/armlab_src"})
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
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd="/workspace/robolab")
    out = Path("/workspace/robolab/output")
    for d in out.glob(f"*{run}*"):
        shutil.copytree(d, Path("/runs") / d.name, dirs_exist_ok=True)
    runs.commit()
    return f"exit code {proc.returncode}; outputs copied to volume robolab-runs"


@app.local_entrypoint()
def main(tasks: str = "BananaInBowlTask", vlm: str = "anthropic:claude-opus-5-5:medium", num_runs: int = 5,
         run: str = "", extra: str = ""):
    import time

    run = run or time.strftime("%Y%m%d-%H%M%S-armlab")
    print(evaluate.remote(tasks.split(","), vlm, num_runs, run, extra.split() if extra else []))
    print(f"Download: modal volume get robolab-runs {run} runs/robolab/{run}")
