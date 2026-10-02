"""Run MuJoCo arm-sim evaluations on Modal, one container per episode (CPU only).

Episodes spend most of their time waiting on the model API, so fanning them out is the cheap way
to run 50+ episodes. Needs a Modal secret named `armlab-llm` holding whichever keys you use
(ANTHROPIC_API_KEY, OPENAI_API_KEY, NVIDIA_API_KEY).

  cd robotics
  modal run modal_apps/armlab_eval.py --policy direct --vlm anthropic:claude-opus-5-5:medium --tasks all --seeds 0-4 --video
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

MENAGERIE_COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libosmesa6", "libgl1", "fonts-dejavu-core")
    .run_commands(
        "mkdir -p /assets/mujoco_menagerie && cd /assets/mujoco_menagerie && git init -q"
        " && git remote add origin https://github.com/google-deepmind/mujoco_menagerie"
        " && git sparse-checkout set franka_emika_panda"
        f" && git fetch -q --depth 1 origin {MENAGERIE_COMMIT} && git checkout -q FETCH_HEAD"
    )
    .uv_pip_install("mujoco>=3.3", "numpy", "pillow", "imageio", "imageio-ffmpeg", "httpx",
                    "anthropic>=0.60", "openai>=1.60")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "ARMLAB_ASSETS": "/assets"})
    .add_local_python_source("armlab")
)

app = modal.App("armlab-eval", image=image)
runs = modal.Volume.from_name("armlab-runs", create_if_missing=True)


@app.function(secrets=[modal.Secret.from_name("armlab-llm")], volumes={"/runs": runs}, timeout=3 * 3600, cpu=2.0,
              memory=4096, retries=0)
def episode(task: str, seed: int, policy: str, vlm: str, clock: str, obs_mode: str, video: bool, run: str) -> dict:
    from dataclasses import asdict

    from armlab.eval.runner import run_episode

    res = run_episode(task, seed, policy, vlm if policy in ("direct", "hybrid") else None, clock, obs_mode, video,
                      run, out_dir=Path("/runs") / run)
    runs.commit()
    return asdict(res) | {"correction_fraction": res.correction_fraction}


@app.local_entrypoint()
def main(policy: str = "direct", vlm: str = "anthropic:claude-opus-5-5:medium", tasks: str = "all", seeds: str = "0-4",
         clock: str = "paused", obs_mode: str = "vision", video: bool = False, run: str = ""):
    import time

    from armlab.eval.runner import RUNS_DIR, EpisodeResult, parse_seeds, summarize
    from armlab.sim.tasks import TASKS

    run = run or time.strftime(f"%Y%m%d-%H%M%S-{policy}-modal")
    task_list = list(TASKS) if tasks == "all" else tasks.split(",")
    jobs = [(t, s, policy, vlm, clock, obs_mode, video, run) for t in task_list for s in parse_seeds(seeds)]
    print(f"Launching {len(jobs)} episodes as run {run!r}")
    results = []
    for r in episode.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("episode failed:", r)
            continue
        results.append(r)
        print(f"{r['task']} seed {r['seed']}: {'SUCCESS' if r['success'] else 'fail'} score {r['score']:.0f} {r['error']}")
    out = RUNS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    fields = EpisodeResult.__dataclass_fields__
    summary = summarize([EpisodeResult(**{k: v for k, v in r.items() if k in fields}) for r in results])
    (out / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nVideos and logs are on the volume: modal volume get armlab-runs {run} {out}")
