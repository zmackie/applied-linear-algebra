"""Run MuJoCo arm-sim evaluations on Modal, one container per episode (CPU only).

Episodes spend most of their time waiting on the model API, so fanning them out is the cheap way to run 50+
episodes. Model keys come only from the Modal secret `armlab-llm` (ANTHROPIC_API_KEY, optional OPENAI_API_KEY);
the machine you launch from needs just a Modal token. Everything a run produces lands on the `armlab-runs`
volume, which the results page (`modal_apps/armlab_results.py`) serves:

  armlab-runs:/<run>/summary.txt, results.json, results.csv, config.json, <task>/seed<k>/{video.mp4,decisions.jsonl,result.json}

  cd robotics
  modal run modal_apps/armlab_eval.py --policy oracle --tasks all --seeds 0 --video           # no model keys used
  modal run modal_apps/armlab_eval.py --policy direct --vlm anthropic:claude-opus-5-5:medium --tasks all --seeds 0-4 --video
  modal run modal_apps/armlab_eval.py --policy oracle --seeds 0 --video --download            # also copy videos back locally
  modal run modal_apps/armlab_eval.py --check-keys                                            # which keys armlab-llm holds (names only)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# BASE_NOTE: workspaces created before 2025 build `debian_slim` on Debian bullseye, whose apt mirrors went
# away when bullseye LTS ended (Aug 2026), so apt_install 404s. Pin a bookworm base explicitly.

MENAGERIE_COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
KEY_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")  # see BASE_NOTE
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
llm = modal.Secret.from_name("armlab-llm")


def _run_one(task, seed, policy, vlm, clock, obs_mode, video, run) -> dict:
    from dataclasses import asdict

    from armlab.eval.runner import run_episode

    res = run_episode(task, seed, policy, vlm if policy in ("direct", "hybrid") else None, clock, obs_mode, video,
                      run, out_dir=Path("/runs") / run)
    # Store the volume-relative video path so the results page and downloads can find it.
    if res.video:
        res.video = str(Path(res.video).relative_to("/runs"))
        (Path("/runs") / run / task / f"seed{seed}" / "result.json").write_text(
            json.dumps(asdict(res) | {"correction_fraction": res.correction_fraction}, indent=2))
    runs.commit()
    return asdict(res) | {"correction_fraction": res.correction_fraction}


@app.function(secrets=[llm], volumes={"/runs": runs}, timeout=3 * 3600, cpu=2.0, memory=4096, retries=0)
def episode(task: str, seed: int, policy: str, vlm: str, clock: str, obs_mode: str, video: bool, run: str) -> dict:
    return _run_one(task, seed, policy, vlm, clock, obs_mode, video, run)


@app.function(volumes={"/runs": runs}, timeout=1800, cpu=2.0, memory=4096, retries=0)
def episode_nokeys(task: str, seed: int, policy: str, vlm: str, clock: str, obs_mode: str, video: bool, run: str) -> dict:
    """oracle / system1 episodes: no model is called, so no secret is mounted."""
    return _run_one(task, seed, policy, vlm, clock, obs_mode, video, run)


@app.function(volumes={"/runs": runs}, timeout=300)
def finalize(run: str, files: dict[str, str]) -> None:
    """Write the run-level summary files onto the volume next to the episode folders."""
    out = Path("/runs") / run
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / Path(name).name).write_text(text)
    runs.commit()


@app.function(secrets=[llm], timeout=120)
def key_status() -> dict:
    """Which model keys the armlab-llm secret provides. Returns booleans only, never values."""
    return {k: bool(os.environ.get(k)) for k in KEY_NAMES}


def _download(run: str, dest: Path) -> int:
    n = 0
    for entry in runs.listdir(run, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = dest / Path(entry.path).relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            runs.read_file_into_fileobj(entry.path, f)
        n += 1
    return n


@app.local_entrypoint()
def main(policy: str = "oracle", vlm: str = "anthropic:claude-opus-5-5:medium", tasks: str = "all", seeds: str = "0-4",
         clock: str = "paused", obs_mode: str = "vision", video: bool = False, run: str = "", download: bool = False,
         check_keys: bool = False):
    import time

    from armlab.eval.runner import RUNS_DIR, EpisodeResult, parse_seeds, summarize, write_results_csv
    from armlab.sim.tasks import TASKS

    if check_keys:
        for k, present in key_status.remote().items():
            print(f"{k}: {'present' if present else 'missing'}")
        return
    run = run or time.strftime(f"%Y%m%d-%H%M%S-{policy}-modal")
    task_list = list(TASKS) if tasks == "all" else tasks.split(",")
    unknown = [t for t in task_list if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; choose from {list(TASKS)}")
    jobs = [(t, s, policy, vlm, clock, obs_mode, video, run) for t in task_list for s in parse_seeds(seeds)]
    fn = episode if policy in ("direct", "hybrid") else episode_nokeys
    print(f"Launching {len(jobs)} episodes as run {run!r}")
    results = []
    for r in fn.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("episode failed:", r)
            continue
        results.append(r)
        print(f"{r['task']} seed {r['seed']}: {'SUCCESS' if r['success'] else 'fail'} score {r['score']:.0f} {r['error']}")
    out = RUNS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    config = {"policy": policy, "vlm": vlm if policy in ("direct", "hybrid") else "", "tasks": tasks, "seeds": seeds,
              "clock": clock, "obs_mode": obs_mode, "video": video, "launched_from": "modal"}
    fields = EpisodeResult.__dataclass_fields__
    eps = [EpisodeResult(**{k: v for k, v in r.items() if k in fields}) for r in results]
    summary = summarize(eps) if eps else "no episodes completed"
    files = {"config.json": json.dumps(config, indent=2), "results.json": json.dumps(results, indent=2),
             "summary.txt": summary + "\n"}
    if eps:
        write_results_csv(out / "results.csv", eps)
        files["results.csv"] = (out / "results.csv").read_text()
    for name, text in files.items():
        (out / name).write_text(text)
    finalize.remote(run, files)
    print("\n" + summary)
    if download:
        print(f"Downloaded {_download(run, out)} files from the volume into {out}")
    else:
        print(f"\nVideos and logs are on the volume: modal volume get armlab-runs {run} {out}")
    try:
        url = f'{modal.Function.from_name("armlab-results", "web").get_web_url()}/runs/{run}'
        (out / "results_url.txt").write_text(url + "\n")  # read by the GitHub workflow (console output may wrap)
        print(f"Results page: {url}")
    except Exception:
        print("Results page: deploy it with `modal deploy modal_apps/armlab_results.py`")
    if not results:
        raise SystemExit(1)
