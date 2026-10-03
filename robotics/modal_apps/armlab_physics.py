"""Physics-plausibility benchmark on Modal: render the labelled sim dataset and score judge models on it.

Model keys stay in Modal secrets (`armlab-llm` for Claude/GPT; Cosmos is the self-hosted `cosmos-reason` app), so the
launching machine needs only a Modal token. Everything lands on the `armlab-runs` volume under `_physics/`:

  _physics/<dataset>/{*.mp4,manifest.jsonl}       labelled clips (plausible: drop/place/carry; implausible: the rest)
  _physics/<dataset>/eval-<tag>.json              metrics + per-clip verdicts for one judge model

  cd robotics
  modal run modal_apps/armlab_physics.py --make --per-scenario 5                       # render the dataset (CPU)
  modal run modal_apps/armlab_physics.py --vlm cosmos                                  # Cosmos3-Nano, native video
  modal run modal_apps/armlab_physics.py --vlm cosmos --mode frames                    # Cosmos3-Nano, sampled frames
  modal run modal_apps/armlab_physics.py --vlm cosmos --mode native --fps 12           # denser video sampling
  modal run modal_apps/armlab_physics.py --vlm anthropic:claude-opus-5-5:medium        # Claude (frames)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal

# BASE_NOTE: pin a bookworm base (bullseye apt mirrors are gone); same image recipe as armlab_eval.py.
MENAGERIE_COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .apt_install("git", "ffmpeg", "libosmesa6", "libgl1", "fonts-dejavu-core")
    .run_commands(
        "mkdir -p /assets/mujoco_menagerie && cd /assets/mujoco_menagerie && git init -q"
        " && git remote add origin https://github.com/google-deepmind/mujoco_menagerie"
        " && git sparse-checkout set franka_emika_panda"
        f" && git fetch -q --depth 1 origin {MENAGERIE_COMMIT} && git checkout -q FETCH_HEAD"
    )
    .uv_pip_install("mujoco>=3.3", "numpy", "pillow", "imageio", "imageio-ffmpeg", "httpx",
                    "anthropic>=0.60", "openai>=1.60", "modal")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "ARMLAB_ASSETS": "/assets"})
    .add_local_python_source("armlab")
)

app = modal.App("armlab-physics", image=image)
runs = modal.Volume.from_name("armlab-runs", create_if_missing=True)
secrets = [modal.Secret.from_name("armlab-llm")]
if os.environ.get("ARMLAB_VLLM_KEY"):
    secrets.append(modal.Secret.from_dict({"ARMLAB_VLLM_KEY": os.environ["ARMLAB_VLLM_KEY"]}))


@app.function(volumes={"/runs": runs}, timeout=3600, cpu=4.0, memory=8192)
def make_dataset(dataset: str, per_scenario: int) -> str:
    from armlab.cosmos.physics_filter import make_dataset as _make

    out = Path("/runs/_physics") / dataset
    _make(out, per_scenario)
    runs.commit()
    return (out / "manifest.jsonl").read_text()


@app.function(volumes={"/runs": runs}, secrets=secrets, timeout=3 * 3600, cpu=2.0, memory=4096)
def evaluate(dataset: str, vlm: str, mode: str, fps: float = 4.0) -> dict:
    from armlab.cosmos.physics_filter import evaluate as _evaluate
    from armlab.cosmos.reason import make_reasoner

    runs.reload()
    src = Path("/runs/_physics") / dataset
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", vlm) + (f"-{mode}" if mode else "") + (f"-fps{fps:g}" if fps != 4.0 else "")
    out = src / f"eval-{tag}.json"
    _evaluate(src, make_reasoner(vlm, mode=mode or None, fps=fps), out)
    runs.commit()
    return json.loads(out.read_text())


def table(doc: dict) -> str:
    m, res = doc["metrics"], doc["results"]
    lines = [f"model {m.get('model')}  mode {m.get('mode')}  n={m['n']}  errors={m.get('errors', 0)}",
             f"accuracy {m['accuracy']:.2f}  reject precision {m['reject_precision']:.2f}  "
             f"reject recall {m['reject_recall']:.2f}  false-reject rate {m['false_reject_rate']:.2f}",
             f"{'scenario':<12} {'label':<11} {'acc':>5}"]
    labels = {r["scenario"]: ("plausible" if r["plausible"] else "IMPLAUSIBLE") for r in res}
    for k, v in m["per_scenario_accuracy"].items():
        lines.append(f"{k:<12} {labels[k]:<11} {v:>5.2f}")
    lat = [r["latency_s"] for r in res if r.get("latency_s")]
    if lat:
        lines.append(f"mean latency per clip {sum(lat) / len(lat):.1f} s")
    return "\n".join(lines)


@app.local_entrypoint()
def main(vlm: str = "cosmos", mode: str = "", dataset: str = "physics-v1", make: bool = False, per_scenario: int = 5,
         fps: float = 4.0):
    if make:
        manifest = make_dataset.remote(dataset, per_scenario)
        print(f"rendered {len(manifest.splitlines())} clips into armlab-runs:_physics/{dataset}")
        return
    print(table(evaluate.remote(dataset, vlm, mode, fps)))
