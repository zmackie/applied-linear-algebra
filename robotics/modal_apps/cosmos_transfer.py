"""Run Cosmos-Transfer2.5 on exported sim controls (sim-to-real variations) on a Modal H100.

Needs a Modal secret `huggingface` with HF_TOKEN; accept the NVIDIA Open Model License for
nvidia/Cosmos-Transfer2.5-2B and nvidia/Cosmos-Guardrail1 on Hugging Face first. The model needs
~65 GB of VRAM, so this uses an 80 GB H100. Expect roughly 5-10 minutes per 93-frame clip.

  python -m armlab.cosmos.transfer_export --task blocks_into_bin --seed 0 --out exports/blocks0 --variations 3
  modal run modal_apps/cosmos_transfer.py --export-dir exports/blocks0
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = "https://github.com/nvidia-cosmos/cosmos-transfer2.5"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "git-lfs", "ffmpeg", "curl", "libx11-dev", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install uv",
        f"git clone --depth 1 {REPO} /opt/ct && cd /opt/ct && git lfs install && git lfs pull",
        "cd /opt/ct && uv sync --extra=cu128",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
app = modal.App("cosmos-transfer25", image=image)
hf = modal.Volume.from_name("hf-cache-transfer", create_if_missing=True)
outputs = modal.Volume.from_name("cosmos-transfer-outputs", create_if_missing=True)


@app.function(gpu="H100", secrets=[modal.Secret.from_name("huggingface")], volumes={"/hf": hf, "/outputs": outputs},
              timeout=2 * 3600, memory=65536)
def transfer(files: dict[str, bytes], spec: dict) -> dict[str, bytes]:
    import subprocess
    import tempfile

    work = Path(tempfile.mkdtemp())
    for name, data in files.items():
        (work / name).write_bytes(data)
    # Make every path in the spec absolute inside the container.
    spec = json.loads(json.dumps(spec))
    spec["video_path"] = str(work / spec["video_path"])
    for key in ("depth", "seg", "edge", "vis"):
        if key in spec and "control_path" in spec[key]:
            spec[key]["control_path"] = str(work / spec[key]["control_path"])
    spec_path = work / "spec.json"
    spec_path.write_text(json.dumps(spec))
    out_dir = Path("/outputs") / spec.get("name", "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["/opt/ct/.venv/bin/python", "examples/inference.py", "-i", str(spec_path), "-o", str(out_dir)],
                   cwd="/opt/ct", check=True)
    outputs.commit()
    hf.commit()
    return {p.name: p.read_bytes() for p in out_dir.rglob("*.mp4")}


@app.local_entrypoint()
def main(export_dir: str, specs: str = ""):
    src = Path(export_dir)
    files = {n: (src / n).read_bytes() for n in ("input_rgb.mp4", "control_depth.mp4", "control_seg.mp4") if (src / n).exists()}
    spec_files = [src / s for s in specs.split(",")] if specs else sorted(src.glob("spec_v*.json"))
    jobs = [(files, json.loads(p.read_text())) for p in spec_files]
    print(f"Running {len(jobs)} Transfer jobs in parallel on H100s...")
    out = src / "transfer_out"
    out.mkdir(exist_ok=True)
    for spec_path, result in zip(spec_files, transfer.starmap(jobs)):
        for name, data in result.items():
            dest = out / f"{spec_path.stem}_{name}"
            dest.write_bytes(data)
            print("wrote", dest)
