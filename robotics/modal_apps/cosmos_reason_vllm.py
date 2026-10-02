"""Serve a Cosmos reasoning model as an OpenAI-compatible endpoint on Modal (vLLM).

Default model: nvidia/Cosmos3-Nano (16B Mixture-of-Transformers) loaded as the text "Reasoner" through NVIDIA's
`vllm-cosmos3` plugin (vLLM 0.19.1 on CUDA 12.8). It replaces Cosmos Reason 2; the hosted build.nvidia.com API
is gone, so the safety monitor, ego video API and physics filter all call this endpoint (`--vlm cosmos`).

Keys: Modal secret `huggingface` with HF_TOKEN (required for all Cosmos work; Cosmos3-Nano is OpenMDW-1.1 and not
gated, Cosmos-Reason2 needs the NVIDIA Open Model License accepted on its HF page). If ARMLAB_VLLM_KEY is set in your
shell when you deploy, the endpoint requires that key as a bearer token.

  modal deploy modal_apps/cosmos_reason_vllm.py                                  # Cosmos3-Nano on an L40S
  ARMLAB_COSMOS_GPU=H100 modal deploy modal_apps/cosmos_reason_vllm.py           # faster / more KV-cache headroom
  ARMLAB_COSMOS_MODEL=nvidia/Cosmos-Reason2-8B modal deploy modal_apps/cosmos_reason_vllm.py   # previous generation
  armlab-safety clips/ --vlm cosmos           # finds https://<workspace>--cosmos-reason-serve.modal.run/v1 itself

Swap in another Cosmos 3 reasoner by setting ARMLAB_COSMOS_MODEL at deploy time (and the same value in the client's
environment, or pass `--vlm cosmos:<model>`).
"""
import os
import subprocess

import modal

# BASE_NOTE: workspaces created before 2025 build `debian_slim` on Debian bullseye, whose apt mirrors went
# away when bullseye LTS ended (Aug 2026), so apt_install 404s. Pin a bookworm base explicitly.
# Python 3.11 because the old builder's pinned Modal client deps (aiohttp) do not build on 3.12.

MODEL = os.environ.get("ARMLAB_COSMOS_MODEL") or os.environ.get("COSMOS_REASON_MODEL") or "nvidia/Cosmos3-Nano"
GPU = os.environ.get("ARMLAB_COSMOS_GPU", "L40S")  # 48 GB: Cosmos3-Nano's reasoner weights fit in bf16; H100 for more headroom
PORT = 8000
IS_COSMOS3 = "cosmos3" in MODEL.lower()
COSMOS_FRAMEWORK_COMMIT = "cf5d68c00d97ccd2480a2320ed652b92dec63102"  # NVIDIA/cosmos-framework (vllm-cosmos3 plugin)

if IS_COSMOS3:
    image = (
        modal.Image.from_registry("python:3.11-slim-bookworm")  # see BASE_NOTE
        .apt_install("git", "ffmpeg", "build-essential")  # Triton JIT-compiles a CUDA helper with gcc at startup
        .run_commands(
            "git init -q /opt/cosmos-framework && cd /opt/cosmos-framework"
            " && git remote add origin https://github.com/NVIDIA/cosmos-framework"
            f" && git fetch -q --depth 1 origin {COSMOS_FRAMEWORK_COMMIT} && git checkout -q FETCH_HEAD",
            # Release-tested combo from the Cosmos3-Nano model card for CUDA 12.8 drivers.
            "pip install -q uv && uv pip install --system --torch-backend=cu128 'vllm==0.19.1'"
            " /opt/cosmos-framework/packages/transformers-cosmos3 /opt/cosmos-framework/packages/vllm-cosmos3"
            " 'huggingface_hub[hf_transfer]'",
        )
    )
else:  # Cosmos-Reason2 (Qwen3-VL based) runs on stock vLLM
    image = (
        modal.Image.from_registry("python:3.11-slim-bookworm")  # see BASE_NOTE
        .apt_install("ffmpeg")
        .uv_pip_install("vllm>=0.11.0", "huggingface_hub[hf_transfer]")
    )
# Bake the choice into the image so the container serves the same model the deploy shell picked.
image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "ARMLAB_COSMOS_MODEL": MODEL, "VLLM_USE_DEEP_GEMM": "0"})

app = modal.App(os.environ.get("ARMLAB_COSMOS_APP", "cosmos-reason"), image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def serve_command(model: str, port: int = PORT, api_key: str | None = None) -> list[str]:
    cmd = ["vllm", "serve", model, "--host", "0.0.0.0", "--port", str(port), "--served-model-name", model,
           "--max-model-len", "16384", "--media-io-kwargs", '{"video": {"num_frames": -1}}']
    if "cosmos3" in model.lower():
        cmd += ["--hf-overrides", '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}',
                "--tensor-parallel-size", "1", "--mm-encoder-tp-mode", "data", "--async-scheduling"]
    else:
        cmd += ["--reasoning-parser", "qwen3"]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface")]
    + ([modal.Secret.from_dict({"ARMLAB_VLLM_KEY": os.environ["ARMLAB_VLLM_KEY"]})] if os.environ.get("ARMLAB_VLLM_KEY") else []),
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    timeout=60 * 60,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=25 * 60)
def serve():
    model = os.environ.get("ARMLAB_COSMOS_MODEL", MODEL)
    subprocess.Popen(serve_command(model, api_key=os.environ.get("ARMLAB_VLLM_KEY")))
