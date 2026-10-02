"""Serve Cosmos-Reason2 as an OpenAI-compatible endpoint on Modal (vLLM).

Needs a Modal secret `huggingface` with HF_TOKEN (accept the model license on Hugging Face first).
If ARMLAB_VLLM_KEY is set in your shell when you deploy, the endpoint requires that key.

  export ARMLAB_VLLM_KEY=$(openssl rand -hex 16)
  modal deploy modal_apps/cosmos_reason_vllm.py
  # then use the printed URL:
  python -m armlab.cosmos.safety clips/ --vlm "vllm:nvidia/Cosmos-Reason2-8B@https://<workspace>--cosmos-reason2-serve.modal.run/v1"
"""
import os
import subprocess

import modal

MODEL = os.environ.get("COSMOS_REASON_MODEL", "nvidia/Cosmos-Reason2-8B")
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("vllm>=0.11.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("cosmos-reason2", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    gpu="L40S",  # 48 GB is plenty for the 8B model in bf16; use "H100" for lower latency
    secrets=[modal.Secret.from_name("huggingface")]
    + ([modal.Secret.from_dict({"ARMLAB_VLLM_KEY": os.environ["ARMLAB_VLLM_KEY"]})] if os.environ.get("ARMLAB_VLLM_KEY") else []),
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    timeout=60 * 60,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--max-model-len", "16384",
        "--media-io-kwargs", '{"video": {"num_frames": -1}}',
        "--reasoning-parser", "qwen3",
        "--served-model-name", MODEL,
    ]
    if os.environ.get("ARMLAB_VLLM_KEY"):
        cmd += ["--api-key", os.environ["ARMLAB_VLLM_KEY"]]
    subprocess.Popen(cmd)
