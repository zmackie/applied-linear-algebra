"""Daily workplace-safety report on a schedule: clips on the `armlab-runs` volume -> `armlab-safety` -> report
on the results page (/safety).

Drop clips in the inbox any time; each morning the job analyzes them with the self-hosted Cosmos endpoint
(`modal deploy modal_apps/cosmos_reason_vllm.py` first), writes `_safety/reports/safety-<date>.md` and moves the
clips to `_safety/clips/processed/<date>/`. An empty inbox still produces a (short) report and never wakes the GPU.

  modal volume put armlab-runs path/to/clips/ _safety/clips/inbox/
  modal deploy modal_apps/armlab_safety_cron.py
  modal run modal_apps/armlab_safety_cron.py          # run once now instead of waiting for the schedule

Config at deploy time: ARMLAB_SAFETY_VLM (default `cosmos`), ARMLAB_SAFETY_SITE, ARMLAB_SAFETY_CRON
(default "0 7 * * *" America/New_York). Keys: only from Modal secrets (`armlab-llm`, and
`armlab-vllm` for the Cosmos endpoint's bearer key).
"""
import os

import modal

# BASE_NOTE: workspaces created before 2025 build `debian_slim` on Debian bullseye, whose apt mirrors went
# away when bullseye LTS ended (Aug 2026), so apt_install 404s. Pin a bookworm base explicitly.

VLM = os.environ.get("ARMLAB_SAFETY_VLM", "cosmos")
SITE = os.environ.get("ARMLAB_SAFETY_SITE", "Robot work cell")
CRON = os.environ.get("ARMLAB_SAFETY_CRON", "0 7 * * *")

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")  # see BASE_NOTE
    .apt_install("ffmpeg")
    .uv_pip_install("numpy", "pillow", "imageio", "imageio-ffmpeg", "httpx", "anthropic>=0.60", "openai>=1.60", "modal")
    .env({"ARMLAB_SAFETY_VLM": VLM, "ARMLAB_SAFETY_SITE": SITE})
    .add_local_python_source("armlab")
)
app = modal.App("armlab-safety-cron", image=image)
runs = modal.Volume.from_name("armlab-runs", create_if_missing=True)
secrets = [modal.Secret.from_name("armlab-llm"), modal.Secret.from_name("armlab-vllm")]  # model keys + Cosmos key


@app.function(volumes={"/runs": runs}, secrets=secrets, timeout=2 * 3600,
              schedule=modal.Cron(CRON, timezone="America/New_York"))
def daily_report() -> str:
    from armlab.cosmos.reason import make_reasoner
    from armlab.cosmos.safety import process_inbox

    runs.reload()
    vlm, site = os.environ["ARMLAB_SAFETY_VLM"], os.environ["ARMLAB_SAFETY_SITE"]
    md = process_inbox("/runs", lambda: make_reasoner(vlm), site)
    runs.commit()
    print(f"wrote {md}")
    return md.read_text()


@app.local_entrypoint()
def main():
    print(daily_report.remote())
