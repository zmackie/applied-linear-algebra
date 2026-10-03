# armlab: robot arms, video models, and testing the hype

A small, runnable testbed for two things:

1. **Testing claims like "GPT-6 Astra cracked RoboLab."** Put any vision-language model (Claude, GPT-6 Astra, Cosmos 3) in a robot-arm control loop, using the two architectures from *GPT 6 Astra as an Embodied Policy* (Su et al., 2026), and measure success, Score, tokens and latency.
2. **Cosmos video apps:** a workplace safety monitor, an egocentric reasoning API, a physics-plausibility filter, and a sim-to-real data generator.

Everything except the GPU pieces runs on a laptop CPU.

## What the paper actually showed (and what this lets you check)

| Claim in the tweet | What the report shows | How to test it here |
|---|---|---|
| "Cracked RoboLab with a near-perfect score" | 49/50 on **10 hand-picked** RoboLab tasks, 5 trials each, zero-shot, sim paused while the model thinks | Run the same "direct" architecture on all 5 local tasks, then on more RoboLab tasks (`modal_apps/robolab_eval.py`) |
| "Infrastructure + scaling beat heuristics" | Not tested in the report. On the bimanual RoboDojo set the *hybrid* (π0.5 + Astra) won: 48% vs 26% | `--policy direct` vs `--policy hybrid` on the same seeds |
| "Brink of physical RSI" | Sim only; authors call latency "a hugely unresolved issue" | `--clock realtime` keeps the world moving while the model thinks (see `media/latency_*.mp4`) |

## Layout

```
armlab/sim/        MuJoCo Franka Panda env: 5 tasks, IK, cameras, captioned video recorder
armlab/policy/     scripted oracle / "System 1", VLM clients, direct + hybrid VLM policies
armlab/eval/       episode loop + `armlab-eval` CLI (paused or realtime clock)
armlab/cosmos/     video reasoner, safety monitor, FastAPI ego API, physics filter, Transfer exporter
armlab/robolab/    RoboLab (Isaac Lab) inference client + runner for the same VLM policies
armlab/web/        phone-friendly results pages (runs, summary tables, videos, safety reports)
armlab/doctor.py   `armlab-doctor`: which credentials and hosts this machine can reach
modal_apps/        Modal apps: parallel sim evals, results page, daily safety cron, RoboLab on L40S, Cosmos 3 vLLM server, Cosmos Transfer on H100
media/             demo videos
../notebooks/robot_eval.py   marimo notebook to browse runs and watch episodes
```

## Setup

```sh
# from the repo root
uv venv && uv pip install -e "robotics[llm,api,cloud,dev]"
cd robotics
pytest                      # ~1 min; all CPU, no API keys needed
```

The first run downloads the Franka model from MuJoCo Menagerie (pinned commit) into `robotics/.assets/`.
Rendering is headless via OSMesa (`apt install libosmesa6` on Linux). Set `MUJOCO_GL=egl` on a GPU box, or `MUJOCO_GL=glfw` on macOS.

## 1. Arm sim + VLM policies

Tasks (`armlab/sim/tasks.py`), modelled on the RoboLab subset in the report:

| task | what it tests |
|---|---|
| `blocks_into_bin` | basic pick and place |
| `stack_in_order` | ordered stacking (procedural) |
| `larger_into_bin` | size comparison (relational) |
| `sort_warm_cool` | semantic classification into two bins |
| `conveyor_pick` | a moving target: the latency test |

```sh
# no model, just to see it work (writes runs/<name>/<task>/seed<k>/video.mp4)
armlab-eval --policy oracle --tasks all --seeds 0 --video

# the report's "direct" architecture: the model emits end-effector waypoints from images + proprioception
export ANTHROPIC_API_KEY=...
armlab-eval --policy direct --vlm anthropic:claude-opus-5-5:medium --tasks blocks_into_bin --seeds 0-4 --video

# GPT-6 Astra (or any OpenAI model): use the model id from OpenAI's docs
export OPENAI_API_KEY=...
armlab-eval --policy direct --vlm openai:<gpt-6-astra-model-id> --tasks all --seeds 0-4

# the report's hybrid: a fluent but error-prone System-1 proposes, the model accepts or corrects
armlab-eval --policy hybrid --vlm anthropic --tasks all --seeds 0-4

# no pause button: the world keeps moving for as long as the model thinks
armlab-eval --policy direct --vlm anthropic --tasks conveyor_pick --clock realtime --video
```

Useful ablations: `--obs-mode state` (give the model object positions, like a perception module), `--no-overlay` (no coordinate grid drawn on the head image), `--extra-latency 6` (add simulated thinking time to any policy).

What the model sees each decision: the head and wrist camera images (head image annotated with the table's x/y grid and the fingertip), proprioception, recent decisions and its own notes. It returns JSON with 1-5 waypoints. A target is either `xyz` or a head-image `pixel`, which the harness grounds with the depth buffer. "System 1" in the hybrid is a scripted controller with injected grounding errors (35% wrong object/bin, 8 mm noise). It stands in for π0.5, which is not used locally.

Outputs per run: `results.csv`, `summary.txt` (success, Score, decisions, model seconds per decision, tokens, % of steps that were model corrections) and per-episode `decisions.jsonl` + `video.mp4`. Browse them with `marimo edit notebooks/robot_eval.py`.

Fan out on Modal (episodes mostly wait on the API, so this is cheap CPU):

```sh
modal secret create armlab-llm ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
modal run modal_apps/armlab_eval.py --policy direct --vlm anthropic:claude-opus-5-5:medium --tasks all --seeds 0-9 --video
```

Every Modal run lands on the `armlab-runs` volume and shows up on the results page (see "How to run from anywhere" below).

### On real RoboLab (Isaac Lab, GPU)

`armlab/robolab/` runs the same policy against NVIDIA's RoboLab-120 benchmark through its absolute end-effector IK action space, with `--enable-gt-state` object poses as the "perception" input:

```sh
modal secret create armlab-llm ANTHROPIC_API_KEY=...
modal run modal_apps/robolab_eval.py --tasks BananaInBowlTask,RubiksCubeAndBananaTask --vlm anthropic:claude-opus-5-5:medium --num-runs 5
```

This path is **untested**: it was written against RoboLab's source with no GPU available. Expect to fix things on the first build. It needs an RTX-class GPU (L40S), not A100/H100.

## 2. Cosmos apps

The Cosmos apps use **Cosmos 3** (`nvidia/Cosmos3-Nano`, served as its text "Reasoner"), self-hosted on Modal with vLLM. NVIDIA's hosted Cosmos Reason API on build.nvidia.com is gone, so there is no `nvidia:` option any more.

```sh
modal deploy modal_apps/cosmos_reason_vllm.py      # L40S; ARMLAB_COSMOS_GPU=H100 for more headroom
```

That serves `https://<workspace>--cosmos-reason-serve.modal.run/v1`. `--vlm cosmos` (the default for the safety monitor, video API and physics filter) finds it through your Modal token, or set `ARMLAB_COSMOS_URL`. It needs the Modal secret `huggingface` (HF_TOKEN). Cosmos3-Nano is released under OpenMDW-1.1 and is not gated on Hugging Face, so there is no license page to click through. The model is configurable: deploy with `ARMLAB_COSMOS_MODEL=<hf id>` and use the same env var (or `--vlm cosmos:<hf id>`) on the client. `ARMLAB_COSMOS_MODEL=nvidia/Cosmos-Reason2-8B` brings back the previous generation, which needs the NVIDIA Open Model License accepted on its HF page. Any other VLM (Claude, GPT) works too, via frame sampling.

**Workplace safety monitor:** describes each clip, flags hazards with severity and writes a daily markdown report. On Modal it runs every morning (`modal_apps/armlab_safety_cron.py`) over clips you drop on the `armlab-runs` volume, and the report shows up on the results page.
```sh
armlab-safety path/to/clips/ --vlm cosmos --site "Loading dock" --out reports/
```

**Egocentric reasoning API:** FastAPI service, video in, "what's happening + next action" out.
```sh
ARMLAB_REASON_VLM=cosmos uvicorn armlab.cosmos.api:app --port 8080
curl -F video=@clip.mp4 -F task="put the block in the bin" localhost:8080/v1/analyze
```

**Physics plausibility filter:** builds a labelled benchmark from the sim, then scores a judge model on it.
```sh
armlab-physics make-dataset --out data/physics --per-scenario 3     # plausible: drop, place, carry
                                                                   # implausible: antigravity, teleport, passthrough, vanish, floating, reverse
armlab-physics evaluate data/physics --vlm cosmos
armlab-physics filter generated_clips/ --vlm ...                     # -> accepted/ and rejected/
```
On Modal, with keys only in Modal secrets (dataset and verdicts land on the `armlab-runs` volume under `_physics/`):
```sh
modal run modal_apps/armlab_physics.py --make --per-scenario 5
modal run modal_apps/armlab_physics.py --vlm cosmos                                  # or --mode frames
modal run modal_apps/armlab_physics.py --vlm anthropic:claude-opus-5-5:medium
```
First benchmark (Oct 2026, 45 clips): Claude Opus 5.5 (frames) accuracy 0.96, reject recall 1.00, false-reject rate 0.13; Cosmos3-Nano 0.40 (native) / 0.47 (frames), reject recall 0.10 / 0.20. Cosmos3-Nano labels nearly everything plausible, so use Claude as the gate for now.

**Sim-to-real data generator:** exports RGB + depth + segmentation of a rollout as Cosmos-Transfer2.5 inputs, with several appearance prompts (lab, warehouse, kitchen, cleanroom, outdoor), then runs them on an H100.
```sh
armlab-transfer-export --task blocks_into_bin --seed 0 --out exports/blocks0 --variations 3
modal run modal_apps/cosmos_transfer.py --export-dir exports/blocks0
```
Then feed the outputs to `armlab-physics filter` to drop the ones with broken physics.

## Where keys live / how to run from anywhere

Model keys live **only** in Modal secrets. Anything you drive runs from (GitHub Actions, a laptop, a phone, an agent box) holds just a Modal token, so you can launch runs from anywhere without copying API keys around.

| Where | Holds | Used by |
|---|---|---|
| Modal secret `armlab-llm` | `ANTHROPIC_API_KEY` (required), `OPENAI_API_KEY` (optional, GPT-6 Astra comparison) | `armlab_eval.py` direct/hybrid episodes, `armlab_safety_cron.py`, `robolab_eval.py` |
| Modal secret `huggingface` | `HF_TOKEN` (required for all Cosmos work) | `cosmos_reason_vllm.py` (Cosmos 3), `cosmos_transfer.py` |
| Modal secret `armlab-vllm` | `ARMLAB_VLLM_KEY` (required): bearer key the Cosmos endpoint always demands (vLLM `--api-key`) | `cosmos_reason_vllm.py` (server); `armlab_safety_cron.py`, `armlab_physics.py`, `armlab_eval.py` mount it; local `--vlm cosmos` clients fetch it in memory from the app's `api_key` function with your Modal token |
| your shell at deploy time (optional) | `ARMLAB_RESULTS_KEY`: key for the results page | stored as a Modal secret by `modal deploy` |
| GitHub repo secrets | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | the `run-eval` workflow |
| laptop / agent box | a Modal token (`modal token new`, or the two env vars) | `modal run` / `modal deploy` |

```sh
modal secret create armlab-llm ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
modal secret create huggingface HF_TOKEN=...
# Cosmos endpoint key, generated and stored without ever being printed:
python -c "import secrets, modal; modal.Secret.objects.create('armlab-vllm', {'ARMLAB_VLLM_KEY': secrets.token_hex(32)})"
```

**Check a machine:** `make doctor` (or `armlab-doctor`, `armlab-doctor --json`) reports whether a Modal token is configured, whether api.modal.com and GitHub are reachable, whether the Modal secrets, the `armlab-runs` volume and the deployed apps exist, and whether model keys are sitting in the local shell. It never prints a secret value, and it still runs (and says what is missing) with no token at all. `modal run modal_apps/armlab_eval.py --check-keys` asks Modal which keys `armlab-llm` holds (names only).

**Deploy once:**

```sh
cd robotics
modal deploy modal_apps/armlab_results.py        # results page: https://<workspace>--armlab-results-web.modal.run
modal deploy modal_apps/armlab_safety_cron.py    # daily safety report at 07:00 America/New_York
modal deploy modal_apps/cosmos_reason_vllm.py    # Cosmos 3 endpoint (GPU only while in use)
```

**Run from:**

- **GitHub (phone-friendly):** Actions > `run-eval` > Run workflow, pick a policy (`oracle` needs no model keys), tasks, seeds, clock and video. Or `gh workflow run run-eval.yml -f policy=oracle -f seeds=0`. The job summary shows the score table and a link to the results page; `results.csv`, `summary.txt` and videos are attached as an artifact.
- **Any machine with a Modal token:** `modal run modal_apps/armlab_eval.py --policy oracle --seeds 0 --video` (`--download` also copies the videos back).
- **Watch:** open the results page on your phone: runs newest first, the summary table per run, and videos that play inline. `/safety` lists the daily safety reports with their clips.

Volume layout (`armlab-runs`): `<run>/summary.txt|results.csv|results.json|config.json`, `<run>/<task>/seed<k>/{video.mp4,decisions.jsonl,result.json}`, `_safety/clips/inbox/` (drop clips with `modal volume put armlab-runs clips/ _safety/clips/inbox/`), `_safety/clips/processed/<date>/`, `_safety/reports/safety-<date>.md`.

The results page URL is public but unguessable and shows only sim videos and scores. To lock it, deploy with `ARMLAB_RESULTS_KEY=...` and open `/?key=...` once per browser.

## Accounts and keys

| Need | For | Where |
|---|---|---|
| `ANTHROPIC_API_KEY` (required) | Claude as the policy or judge | console.anthropic.com, stored in Modal secret `armlab-llm` |
| `OPENAI_API_KEY` (optional) + the GPT-6 Astra model id | the GPT-6 Astra comparison | platform.openai.com, stored in `armlab-llm` |
| Modal account (`modal token new`) | everything in the cloud: evals, results page, safety cron, RoboLab GPU, Cosmos serving and Transfer | modal.com |
| Hugging Face token (`HF_TOKEN`) | Cosmos 3 / Cosmos-Transfer2.5 weights on Modal; accept the NVIDIA Open Model License for Cosmos-Transfer2.5 (and Cosmos-Reason2 if you switch back) | huggingface.co, stored in Modal secret `huggingface` |

Rough GPU sizing: RoboLab needs an RTX-class GPU with 48 GB (L40S). Cosmos3-Nano (16B; ~31 GB of bf16 weights in the checkpoint) is set up for an L40S, with H100 as the option if memory gets tight. Cosmos Transfer 2.5 needs ~65 GB (H100 80 GB) and takes several minutes per 93-frame clip.

## What has and has not been verified

- Verified here (CPU, no keys): the sim, IK and grasping; the oracle solves all 5 tasks; video recording; both VLM policies end to end with a scripted fake model (parsing, pixel grounding, locate queries, accept/correct); the realtime-clock latency effect; the physics dataset renders; the Transfer control export; the safety report, API and judge logic with a fake model; the RoboLab client's pose math against a stub; `armlab-doctor` with and without a token; the results page against a fake runs directory.
- Verified on Modal (Oct 2026): `armlab_eval.py` with the oracle (10/10) and with live Claude in both the direct and hybrid architectures (full 5 tasks x 5 seeds: direct 24/25, hybrid 25/25 paused; direct 1/10 on conveyor + blocks with `--clock realtime`); the results page (runs, tables, inline video with HTTP Range); the safety cron end to end with Cosmos3-Nano served by `cosmos_reason_vllm.py` on an L40S; the ego API (`/v1/analyze`) against the live Cosmos3-Nano endpoint, with native `video_url` input (no `--mode frames` needed); the physics filter benchmark (`modal_apps/armlab_physics.py`) with Cosmos3-Nano and Claude; the `run-eval` GitHub workflow (run 37065442935, oracle, passed).
- Not verified: GPT-6 Astra; Cosmos Transfer inference.
