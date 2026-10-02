# armlab: robot arms, video models, and testing the hype

A small, runnable testbed for two things:

1. **Testing claims like "GPT-6 Astra cracked RoboLab."** Put any vision-language model (Claude, GPT-6 Astra, Cosmos Reason) in a robot-arm control loop, using the two architectures from *GPT 6 Astra as an Embodied Policy* (Su et al., 2026), and measure success, Score, tokens and latency.
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
modal_apps/        Modal apps: parallel sim evals, RoboLab on L40S, Cosmos Reason vLLM server, Cosmos Transfer on H100
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

### On real RoboLab (Isaac Lab, GPU)

`armlab/robolab/` runs the same policy against NVIDIA's RoboLab-120 benchmark through its absolute end-effector IK action space, with `--enable-gt-state` object poses as the "perception" input:

```sh
modal secret create armlab-llm ANTHROPIC_API_KEY=...
modal run modal_apps/robolab_eval.py --tasks BananaInBowlTask,RubiksCubeAndBananaTask --vlm anthropic:claude-opus-5-5:medium --num-runs 5
```

This path is **untested**: it was written against RoboLab's source with no GPU available. Expect to fix things on the first build. It needs an RTX-class GPU (L40S), not A100/H100.

## 2. Cosmos apps

Cosmos Reason 2 is reachable two ways:
- **Hosted:** `--vlm nvidia:nvidia/cosmos-reason2-8b` with `NVIDIA_API_KEY` from build.nvidia.com. Uses frame sampling unless `--mode native`.
- **Self-hosted on Modal:** `modal deploy modal_apps/cosmos_reason_vllm.py`, then `--vlm vllm:nvidia/Cosmos-Reason2-8B@https://<your-endpoint>/v1`. This sends the actual video.

Any other VLM (Claude, GPT) works too, via frame sampling.

**Workplace safety monitor:** describes each clip, flags hazards with severity and writes a daily markdown report.
```sh
armlab-safety path/to/clips/ --vlm nvidia:nvidia/cosmos-reason2-8b --site "Loading dock" --out reports/
```

**Egocentric reasoning API:** FastAPI service, video in, "what's happening + next action" out.
```sh
ARMLAB_REASON_VLM=nvidia:nvidia/cosmos-reason2-8b uvicorn armlab.cosmos.api:app --port 8080
curl -F video=@clip.mp4 -F task="put the block in the bin" localhost:8080/v1/analyze
```

**Physics plausibility filter:** builds a labelled benchmark from the sim, then scores a judge model on it.
```sh
armlab-physics make-dataset --out data/physics --per-scenario 3     # plausible: drop, place, carry
                                                                   # implausible: antigravity, teleport, passthrough, vanish, floating, reverse
armlab-physics evaluate data/physics --vlm nvidia:nvidia/cosmos-reason2-8b
armlab-physics filter generated_clips/ --vlm ...                     # -> accepted/ and rejected/
```

**Sim-to-real data generator:** exports RGB + depth + segmentation of a rollout as Cosmos-Transfer2.5 inputs, with several appearance prompts (lab, warehouse, kitchen, cleanroom, outdoor), then runs them on an H100.
```sh
armlab-transfer-export --task blocks_into_bin --seed 0 --out exports/blocks0 --variations 3
modal run modal_apps/cosmos_transfer.py --export-dir exports/blocks0
```
Then feed the outputs to `armlab-physics filter` to drop the ones with broken physics.

## Accounts and keys

| Need | For | Where |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude as the policy or judge | console.anthropic.com |
| `OPENAI_API_KEY` + the GPT-6 Astra model id | the GPT-6 Astra comparison | platform.openai.com |
| `NVIDIA_API_KEY` | hosted Cosmos Reason 2 | build.nvidia.com |
| Modal account (`modal token new`) | parallel evals, RoboLab GPU, Cosmos serving and Transfer | modal.com |
| Hugging Face token with the NVIDIA Open Model License accepted | Cosmos-Reason2 / Cosmos-Transfer2.5 weights on Modal | huggingface.co |

Modal secrets used: `armlab-llm` (LLM keys) and `huggingface` (`HF_TOKEN`).

Rough GPU sizing: RoboLab needs an RTX-class GPU with 48 GB (L40S). Cosmos Reason 2-8B fits an L40S. Cosmos Transfer 2.5 needs ~65 GB (H100 80 GB) and takes several minutes per 93-frame clip.

## What has and has not been verified

- Verified here (CPU, no keys): the sim, IK and grasping; the oracle solves all 5 tasks; video recording; both VLM policies end to end with a scripted fake model (parsing, pixel grounding, locate queries, accept/correct); the realtime-clock latency effect; the physics dataset renders; the Transfer control export; the safety report, API and judge logic with a fake model; the RoboLab client's pose math against a stub.
- Not verified (needs keys or a GPU): live Claude, GPT and Cosmos calls; the Modal apps; RoboLab on Isaac Lab; Cosmos Transfer inference.
