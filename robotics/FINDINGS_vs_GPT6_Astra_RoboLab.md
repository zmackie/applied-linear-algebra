# Findings vs "GPT-6 Astra cracked RoboLab"

One-page comparison of the tweet/paper claims with our measurements (Oct 2026). Sources: [NIM-355](https://linear.app/mainmain/issue/NIM-355), [NIM-356](https://linear.app/mainmain/issue/NIM-356), [NIM-361](https://linear.app/mainmain/issue/NIM-361). Same-seed Claude vs GPT-6 Astra on our arm-sim ([NIM-357](https://linear.app/mainmain/issue/NIM-357)) is still pending an OpenAI key in Modal secrets.

## RoboLab paper-10 subset (paused sim)

| | Paper (GPT-6 Astra direct) | Ours (Claude Opus 5.5 medium, direct) |
|---|---|---|
| Score | 49/50 | **48/50** (96%) |
| Setup | Table 2; GT object poses | `--enable-gt-state`; Modal L40S (R580 hosts) |

Per-task (5 runs each): BlocksInBin 4/5, ClutterPumpkin 4/5, ButterAboveRaisin 5/5, BlockStackingSpecifiedOrder 5/5, ReorientRedMug 5/5, LargerObjectRaisinBoxInBin 5/5, SauceBottlesCrate 5/5, CannedFoodInBin 5/5, YogurtInBowl 5/5, RubiksCube 5/5.

Failures: large pumpkin near reach envelope (timeout); green block knocked to the floor (unrecoverable). Early raisin-box / butter grasp failures were fixed by putting Robotiq opening, reach, and per-object bbox/yaw in the prompt (v1 → v2).

Results: https://zmackie--armlab-results-web.modal.run/runs/robolab-paper10-v2-claude-direct

**Takeaway:** on the paper's own 10-task subset, Claude direct is within 1 episode of the reported Astra number. That does not by itself validate or refute the tweet; it says a strong VLM policy can land near-ceiling when the sim is paused and poses are given.

## Direct vs hybrid (our 5 arm-sim tasks, paused)

Claude Opus 5.5 medium, seeds 0–4, all 5 tasks:

| Architecture | Success | Mean model s/decision | Notes |
|---|---|---|---|
| Direct | 24/25 (96%) | 11.6 | One conveyor miss |
| Hybrid | **25/25 (100%)** | **8.4** | ~30% faster decisions; 31% of steps were corrections |

The paper reported hybrid winning on bimanual RoboDojo (48% vs 26%). On these local Franka tasks the gap is tiny at ceiling; hybrid's win here is mostly latency and catching System-1 grounding errors (never reached a failure).

Results: [direct](https://zmackie--armlab-results-web.modal.run/runs/claude-direct-20261003), [hybrid](https://zmackie--armlab-results-web.modal.run/runs/claude-hybrid-20261003)

## Paused vs realtime clock (latency)

Same model (Claude direct), same seeds, conveyor_pick + blocks_into_bin:

| Clock | Success | What happens |
|---|---|---|
| Paused | **9/10** | World waits for the model |
| Realtime | **1/10** | Conveyor 0/5; blocks 1/5 |

The model still thinks ~10–13 s per decision. With the clock running, the conveyor target is gone before the intercept plan executes; static pick-and-place runs out of wall-clock on the third object. The report's 49/50 depends on pausing the sim. Holding the policy fixed, latency alone collapses success.

Results: https://zmackie--armlab-results-web.modal.run/runs/claude-realtime-20261003

## Claude vs GPT-6 Astra (status)

- **RoboLab paper-10 (cross-paper):** Claude direct 48/50 vs reported Astra 49/50 (not same seeds / not re-run by us).
- **Our arm-sim, same seeds:** not run yet ([NIM-357](https://linear.app/mainmain/issue/NIM-357); needs `OPENAI_API_KEY` + Astra model id in the `armlab-llm` Modal secret).

## Caveats

- Sim only (MuJoCo Franka armlab + Isaac Lab RoboLab). No real robot.
- Small seed counts (5 per task). Task selection matches the paper-10 subset for RoboLab; armlab tasks are our own five.
- RoboLab runs used GT object poses (`--enable-gt-state`), as the paper did.
- Hybrid System-1 is a scripted stand-in for π0.5, not the paper's policy.
- Isaac Sim 5.0 on Modal needed R580 hosts; R610 hosts crash at CUDA init.
- No Astra head-to-head on our arm-sim yet, so the tweet's model-specific claim is only lightly tested here.

## Bottom line

Paused-sim, GT-pose RoboLab is near-solved for a strong VLM (Claude 48/50 ≈ Astra 49/50). Architecture (direct vs hybrid) barely moves ceiling scores on easy local tasks but does cut decision latency. Realtime clock is the actual cliff: 9/10 → 1/10 from latency alone. The interesting open question is whether Astra (or hybrid-with-System-1-acting-between-calls) recovers any of that realtime gap on matched seeds once NIM-357 can run.
