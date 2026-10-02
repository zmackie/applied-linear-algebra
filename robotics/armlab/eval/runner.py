"""Episode loop + CLI for evaluating policies in the MuJoCo arm sim.

Examples
  armlab-eval --policy oracle --tasks all --seeds 0-2 --video
  armlab-eval --policy direct --vlm anthropic:claude-opus-5-5:medium --tasks blocks_into_bin --seeds 0-4
  armlab-eval --policy hybrid --vlm openai:<model-id> --tasks all --seeds 0-4
  armlab-eval --policy direct --vlm anthropic --tasks conveyor_pick --clock realtime --video
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..policy.types import Decision, Policy, Usage
from ..sim.env import ArmEnv
from ..sim.recorder import Recorder
from ..sim.tasks import TASKS

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


@dataclass
class EpisodeResult:
    run: str
    task: str
    seed: int
    policy: str
    vlm: str
    clock: str
    obs_mode: str
    success: bool
    score: float
    steps: int
    sim_time_s: float
    decisions: int
    steps_by_source: dict = field(default_factory=dict)
    model_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    model_latency_s: float = 0.0
    wall_time_s: float = 0.0
    false_done_claims: int = 0
    error: str = ""
    video: str = ""

    @property
    def correction_fraction(self) -> float:
        total = sum(self.steps_by_source.values())
        return self.steps_by_source.get("llm-correction", 0) / total if total else 0.0


def make_policy(kind: str, env: ArmEnv, vlm_spec: str | None, seed: int, overlay: bool = True,
                system1_error: float = 0.35) -> Policy:
    from ..policy.scripted import ScriptedPolicy

    if kind == "oracle":
        return ScriptedPolicy(env, source="oracle")
    if kind == "system1":
        return ScriptedPolicy(env, wrong_target_prob=system1_error, xy_noise=0.008, seed=seed, source="system1")
    from ..policy.llm import DirectVLMPolicy, HybridPolicy
    from ..policy.vlm import make_vlm

    vlm = make_vlm(vlm_spec or "anthropic")
    if kind == "direct":
        return DirectVLMPolicy(vlm=vlm, overlay=overlay)
    if kind == "hybrid":
        s1 = ScriptedPolicy(env, wrong_target_prob=system1_error, xy_noise=0.008, seed=seed, source="system1")
        return HybridPolicy(vlm=vlm, overlay=overlay, system1=s1)
    raise ValueError(kind)


def _feedback(env: ArmEnv, info: dict, decision: Decision) -> str:
    parts = [f"executed {info['executed']}/{len(decision.waypoints)} waypoints in {info['steps']} steps",
             f"fingertip now {np.round(env.eef_xyz(), 3).tolist()}", f"gripper width {env.gripper_width():.3f} m"]
    if info.get("clamped"):
        parts.append("a target was outside the reachable area and was clamped")
    if info.get("final_error_m") and info["final_error_m"] > 0.01:
        parts.append(f"tracking error {info['final_error_m']:.3f} m (motion may be blocked)")
    return "; ".join(parts)


def run_episode(task: str, seed: int, policy_kind: str = "oracle", vlm_spec: str | None = None,
                clock: str = "paused", obs_mode: str = "vision", video: bool = False, run_name: str = "adhoc",
                max_decisions: int = 60, overlay: bool = True, policy: Policy | None = None,
                out_dir: Path | None = None, verbose: bool = True, extra_latency_s: float = 0.0) -> EpisodeResult:
    """Run one episode. clock='paused' freezes the world while the model thinks (as in the report);
    clock='realtime' keeps the world moving for the model's measured latency.
    extra_latency_s adds a fixed per-decision delay (in sim time only) to emulate a slower model."""
    env = ArmEnv(task, seed=seed, obs_mode=obs_mode)
    pol = policy or make_policy(policy_kind, env, vlm_spec, seed, overlay=overlay)
    out = (out_dir or RUNS_DIR / run_name) / task / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    vlm_name = getattr(getattr(pol, "vlm", None), "name", "")
    res = EpisodeResult(run_name, task, seed, policy_kind, vlm_name, clock, obs_mode, False, 0.0, 0, 0.0, 0)
    lat_note = f" | +{extra_latency_s:g}s/decision simulated latency" if extra_latency_s else ""
    rec = Recorder(env, out / "video.mp4", f"{policy_kind} | {vlm_name or 'no model'} | {clock} clock{lat_note} | {task}: {env.task.instruction}") if video else None
    log = open(out / "decisions.jsonl", "w")
    usage = Usage()
    t_start = time.time()
    pol.reset(env.task.instruction)
    obs = env.observe()
    if rec:
        rec.write_frame(repeat=10)
    try:
        for k in range(1, max_decisions + 1):
            if obs.task_done or env.budget_left() <= 0:
                break
            t0 = time.time()
            decision = pol.act(obs)
            think_s = (decision.usage.latency_s or (time.time() - t0)) + extra_latency_s
            usage.add(decision.usage)
            if clock == "realtime" and think_s > 0:
                if rec:
                    rec.set_decision(k, "think", f"Model deliberating for {think_s:.1f} s while the world keeps moving.", "")
                env.idle(think_s, phase="think")
            elif rec and think_s > 0.5:
                rec.pause_card(think_s)
            if decision.done and not decision.waypoints:
                res.false_done_claims += 1
                if res.false_done_claims >= 3:
                    break
            if rec:
                cmd = " > ".join(f"[{w.xyz[0]:.2f},{w.xyz[1]:.2f},{w.xyz[2]:.2f}] {w.gripper or ''}".strip() for w in decision.waypoints)
                rec.set_decision(k, decision.source, decision.rationale, cmd)
            info = env.execute(decision.waypoints, phase_label=decision.source)
            res.steps_by_source[decision.source] = res.steps_by_source.get(decision.source, 0) + info["steps"]
            fb = _feedback(env, info, decision)
            if hasattr(pol, "record"):
                pol.record(decision, fb)
            st = env.status()
            entry = {"decision": k, "source": decision.source, "rationale": decision.rationale,
                     "waypoints": [w.to_dict() for w in decision.waypoints],
                     "candidate": [w.to_dict() for w in decision.candidate] if decision.candidate is not None else None,
                     "think_s": round(think_s, 3), "usage": asdict(decision.usage), "exec": info,
                     "step": env.step_count, "score": st.score, "success": st.success, "notes": decision.notes}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            res.decisions = k
            if verbose:
                print(f"  [{task} s{seed}] #{k:02d} {decision.source:<14} think {think_s:5.1f}s  step {env.step_count:4d}  "
                      f"score {st.score:5.1f}  {decision.rationale[:90]}")
            obs = env.observe()
            obs.feedback = fb
    except Exception as e:  # keep the run going; record the failure
        res.error = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"  [{task} s{seed}] ERROR {res.error}")
    finally:
        st = env.status()
        res.success, res.score = st.success, st.score
        res.steps, res.sim_time_s = env.step_count, round(env.sim_time, 2)
        res.model_calls, res.input_tokens, res.output_tokens = usage.calls, usage.input_tokens, usage.output_tokens
        res.cached_input_tokens, res.model_latency_s = usage.cached_input_tokens, round(usage.latency_s, 2)
        res.wall_time_s = round(time.time() - t_start, 2)
        if rec:
            rec.write_frame(banner="SUCCESS" if st.success else "EPISODE END", repeat=25)
            rec.close()
            res.video = str(out / "video.mp4")
        log.close()
        (out / "result.json").write_text(json.dumps(asdict(res) | {"correction_fraction": res.correction_fraction}, indent=2))
        env.close()
    return res


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def write_results_csv(path: Path, results: list[EpisodeResult]) -> None:
    fields = [*EpisodeResult.__dataclass_fields__, "correction_fraction"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r) | {"steps_by_source": json.dumps(r.steps_by_source), "correction_fraction": round(r.correction_fraction, 4)})


def summarize(results: list[EpisodeResult]) -> str:
    lines = [f"{'task':<18}{'success':>10}{'score':>8}{'decisions':>11}{'model s/dec':>13}{'tokens':>12}{'corr%':>7}"]
    by_task: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)
    for t, rs in [*by_task.items(), ("OVERALL", results)]:
        n = len(rs)
        succ = sum(r.success for r in rs)
        dec = sum(r.decisions for r in rs)
        lat = sum(r.model_latency_s for r in rs) / dec if dec else 0
        tok = sum(r.input_tokens + r.output_tokens for r in rs)
        corr = 100 * sum(r.steps_by_source.get("llm-correction", 0) for r in rs) / max(1, sum(sum(r.steps_by_source.values()) for r in rs))
        lines.append(f"{t:<18}{f'{succ}/{n}':>10}{np.mean([r.score for r in rs]):>8.1f}{dec / n:>11.1f}{lat:>13.1f}{tok:>12,}{corr:>7.1f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["oracle", "system1", "direct", "hybrid"], default="oracle")
    ap.add_argument("--vlm", default="anthropic:claude-opus-5-5:medium",
                    help="anthropic[:model[:effort]] | openai:<model> | cosmos[:<model>] | vllm:<model>@<url>")
    ap.add_argument("--tasks", default="all", help=f"comma list or 'all': {', '.join(TASKS)}")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--clock", choices=["paused", "realtime"], default="paused")
    ap.add_argument("--obs-mode", choices=["vision", "state"], default="vision")
    ap.add_argument("--no-overlay", action="store_true", help="do not draw the coordinate grid on the head image")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-decisions", type=int, default=60)
    ap.add_argument("--extra-latency", type=float, default=0.0,
                    help="add this many seconds of (sim-time) thinking per decision, e.g. to show what latency does under --clock realtime")
    args = ap.parse_args(argv)

    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    run = args.run_name or time.strftime(f"%Y%m%d-%H%M%S-{args.policy}")
    out_dir = RUNS_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    results = []
    for t in tasks:
        for s in parse_seeds(args.seeds):
            r = run_episode(t, s, args.policy, args.vlm if args.policy in ("direct", "hybrid") else None, args.clock,
                            args.obs_mode, args.video, run, args.max_decisions, overlay=not args.no_overlay,
                            extra_latency_s=args.extra_latency)
            results.append(r)
            print(f"{t} seed {s}: {'SUCCESS' if r.success else 'fail'} score {r.score:.0f} "
                  f"({r.decisions} decisions, {r.sim_time_s:.1f}s sim, {r.wall_time_s:.0f}s wall){' ' + r.error if r.error else ''}")
    write_results_csv(out_dir / "results.csv", results)
    summary = summarize(results)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary + f"\n\nResults in {out_dir}")


if __name__ == "__main__":
    main()
