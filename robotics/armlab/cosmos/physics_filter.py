"""Physics-plausibility quality gate for generated video.

1. `make-dataset`: renders a labelled benchmark from the MuJoCo sim. Plausible clips are normal
   physics (drops, placements, carries); implausible clips break physics on purpose (anti-gravity,
   teleporting, passing through the table, vanishing, floating, time reversal).
2. `judge` / `evaluate`: asks a video VLM (self-hosted Cosmos 3 Nano by default) whether each clip is
   physically plausible and scores it against the labels.
3. `filter`: splits a directory of generated clips into accepted/rejected before they reach training.

  python -m armlab.cosmos.physics_filter make-dataset --out data/physics --per-scenario 3
  python -m armlab.cosmos.physics_filter evaluate data/physics --vlm cosmos
  python -m armlab.cosmos.physics_filter filter generated/ --vlm cosmos:nvidia/Cosmos3-Nano@https://<endpoint>/v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np

from ..policy.scripted import ScriptedPolicy
from ..policy.types import Waypoint
from ..sim.env import ArmEnv
from .reason import VideoReasoner, make_reasoner, write_video

PLAUSIBLE = ["drop", "place", "carry"]
IMPLAUSIBLE = ["antigravity", "teleport", "passthrough", "vanish", "floating", "reverse"]
CLIP_SIZE = (384, 512)


def _held_block_env(seed: int) -> tuple[ArmEnv, str]:
    env = ArmEnv("blocks_into_bin", seed=seed)
    oracle = ScriptedPolicy(env)
    d = oracle.act(env.observe())
    env.execute(d.waypoints)
    obj = env.held_objects()[0] if env.held_objects() else env.layout.blocks[0].name
    env.execute([Waypoint((0.45, -0.05, 0.25), 0.0, None)])
    return env, obj


def render_scenario(scenario: str, seed: int, fps: int = 25) -> list[np.ndarray]:
    env, obj = _held_block_env(seed)
    m, d = env.model, env.data
    gid, bid = env._block_geom[obj], env._block_body[obj]
    frames: list[np.ndarray] = []
    env.frame_callbacks.append(lambda e, fi: frames.append(e.render("closeup", *CLIP_SIZE)))
    adr = m.jnt_qposadr[m.body_jntadr[bid]]

    def release_and_retract():
        env.execute([Waypoint((0.45, -0.05, 0.25), 0.0, "open"), Waypoint((0.35, 0.1, 0.4), 0.0, None)])

    env.idle(0.4, "move")
    if scenario in ("drop", "reverse"):
        release_and_retract()
        env.idle(1.5, "move")
        if scenario == "reverse":
            frames.reverse()
    elif scenario == "place":
        env.execute([Waypoint((0.45, -0.05, 0.021), 0.0, None), Waypoint((0.45, -0.05, 0.021), 0.0, "open"),
                     Waypoint((0.45, -0.05, 0.2), 0.0, None)])
        env.idle(0.6, "move")
    elif scenario == "carry":
        env.execute([Waypoint((0.55, 0.15, 0.25), 0.0, None), Waypoint((0.55, 0.15, 0.03), 0.0, "open"),
                     Waypoint((0.5, 0.15, 0.25), 0.0, None)])
        env.idle(0.5, "move")
    elif scenario in ("antigravity", "floating"):
        # Drop normally, then apply an unphysical force to just this block once it rests on the table:
        # a constant 1.05x its weight (it lifts off by itself) or a hover controller (it rises and hangs in mid-air).
        release_and_retract()
        env.idle(0.5, "move")
        weight = m.body_mass[bid] * -m.opt.gravity[2]
        dof = m.body_dofadr[bid]
        if scenario == "antigravity":
            d.xfrc_applied[bid, 2] = 1.05 * weight
        else:
            def hover(e, fi):
                z, vz = d.xpos[bid][2], d.qvel[dof + 2]
                d.xfrc_applied[bid, 2] = weight + 3.0 * (0.14 - z) * m.body_mass[bid] * 10 - 1.0 * vz
            env.frame_callbacks.insert(0, hover)
        env.idle(1.8, "move")
    elif scenario == "passthrough":
        m.geom_contype[gid] = 0
        m.geom_conaffinity[gid] = 0
        release_and_retract()
        env.idle(1.2, "move")
    elif scenario in ("teleport", "vanish"):
        release_and_retract()
        env.idle(0.8, "move")
        if scenario == "teleport":
            d.qpos[adr:adr + 2] += [0.12, 0.18]
            mujoco.mj_forward(m, d)
        else:
            m.geom_rgba[gid, 3] = 0.0
        env.idle(1.0, "move")
    else:
        raise ValueError(scenario)
    env.close()
    return frames


def make_dataset(out: Path, per_scenario: int = 3, fps: int = 25) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for scen in PLAUSIBLE + IMPLAUSIBLE:
        for i in range(per_scenario):
            frames = render_scenario(scen, seed=100 + i, fps=fps)
            path = write_video(frames, out / f"{scen}_{i}.mp4", fps)
            manifest.append({"clip": path.name, "scenario": scen, "plausible": scen in PLAUSIBLE})
            print(f"  wrote {path.name} ({len(frames)} frames)")
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    return out / "manifest.jsonl"


JUDGE_QUESTION = """You are a quality gate for a dataset of robot videos. Decide whether this video is physically plausible: could it have been filmed in the real world?
Watch for violations such as objects falling upward or hovering without support, objects passing through solid surfaces, objects teleporting or suddenly appearing/disappearing, objects changing shape or size, motion that only makes sense played backwards, or impossible contacts.
Return JSON: {"plausible": true|false, "confidence": <0.0-1.0>, "violations": ["<what is wrong and when>"]}"""


def judge(reasoner: VideoReasoner, clip: Path) -> dict:
    out = reasoner.ask(clip, JUDGE_QUESTION)
    j = out["json"] or {}
    verdict = j.get("plausible")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower() in ("true", "yes", "plausible")
    if verdict is None:  # fall back to reading the prose answer
        a = out["answer"].lower()
        verdict = "implausible" not in a and "not plausible" not in a
    return {"plausible": bool(verdict), "confidence": float(j.get("confidence") or 0.5),
            "violations": j.get("violations") or [], "reasoning": out["reasoning"],
            "latency_s": round(out["usage"].latency_s, 2)}


def evaluate(dataset: Path, reasoner: VideoReasoner, out: Path | None = None, retries: int = 2) -> dict:
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines() if line.strip()]
    results = []
    for r in rows:
        v, err = None, None
        for attempt in range(retries + 1):  # e.g. a scale-from-zero endpoint dropping queued requests
            try:
                v = judge(reasoner, dataset / r["clip"])
                break
            except Exception as e:  # one bad request should not sink the whole benchmark
                err = e
                time.sleep(5 * (attempt + 1))
        if v is None:
            print(f"  ERR  {r['clip']}: {type(err).__name__}: {str(err)[:200]}")
            results.append(r | {"pred_plausible": None, "error": f"{type(err).__name__}: {err}"[:500]})
            continue
        results.append(r | {"pred_plausible": v["plausible"], "confidence": v["confidence"], "violations": v["violations"],
                            "latency_s": v["latency_s"]})
        mark = "ok " if v["plausible"] == r["plausible"] else "MISS"
        print(f"  {mark} {r['clip']:<20} label={'plausible' if r['plausible'] else 'IMPLAUSIBLE':<11} pred={'plausible' if v['plausible'] else 'IMPLAUSIBLE'}")
    metrics = score([r for r in results if r["pred_plausible"] is not None])
    metrics["errors"] = sum(1 for r in results if r["pred_plausible"] is None)
    metrics["model"] = getattr(reasoner.vlm, "name", "")
    metrics["mode"] = reasoner.mode
    (out or dataset / "eval_results.json").write_text(json.dumps({"metrics": metrics, "results": results}, indent=2))
    return metrics


def score(results: list[dict]) -> dict:
    """Treat 'reject' (implausible) as the positive class: precision = rejected clips that were truly bad."""
    tp = sum(1 for r in results if not r["plausible"] and not r["pred_plausible"])
    fp = sum(1 for r in results if r["plausible"] and not r["pred_plausible"])
    fn = sum(1 for r in results if not r["plausible"] and r["pred_plausible"])
    tn = sum(1 for r in results if r["plausible"] and r["pred_plausible"])
    n = len(results)
    per = {}
    for r in results:
        s = per.setdefault(r["scenario"], [0, 0])
        s[0] += int(r["pred_plausible"] == r["plausible"])
        s[1] += 1
    return {"n": n, "accuracy": (tp + tn) / n if n else 0.0,
            "reject_precision": tp / (tp + fp) if tp + fp else 0.0,
            "reject_recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_reject_rate": fp / (fp + tn) if fp + tn else 0.0,
            "per_scenario_accuracy": {k: v[0] / v[1] for k, v in per.items()}}


def filter_dir(src: Path, reasoner: VideoReasoner, min_confidence: float = 0.0) -> dict:
    acc, rej = src / "accepted", src / "rejected"
    acc.mkdir(exist_ok=True)
    rej.mkdir(exist_ok=True)
    log = {}
    for clip in sorted(src.glob("*.mp4")):
        v = judge(reasoner, clip)
        keep = v["plausible"] or v["confidence"] < min_confidence
        shutil.copy2(clip, (acc if keep else rej) / clip.name)
        log[clip.name] = v
        print(f"  {'ACCEPT' if keep else 'REJECT'} {clip.name} {'; '.join(map(str, v['violations']))[:100]}")
    (src / "filter_log.json").write_text(json.dumps(log, indent=2))
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-dataset")
    a.add_argument("--out", default="data/physics")
    a.add_argument("--per-scenario", type=int, default=3)
    for name in ("evaluate", "filter"):
        b = sub.add_parser(name)
        b.add_argument("path")
        b.add_argument("--vlm", default="cosmos")
        b.add_argument("--mode", choices=["native", "frames"], default=None)
        b.add_argument("--fps", type=float, default=4.0, help="native mode: frames per second the server samples")
        if name == "evaluate":
            b.add_argument("--out", default=None, help="results JSON (default <path>/eval_results.json)")
    args = ap.parse_args(argv)
    if args.cmd == "make-dataset":
        print(make_dataset(Path(args.out), args.per_scenario))
        return
    reasoner = make_reasoner(args.vlm, mode=args.mode, fps=args.fps)
    if args.cmd == "evaluate":
        print(json.dumps(evaluate(Path(args.path), reasoner, Path(args.out) if args.out else None), indent=2))
    else:
        filter_dir(Path(args.path), reasoner)


if __name__ == "__main__":
    main()
