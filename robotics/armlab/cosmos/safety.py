"""Workplace safety monitor: describe each camera clip, flag hazards, write a daily report.

  armlab-safety clips/ --vlm nvidia:nvidia/cosmos-reason2-8b --site "Assembly cell 3" --out reports/
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .reason import VideoReasoner, make_reasoner

SEVERITIES = ("low", "medium", "high")

SAFETY_QUESTION = """You are a workplace safety monitor reviewing a security-camera clip from: {site}.
Describe what is happening, then list every safety hazard you can see. Consider: people near moving machinery or robots, missing PPE (hard hat, eye protection, gloves, hi-vis), objects falling or likely to fall, spills and trip hazards, blocked walkways or exits, unsafe lifting, smoke or fire, vehicles near pedestrians, and anything unstable.

Return JSON with exactly these keys:
{{"summary": "<one or two sentences>",
  "people": <integer count of people visible>,
  "hazards": [{{"category": "<short category>", "severity": "low|medium|high", "time_s": <approximate time in the clip>, "description": "<what and where>"}}],
  "recommended_actions": ["<short action>"]}}
Use an empty list when there are no hazards. Only report what is visible."""


def analyze_clip(reasoner: VideoReasoner, clip: str | Path, site: str = "an industrial work area") -> dict:
    out = reasoner.ask(clip, SAFETY_QUESTION.format(site=site))
    data = out["json"] or {"summary": out["answer"][:500], "people": None, "hazards": [], "recommended_actions": [],
                           "parse_error": True}
    hazards = []
    for h in data.get("hazards") or []:
        sev = str(h.get("severity", "low")).lower()
        hazards.append({"category": str(h.get("category", "other")), "severity": sev if sev in SEVERITIES else "low",
                        "time_s": h.get("time_s"), "description": str(h.get("description", ""))})
    data["hazards"] = hazards
    data["clip"] = str(clip)
    data["reasoning"] = out["reasoning"]
    data["tokens"] = out["usage"].input_tokens + out["usage"].output_tokens
    data["latency_s"] = round(out["usage"].latency_s, 2)
    return data


def daily_report(results: list[dict], site: str, date: dt.date | None = None) -> str:
    date = date or dt.date.today()
    all_h = [(r, h) for r in results for h in r["hazards"]]
    counts = {s: sum(1 for _, h in all_h if h["severity"] == s) for s in SEVERITIES}
    lines = [f"# Daily safety report: {site}", f"**Date:** {date.isoformat()}  ",
             f"**Clips reviewed:** {len(results)}  ",
             f"**Hazards flagged:** {len(all_h)} (high {counts['high']}, medium {counts['medium']}, low {counts['low']})", ""]
    high = [(r, h) for r, h in all_h if h["severity"] == "high"]
    if high:
        lines += ["## Needs attention today", ""]
        for r, h in high:
            t = f" at {h['time_s']} s" if h.get("time_s") is not None else ""
            lines.append(f"- **{h['category']}** in `{Path(r['clip']).name}`{t}: {h['description']}")
        lines.append("")
    cats: dict[str, int] = {}
    for _, h in all_h:
        cats[h["category"]] = cats.get(h["category"], 0) + 1
    if cats:
        lines += ["## Hazards by category", "", "| Category | Count |", "|---|---|"]
        lines += [f"| {c} | {n} |" for c, n in sorted(cats.items(), key=lambda kv: -kv[1])]
        lines.append("")
    lines += ["## Clip log", ""]
    for r in results:
        sev = max((SEVERITIES.index(h["severity"]) for h in r["hazards"]), default=-1)
        flag = ["low", "medium", "HIGH"][sev] if sev >= 0 else "clear"
        lines.append(f"### `{Path(r['clip']).name}`: {flag}")
        lines.append(r.get("summary", ""))
        for h in r["hazards"]:
            lines.append(f"- [{h['severity']}] {h['category']}: {h['description']}")
        for a in r.get("recommended_actions") or []:
            lines.append(f"- action: {a}")
        lines.append("")
    lines.append("_Generated automatically by a vision-language model; verify flagged items before acting on them._")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", help="directory of .mp4 clips (or a single clip)")
    ap.add_argument("--vlm", default="nvidia:nvidia/cosmos-reason2-8b")
    ap.add_argument("--mode", choices=["native", "frames"], default=None)
    ap.add_argument("--site", default="an industrial work area")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    src = Path(args.clips)
    clips = sorted(src.glob("*.mp4")) if src.is_dir() else [src]
    reasoner = make_reasoner(args.vlm, mode=args.mode)
    results = []
    for c in clips:
        r = analyze_clip(reasoner, c, args.site)
        print(f"{c.name}: {len(r['hazards'])} hazards  {r.get('summary', '')[:100]}")
        results.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().isoformat()
    (out / f"safety-{day}.json").write_text(json.dumps(results, indent=2))
    (out / f"safety-{day}.md").write_text(daily_report(results, args.site))
    print(f"Report: {out / f'safety-{day}.md'}")


if __name__ == "__main__":
    main()
