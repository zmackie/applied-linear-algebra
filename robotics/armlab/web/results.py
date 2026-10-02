"""Results web app: lists eval runs, their summary tables and videos, plus the daily safety reports.

It reads a plain directory, so the same code serves `robotics/runs/` on a laptop and the `armlab-runs` Modal volume
(see `modal_apps/armlab_results.py`). Layout it understands:

  <root>/<run>/summary.txt, results.json | results.csv, config.json
  <root>/<run>/<task>/seed<k>/result.json, video.mp4, decisions.jsonl
  <root>/_safety/reports/safety-<YYYY-MM-DD>.md (+ .json)        written by the daily safety cron
  <root>/_safety/clips/inbox/*.mp4                                clips waiting for the next report

Local preview:
  uvicorn --factory armlab.web.results:app_from_env --port 8000      # ARMLAB_RUNS_ROOT defaults to robotics/runs
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

SAFETY_DIR = "_safety"
SAFETY_REPORTS = f"{SAFETY_DIR}/reports"
SAFETY_INBOX = f"{SAFETY_DIR}/clips/inbox"
SAFETY_PROCESSED = f"{SAFETY_DIR}/clips/processed"
_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


# ----------------------------------------------------------------------------------------------- data loading

@dataclass
class Episode:
    task: str
    seed: int
    success: bool | None = None
    score: float | None = None
    decisions: int | None = None
    tokens: int | None = None
    model_latency_s: float | None = None
    error: str = ""
    video: str | None = None  # path relative to the runs root


@dataclass
class Run:
    name: str
    mtime: float
    summary: str = ""
    config: dict = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def policy(self) -> str:
        if self.config.get("policy"):
            return str(self.config["policy"])
        return self.name.split("-")[2] if self.name.count("-") >= 2 else ""


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _num(v, cast=float):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def _bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    return None


def safe_name(name: str) -> bool:
    return bool(_NAME_OK.match(name)) and ".." not in name


def list_runs(root: Path) -> list[Run]:
    """Every run directory under root (newest first), without loading episodes."""
    root = Path(root)
    if not root.is_dir():
        return []
    runs = []
    for d in root.iterdir():
        if d.is_dir() and safe_name(d.name) and not d.name.startswith(("_", ".")):
            runs.append(Run(d.name, d.stat().st_mtime, summary=_safe_text(d / "summary.txt")))
    return sorted(runs, key=lambda r: (r.mtime, r.name), reverse=True)


def _safe_text(p: Path) -> str:
    try:
        return p.read_text()
    except OSError:
        return ""


def _episode_from(d: dict, root: Path, ep_dir: Path | None) -> Episode:
    tokens = None
    if d.get("input_tokens") not in (None, "") or d.get("output_tokens") not in (None, ""):
        tokens = (_num(d.get("input_tokens"), int) or 0) + (_num(d.get("output_tokens"), int) or 0)
    video = None
    if ep_dir is not None and (ep_dir / "video.mp4").is_file():
        video = (ep_dir / "video.mp4").relative_to(root).as_posix()
    return Episode(task=str(d.get("task", "?")), seed=_num(d.get("seed"), int) or 0, success=_bool(d.get("success")),
                   score=_num(d.get("score")), decisions=_num(d.get("decisions"), int), tokens=tokens,
                   model_latency_s=_num(d.get("model_latency_s")), error=str(d.get("error") or ""), video=video)


def load_run(root: Path, name: str) -> Run | None:
    root = Path(root)
    if not safe_name(name) or name.startswith("_"):
        return None
    d = root / name
    if not d.is_dir():
        return None
    run = Run(name, d.stat().st_mtime, summary=_safe_text(d / "summary.txt"), config=_read_json(d / "config.json") or {})
    # Per-episode result.json files are written as each episode finishes, so in-progress runs show up too.
    seen = set()
    for rj in sorted(d.glob("*/seed*/result.json")):
        data = _read_json(rj)
        if isinstance(data, dict):
            ep = _episode_from(data, root, rj.parent)
            run.episodes.append(ep)
            seen.add((ep.task, ep.seed))
    # Fall back to the aggregate tables (e.g. a run copied without its episode folders).
    rows = _read_json(d / "results.json")
    if not isinstance(rows, list) and (d / "results.csv").is_file():
        with open(d / "results.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict):
            ep_dir = d / str(r.get("task", "")) / f"seed{r.get('seed', '')}"
            ep = _episode_from(r, root, ep_dir if ep_dir.is_dir() else None)
            if (ep.task, ep.seed) not in seen:
                run.episodes.append(ep)
    run.episodes.sort(key=lambda e: (e.task, e.seed))
    return run


def list_safety_reports(root: Path) -> list[str]:
    """Report dates (YYYY-MM-DD), newest first."""
    rd = Path(root) / SAFETY_REPORTS
    if not rd.is_dir():
        return []
    dates = [m.group(1) for p in rd.glob("safety-*.md") if (m := re.fullmatch(r"safety-(\d{4}-\d{2}-\d{2})\.md", p.name))]
    return sorted(dates, reverse=True)


def pending_clips(root: Path) -> int:
    inbox = Path(root) / SAFETY_INBOX
    return len(list(inbox.glob("*.mp4"))) if inbox.is_dir() else 0


# ----------------------------------------------------------------------------------------------- rendering

CSS = """
:root{color-scheme:light dark;--fg:#111;--bg:#fff;--mut:#666;--ok:#137333;--bad:#b3261e;--card:#f4f4f6}
@media (prefers-color-scheme:dark){:root{--fg:#eee;--bg:#121212;--mut:#9a9a9a;--ok:#81c995;--bad:#f28b82;--card:#1e1e22}}
*{box-sizing:border-box}body{margin:0;font:16px/1.45 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
main{max-width:900px;margin:auto;padding:12px 14px 40px}a{color:inherit}h1{font-size:1.35em;margin:.4em 0}h2{font-size:1.1em;margin:1.2em 0 .4em}
nav{font-size:.95em;color:var(--mut)}nav a{margin-right:12px}.mut{color:var(--mut);font-size:.9em}
.card{display:block;background:var(--card);border-radius:12px;padding:10px 12px;margin:8px 0;text-decoration:none}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}table{border-collapse:collapse;width:100%;font-size:.92em}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid rgba(127,127,127,.25);white-space:nowrap}
pre{background:var(--card);padding:10px;border-radius:10px;overflow-x:auto;font-size:.8em}
video{width:100%;max-height:70vh;border-radius:10px;background:#000}figure{margin:14px 0}figcaption{font-size:.9em;color:var(--mut)}
"""


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><main><nav><a href='/'>Runs</a><a href='/safety'>Safety</a></nav>"
            f"{body}</main></body></html>")


def _when(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _overall_line(summary: str) -> str:
    for line in summary.splitlines():
        if line.startswith("OVERALL"):
            return line
    return ""


def render_index(root: Path, limit: int = 200) -> str:
    runs = list_runs(root)
    reports = list_safety_reports(root)
    parts = ["<h1>armlab runs</h1>"]
    if reports:
        parts.append(f"<a class=card href='/safety/{reports[0]}'>Latest safety report: <b>{reports[0]}</b></a>")
    if not runs:
        parts.append("<p class=mut>No runs yet. Trigger the <code>run-eval</code> GitHub workflow or "
                     "<code>modal run modal_apps/armlab_eval.py --policy oracle --seeds 0</code>.</p>")
    for r in runs[:limit]:
        ov = _overall_line(r.summary)
        detail = html.escape(" ".join(ov.split()[1:3])) if ov else "in progress or no summary yet"
        parts.append(f"<a class=card href='/runs/{quote(r.name)}'><b>{html.escape(r.name)}</b><br>"
                     f"<span class=mut>{_when(r.mtime)} &middot; success/score: {detail}</span></a>")
    return _page("armlab runs", "".join(parts))


def _fmt(v, spec: str = "") -> str:
    if v is None:
        return ""
    return format(v, spec) if spec else str(v)


def render_run(root: Path, name: str) -> str | None:
    run = load_run(root, name)
    if run is None:
        return None
    e_name = html.escape(run.name)
    parts = [f"<h1>{e_name}</h1>",
             f"<p class=mut>{_when(run.mtime)} &middot; {run.n_success}/{len(run.episodes)} episodes succeeded</p>"]
    if run.config:
        keys = ("policy", "vlm", "tasks", "seeds", "clock", "obs_mode", "video")
        cfg = " &middot; ".join(f"{k}: {html.escape(str(run.config[k]))}" for k in keys if k in run.config)
        parts.append(f"<p class=mut>{cfg}</p>")
    if run.summary:
        parts.append(f"<h2>Summary</h2><pre>{html.escape(run.summary)}</pre>")
    if run.episodes:
        rows = []
        for e in run.episodes:
            res = "<span class=ok>success</span>" if e.success else ("<span class=bad>fail</span>" if e.success is False else "")
            vid = f"<a href='#v-{html.escape(e.task)}-{e.seed}'>video</a>" if e.video else ""
            rows.append(f"<tr><td>{html.escape(e.task)}</td><td>{e.seed}</td><td>{res}</td><td>{_fmt(e.score, '.0f')}</td>"
                        f"<td>{_fmt(e.decisions)}</td><td>{_fmt(e.tokens, ',')}</td><td>{vid}</td>"
                        f"<td>{html.escape(e.error[:80])}</td></tr>")
        parts.append("<h2>Episodes</h2><div class=tw><table><tr><th>task</th><th>seed</th><th>result</th><th>score</th>"
                     "<th>decisions</th><th>tokens</th><th></th><th>error</th></tr>" + "".join(rows) + "</table></div>")
    vids = [e for e in run.episodes if e.video]
    if vids:
        parts.append("<h2>Videos</h2>")
        for e in vids:
            src = "/files/" + quote(e.video)
            parts.append(f"<figure id='v-{html.escape(e.task)}-{e.seed}'><video controls playsinline preload=metadata "
                         f"src='{src}'></video><figcaption>{html.escape(e.task)} seed {e.seed} &middot; "
                         f"<a href='{src}' download>download</a></figcaption></figure>")
    return _page(run.name, "".join(parts))


def markdown_to_html(md: str) -> str:
    """Just enough Markdown for the safety reports: headings, lists, tables, bold, italics, inline code."""
    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\w*])_([^_]+)_(?![\w*])", r"<i>\1</i>", s)
        return s

    out, in_list, table = [], False, []

    def flush_table():
        if not table:
            return
        rows = [r for r in table if not re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?", r)]
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        head, body = cells[0], cells[1:]
        out.append("<div class=tw><table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
                   + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                   + "</table></div>")
        table.clear()

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if m := re.match(r"^(#{1,4}) (.*)", line):
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
        elif line.strip():
            out.append(f"<p>{inline(line.strip())}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def render_safety_index(root: Path) -> str:
    reports = list_safety_reports(root)
    n = pending_clips(root)
    parts = ["<h1>Daily safety reports</h1>",
             f"<p class=mut>{n} clip(s) waiting in <code>{SAFETY_INBOX}/</code> for the next scheduled report.</p>"]
    if not reports:
        parts.append("<p class=mut>No reports yet. The <code>armlab-safety-cron</code> Modal app writes one per day.</p>")
    parts += [f"<a class=card href='/safety/{d}'>{d}</a>" for d in reports]
    return _page("Safety reports", "".join(parts))


def render_safety_report(root: Path, date: str) -> str | None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return None
    p = Path(root) / SAFETY_REPORTS / f"safety-{date}.md"
    if not p.is_file():
        return None
    body = markdown_to_html(p.read_text())
    clips_dir = Path(root) / SAFETY_PROCESSED / date
    clips = sorted(clips_dir.glob("*.mp4")) if clips_dir.is_dir() else []
    if clips:
        body += "<h2>Clips</h2>"
        for c in clips:
            src = "/files/" + quote(c.relative_to(root).as_posix())
            body += (f"<figure><video controls playsinline preload=metadata src='{src}'></video>"
                     f"<figcaption>{html.escape(c.name)}</figcaption></figure>")
    return _page(f"Safety {date}", body)


def resolve_file(root: Path, rel: str) -> Path | None:
    """Map a /files/<rel> URL to a file under root, refusing anything that escapes it."""
    root = Path(root).resolve()
    try:
        p = (root / rel).resolve()
    except (OSError, ValueError):
        return None
    if root not in p.parents or not p.is_file():
        return None
    return p


# ----------------------------------------------------------------------------------------------- web app

def make_app(root: str | Path, access_key: str | None = None, refresh=None):
    """FastAPI app over `root`. `refresh` is called before each page (on Modal: reload the volume).
    If `access_key` is set, visitors must open `/?key=<access_key>` once; a cookie remembers them."""
    import hmac

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

    root = Path(root)
    app = FastAPI(title="armlab results", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        if access_key and request.url.path != "/health":
            given = request.query_params.get("key") or request.cookies.get("armlab_key") or ""
            if not hmac.compare_digest(given, access_key):
                return HTMLResponse(_page("Locked", "<h1>Locked</h1><p>Open this page with <code>?key=...</code>.</p>"),
                                    status_code=401)
            if request.query_params.get("key"):
                resp = RedirectResponse(request.url.path, status_code=303)
                resp.set_cookie("armlab_key", access_key, max_age=90 * 24 * 3600, httponly=True, secure=True,
                                samesite="lax")
                return resp
        return await call_next(request)

    def _refresh():
        if refresh:
            try:
                refresh()
            except Exception:  # a stale view is better than an error page
                pass

    @app.get("/health")
    def health():
        import starlette

        return {"ok": True, "starlette": starlette.__version__}

    @app.get("/", response_class=HTMLResponse)
    def index():
        _refresh()
        return render_index(root)

    @app.get("/runs/{name}", response_class=HTMLResponse)
    def run_page(name: str):
        _refresh()
        page = render_run(root, name)
        if page is None:
            raise HTTPException(404, "no such run")
        return page

    @app.get("/api/runs")
    def api_runs():
        _refresh()
        return JSONResponse([{"name": r.name, "mtime": r.mtime, "overall": _overall_line(r.summary)} for r in list_runs(root)])

    @app.get("/safety", response_class=HTMLResponse)
    def safety_index():
        _refresh()
        return render_safety_index(root)

    @app.get("/safety/{date}", response_class=HTMLResponse)
    def safety_report(date: str):
        _refresh()
        page = render_safety_report(root, date)
        if page is None:
            raise HTTPException(404, "no such report")
        return page

    @app.get("/files/{rel:path}")
    def files(rel: str):
        p = resolve_file(root, rel)
        if p is None:
            raise HTTPException(404, "not found")
        media = {".mp4": "video/mp4", ".md": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                 ".json": "application/json", ".jsonl": "application/x-ndjson", ".csv": "text/csv"}.get(p.suffix)
        return FileResponse(p, media_type=media)  # Starlette serves HTTP Range requests, which iOS Safari needs for video

    return app


def app_from_env():
    default = Path(__file__).resolve().parents[2] / "runs"
    return make_app(os.environ.get("ARMLAB_RUNS_ROOT", str(default)), os.environ.get("ARMLAB_RESULTS_KEY") or None)
