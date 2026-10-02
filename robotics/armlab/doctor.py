"""armlab-doctor: what can this machine (laptop, CI runner, agent box) reach, and what is missing?

Model keys are meant to live only in Modal secrets (`armlab-llm`, `huggingface`); a control surface only needs a
Modal token. This checks, without ever printing a secret value:

  - is a Modal token configured (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars or ~/.modal.toml)?
  - are api.modal.com and GitHub reachable over HTTPS?
  - with a token: do the Modal secrets, the `armlab-runs` volume and the deployed armlab apps exist?
  - are model keys sitting in this shell's environment (fine for local runs, not needed for Modal runs)?

  armlab-doctor            # human-readable; exits 0 unless --strict and something required is missing
  armlab-doctor --json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_SECRETS = ("armlab-llm", "huggingface")
REQUIRED_VOLUMES = ("armlab-runs",)
EXPECTED_APPS = ("armlab-results", "armlab-safety-cron")
HOSTS = {"api.modal.com": "https://api.modal.com", "github.com": "https://github.com"}
MODEL_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN")

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    required: bool = True


def modal_config_path() -> Path:
    return Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml")


def _load_toml(p: Path) -> dict:
    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return {}
    try:
        return tomllib.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def modal_token_source(env: dict | None = None) -> str | None:
    """Where a usable Modal token comes from ('env' or 'config:<profile>'), or None. Never returns the token."""
    env = os.environ if env is None else env
    if env.get("MODAL_TOKEN_ID") and env.get("MODAL_TOKEN_SECRET"):
        return "env"
    cfg = _load_toml(modal_config_path())
    profiles = {k: v for k, v in cfg.items() if isinstance(v, dict)}
    want = env.get("MODAL_PROFILE")
    if not want:
        active = [k for k, v in profiles.items() if v.get("active")]
        want = active[0] if active else (next(iter(profiles)) if len(profiles) == 1 else None)
    prof = profiles.get(want or "", {})
    if prof.get("token_id") and prof.get("token_secret"):
        return f"config:{want}"
    return None


def check_host(name: str, url: str, timeout: float = 5.0) -> Check:
    import httpx

    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=False)
        return Check(f"reach {name}", OK, f"HTTPS ok (status {r.status_code})")
    except Exception as e:  # DNS failure, proxy block, timeout...
        return Check(f"reach {name}", FAIL, f"unreachable: {type(e).__name__}")


def _modal_cli(*args: str, timeout: float = 30.0) -> list[dict] | str:
    """Run `modal <args> --json`; returns parsed rows or an error string (stderr is not echoed: it can hold URLs/ids)."""
    try:
        p = subprocess.run([sys.executable, "-m", "modal", *args, "--json"], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timed out"
    if p.returncode != 0:
        last = (p.stderr.strip().splitlines() or ["error"])[-1]
        return f"modal {' '.join(args)} failed: {last[:160]}"
    try:
        rows = json.loads(p.stdout)
    except ValueError:
        return "could not parse modal output"
    return rows if isinstance(rows, list) else []


def _names(rows: list[dict], *keys: str) -> set[str]:
    out = set()
    for r in rows:
        for k in keys:
            if r.get(k):
                out.add(str(r[k]))
    return out


def run_checks(env: dict | None = None, network: bool = True, modal_cli=_modal_cli, host_check=check_host) -> list[Check]:
    env = os.environ if env is None else env
    checks: list[Check] = []

    try:
        import modal

        checks.append(Check("modal client", OK, f"modal {modal.__version__} installed"))
        have_modal = True
    except ModuleNotFoundError:
        checks.append(Check("modal client", FAIL, "not installed: pip install -e 'robotics[cloud]'"))
        have_modal = False

    src = modal_token_source(env)
    if src:
        checks.append(Check("modal token", OK, f"present (from {src}); value not shown"))
    else:
        checks.append(Check("modal token", FAIL, "missing: set MODAL_TOKEN_ID + MODAL_TOKEN_SECRET or run `modal token new`"))

    if network:
        checks += [host_check(n, u) for n, u in HOSTS.items()]
    else:
        checks += [Check(f"reach {n}", SKIP, "network checks disabled") for n in HOSTS]

    modal_reachable = any(c.name == "reach api.modal.com" and c.status == OK for c in checks)
    if not (src and have_modal and network and modal_reachable):
        why = "no Modal token" if not src else ("modal not installed" if not have_modal else "api.modal.com not reachable")
        for s in REQUIRED_SECRETS:
            checks.append(Check(f"modal secret {s}", SKIP, f"not checked ({why})"))
        for v in REQUIRED_VOLUMES:
            checks.append(Check(f"modal volume {v}", SKIP, f"not checked ({why})"))
        for a in EXPECTED_APPS:
            checks.append(Check(f"modal app {a}", SKIP, f"not checked ({why})", required=False))
    else:
        for kind, wanted, keys, required, hint in (
            ("secret", REQUIRED_SECRETS, ("Name", "name"), True, "create it with `modal secret create {n} KEY=...`"),
            ("volume", REQUIRED_VOLUMES, ("Name", "name"), True, "created on first `modal run modal_apps/armlab_eval.py`"),
            ("app", EXPECTED_APPS, ("Description", "description", "Name", "name"), False, "deploy with `modal deploy`"),
        ):
            rows = modal_cli(kind, "list")
            if isinstance(rows, str):
                checks += [Check(f"modal {kind} {n}", WARN, rows, required) for n in wanted]
                continue
            names = _names(rows, *keys)
            if kind == "app":  # only count apps that are currently deployed
                names = {str(r.get("Description") or r.get("description") or r.get("Name") or "") for r in rows
                         if str(r.get("State") or r.get("state") or "deployed").lower().startswith("deployed")}
            for n in wanted:
                if n in names:
                    checks.append(Check(f"modal {kind} {n}", OK, "present", required))
                else:
                    checks.append(Check(f"modal {kind} {n}", FAIL if required else WARN, "missing: " + hint.format(n=n),
                                        required))

    local = [k for k in MODEL_KEYS if env.get(k)]
    checks.append(Check("model keys in this shell", OK if not local else WARN,
                        ("none (good: Modal runs read them from the armlab-llm/huggingface secrets)" if not local else
                         f"{', '.join(local)} set locally (values not shown); only needed for local non-Modal runs"),
                        required=False))
    if shutil.which("gh") and network:
        checks.append(_gh_repo_secrets())
    return checks


def _gh_repo_secrets() -> Check:
    """GitHub Actions needs MODAL_TOKEN_ID/MODAL_TOKEN_SECRET repo secrets (names only are listed by gh)."""
    try:
        p = subprocess.run(["gh", "secret", "list", "--json", "name"], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return Check("github repo secrets", WARN, "gh timed out", required=False)
    if p.returncode != 0:
        return Check("github repo secrets", SKIP, "gh not authenticated or not in a GitHub repo", required=False)
    try:
        names = {r["name"] for r in json.loads(p.stdout)}
    except (ValueError, KeyError, TypeError):
        return Check("github repo secrets", WARN, "could not parse gh output", required=False)
    missing = [n for n in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") if n not in names]
    if missing:
        return Check("github repo secrets", WARN, f"missing {', '.join(missing)} (needed by the run-eval workflow)",
                     required=False)
    return Check("github repo secrets", OK, "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET present", required=False)


def format_report(checks: list[Check]) -> str:
    icon = {OK: "[ok]  ", WARN: "[warn]", FAIL: "[FAIL]", SKIP: "[skip]"}
    w = max(len(c.name) for c in checks)
    lines = [f"{icon[c.status]} {c.name:<{w}}  {c.detail}" for c in checks]
    bad = [c for c in checks if c.required and c.status == FAIL]
    lines.append("")
    lines.append("All required checks passed." if not bad else
                 f"{len(bad)} required check(s) failing: {', '.join(c.name for c in bad)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--offline", action="store_true", help="skip network checks")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a required check fails")
    args = ap.parse_args(argv)
    checks = run_checks(network=not args.offline)
    print(json.dumps([asdict(c) for c in checks], indent=2) if args.json else format_report(checks))
    failing = any(c.required and c.status == FAIL for c in checks)
    return 1 if (args.strict and failing) else 0


if __name__ == "__main__":
    sys.exit(main())
