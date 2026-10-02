"""Fetch the Franka Panda model from MuJoCo Menagerie (pinned commit, sparse checkout)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
ASSET_DIR = Path(os.environ.get("ARMLAB_ASSETS", Path(__file__).resolve().parent.parent / ".assets"))


def panda_dir() -> Path:
    """Return the directory containing panda.xml, downloading it on first use."""
    target = ASSET_DIR / "mujoco_menagerie" / "franka_emika_panda"
    if (target / "panda.xml").exists():
        return target
    repo = ASSET_DIR / "mujoco_menagerie"
    repo.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    if not (repo / ".git").exists():
        git("init", "-q")
        git("remote", "add", "origin", MENAGERIE_URL)
    git("sparse-checkout", "set", "franka_emika_panda")
    git("fetch", "-q", "--depth", "1", "origin", MENAGERIE_COMMIT)
    git("checkout", "-q", "FETCH_HEAD")
    if not (target / "panda.xml").exists():
        raise RuntimeError(f"Failed to fetch Panda model into {target}")
    return target
