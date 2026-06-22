"""Project-wide configuration and paths."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root (…/cinemap) and where projects + render assets live.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTS_DIR = REPO_ROOT / "projects"
PROJECTS_DIR = Path(os.environ.get("CINEMAP_PROJECTS_DIR", DEFAULT_PROJECTS_DIR))
FRONTEND_DIR = REPO_ROOT / "frontend"

# The env's python (used to launch the Blender `bpy` subprocess in isolation).
ENV_PYTHON = os.environ.get("CINEMAP_PYTHON", "")

# World scale: how many nanometres map to one Blender unit. Mesh vertices and EM
# slice positions are in nm; dividing by this keeps Blender working at a sane scale.
NM_PER_BU = float(os.environ.get("CINEMAP_NM_PER_BU", "1000.0"))

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Persisted secrets live in the user's PRIVATE home config dir (NOT the shared repo
# on /groups), with 0600 perms — so the Anthropic key survives restarts safely.
CONFIG_DIR = Path(os.environ.get("CINEMAP_CONFIG_DIR", Path.home() / ".config" / "cinemap"))
_KEY_FILE = CONFIG_DIR / "anthropic_key"


def load_saved_key() -> str | None:
    try:
        return _KEY_FILE.read_text().strip() or None
    except OSError:
        return None


def save_key(key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Create the file already restricted (0600) rather than write-then-chmod, which
    # leaves a window where the secret is world-readable under a 0022 umask.
    fd = os.open(_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key.strip())


def clear_key() -> None:
    try:
        _KEY_FILE.unlink()
    except FileNotFoundError:
        pass
