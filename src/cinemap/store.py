"""project.json persistence — the single source of truth on disk."""
from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECTS_DIR
from .models import Project


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def save(project: Project) -> None:
    d = project_dir(project.id)
    d.mkdir(parents=True, exist_ok=True)
    project_file(project.id).write_text(project.model_dump_json(indent=2))


def load(project_id: str) -> Project:
    return Project.model_validate_json(project_file(project_id).read_text())


def exists(project_id: str) -> bool:
    return project_file(project_id).exists()


def list_projects() -> list[dict]:
    out = []
    for p in sorted(PROJECTS_DIR.glob("*/project.json")):
        try:
            data = json.loads(p.read_text())
            out.append({"id": data["id"], "name": data.get("name", data["id"]),
                        "keyframes": len(data.get("keyframes", []))})
        except Exception:
            continue
    return out
