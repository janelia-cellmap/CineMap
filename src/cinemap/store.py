"""project.json persistence — the single source of truth on disk.

Within one server process the parsed Project is cached in memory: load()ing it
costs an NFS read + a full pydantic validation (~5 ms, growing with mesh/segment
count), and the UI reads it on nearly every click (goto, thumbnail, status polls).
The cache holds the live object; save() refreshes it. Assumes a single writer
process (the server) — separate render/CLI processes load fresh from disk.

Each keyframe's `ng_state` (the originating neuroglancer round-trip state, the
single largest field and one that grows with the selected-segment count) is kept
in a per-keyframe sidecar file under `<pid>/ng_states/<kid>.json` and EXCLUDED
from project.json. The hot file the UI saves on every interaction is therefore
small and roughly flat in mesh count, while the in-memory Project still carries
ng_state fully (load() rehydrates it). Sidecars are rewritten only when their
content changes, so opacity/reorder/render saves don't touch them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import PROJECTS_DIR
from .models import Project

# pid -> parsed Project (the live object the UI mutates and save() persists)
_cache: dict[str, Project] = {}
# "pid:kid" -> md5 of the ng_state last written to its sidecar (skip no-op writes)
_ng_hash: dict[str, str] = {}

# exclude ng_state from every keyframe when serializing project.json
_EXCLUDE_NG = {"keyframes": {"__all__": {"ng_state"}}}


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def ng_states_dir(project_id: str) -> Path:
    return project_dir(project_id) / "ng_states"


def _ng_digest(state: dict) -> str:
    return hashlib.md5(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_ng_sidecars(project: Project) -> None:
    """Persist each keyframe's ng_state to its sidecar, but only when changed
    (covers first-time migration of inline ng_state too); prune orphans."""
    nd = ng_states_dir(project.id)
    live_ids = set()
    for kf in project.keyframes:
        if kf.ng_state is None:
            continue
        live_ids.add(kf.id)
        key = f"{project.id}:{kf.id}"
        digest = _ng_digest(kf.ng_state)
        if _ng_hash.get(key) == digest:
            continue  # unchanged since last write — skip the NFS write
        nd.mkdir(parents=True, exist_ok=True)
        (nd / f"{kf.id}.json").write_text(json.dumps(kf.ng_state, separators=(",", ":")))
        _ng_hash[key] = digest
    # prune sidecars (and hashes) for keyframes that no longer exist
    if nd.exists():
        for f in nd.glob("*.json"):
            if f.stem not in live_ids:
                f.unlink(missing_ok=True)
                _ng_hash.pop(f"{project.id}:{f.stem}", None)


def _read_ng_sidecars(project: Project) -> None:
    """Rehydrate ng_state from sidecars for keyframes that don't already carry it
    (inline ng_state in a not-yet-migrated project.json is left as-is)."""
    nd = ng_states_dir(project.id)
    for kf in project.keyframes:
        if kf.ng_state is not None:
            continue  # inline (pre-migration) — keep it; next save migrates it
        f = nd / f"{kf.id}.json"
        if f.exists():
            kf.ng_state = json.loads(f.read_text())
            # came from a sidecar => already persisted; don't rewrite next save
            _ng_hash[f"{project.id}:{kf.id}"] = _ng_digest(kf.ng_state)


def save(project: Project) -> None:
    d = project_dir(project.id)
    d.mkdir(parents=True, exist_ok=True)
    _write_ng_sidecars(project)
    # Compact (no indent) + ng_state in sidecars: project.json is loaded+saved on
    # every interaction over NFS; keeping it small (and flat in mesh count) is the
    # whole point. The in-memory object still carries ng_state.
    project_file(project.id).write_text(project.model_dump_json(exclude=_EXCLUDE_NG))
    _cache[project.id] = project


def load(project_id: str) -> Project:
    cached = _cache.get(project_id)
    if cached is not None:
        return cached
    project = Project.model_validate_json(project_file(project_id).read_text())
    _read_ng_sidecars(project)
    _cache[project_id] = project
    return project


def invalidate(project_id: str) -> None:
    """Drop a project from the in-memory cache (forces a fresh disk read next load)."""
    _cache.pop(project_id, None)


def exists(project_id: str) -> bool:
    return project_id in _cache or project_file(project_id).exists()


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
