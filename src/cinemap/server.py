"""FastAPI server — REST + websocket + the neuroglancer scouting viewer.

Routes are thin wrappers over operations.* (the single funnel). Renders run in a
background thread; progress streams over a websocket to the timeline.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response


def _image_response(path: str, media: str = "image/png", cacheable: bool = False) -> Response:
    """Serve a small image from in-memory bytes so Content-Length always matches —
    these files (thumbnails, live frames) can be overwritten while being served.
    `cacheable` lets the browser cache (used with versioned thumbnail URLs)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        raise HTTPException(404, "not available")
    cc = "public, max-age=31536000, immutable" if cacheable else "no-store"
    return Response(content=data, media_type=media, headers={"Cache-Control": cc})
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, operations as ops
from . import scouting, store
from .config import FRONTEND_DIR
from .models import Project, RenderJob, RenderPrefs, RenderSettings
from .render.worker import RenderCancelled, RenderWorker

app = FastAPI(title="CineMap")

# Restore a previously-saved Anthropic key (from the private home config dir) so
# the agent works after a restart without re-entering it.
if not os.environ.get("ANTHROPIC_API_KEY"):
    _saved = config.load_saved_key()
    if _saved:
        os.environ["ANTHROPIC_API_KEY"] = _saved

# in-memory render progress: job_id -> {progress, message, status, output}
_render_state: dict[str, dict] = {}
# live workers (so a render can be cancelled): job_id -> RenderWorker (thumbnails)
# OR _SubprocHandle (full video renders). Both expose .terminate().
_workers: dict[str, Any] = {}


class _SubprocHandle:
    """Wraps the Popen of a `cinemap.render.run_job` subprocess so it presents the same
    `.terminate()` + `.frames_dir` interface as an in-process RenderWorker.
    terminate() sends SIGKILL to the entire process group — Blender + cloud-volume
    mp pool + asset threads all die in one syscall, no orphans, no half-cancelled
    tasks."""

    def __init__(self, proc, frames_dir):
        self.proc = proc
        # Mirror RenderWorker.frames_dir so `latest_frame` works the same way for
        # both handle types — no special case in the live-preview endpoint.
        self.frames_dir = frames_dir

    def terminate(self) -> None:
        import signal as _signal
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(p.pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass


@app.on_event("shutdown")
def _shutdown_cleanup() -> None:
    """On a graceful stop (SIGTERM/Ctrl-C, e.g. a restart), tear down in-flight renders so
    nothing they launched (Blender, cloud-volume mp pool, etc.) gets reparented to init
    and keeps eating CPU. Subprocess renders terminate their whole group; thumbnail
    threads cancel their Blender child. Then SIGKILL any leftover direct children."""
    import signal as _signal
    for w in list(_workers.values()):
        try:
            w.terminate()
        except Exception:  # noqa: BLE001
            pass
    try:  # kill any child processes this server spawned that didn't exit on their own
        for child_pid in _own_child_pids():
            try:
                os.kill(child_pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:  # noqa: BLE001
        pass


def _own_child_pids() -> list[int]:
    """PIDs whose parent is this process (Blender subprocess, multiprocessing workers)."""
    me = os.getpid()
    out = []
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/stat") as f:
                    ppid = int(f.read().split(") ", 1)[1].split()[1])
                if ppid == me:
                    out.append(int(pid))
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass
    return out


# ----------------------------- request bodies -----------------------------
class CreateProject(BaseModel):
    name: str
    data_path: str


class OrbitReq(BaseModel):
    degrees: float = 360.0
    n: int = 12
    elevation_deg: float = 22.0
    total_duration_s: float | None = None
    axis: str = "z"          # orbit pole: 'z' circles in XY, 'y' in XZ, 'x' in YZ


class SweepReq(BaseModel):
    axis: str = "z"
    n: int = 12


class SweepClipReq(BaseModel):
    mesh_name: str | None = None
    axis: str = "z"
    n: int = 12
    side: int = 1


class PlaneMoveReq(BaseModel):
    axis: str = "z"
    mode: str = "slice"          # 'slice' | 'cull' | 'both'
    start_nm: float | None = None
    stop_nm: float | None = None
    from_xyz: list[float] | None = None   # oblique scan: A -> B points (nm)
    to_xyz: list[float] | None = None
    from_ng: list[float] | None = None    # A/B as neuroglancer coords (its units + order)
    to_ng: list[float] | None = None
    n: int = 12
    mesh_name: str | None = None
    side: int = 1
    total_duration_s: float | None = None   # duration-driven scan (spans the whole sweep)


class RenderReq(BaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 30
    samples: int = 48
    noise_threshold: float = 0.01   # adaptive-sampling bail threshold (higher = faster)
    # render engine: CYCLES (path-traced, photoreal, slow) or BLENDER_EEVEE_NEXT
    # (rasterized, ~10-50x faster per frame — ideal for fast previews / movie drafts).
    engine: Literal["CYCLES", "BLENDER_EEVEE_NEXT"] = "CYCLES"
    kf_range: list[int] | None = None
    export_blend: bool = False  # produce a self-contained .blend instead of a video
    draft: bool = False         # fast low-res preview (coarse EM + low-voxel meshes)
    mesh_detail: float = 1.0    # per-layer vertex-budget multiplier (hard-capped)
    mesh_from_labels: bool = False  # regenerate render meshes from labels with zmesh
    auto_direct: bool = True    # non-destructive presentation pass (lighting/material/DOF)
    lod_mode: str = "frame"     # mesh LOD: "single" | "frame" (per-frame adaptive) | "chunk"
    show_bbox: bool = False     # draw a wireframe box around each data source's extent
    bbox_color: list[float] | None = None   # wireframe rgb (0–1); None = default gray
    bbox_source: str = ""       # layer name to box (even if hidden); "" = auto


class ThumbnailReq(BaseModel):
    mesh_from_labels: bool = False  # keep thumbnails source-compatible with previews


class ChatReq(BaseModel):
    message: str
    history: list[dict] = []


class KeyReq(BaseModel):
    key: str


# ----------------------------- config / agent key -----------------------------
@app.get("/api/config")
def get_config():
    return {"has_anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "agent_model": os.environ.get("CINEMAP_AGENT_MODEL", "claude-sonnet-4-6")}


@app.post("/api/config/anthropic_key")
def set_anthropic_key(req: KeyReq):
    """Set the Anthropic key and persist it (private home config dir, 0600) so it
    survives restarts."""
    key = req.key.strip()
    if not key:
        raise HTTPException(400, "empty key")
    os.environ["ANTHROPIC_API_KEY"] = key
    config.save_key(key)
    return {"ok": True, "has_anthropic_key": True}


@app.delete("/api/config/anthropic_key")
def clear_anthropic_key():
    """Forget the saved key (remove from disk + this process)."""
    config.clear_key()
    os.environ.pop("ANTHROPIC_API_KEY", None)
    return {"ok": True, "has_anthropic_key": False}


# ----------------------------- projects -----------------------------
@app.get("/api/projects")
def list_projects():
    return store.list_projects()


@app.post("/api/projects")
def create_project(req: CreateProject):
    try:
        project = ops.create_project(req.name, req.data_path)
        scouting.load_dataset(req.data_path)
        return _light_project(project)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not analyze dataset: {e}") from e


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    if not store.exists(pid):
        raise HTTPException(404, "no such project")
    return _light_project(store.load(pid))


def _light_project(project) -> dict:
    """Project dict without each keyframe's huge neuroglancer state (the frontend
    never uses it) — keeps these responses small and the browser memory low."""
    d = project.model_dump()
    for k in d.get("keyframes", []):
        k.pop("ng_state", None)
    return d


@app.get("/api/projects/{pid}/export")
def export_project(pid: str):
    """Download the project as a portable .cinemap.json file (state, not video)."""
    if not store.exists(pid):
        raise HTTPException(404, "no such project")
    p = store.load(pid)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (p.name or pid))
    # Serialize the in-memory project (ng_state rehydrated from sidecars) rather
    # than streaming project.json — on disk ng_state lives in sidecars, so the
    # raw file alone would export an incomplete (non-round-tripping) state.
    return Response(content=p.model_dump_json(), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.cinemap.json"'})


@app.post("/api/projects/import")
def import_project(body: dict):
    """Create a project from an uploaded .cinemap.json (a fresh copy with a new id)."""
    try:
        p = Project.model_validate(body)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"invalid project file: {e}") from e
    p.id = ops.project_id(p.name)  # fresh id so import never clobbers an existing project
    p.renders = []              # drop render history (output files won't exist)
    for kf in p.keyframes:      # thumbnails live in the OLD project dir -> stale paths; regenerate
        kf.thumbnail_path = None
    store.save(p)
    try:
        scouting.load_dataset(p.data_path)
    except Exception as e:  # noqa: BLE001
        print(f"[import] load_dataset failed: {e}", flush=True)
    # Drive the viewer to the first keyframe right away — without this, an import
    # leaves the iframe on whatever the dataset's default state is (often empty),
    # which looks broken to the user. Same UX as clicking the first keyframe row.
    # NOTE: deliberately NOT inside the try above — load_dataset failing shouldn't
    # block goto, and goto failing shouldn't be swallowed silently.
    if p.keyframes:
        try:
            scouting.goto_keyframe(p, p.keyframes[0].id)
        except Exception as e:  # noqa: BLE001
            print(f"[import] goto first keyframe failed: {e}", flush=True)
    return p.model_dump()


@app.put("/api/projects/{pid}/render_prefs")
def set_render_prefs(pid: str, body: dict):
    """Persist the UI's render-control choices on the project (so they survive reload
    and travel with export). Merges into any existing prefs."""
    if not store.exists(pid):
        raise HTTPException(404, "no such project")
    p = store.load(pid)
    base = p.render_prefs.model_dump() if p.render_prefs else {}
    p.render_prefs = RenderPrefs.model_validate({**base, **body})
    store.save(p)
    return p.render_prefs.model_dump()


def _ng_url_for(request: Request) -> str:
    """The neuroglancer viewer URL with its host rewritten to whatever host the
    browser used to reach this app. NG binds to 0.0.0.0 and would otherwise hand
    back a 0.0.0.0/127.0.0.1 URL that a remote browser can't open."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(scouting.viewer_url())
    host = request.url.hostname or "127.0.0.1"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


@app.post("/api/projects/{pid}/open")
def open_project(pid: str, request: Request):
    p = store.load(pid)
    scouting.load_dataset(p.data_path)
    # Drive the viewer to the first keyframe so opening a project shows something
    # immediately — otherwise the iframe is blank/default until the user clicks a row.
    if p.keyframes:
        try:
            scouting.goto_keyframe(p, p.keyframes[0].id)
        except Exception:  # noqa: BLE001
            pass
    return {"ng_url": _ng_url_for(request)}


# ----------------------------- scouting -----------------------------
@app.get("/api/ng_url")
def ng_url(request: Request):
    return {"ng_url": _ng_url_for(request)}


@app.post("/api/projects/{pid}/bake")
def bake(pid: str):
    p = store.load(pid)
    kf = scouting.bake_keyframe(p)
    return kf.model_dump()


@app.post("/api/projects/{pid}/sync_layers")
def sync_layers(pid: str):
    """Union the live neuroglancer view's layers into the manifest (no bake) so a
    volume just added in the viewer becomes sliceable. Returns the updated project."""
    if not store.exists(pid):
        raise HTTPException(404, "no such project")
    p = scouting.sync_manifest_from_view(store.load(pid))
    return _light_project(p)


@app.post("/api/projects/import_states")
def import_states_new_project(body: dict):
    """Create a project from an uploaded states list and bake a keyframe per state.
    The project's dataset is taken from the FIRST state, so meshes/segments resolve
    against the same data the states use (no project needs to exist first)."""
    from .data.manifest import parse_state_links

    links = parse_state_links(body.get("text", ""))
    if not links:
        raise HTTPException(400, "no states found in the uploaded file")
    try:
        project = ops.create_project(body.get("name") or "imported states", links[0][1])
        scouting.load_dataset(links[0][1])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not analyze the first state: {e}") from e
    created, errors = scouting.import_states(project, links)
    # `import_states` bakes each state by driving the viewer to it -> the viewer ends
    # up sitting on the LAST state. Send it back to the FIRST keyframe so the user
    # sees the start of their timeline, matching what /api/projects/import does.
    final = store.load(project.id)
    if final.keyframes:
        try:
            scouting.goto_keyframe(final, final.keyframes[0].id)
        except Exception:  # noqa: BLE001
            pass
    return {"project": _light_project(final),
            "count": len(created), "errors": errors}


@app.post("/api/projects/{pid}/keyframes/import_states")
def import_states(pid: str, body: dict):
    """Bake a keyframe per state in an uploaded list. `body.text` is the raw file
    content: one neuroglancer state link per line, a neuroglancer video_tool script,
    or a CSV with a state column. Each becomes a keyframe just like a hand-baked view."""
    from .data.manifest import parse_state_links

    p = store.load(pid)
    links = parse_state_links(body.get("text", ""))
    if not links:
        raise HTTPException(400, "no states found in the uploaded file")
    created, errors = scouting.import_states(p, links)
    return {"created": [k.id for k in created], "count": len(created), "errors": errors}


@app.post("/api/projects/{pid}/keyframes/{kid}/goto")
def goto_keyframe(pid: str, kid: str):
    p = store.load(pid)
    ok = scouting.goto_keyframe(p, kid)
    if not ok:
        raise HTTPException(404, "no such keyframe")
    return {"ok": True}


@app.post("/api/projects/{pid}/keyframes/{kid}/thumbnail")
def render_thumbnail(pid: str, kid: str, req: ThumbnailReq | None = None):
    """Render this single keyframe to a still and keep it as the frame's thumbnail."""
    p = store.load(pid)
    idx = next((i for i, k in enumerate(p.keyframes) if k.id == kid), None)
    if idx is None:
        raise HTTPException(404, "no such keyframe")
    req = req or ThumbnailReq()
    settings = RenderSettings(width=640, height=480, samples=24, fps=1, draft=True,
                              still=True, mesh_from_labels=req.mesh_from_labels)
    job_id = _start_render(pid, settings, kf_range=[idx, idx], thumbnail_for=kid)
    return {"job_id": job_id}


@app.get("/api/projects/{pid}/keyframes/{kid}/thumbnail")
def get_thumbnail(pid: str, kid: str):
    p = store.load(pid)
    kf = next((k for k in p.keyframes if k.id == kid), None)
    if not kf or not kf.thumbnail_path or not os.path.exists(kf.thumbnail_path):
        raise HTTPException(404, "no thumbnail yet")
    # tolerate old .png thumbnails alongside new compressed .jpg ones
    media = "image/jpeg" if kf.thumbnail_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return _image_response(kf.thumbnail_path, media=media, cacheable=True)  # URL is versioned (?v=)


@app.post("/api/projects/{pid}/keyframes/{kid}/update_from_ng")
def update_from_ng(pid: str, kid: str):
    """Overwrite this keyframe with the current Neuroglancer state (camera + layers
    + segments), keeping its timing — i.e. 'update current frame'. Also returns the
    per-layer setting `changes` (color/opacity/etc.) vs the previous version, so the UI
    can offer to propagate them to other keyframes."""
    p = store.load(pid)
    kf_old = next((k for k in p.keyframes if k.id == kid), None)
    old_meshes = [m.model_copy(deep=True) for m in kf_old.meshes] if kf_old else []
    kf = scouting.update_keyframe_from_view(p, kid)
    if kf is None:
        raise HTTPException(404, "no such keyframe")
    changes = ops.diff_layer_settings(old_meshes, kf.meshes)
    return {**kf.model_dump(), "changes": changes}


# ----------------------------- keyframes -----------------------------
class DuplicateKeyframeReq(BaseModel):
    after_id: str | None = None


@app.post("/api/projects/{pid}/keyframes")
def add_keyframe(pid: str):
    p = store.load(pid)
    return ops.add_keyframe(p).model_dump()


@app.post("/api/projects/{pid}/keyframes/{kid}/duplicate")
def duplicate_keyframe(pid: str, kid: str, req: DuplicateKeyframeReq):
    p = store.load(pid)
    try:
        return ops.duplicate_keyframe(p, kid, after_id=req.after_id).model_dump()
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@app.delete("/api/projects/{pid}/keyframes/{kid}")
def delete_keyframe(pid: str, kid: str):
    p = store.load(pid)
    ops.delete_keyframe(p, kid)
    return {"ok": True}


class ReorderReq(BaseModel):
    order: list[str]  # keyframe ids in the new order


class MeshOpacityReq(BaseModel):
    opacity: float


class PropagateReq(BaseModel):
    mesh_name: str
    field: str                       # MeshInstance field, or "segment_color"
    value: Any = None                # new value (e.g. [r,g,b], float, int, bool)
    segment_id: int | None = None    # for field == "segment_color"
    direction: str = "right"         # this | right (later) | left (earlier) | all
    match_old: bool = True           # only change keyframes currently holding the OLD value
    match_value: Any = None          # explicit OLD value to match (used when the source was
    match_value_set: bool = False    # already updated, e.g. from the NG view)


@app.get("/api/projects/{pid}/keyframes/{kid}/layers")
def keyframe_layers(pid: str, kid: str):
    """The editable per-layer settings of one keyframe (for the propagate editor)."""
    p = store.load(pid)
    kf = next((k for k in p.keyframes if k.id == kid), None)
    if kf is None:
        raise HTTPException(404, "no such keyframe")
    return {"layers": [
        {"mesh_name": m.mesh_name, "render_3d": m.render_3d, "visible": m.visible,
         "color_seed": m.color_seed, "default_color": m.default_color,
         "segment_colors": m.segment_colors, "segment_ids": m.segment_ids[:200],
         "object_alpha": m.object_alpha, "silhouette": m.silhouette,
         "saturation": m.saturation, "color": m.color}
        for m in kf.meshes]}


@app.post("/api/projects/{pid}/keyframes/{kid}/propagate")
def propagate_layer(pid: str, kid: str, req: PropagateReq):
    """Edit a layer setting on this keyframe and propagate it to others (replace-where-
    matching by default), so a change (e.g. a segment's color) carries to later/earlier
    snapshots without editing each by hand."""
    p = store.load(pid)
    kw = {"match_value": req.match_value} if req.match_value_set else {}
    try:
        res = ops.propagate_layer_field(
            p, kid, req.mesh_name, req.field, req.value, segment_id=req.segment_id,
            direction=req.direction, match_old=req.match_old, **kw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, **res, "count": len(res["changed"])}


class LookReq(BaseModel):
    look: dict[str, Any] = {}


@app.get("/api/projects/{pid}/look")
def get_look(pid: str):
    """The project's render-look overrides (preset + glow/roughness/etc.)."""
    return {"look": store.load(pid).look or {}}


@app.post("/api/projects/{pid}/look")
def set_look(pid: str, req: LookReq):
    """Set the project's render-look overrides; affects subsequent renders/thumbnails."""
    p = store.load(pid)
    p.look = req.look or {}
    store.save(p)
    return {"ok": True, "look": p.look}


@app.get("/api/projects/{pid}/keyframes/diffs")
def keyframe_diffs(pid: str):
    """Per-layer setting changes for each keyframe vs the PREVIOUS one (color/opacity/
    silhouette/seed/per-segment color). Powers the timeline 'what changed' hover badges."""
    p = store.load(pid)
    out = []
    for i, k in enumerate(p.keyframes):
        changes = ops.diff_layer_settings(p.keyframes[i - 1].meshes, k.meshes) if i > 0 else []
        out.append({"keyframe_id": k.id, "changes": changes})
    return {"diffs": out}


@app.post("/api/projects/{pid}/keyframes/{kid}/mesh_opacity")
def set_mesh_opacity(pid: str, kid: str, req: MeshOpacityReq):
    """Set the 3D mesh opacity for a keyframe (0 = hidden -> EM slice + its
    segmentation overlay show without the 3D meshes occluding)."""
    p = store.load(pid)
    kf = next((k for k in p.keyframes if k.id == kid), None)
    if kf is None:
        raise HTTPException(404, "no such keyframe")
    op = max(0.0, min(1.0, req.opacity))
    meshes = [m.model_copy(update={"opacity": op, "visible": op > 0.001}) for m in kf.meshes]
    ops.update_keyframe(p, kid, meshes=meshes)
    return {"ok": True, "opacity": op}


class DurationReq(BaseModel):
    duration_in_s: float | None = None
    hold_in_s: float | None = None
    easing: str | None = None
    transition: str | None = None
    layer_transition: str | None = None
    layer_transition_at: float | None = None


@app.put("/api/projects/{pid}/keyframes/{kid}/duration")
def set_keyframe_duration(pid: str, kid: str, req: DurationReq):
    """Set a keyframe's transition TIMING — duration (seconds the camera/slice glides INTO
    this keyframe), hold time after it, camera transition, and layer transition."""
    p = store.load(pid)
    kf = next((k for k in p.keyframes if k.id == kid), None)
    if kf is None:
        raise HTTPException(404, "no such keyframe")
    fields = {}
    if req.duration_in_s is not None:
        fields["duration_in_s"] = max(0.0, float(req.duration_in_s))
    if req.hold_in_s is not None:
        fields["hold_in_s"] = max(0.0, float(req.hold_in_s))
    if req.easing:
        fields["easing"] = req.easing
    if req.transition:
        if req.transition not in ("glide", "cut", "fade"):
            raise HTTPException(400, "transition must be glide, cut, or fade")
        fields["transition"] = req.transition
    if req.layer_transition:
        if req.layer_transition not in ("fade", "cut"):
            raise HTTPException(400, "layer_transition must be fade or cut")
        fields["layer_transition"] = req.layer_transition
    if req.layer_transition_at is not None:
        fields["layer_transition_at"] = max(0.0, min(1.0, float(req.layer_transition_at)))
    ops.update_keyframe(p, kid, **fields)
    return {"ok": True, **fields}


@app.post("/api/projects/{pid}/keyframes/reorder")
def reorder_keyframes(pid: str, req: ReorderReq):
    p = store.load(pid)
    ops.reorder_keyframes(p, req.order)
    return {"ok": True, "count": len(p.keyframes)}


def _scan_resp(kfs):
    """Response for a generated scan/orbit. Also names the representative (middle)
    keyframe so the client can render just that one thumbnail — the collapsed/stacked
    timeline card shows the middle frame, so without this a fresh scan's card is blank."""
    mid = kfs[len(kfs) // 2] if kfs else None
    return {"added": len(kfs), "group": getattr(mid, "group", None),
            "thumb_id": getattr(mid, "id", None)}


@app.post("/api/projects/{pid}/orbit")
def orbit(pid: str, req: OrbitReq):
    p = store.load(pid)
    kfs = ops.make_orbit(p, degrees=req.degrees, n=req.n, elevation_deg=req.elevation_deg,
                         total_duration_s=req.total_duration_s, axis=req.axis)
    return _scan_resp(kfs)


@app.post("/api/projects/{pid}/sweep_slice")
def sweep(pid: str, req: SweepReq):
    p = store.load(pid)
    kfs = ops.sweep_slice(p, axis=req.axis, n=req.n)
    return _scan_resp(kfs)


@app.post("/api/projects/{pid}/sweep_clip")
def sweep_clip(pid: str, req: SweepClipReq):
    p = store.load(pid)
    kfs = ops.sweep_clip(p, mesh_name=req.mesh_name, axis=req.axis, n=req.n, side=req.side)
    return _scan_resp(kfs)


@app.post("/api/projects/{pid}/plane_move")
def plane_move(pid: str, req: PlaneMoveReq):
    p = store.load(pid)
    kfs = ops.plane_move(p, axis=req.axis, mode=req.mode, start_nm=req.start_nm,
                         stop_nm=req.stop_nm, from_xyz=req.from_xyz, to_xyz=req.to_xyz,
                         from_ng=req.from_ng, to_ng=req.to_ng,
                         n=req.n, mesh_name=req.mesh_name, side=req.side,
                         total_duration_s=req.total_duration_s)
    return _scan_resp(kfs)


class SweepReqNew(BaseModel):
    kind: str = "cutaway"                # 'cutaway' (clip a mesh layer) or 'slice' (EM plane)
    layer: str = ""                      # mesh layer (cutaway)
    em_name: str = ""                    # EM layer (slice)
    axis: str = "z"
    side: int = 1
    from_ng: list[float] | None = None   # NG coords: triple => oblique A->B, single => depth
    to_ng: list[float] | None = None
    from_nm: float | None = None
    to_nm: float | None = None
    start_s: float | None = None
    duration_s: float | None = None
    easing: str = "linear"
    mirror: bool = False
    cap: bool = True


@app.post("/api/projects/{pid}/sweeps")
def add_sweep(pid: str, req: SweepReqNew):
    """Create an independent plane sweep (cutaway or EM slice) on its own timeline,
    decoupled from the camera keyframes."""
    p = store.load(pid)
    sw = ops.add_sweep(p, kind=req.kind, layer=req.layer, em_name=req.em_name,
                       axis=req.axis, side=req.side,
                       from_ng=req.from_ng, to_ng=req.to_ng,
                       from_nm=req.from_nm, to_nm=req.to_nm,
                       start_s=req.start_s, duration_s=req.duration_s, easing=req.easing,
                       mirror=req.mirror, cap=req.cap)
    return sw.model_dump()


class SweepPatch(BaseModel):
    start_s: float | None = None
    duration_s: float | None = None
    side: int | None = None
    easing: str | None = None
    from_nm: float | None = None
    to_nm: float | None = None
    axis: str | None = None
    mirror: bool | None = None
    cap: bool | None = None
    enabled: bool | None = None


@app.put("/api/projects/{pid}/sweeps/{sid}")
def update_sweep(pid: str, sid: str, req: SweepPatch):
    p = store.load(pid)
    sw = ops.update_sweep(p, sid, **req.model_dump(exclude_none=True))
    if sw is None:
        raise HTTPException(404, "no such sweep")
    return sw.model_dump()


@app.delete("/api/projects/{pid}/sweeps/{sid}")
def delete_sweep(pid: str, sid: str):
    p = store.load(pid)
    ops.remove_sweep(p, sid)
    return {"ok": True}


# clip snapshot render state: "pid/sid" -> {status, count}
_snap_state: dict[str, dict] = {}


def _snapshot_times(sw) -> list[float]:
    """Global times (s) to sample a sweep for its preview stills — 3 across the sweep
    (5 for a mirror, since start≈end) so the strip shows the progression (cutaway
    revealing, slice sweeping). The mesh geometry is decoded once and reused across
    these frames, so the cost is dominated by the first decode, not the frame count."""
    n = 5 if getattr(sw, "mirror", False) else 3
    dur = sw.duration_s or 1e-9
    return [sw.start_s + dur * k / (n - 1) for k in range(n)]


@app.post("/api/projects/{pid}/sweeps/{sid}/snapshots")
def render_sweep_snapshots(pid: str, sid: str):
    """Kick off (in the background) a few small preview stills of a clip so the timeline
    bar can show what the sweep looks like. A MIRROR sweep gets 5 evenly-spaced frames
    (0/25/50/75/100%) since start & end look identical; others get 3 (start/mid/end).
    Returns immediately; poll …/snapshots/status."""
    import shutil
    p = store.load(pid)
    sw = next((s for s in p.sweeps if s.id == sid), None)
    if sw is None:
        raise HTTPException(404, "no such sweep")
    times = _snapshot_times(sw)
    n = len(times)
    out = config.PROJECTS_DIR / pid / "assets" / "snapshots" / sid
    key = f"{pid}/{sid}"
    _snap_state[key] = {"status": "running", "count": n}

    job_id = f"snap_{sid}"

    def _run():
        settings = RenderSettings(width=240, height=160, samples=12, fps=2, draft=True)
        worker = RenderWorker(p, RenderJob(id=job_id, settings=settings))
        _workers[job_id] = worker   # registered so Stop/cancel + shutdown can kill it
        try:
            shutil.rmtree(out, ignore_errors=True)
            paths = worker.render_snapshots(times, out)
            _snap_state[key] = ({"status": "cancelled", "count": 0} if worker.cancel.is_set()
                                else {"status": "done", "count": len(paths)})
        except Exception as e:  # noqa: BLE001
            _snap_state[key] = {"status": "error", "error": str(e)}
        finally:
            _workers.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "running", "count": n, "job_id": job_id}


@app.get("/api/projects/{pid}/sweeps/{sid}/snapshots/status")
def snapshot_status(pid: str, sid: str):
    return _snap_state.get(f"{pid}/{sid}", {"status": "idle", "count": 0})


@app.post("/api/projects/{pid}/sweeps/{sid}/adopt_preview")
def adopt_preview(pid: str, sid: str):
    """Hand the just-rendered '_preview' stills to a newly-created sweep, so the preview
    frames stick to the clip without re-rendering."""
    base = config.PROJECTS_DIR / pid / "assets" / "snapshots"
    prev, dest = base / "_preview", base / sid
    if not prev.exists():
        return {"ok": False, "count": 0}
    import shutil
    shutil.rmtree(dest, ignore_errors=True)
    prev.rename(dest)
    st = _snap_state.pop(f"{pid}/_preview", {"status": "done", "count": 0})
    _snap_state[f"{pid}/{sid}"] = st
    return {"ok": True, "count": st.get("count", 0)}


@app.post("/api/projects/{pid}/sweeps/preview")
def preview_sweep(pid: str, req: SweepReqNew):
    """Render preview stills for a sweep BEFORE committing it — so you can see a cutaway/
    slice in the create panel without adding it to the timeline first. Renders into the
    '_preview' snapshot slot (served via …/sweeps/_preview/snapshot/{i})."""
    import shutil
    p = store.load(pid)
    pc = p.model_copy(deep=True)               # work on a COPY; never saved
    sw = ops.add_sweep(pc, kind=req.kind, layer=req.layer, em_name=req.em_name, axis=req.axis,
                       side=req.side, from_ng=req.from_ng, to_ng=req.to_ng,
                       from_nm=req.from_nm, to_nm=req.to_nm, start_s=req.start_s,
                       duration_s=req.duration_s, easing=req.easing, mirror=req.mirror,
                       cap=req.cap, commit=False)
    times = _snapshot_times(sw)
    n = len(times)
    out = config.PROJECTS_DIR / pid / "assets" / "snapshots" / "_preview"
    key = f"{pid}/_preview"
    _snap_state[key] = {"status": "running", "count": n}

    def _run():
        try:
            shutil.rmtree(out, ignore_errors=True)
            settings = RenderSettings(width=240, height=160, samples=12, fps=2, draft=True)
            worker = RenderWorker(pc, RenderJob(id="snap_preview", settings=settings))
            paths = worker.render_snapshots(times, out)
            _snap_state[key] = {"status": "done", "count": len(paths)}
        except Exception as e:  # noqa: BLE001
            _snap_state[key] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "running", "count": n}


@app.get("/api/projects/{pid}/sweeps/{sid}/snapshot/{idx}")
def get_sweep_snapshot(pid: str, sid: str, idx: int):
    path = config.PROJECTS_DIR / pid / "assets" / "snapshots" / sid / f"frame_{idx:05d}.png"
    return _image_response(str(path), cacheable=False)


# ----------------------------- render -----------------------------
def _run_thumbnail_render(pid: str, job_id: str, worker: RenderWorker, thumbnail_for: str):
    """Thumbnail/preview path — in-process daemon thread. Sub-second jobs where the
    subprocess fork overhead would dominate; full video renders go through
    `_launch_render_subprocess` instead."""
    def cb(pr, msg):
        # keep status as "cancelling" once requested, until the worker bails out
        cur = _render_state.get(job_id, {}).get("status")
        status = "cancelling" if cur == "cancelling" else "running"
        _render_state[job_id] = {"progress": pr, "message": msg, "status": status}

    # A thumbnail job is NOT in project.renders — it only ever touches the project if
    # it sets a keyframe's thumbnail_path. Track that so we don't rewrite the whole
    # project.json over NFS when nothing changed (cancelled/failed thumbnails, or
    # thumbnails where the keyframe is gone).
    dirty = False
    try:
        out = worker.run(progress=cb)
        if out:  # keep a per-keyframe thumbnail and point the kf at it
            from PIL import Image

            tdir = config.PROJECTS_DIR / pid / "thumbnails"
            tdir.mkdir(parents=True, exist_ok=True)
            dst = tdir / f"{thumbnail_for}.jpg"
            tmp = tdir / f".{thumbnail_for}.jpg.tmp"
            # Re-encode the lossless render PNG to a compressed JPEG: it's shown at
            # 150px in the timeline card and ~medium in the preview pane, so the
            # full PNG (hundreds of KB) is wasteful. JPEG q82 keeps it sharp at a
            # fraction of the bytes (no alpha needed — the render has an opaque bg).
            with Image.open(out) as im:
                im.convert("RGB").save(tmp, "JPEG", quality=82, optimize=True)
            os.replace(tmp, dst)  # atomic swap so a concurrent GET never reads a half-written file
            for k in worker.project.keyframes:
                if k.id == thumbnail_for:
                    k.thumbnail_path = str(dst)
                    dirty = True
            out = str(dst)
        _render_state[job_id] = {"progress": 1.0, "message": "done", "status": "done", "output": out}
    except RenderCancelled:
        _render_state[job_id] = {"progress": 0.0, "message": "cancelled", "status": "cancelled"}
    except Exception as e:  # noqa: BLE001
        _render_state[job_id] = {"progress": 0.0, "message": str(e), "status": "error"}
    finally:
        if dirty:
            store.save(worker.project)
        _workers.pop(job_id, None)


def _start_render(pid: str, settings: RenderSettings, kf_range=None, thumbnail_for=None) -> str:
    """Kick off a render. THUMBNAIL jobs (sub-second, frequent) run in-process on a
    daemon thread — subprocess fork overhead would dominate. FULL VIDEO renders run
    in their own process group via `cinemap.render.run_job`, so Stop can kill the
    whole tree (Blender + cloud-volume mp pool + asset threads) in one syscall."""
    p = store.load(pid)
    _evict_finished_states()  # keep the in-memory job table from growing forever
    if thumbnail_for:
        job = RenderJob(id=ops.render_id(p, "thumb"), kf_range=kf_range, settings=settings)
        worker = RenderWorker(p, job)
        _workers[job.id] = worker
        _render_state[job.id] = {"progress": 0.0, "message": "queued", "status": "pending"}
        threading.Thread(target=_run_thumbnail_render,
                         args=(pid, job.id, worker, thumbnail_for), daemon=True).start()
        return job.id
    # Full video: persist the job to project.renders first (so the subprocess can find
    # it by id), then spawn run_job in its own session.
    job = ops.create_render_job(p, settings, kf_range=kf_range)
    _render_state[job.id] = {"progress": 0.0, "message": "queued", "status": "pending"}
    _launch_render_subprocess(pid, job.id)
    return job.id


def _launch_render_subprocess(pid: str, job_id: str) -> None:
    """Spawn `python -m cinemap.render.run_job <pid> <job_id>` in its own session,
    register a `_SubprocHandle` (so cancel can SIGKILL the whole group), and start a
    reader thread that translates the subprocess's JSON-line progress into the
    in-memory render state."""
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-m", "cinemap.render.run_job", pid, job_id],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,   # own process group -> SIGKILL takes everything down
    )
    # frames_dir must mirror RenderWorker's construction (worker.py:100-101).
    frames_dir = config.PROJECTS_DIR / pid / "renders" / job_id / "frames"
    _workers[job_id] = _SubprocHandle(proc, frames_dir)
    threading.Thread(target=_drain_render_subprocess,
                     args=(pid, job_id, proc), daemon=True).start()


def _drain_render_subprocess(pid: str, job_id: str, proc) -> None:
    """Read JSON-line progress from the render subprocess and mirror onto `_render_state`.
    On `final` line OR EOF, finalize the state and unregister the handle. stderr is
    drained on a sibling thread so a chatty subprocess can't deadlock on a full pipe."""
    import json as _json

    def _drain_stderr():
        for line in proc.stderr:  # type: ignore
            print(f"[run_job {job_id}] {line.rstrip()}", flush=True)

    threading.Thread(target=_drain_stderr, daemon=True).start()

    final: dict | None = None
    try:
        for line in proc.stdout:  # type: ignore
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                # any non-JSON output from the subprocess goes to the server log
                print(f"[run_job {job_id}] {line}", flush=True)
                continue
            if "final" in obj:
                final = obj
                break
            if "p" in obj or "m" in obj:
                # cancel requests flip status to 'cancelling' — keep that label until
                # the subprocess actually exits, then we overwrite below.
                cur = _render_state.get(job_id, {}).get("status")
                status = "cancelling" if cur == "cancelling" else "running"
                _render_state[job_id] = {"progress": float(obj.get("p", 0.0)),
                                         "message": str(obj.get("m", "")),
                                         "status": status}
    finally:
        proc.wait()
        # Cancel/normal exit: translate the final marker (or exit code) into a status.
        if final and final.get("final") == "done":
            _render_state[job_id] = {"progress": 1.0, "message": "done",
                                     "status": "done", "output": final.get("output")}
        elif final and final.get("final") == "cancelled":
            _render_state[job_id] = {"progress": 0.0, "message": "cancelled",
                                     "status": "cancelled"}
        elif final and final.get("final") == "error":
            _render_state[job_id] = {"progress": 0.0,
                                     "message": str(final.get("message", "render failed")),
                                     "status": "error"}
        else:
            # process died without emitting a final marker (e.g. SIGKILL'd) — infer
            # from the status that was set when cancel was requested, else error.
            cur = _render_state.get(job_id, {}).get("status")
            if cur == "cancelling" or proc.returncode in (-9, 143):  # SIGKILL / SIGTERM
                _render_state[job_id] = {"progress": 0.0, "message": "cancelled",
                                         "status": "cancelled"}
            else:
                _render_state[job_id] = {"progress": 0.0,
                                         "message": f"render exited {proc.returncode}",
                                         "status": "error"}
        _workers.pop(job_id, None)


def _evict_finished_states(keep: int = 200) -> None:
    """Drop the oldest finished entries from `_render_state` once it exceeds `keep`
    (every render + thumbnail leaves a record; nothing else removes them)."""
    if len(_render_state) <= keep:
        return
    finished = [k for k, v in _render_state.items()
                if v.get("status") in ("done", "error", "cancelled")]
    for k in finished[: len(_render_state) - keep]:
        _render_state.pop(k, None)


@app.post("/api/projects/{pid}/render")
def render(pid: str, req: RenderReq):
    settings = RenderSettings(width=req.width, height=req.height, fps=req.fps,
                              samples=req.samples, noise_threshold=req.noise_threshold,
                              engine=req.engine,
                              export_blend=req.export_blend, draft=req.draft,
                              mesh_detail=req.mesh_detail, mesh_from_labels=req.mesh_from_labels,
                              auto_direct=req.auto_direct, lod_mode=req.lod_mode,
                              show_bbox=req.show_bbox,
                              bbox_color=req.bbox_color or [0.62, 0.66, 0.74],
                              bbox_source=req.bbox_source or "")
    return {"job_id": _start_render(pid, settings, kf_range=req.kf_range)}


@app.post("/api/projects/{pid}/chat")
def chat(pid: str, req: ChatReq):
    """Run one turn of the Claude director agent over this project."""
    from . import agent

    if not store.exists(pid):
        raise HTTPException(404, "no such project")
    result = agent.run_agent(pid, req.message, req.history,
                             render_fn=lambda s: _start_render(pid, s))
    return result


@app.post("/api/renders/{job_id}/cancel")
def cancel_render(job_id: str):
    worker = _workers.get(job_id)
    if worker is None:
        # nothing running (already finished/unknown)
        return {"ok": False, "status": _render_state.get(job_id, {}).get("status", "unknown")}
    worker.terminate()
    cur = _render_state.get(job_id, {})
    _render_state[job_id] = {**cur, "status": "cancelling", "message": "cancelling…"}
    return {"ok": True, "status": "cancelling"}


@app.get("/api/renders/{job_id}")
def render_status(job_id: str):
    return _render_state.get(job_id, {"status": "unknown"})


@app.get("/api/projects/{pid}/renders/{job_id}/output")
def render_output(pid: str, job_id: str):
    st = _render_state.get(job_id, {})
    out = st.get("output")
    if not out:
        # fall back to project record
        p = store.load(pid)
        j = next((j for j in p.renders if j.id == job_id), None)
        out = j.output_path if j else None
    if not out:
        raise HTTPException(404, "no output yet")
    if out.endswith(".mp4"):
        return FileResponse(out, media_type="video/mp4")  # large, not overwritten
    if out.endswith(".blend"):
        return FileResponse(out, media_type="application/octet-stream",
                            filename=f"{pid}.blend")  # Content-Disposition -> download
    return _image_response(out)  # PNG output may be overwritten by a re-render


@app.get("/api/renders/{job_id}/latest_frame")
def latest_frame(job_id: str):
    """The most-recently-written frame of a *running* render, for live preview."""
    worker = _workers.get(job_id)
    if worker is None:
        raise HTTPException(404, "no running render")
    frames = sorted(worker.frames_dir.glob("frame_*.png"))
    if not frames:
        raise HTTPException(404, "no frame yet")
    # last fully-written frame (the highest-numbered one already on disk)
    return _image_response(str(frames[-1]))


@app.websocket("/api/renders/{job_id}/ws")
async def render_ws(ws: WebSocket, job_id: str):
    await ws.accept()
    try:
        while True:
            st = _render_state.get(job_id, {"status": "unknown"})
            await ws.send_json(st)
            if st.get("status") in ("done", "error", "cancelled", "unknown"):
                break
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        pass


# ----------------------------- frontend (static) -----------------------------
@app.get("/")
def index():
    # no-store so the browser always loads the latest UI (avoids stale cached JS)
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
