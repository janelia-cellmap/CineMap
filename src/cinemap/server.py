"""FastAPI server — REST + websocket + the neuroglancer scouting viewer.

Routes are thin wrappers over operations.* (the single funnel). Renders run in a
background thread; progress streams over a websocket to the timeline.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import threading
from typing import Any

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
from .models import Project, RenderJob, RenderSettings
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
# live workers (so a render can be cancelled): job_id -> RenderWorker
_workers: dict[str, RenderWorker] = {}


# ----------------------------- request bodies -----------------------------
class CreateProject(BaseModel):
    name: str
    data_path: str


class OrbitReq(BaseModel):
    degrees: float = 360.0
    n: int = 12
    elevation_deg: float = 22.0


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


class RenderReq(BaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 30
    samples: int = 48
    kf_range: list[int] | None = None
    export_blend: bool = False  # produce a self-contained .blend instead of a video
    draft: bool = False         # fast low-res preview (coarse EM + low-voxel meshes)
    mesh_detail: float = 1.0    # per-layer vertex-budget multiplier (hard-capped)
    mesh_from_labels: bool = False  # regenerate watertight meshes from labels vs precomputed
    auto_direct: bool = True    # non-destructive presentation pass (lighting/material/DOF)
    lod_mode: str = "frame"     # mesh LOD: "single" | "frame" (per-frame adaptive) | "chunk"


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
    store.save(p)
    try:
        scouting.load_dataset(p.data_path)
    except Exception:  # noqa: BLE001
        pass
    return p.model_dump()


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
    return {"project": _light_project(store.load(project.id)),
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
def render_thumbnail(pid: str, kid: str):
    """Render this single keyframe to a still and keep it as the frame's thumbnail."""
    p = store.load(pid)
    idx = next((i for i, k in enumerate(p.keyframes) if k.id == kid), None)
    if idx is None:
        raise HTTPException(404, "no such keyframe")
    settings = RenderSettings(width=640, height=480, samples=24, fps=1, draft=True)
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
@app.post("/api/projects/{pid}/keyframes")
def add_keyframe(pid: str):
    p = store.load(pid)
    return ops.add_keyframe(p).model_dump()


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


@app.post("/api/projects/{pid}/keyframes/reorder")
def reorder_keyframes(pid: str, req: ReorderReq):
    p = store.load(pid)
    ops.reorder_keyframes(p, req.order)
    return {"ok": True, "count": len(p.keyframes)}


@app.post("/api/projects/{pid}/orbit")
def orbit(pid: str, req: OrbitReq):
    p = store.load(pid)
    kfs = ops.make_orbit(p, degrees=req.degrees, n=req.n, elevation_deg=req.elevation_deg)
    return {"added": len(kfs)}


@app.post("/api/projects/{pid}/sweep_slice")
def sweep(pid: str, req: SweepReq):
    p = store.load(pid)
    kfs = ops.sweep_slice(p, axis=req.axis, n=req.n)
    return {"added": len(kfs)}


@app.post("/api/projects/{pid}/sweep_clip")
def sweep_clip(pid: str, req: SweepClipReq):
    p = store.load(pid)
    kfs = ops.sweep_clip(p, mesh_name=req.mesh_name, axis=req.axis, n=req.n, side=req.side)
    return {"added": len(kfs)}


@app.post("/api/projects/{pid}/plane_move")
def plane_move(pid: str, req: PlaneMoveReq):
    p = store.load(pid)
    kfs = ops.plane_move(p, axis=req.axis, mode=req.mode, start_nm=req.start_nm,
                         stop_nm=req.stop_nm, from_xyz=req.from_xyz, to_xyz=req.to_xyz,
                         from_ng=req.from_ng, to_ng=req.to_ng,
                         n=req.n, mesh_name=req.mesh_name, side=req.side)
    return {"added": len(kfs)}


# ----------------------------- render -----------------------------
def _run_render(pid: str, job_id: str, worker: RenderWorker, thumbnail_for: str | None = None):
    def cb(pr, msg):
        # keep status as "cancelling" once requested, until the worker bails out
        cur = _render_state.get(job_id, {}).get("status")
        status = "cancelling" if cur == "cancelling" else "running"
        _render_state[job_id] = {"progress": pr, "message": msg, "status": status}

    # A full video render mutates the job (status/output) in project.renders, so it
    # must be persisted. A thumbnail job is NOT in project.renders — it only ever
    # touches the project if it sets a keyframe's thumbnail_path. Track that so we
    # don't rewrite the whole project.json over NFS when nothing changed
    # (cancelled/failed thumbnails, or thumbnails where the keyframe is gone).
    dirty = thumbnail_for is None
    try:
        out = worker.run(progress=cb)
        if thumbnail_for and out:  # keep a per-keyframe thumbnail and point the kf at it
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
    p = store.load(pid)
    if thumbnail_for:  # don't clutter the render history with thumbnail jobs
        job = RenderJob(id=ops.render_id(p, "thumb"), kf_range=kf_range, settings=settings)
    else:
        job = ops.create_render_job(p, settings, kf_range=kf_range)
    worker = RenderWorker(p, job)
    _workers[job.id] = worker
    _evict_finished_states()  # keep the in-memory job table from growing forever
    _render_state[job.id] = {"progress": 0.0, "message": "queued", "status": "pending"}
    threading.Thread(target=_run_render, args=(pid, job.id, worker, thumbnail_for), daemon=True).start()
    return job.id


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
                              samples=req.samples, export_blend=req.export_blend, draft=req.draft,
                              mesh_detail=req.mesh_detail, mesh_from_labels=req.mesh_from_labels,
                              auto_direct=req.auto_direct, lod_mode=req.lod_mode)
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
