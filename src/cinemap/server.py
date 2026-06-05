"""FastAPI server — REST + websocket + the neuroglancer scouting viewer.

Routes are thin wrappers over operations.* (the single funnel). Renders run in a
background thread; progress streams over a websocket to the timeline.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import threading

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


class RenderReq(BaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 30
    samples: int = 48
    kf_range: list[int] | None = None
    export_blend: bool = False  # produce a self-contained .blend instead of a video
    draft: bool = False         # fast low-res preview (coarse EM + low-voxel meshes)


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
    p.id = ops._uid("proj")     # fresh id so import never clobbers an existing project
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
    + segments), keeping its timing — i.e. 'update current frame'."""
    p = store.load(pid)
    kf = scouting.update_keyframe_from_view(p, kid)
    if kf is None:
        raise HTTPException(404, "no such keyframe")
    return kf.model_dump()


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
        job = RenderJob(id=ops._uid("thumb"), kf_range=kf_range, settings=settings)
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
                              samples=req.samples, export_blend=req.export_blend, draft=req.draft)
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
