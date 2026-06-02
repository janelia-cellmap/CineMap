"""FastAPI server — REST + websocket + the neuroglancer scouting viewer.

Routes are thin wrappers over operations.* (the single funnel). Renders run in a
background thread; progress streams over a websocket to the timeline.
"""
from __future__ import annotations

import asyncio
import os
import threading

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, operations as ops
from . import scouting, store
from .config import FRONTEND_DIR
from .models import Project, RenderSettings
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
        return project.model_dump()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not analyze dataset: {e}") from e


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    if not store.exists(pid):
        raise HTTPException(404, "no such project")
    return store.load(pid).model_dump()


@app.get("/api/projects/{pid}/export")
def export_project(pid: str):
    """Download the project as a portable .cinemap.json file (state, not video)."""
    if not store.exists(pid):
        raise HTTPException(404, "no such project")
    p = store.load(pid)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (p.name or pid))
    return FileResponse(str(store.project_file(pid)), media_type="application/json",
                        filename=f"{safe}.cinemap.json")


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


@app.post("/api/projects/{pid}/open")
def open_project(pid: str):
    p = store.load(pid)
    scouting.load_dataset(p.data_path)
    return {"ng_url": scouting.viewer_url()}


# ----------------------------- scouting -----------------------------
@app.get("/api/ng_url")
def ng_url():
    return {"ng_url": scouting.viewer_url()}


@app.post("/api/projects/{pid}/bake")
def bake(pid: str):
    p = store.load(pid)
    kf = scouting.bake_keyframe(p)
    return kf.model_dump()


@app.post("/api/projects/{pid}/keyframes/{kid}/goto")
def goto_keyframe(pid: str, kid: str):
    p = store.load(pid)
    ok = scouting.goto_keyframe(p, kid)
    if not ok:
        raise HTTPException(404, "no such keyframe")
    return {"ok": True}


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
def _run_render(pid: str, job_id: str, worker: RenderWorker):
    def cb(pr, msg):
        # keep status as "cancelling" once requested, until the worker bails out
        cur = _render_state.get(job_id, {}).get("status")
        status = "cancelling" if cur == "cancelling" else "running"
        _render_state[job_id] = {"progress": pr, "message": msg, "status": status}

    try:
        out = worker.run(progress=cb)
        _render_state[job_id] = {"progress": 1.0, "message": "done", "status": "done", "output": out}
    except RenderCancelled:
        _render_state[job_id] = {"progress": 0.0, "message": "cancelled", "status": "cancelled"}
    except Exception as e:  # noqa: BLE001
        _render_state[job_id] = {"progress": 0.0, "message": str(e), "status": "error"}
    finally:
        store.save(worker.project)
        _workers.pop(job_id, None)


def _start_render(pid: str, settings: RenderSettings, kf_range=None) -> str:
    p = store.load(pid)
    job = ops.create_render_job(p, settings, kf_range=kf_range)
    worker = RenderWorker(p, job)
    _workers[job.id] = worker
    _render_state[job.id] = {"progress": 0.0, "message": "queued", "status": "pending"}
    threading.Thread(target=_run_render, args=(pid, job.id, worker), daemon=True).start()
    return job.id


@app.post("/api/projects/{pid}/render")
def render(pid: str, req: RenderReq):
    settings = RenderSettings(width=req.width, height=req.height, fps=req.fps, samples=req.samples)
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
    media = "video/mp4" if out.endswith(".mp4") else "image/png"
    return FileResponse(out, media_type=media)


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
    return FileResponse(str(frames[-1]), media_type="image/png",
                        headers={"Cache-Control": "no-store"})


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
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
