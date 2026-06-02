# CineMap

Agentic video generation for large-scale EM segmentation data. **Scout in
Neuroglancer, render in Blender.** See [plan.md](plan.md) for the full design.

Neuroglancer is the interactive scouting tool; Blender is the single render
engine. EM context comes from cross-section **slice planes sampled from zarr**;
the segmentation **meshes** are loaded from precomputed sources — composited in
one Blender scene, one camera, one coordinate system. Videos are built from a
**keyframe timeline** that both the UI and (later) a Claude agent edit through one
`operations` funnel.

## Status — Phase 1 (editor, no AI) working end-to-end on real data

- Analyze a neuroglancer state → project (EM + mesh sources).
- Embedded Neuroglancer **scouting** viewer; "bake" a view into a keyframe.
- Preset shots: **orbit**, **slice sweep**; manual keyframes.
- Real **EM slice loader** (OME-Zarr multiscale, region-cropped, over https) +
  **mesh loader** (precomputed multilod-draco via cloud-volume).
- **Blender GPU (OPTIX Cycles)** render worker → frames → mp4, with live progress.
- FastAPI + WebSocket backend; no-build static frontend.

## Setup

Everything runs in the **`mv_env`** conda env; deps are tracked in
[pyproject.toml](pyproject.toml).

```bash
# (one time) create + install
mamba create -y -n mv_env python=3.11
conda run -n mv_env pip install -e .
conda run -n mv_env pip install bpy   # Blender as a Python module (Cycles)
```

## Run

```bash
export ANTHROPIC_API_KEY=sk-...   # optional: enables the Claude director chat panel
./run.sh                          # -> http://0.0.0.0:8000
# or: conda run -n mv_env uvicorn cinemap.server:app --host 127.0.0.1 --port 8000
```

The **Claude director** (right-hand chat panel) turns plain requests
("orbit the nuclei, then sweep a z-slice") into keyframe edits via the same
`operations` funnel the UI uses. It needs `ANTHROPIC_API_KEY`; without it the
panel explains how to set it. Model via `CINEMAP_AGENT_MODEL` (default
`claude-sonnet-4-6`).

Open the URL, paste a neuroglancer state link (the example is pre-filled), click
**Create**, scout in the Neuroglancer pane, **Bake** keyframes (or **Orbit
360°**), then **Render ▸**. The mp4 plays in the preview pane.

Requirements observed on the dev box: an NVIDIA GPU (OPTIX), network access to the
data host (data is read over https; `/nrs` need not be mounted). ffmpeg is bundled via imageio-ffmpeg.

## Layout

```
src/cinemap/
  models.py            pydantic project/keyframe/render schemas
  operations.py        the single funnel (UI + agent call through this)
  store.py             project.json persistence
  scouting.py          neuroglancer scouting viewer + bake_keyframe
  server.py            FastAPI: REST + websocket
  data/
    manifest.py        neuroglancer state -> EM + mesh sources
    slice_loader.py    OME-Zarr multiscale EM cross-section reader (https)
    mesh_loader.py     precomputed multilod-draco meshes (cloud-volume)
  render/
    interpolate.py     keyframes -> per-frame states
    worker.py          assets -> scene.json -> blender subprocess -> mp4
    blender_script.py  bpy: build scene (slices+meshes), render frames
frontend/index.html    no-build UI (NG iframe + timeline + render)
spikes/                Phase-0 validation scripts
```

## Known follow-ups (Phase 3 — fidelity)

- Hero-mesh lighting/material tuning; smarter framing for concave meshes.
- Generate meshes for label-only layers (e.g. `er`).
- React/Vite frontend (needs Node, not installed on the dev box yet).
- Claude agent (Phase 2): wrap `operations.*` as tools.
- GPU headless Neuroglancer capture (only if NG thumbnails are wanted).
