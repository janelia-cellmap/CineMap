# mvideo — Agentic Video Generation for Segmentation Data

An interactive web app for authoring videos of large 3D EM datasets and their
instance segmentations / meshes. **Neuroglancer** is the interactive *scouting*
tool; **Blender** is the single render engine; a **Claude agent** plus a manual
**keyframe timeline** direct the shots.

Example dataset (Neuroglancer state link, used for scouting):
`https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/jrc_mus-salivary/jrc_mus-salivary-1.json`

## Core concept

**Scout in Neuroglancer, render in Blender.** Neuroglancer is unbeaten at flying
through huge multiscale EM+segmentation interactively — so you use it to *find*
the shot (a view, a region, a set of segments). That view is then "baked" into a
**Blender scene**, where the actual video is rendered: the segmentation **meshes**
plus the **EM context as cross-section slice planes sampled from zarr**, all in
**one engine, one camera, one coordinate system**.

This deletes the hardest problem of the earlier two-engine design (registering a
Neuroglancer camera against a Blender camera) — there is nothing to align because
nothing from Neuroglancer is composited into the video.

The keyframe primitive:

- **Keyframe** = a Blender-renderable scene state: camera (position, orientation,
  FOV), which **EM slice planes** are shown (axis + position + source/scale),
  which **meshes** are visible + their materials, and lighting. A keyframe can be
  *derived* from a Neuroglancer scouting state (NG camera → Blender camera, NG
  cross-section → slice plane, NG selected segments → visible meshes).
- **Video** = interpolation between consecutive keyframes (slerp camera
  orientation, lerp position/FOV, move slice planes, fade mesh opacity), each
  interpolated frame rendered by Blender.
- **Claude and the human edit the same ordered list of keyframes** through one
  API, so agent actions and manual tweaks compose freely.

## Decisions (locked)

| Question | Choice |
|---|---|
| Render engine | **Blender only.** One scene, one camera, one coordinate system. |
| EM context | **Cross-section slice planes** sampled from zarr and textured onto planes in 3D (meshes intersect them). |
| Neuroglancer role | **Interactive scouting only** — never rendered into the video; used to find shots and bake keyframes. |
| Deployment | **Local / cluster web app** — reads `/groups` & `/nrs` directly, single/few users. |
| Agent role | **Hybrid** — structured tool calls for common ops + `run_code` escape hatch (`bpy`). |
| Data formats | **Zarr / N5 / OME-Zarr** EM volumes + **precomputed** multires meshes. |
| Timeline | **Keyframe timeline** — interpolated transitions between saved scene states. |

## Architecture

```
┌─────────────────────────── Frontend (React + TS + Vite) ───────────────────────────┐
│  ┌────────────────────────┐  ┌──────────────────────┐   ┌────────────────────────┐  │
│  │  Neuroglancer viewer   │  │  Blender preview pane │   │   Claude chat panel     │  │
│  │  (SCOUTING — explore    │→ │  (rendered frames /   │   │   (streaming, tool log) │  │
│  │   huge data, "bake KF") │  │   EEVEE thumbnails)   │   │                         │  │
│  └────────────────────────┘  └──────────────────────┘   └────────────────────────┘  │
│  ┌──────────────────────────────── Keyframe timeline (bottom strip) ──────────────┐  │
│  │ [KF0]──trans──[KF1]──trans──[KF2] ...  thumbnails · drag/reorder · duration/ease │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │ REST + WebSocket (state, agent, render progress)
┌─────────────────────────────────────────┴─────────────────────────── Backend (FastAPI) ──┐
│  Project svc · Data-analyze svc · Agent svc (Claude tools) · Render svc (Blender) · encode │
│      │              │                       │                        │               │      │
│   project.json  probe zarr/n5/         Anthropic SDK          Blender bpy worker     ffmpeg │
│   + sqlite      precomputed →          tool-use + run_code    (GPU OPTIX Cycles):           │
│                 manifest                                       slice planes + meshes         │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │ reads
              /groups · /nrs   →  zarr EM slices (tensorstore/zarr)  +  meshes (cloud-volume)
```

### Single source of truth
`project.json` holds the base scene + the ordered keyframe list. Every mutation —
Claude tool call or manual UI edit — goes through the same backend API and updates
this file. The frontend re-renders from it; Claude reads it back before each
action. That is what makes the timeline a shared editing surface.

## Data model

```jsonc
Project {
  id, name, data_path,
  manifest,            // analyze: scales, voxel_size, bbox, segment_ids, meshes
  scene,               // base lighting, world scale (nm per Blender unit), defaults
  keyframes: [Keyframe],
  renders: [RenderJob]
}

Keyframe {
  id, label,
  camera: { position, orientation /*quat*/, fov_deg },
  slices: [ { axis: x|y|z, position_nm, source, scale_level, opacity } ],
  meshes: [ { id, visible, color, opacity } ],
  lighting,            // optional per-keyframe overrides
  ng_state,            // optional: originating NG scouting state (for round-trip)
  thumbnail_path,
  duration_in_s,       // transition INTO this keyframe
  easing               // linear | ease-in-out | ...
}

RenderJob { id, range, settings /*res,fps,samples,codec*/, status, progress, output_path }
```

## Agent design (hybrid)

Claude is a **director** calling structured tools, with a `run_code` escape hatch
(`bpy`) for custom shots. Tools mutate the same API the UI uses.

- `analyze_data(path)` → manifest
- `set_camera(position, orientation, fov)` · `frame_object(mesh_id)`
- `add_slice(axis, position_nm, scale)` · `move_slice(id, position_nm)` · `set_slice_opacity(id, v)`
- `set_mesh(id, visible, color, opacity)`
- `add_keyframe(label, from_current)` · `update_keyframe(id, …)` · `delete_keyframe(id)` · `reorder_keyframes(order)`
- `set_transition(keyframe_id, duration, easing)`
- `make_orbit(target, degrees, n)` · `make_flythrough(path)` · `sweep_slice(axis, from_nm, to_nm)` · `reveal_meshes(sequence)`  ← preset shot generators
- `bake_keyframe_from_ng(ng_state)` — convert a scouting view into a keyframe
- `preview_keyframe(id)` → EEVEE thumbnail · `render(range, settings)` → job_id · `get_render_status(job_id)`
- `run_code(bpy_snippet)` — sandboxed `bpy` for anything the tools don't cover

Example: *"orbit the mito mesh 360° while sweeping a z-slice through it"* →
`frame_object(mito)` → `make_orbit(mito, 360, n=12)` + `sweep_slice(z, …)` →
keyframes appear in the timeline → user nudges any → `render(1080p60)`.

The Anthropic SDK call uses **prompt caching** on tool defs + manifest.

## Rendering pipeline (single engine: Blender)

Every frame is a Blender render of one scene; there is no second engine and no
compositing/alignment.

1. **Meshes:** fetch precomputed multires meshes (`cloud-volume`) → `trimesh` →
   import to Blender (`bpy`), at the project's voxel→world scale.
2. **EM slice planes:** for each visible slice, read that 2D cross-section from
   zarr at the appropriate **multiscale level** (`tensorstore`/`zarr`) → image →
   texture an emission/unlit plane positioned at the slice's world coordinate.
   A *slice* is a cheap 2D read — this is why we avoid the "huge EM volume" cost.
3. **Camera & animation:** interpolate camera + slice positions + mesh opacity
   across the keyframe range; render with **GPU OPTIX Cycles** (final) or EEVEE
   (fast preview/thumbnails).
4. **Encode:** `ffmpeg` frames → mp4.

Renders run in a **Blender worker** (subprocess for isolation); a job queue
(RQ/Redis or a process pool) streams progress over WebSocket to the timeline.

## Integration layers (how the app talks to NG / Blender / data)

**MCP is optional and lives only at the agent layer.** Everything else is direct
Python libraries / subprocesses.

```
AGENT–TOOL layer   how Claude invokes capabilities
                   set_camera() · add_keyframe() · render() · run_code()
                   ❖ native Anthropic tool-use now; MCP-able later (thin wrapper)
        │  every tool call invokes one operations.* function
EXECUTION layer    direct, NOT MCP
                   • Neuroglancer → `neuroglancer` python pkg, SCOUTING ONLY
                     (websocket state sync to the embedded browser; not rendered)
                   • Blender      → `bpy` render worker (subprocess), THE renderer
                   • EM slices    → `tensorstore` / `zarr` (2D cross-section reads)
                   • meshes       → `cloud-volume` → `trimesh`
```

- **Neuroglancer (scouting):** the `neuroglancer` python package owns a live state
  JSON mirrored to the embedded browser viewer over a websocket (bidirectional).
  `bake_keyframe_from_ng` converts that state into a Blender keyframe. NG is **not**
  in the render path.
- **Blender (render):** a `bpy` worker subprocess builds the scene (meshes + slice
  planes) and writes frame PNGs.
- **Data:** the backend reads EM **slices** (`tensorstore`/`zarr`) and **meshes**
  (`cloud-volume`) directly from `/groups` & `/nrs`. For NG scouting, the browser
  fetches data from source URLs (backend serves bytes with CORS).

### The `operations` module — single funnel
All capabilities live in one internal `operations` module. **Both the UI and the
agent call through it** (the single-source-of-truth property). Exposing it as an
MCP server later is a thin wrapper. Native Anthropic tool-use for v1.

## Tech stack

- **Frontend:** React + TypeScript + Vite · embedded Neuroglancer (scouting) ·
  Blender preview pane · custom timeline component.
- **Backend:** Python + FastAPI + WebSocket.
- **Agent:** Anthropic SDK (Opus/Sonnet), tool use + prompt caching.
- **Render:** Blender `bpy` (GPU OPTIX Cycles / EEVEE), `cloud-volume` + `trimesh`
  (meshes), `tensorstore`/`zarr` (EM slices), `ffmpeg` (encode).
- **Scouting (optional headless NG thumbnails):** `selenium` + auto-fetched Chrome
  — *not on the critical path* now that NG isn't rendered into videos.
- **Storage:** per-project directory + sqlite index. **Queue:** RQ+Redis or pool.

## Spike results (Phase 0)

Environment: cluster node, **RTX 5090 32 GB**, `mv_env` conda env (Python 3.11),
all deps via `pyproject.toml`. Blender ships as the `bpy` PyPI module.

| Spike | Status | Finding |
|---|---|---|
| #3 Blender GPU render | ✅ **PASS** | `bpy` renders with **OPTIX** Cycles on the RTX 5090, transparent RGBA, ~2 s/frame. `spikes/spike3_blender_mesh.py`. |
| #2 zarr slice + mesh in Blender | ✅ **PASS** | **The new core mechanism.** Wrote a synthetic EM zarr, read a 2D cross-section, textured it onto a plane, intersected it with a mesh, rendered in one engine/one camera (GPU, ~1 s). `spikes/spike2_zarr_slice_in_blender.py` → `out/spike2_composite.png`. |
| #1 NG headless capture | ✅ **PASS** (now non-critical) | Headless NG capture works (`spikes/spike1_ng_screenshot.py`); only relevant for optional NG thumbnails, since NG is no longer rendered into videos. |

**Headless-Chrome findings (kept for the optional NG-thumbnail path):**
- **Never use `docker=True`** — it adds `--disable-gpu`, killing WebGL so
  `screenshot()` hangs forever. Pass only `--no-sandbox --disable-dev-shm-usage`;
  default GL gives WebGL2 via **SwiftShader**. Set `print_logs=False` (BiDi
  log-listener blocks under xvfb). GPU EGL headless does not work under xvfb.

## Risks / open items

1. **EM slice fidelity & multiscale selection** — pick the right zarr scale level
   for a slice's on-screen extent; handle OME-Zarr multiscale groups + voxel-size
   metadata. (Real-data version of the passed synthetic spike.)
2. **Blender mesh import** from real precomputed sharded meshes
   (`cloud-volume` → `trimesh` → `bpy`) at scale.
3. **NG → Blender camera bake** — one-way conversion of a scouting view into a
   Blender camera (we own both sides; far easier than the old two-engine align).
4. **Interpolation quality** — quaternion slerp + easing for smooth motion;
   moving slice planes without popping.

## Phasing

- **Phase 0 — Spikes:** ✅ core mechanisms validated (Blender GPU, zarr-slice+mesh).
- **Phase 1 — Editor (no AI):** new project → analyze → embedded NG scouting →
  bake/capture keyframes → Blender render (meshes + slice planes) → mp4. Proves
  the loop on real data.
- **Phase 2 — Agent:** Claude hybrid tools + `run_code`, driving the Phase-1 API.
- **Phase 3 — Fidelity:** multiscale EM slices, real precomputed meshes, materials
  & lighting presets, EEVEE thumbnails.
- **Phase 4 — Polish:** easing, preset shots (orbit/flythrough/slice-sweep/
  mesh-reveal), export presets, project save/load.

## Open questions

- Expected output specs (resolution, length, fps) — drives render-time budget.
- Real data layout: is EM OME-Zarr multiscale, and where do the precomputed
  meshes live relative to it? (drives analyze + slice/mesh loaders.)
- Slice look: single moving cross-section, fixed orthoslices, or both? Opaque vs
  semi-transparent EM planes?
- Multi-user later? (auth/sandboxing deferred under the local-app decision.)
