<p align="center"><img src="imgs/logo.png" width="110" alt="CineMap logo"></p>

<h1 align="center">CineMap</h1>

<p align="center"><b>Cinematic videos from large-scale EM segmentation data.</b><br>
Scout in Neuroglancer, render in Blender.</p>

![CineMap interface](imgs/screenshot.png)

---

## How it works

CineMap turns Neuroglancer views into a keyframe timeline that Blender renders
into a smooth video.

**Neuroglancer is not the renderer — it is your control surface.** You arrange
the scene exactly how you want it in Neuroglancer (and frame it in the 3D
preview), then capture it as a keyframe. Blender generates the actual frames.

## Quick start

**1. Paste your Neuroglancer state link into the field and click `Create`.**

![Create a project](imgs/create_project_button.png)

**2. Scout freely in the Neuroglancer pane** — choose layers, segments, colors,
whatever you want to show.

**3. Capture the view.** Click `★ Bake keyframe` to save the current view as a new
keyframe, or `⟳ Update current frame` to overwrite the selected one. Each keyframe
becomes one Blender frame.

![Framing buttons](imgs/framing_buttons.png)

## What a keyframe captures

Baking (or updating) a keyframe records the full state of your view:

- **Visible layers** — which Neuroglancer layers are shown
- **Visible segments** — for mesh layers, which segment IDs are on
- **Segment colors** — the exact colors from Neuroglancer
- **3D mesh opacity & silhouette** — the mesh rendering style
- **Camera** — position, zoom, and rotation, taken from the **3D preview** panel

To frame a shot, use the **3D preview** (right pane): zoom in, zoom out, and rotate
there — that camera is what gets baked.

## Working with the timeline

- **Preview the motion** — click `▷ Preview` to see the interpolated animation
  between your keyframes.
- **Update a frame** — select a keyframe and `⟳ Update current frame` to replace it
  with your latest view.
- **Reorder frames** — drag keyframes along the timeline to change their order.
- **Import / Export project** — save and reload a whole project with `⭱ Import` /
  `⭳ Export`.

## Exporting

- `⬇ Export video` — render the full shot to an mp4.
- `⬇ Export .blend` — a self-contained Blender file (camera, meshes, EM slices,
  packed textures) for manual finishing.

![Export buttons](imgs/export_buttons.png)

## Run locally

```bash
mamba create -y -n mv_env python=3.11
conda run -n mv_env pip install -e .     # installs all deps incl. bpy (Blender/Cycles)
./run.sh                                 # -> http://0.0.0.0:8000
```

Requires an NVIDIA GPU (OPTIX / Cycles) and network access to the data host (data
is read over https; `/nrs` need not be mounted).

## Let Claude help

Click `✦ Help me! Claude` to open the director panel and just ask in plain
language — *"orbit the nuclei, then sweep a z-slice"*. Claude builds the
keyframes for you.

![Help me! Claude](imgs/claude_help_me.png)

Everything Claude creates is normal keyframes, so you stay in control — **tweak,
reframe, reorder, or re-bake any of them afterward** exactly as if you'd made them
by hand. (Requires `ANTHROPIC_API_KEY`.)
