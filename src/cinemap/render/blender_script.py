"""Blender (bpy) render script — runs as an isolated subprocess.

Reads a scene spec JSON (produced by worker.py, all coordinates already in
Blender units) and renders one PNG per frame:

  - imports each mesh OBJ once (static geometry),
  - per frame: positions the camera, (re)builds the EM slice planes with that
    frame's image + placement, sets mesh/slice opacity & visibility, renders.

Invoke:  python -m cinemap.render.blender_script <scene.json>
(launched via the env python so `import bpy` resolves.)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import bpy
from mathutils import Euler, Matrix, Vector


def _srgb_channel_to_linear(c: float) -> float:
    c = max(0.0, min(1.0, float(c)))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _color_for_blender(color, color_space: str = "linear") -> tuple[float, float, float, float]:
    vals = list(color or [1.0, 1.0, 1.0])
    while len(vals) < 4:
        vals.append(1.0)
    rgba = [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]
    if str(color_space).lower() == "srgb":
        rgba[:3] = [_srgb_channel_to_linear(c) for c in rgba[:3]]
    return (rgba[0], rgba[1], rgba[2], rgba[3])


def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _setup_render(scene_spec: dict) -> None:
    scene = bpy.context.scene
    r = scene_spec["render"]
    # Resolve the requested engine to an id this Blender build actually exposes. Eevee was
    # renamed across versions: "BLENDER_EEVEE" (≤4.1 and again in ≥4.4/5.0) vs
    # "BLENDER_EEVEE_NEXT" (only 4.2–4.3). Assigning an id not in the enum raises and the
    # whole render dies, so pick whichever Eevee this build has and fall back to Cycles.
    req_engine = r.get("engine", "CYCLES")
    valid = {i.identifier for i in
             bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if req_engine not in valid:
        if req_engine.startswith("BLENDER_EEVEE"):
            req_engine = ("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in valid
                          else "BLENDER_EEVEE")
        else:
            req_engine = "CYCLES"
        print(f"[blender] engine '{r.get('engine')}' -> '{req_engine}' (not in this build)",
              flush=True)
    scene.render.engine = req_engine
    if scene.render.engine == "CYCLES":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.get_devices()
            # CINEMAP_GPU picks which OPTIX GPU(s) to render on when several exist:
            # an index or comma-separated indices into the OPTIX device list below
            # (e.g. "1" or "0,2"). Unset/blank -> use all GPUs.
            sel = os.environ.get("CINEMAP_GPU", "").strip()
            want = None
            if sel:
                try:
                    want = {int(x) for x in sel.replace(" ", "").split(",") if x}
                except ValueError:
                    print(f"[blender] bad CINEMAP_GPU={sel!r}; using all GPUs", flush=True)
            optix = [d for d in prefs.devices if d.type == "OPTIX"]
            for i, d in enumerate(optix):
                d.use = want is None or i in want
            for d in prefs.devices:
                if d.type == "CPU":
                    d.use = True            # CPU helps alongside the GPU(s)
                elif d.type != "OPTIX":
                    d.use = False
            scene.cycles.device = "GPU"
            # make it obvious in the log which GPUs are available (with the index to
            # pass via CINEMAP_GPU) and which we're actually on — a silent CPU
            # fallback is a ~10-50x slowdown.
            print(f"[blender] OPTIX GPUs available: {[f'{i}:{d.name}' for i, d in enumerate(optix)]}",
                  flush=True)
            gpus = [d.name for d in optix if d.use]
            print(f"[blender] cycles device=GPU compute=OPTIX using={gpus or 'NONE -> CPU fallback!'}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[blender] GPU unavailable, CPU: {e}", flush=True)
        scene.cycles.samples = r.get("samples", 64)
        material_profile = scene_spec.get("direction", {}).get("material", {})
        if material_profile.get("transparent_max_bounces") is not None:
            try:
                scene.cycles.transparent_max_bounces = int(material_profile["transparent_max_bounces"])
            except Exception as e:  # noqa: BLE001
                print(f"[blender] transparent_max_bounces: {e}", flush=True)
        # Adaptive sampling + a denoiser do most of the work: `samples` becomes a CEILING,
        # Cycles stops early on pixels that already look clean (flat areas, the black bg)
        # and spends rays only where it's still noisy (shadows/edges), then the denoiser
        # removes the leftover grain. Lets us render far fewer samples for equal/cleaner
        # output — much faster, and the deep-shadow "drama" look stays clean.
        try:
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = float(r.get("noise_threshold", 0.01))
            scene.cycles.use_denoising = True
            for dn in ("OPTIX", "OPENIMAGEDENOISE"):   # GPU denoiser first, then CPU OIDN
                try:
                    scene.cycles.denoiser = dn
                    break
                except Exception:  # noqa: BLE001
                    continue
            print(f"[blender] sampling: max_samples={scene.cycles.samples} "
                  f"adaptive_threshold={scene.cycles.adaptive_threshold} "
                  f"denoiser={scene.cycles.denoiser}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[blender] denoise/adaptive unavailable: {e}")
    elif scene.render.engine.startswith("BLENDER_EEVEE"):
        # Eevee = rasterizer (no path tracing) -> ~10-50x faster per frame. `samples`
        # here is TAA samples (anti-alias accumulation), NOT light bounces. The look is
        # approximate (screen-space shadows/AO, no true transmission) but ideal for fast
        # previews and movie drafts where Cycles' per-frame cost dominates. The shared
        # material edge-glow / emission / compositor bloom still apply (engine-agnostic).
        try:
            scene.eevee.taa_render_samples = int(r.get("samples", 64))
            print(f"[blender] engine=EEVEE taa_render_samples={scene.eevee.taa_render_samples}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[blender] eevee config: {e}", flush=True)
    scene.render.resolution_x = r["width"]
    scene.render.resolution_y = r["height"]
    scene.render.image_settings.file_format = "PNG"
    # View transform from the preset. Empty fields leave Blender's defaults alone; the
    # Neuroglancer preset uses Raw so Blender does not apply display-tone mapping to
    # WebGL/display-space colors.
    view = scene_spec.get("direction", {}).get("view", {})
    transform = view.get("transform")
    if transform:
        try:
            scene.view_settings.view_transform = transform
        except Exception as e:  # noqa: BLE001
            print(f"[blender] view transform: {e}")
    if "look" in view:
        try:
            scene.view_settings.look = view.get("look") or "None"
        except Exception as e:  # noqa: BLE001
            print(f"[blender] view look: {e}")

    # World gives EVEN ambient fill from all directions (so surfaces facing away from
    # the key aren't pure black — like neuroglancer's even lighting), while the CAMERA
    # still sees the dark background. A Light-Path "Is Camera Ray" mix separates the two:
    # camera ray -> dark bg color; diffuse/AO rays -> gray ambient.
    c = scene_spec["world"].get("background", [0.0, 0.0, 0.0])  # black, like neuroglancer
    lrig = scene_spec.get("direction", {}).get("lighting", {})
    amb = float(lrig.get("ambient", 0.3))
    # cool-tinted ambient (warm key / cool fill = a subtle studio/MeshLab dimension)
    ac = lrig.get("ambient_color", [1.0, 1.0, 1.0])
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg_cam = nt.nodes.new("ShaderNodeBackground")
    bg_cam.inputs[0].default_value = (c[0], c[1], c[2], 1.0)
    global _world_bg_cam   # the camera-ray background node, animated per frame
    _world_bg_cam = bg_cam
    bg_amb = nt.nodes.new("ShaderNodeBackground")
    bg_amb.inputs[0].default_value = (amb * ac[0], amb * ac[1], amb * ac[2], 1.0)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])  # 0 -> ambient, 1 -> bg
    nt.links.new(bg_amb.outputs[0], mix.inputs[1])
    nt.links.new(bg_cam.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    scene.world = world
    if amb <= 0.0:
        try:
            world.cycles_visibility.diffuse = False
        except Exception:  # noqa: BLE001
            pass

    _setup_freestyle(scene_spec)


_world_bg_cam = None   # the world's camera-ray ShaderNodeBackground (set in _setup_render)


def _set_world_bg(frame: dict, f: int | None = None) -> None:
    """Apply this frame's NG 3D background color to the world's camera-ray node.

    Mirrors `_set_fade_overlay`: the PNG path calls with ``f=None`` (set directly per
    frame), the .blend export path passes the scene frame so the color is keyframed
    onto an F-curve and the saved file animates the background. Colors are LINEAR (they
    round-trip through the Standard view transform back to the NG color)."""
    if _world_bg_cam is None:
        return
    bg = frame.get("background")
    if not bg:
        return
    _world_bg_cam.inputs[0].default_value = (float(bg[0]), float(bg[1]), float(bg[2]), 1.0)
    if f is not None:
        _world_bg_cam.inputs[0].keyframe_insert("default_value", frame=f)


def _setup_freestyle(scene_spec: dict) -> None:
    """Optional true outline (Freestyle) pass — crisp dark lines on object silhouettes,
    borders and creases, so overlapping structures are unmistakably separated. Off unless
    the director asks (direction.freestyle.enabled); the material edge-darken is the
    subtler default. Best-effort: never fail the render if the API differs."""
    fs = scene_spec.get("direction", {}).get("freestyle") or {}
    if not fs.get("enabled"):
        return
    try:
        scene = bpy.context.scene
        scene.render.use_freestyle = True
        col = fs.get("color", [0.0, 0.0, 0.0])
        thick = float(fs.get("thickness", 1.5))
        for vl in scene.view_layers:
            vl.use_freestyle = True
            fset = vl.freestyle_settings
            lineset = fset.linesets[0] if len(fset.linesets) else fset.linesets.new("ls")
            lineset.select_silhouette = True
            lineset.select_border = True
            lineset.select_crease = True
            ls = lineset.linestyle
            ls.color = (col[0], col[1], col[2])
            ls.thickness = thick
    except Exception as e:  # noqa: BLE001
        print(f"[blender] freestyle skipped: {e}")


def _scene_center_radius() -> tuple[Vector, float]:
    coords = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith("slice_") or obj.name.startswith("cm_"):
            continue
        try:
            for c in obj.bound_box:
                coords.append(obj.matrix_world @ Vector(c))
        except Exception:  # noqa: BLE001
            continue
    if not coords:
        return Vector((0.0, 0.0, 0.0)), 10.0
    lo = Vector((min(p[i] for p in coords) for i in range(3)))
    hi = Vector((max(p[i] for p in coords) for i in range(3)))
    center = (lo + hi) * 0.5
    radius = max((p - center).length for p in coords) or 10.0
    return center, radius


def _add_neuvid_lights(scene_spec: dict, rig: dict) -> None:
    """neuVid's render.py uses three fixed AREA lights around the neuron bounds.

    Directions/colors/power scaling are ported from neuVid commit be416cba:
    lamp distance = radius * 2.5, size = radius, energy =
    2400000 * (distance / 425.1282)^2 * lightPowerScale[i].
    """
    center, radius = _scene_center_radius()
    specs = [
        {"direction": (-0.892, 0.3, 0.9), "color": rig.get("key_color", [1.0, 1.0, 1.0])},
        {"direction": (0.588, 0.46, 0.248), "color": rig.get("fill_color", [1.0, 1.0, 1.0])},
        {"direction": (0.216, -0.392, -0.216), "color": rig.get("rim_color", [1.0, 1.0, 1.0])},
    ]
    rot = rig.get("neuvid_light_rotation", [0.0, 0.0, 0.0])
    try:
        # Match neuVid render.py exactly: it converts the JSON X/Y values to radians
        # but accidentally builds the Euler with the original X/Y values and only the
        # converted Z value. Defaults are zero, so this matters only when a neuVid-style
        # lightRotation override is used.
        rot_euler = Euler(
            (float(rot[0]), float(rot[1]), math.radians(float(rot[2]))),
            "XYZ",
        ).to_matrix()
    except Exception:  # noqa: BLE001
        rot_euler = Matrix.Identity(3)
    # neuVid parents all lights to a rotator with Y=180 degrees for the standard view.
    light_rotation = (
        Matrix.Rotation(math.radians(180.0), 4, "Y").to_3x3() @ rot_euler
    ).to_3x3()
    power_scale = list(rig.get("neuvid_power_scale", [1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0])
    size_scale = float(rig.get("neuvid_size_scale", 1.0) or 1.0)
    distance_scale = float(rig.get("neuvid_distance_scale", 1.0) or 1.0)
    distance = radius * 2.5 * distance_scale
    for i, spec in enumerate(specs):
        data = bpy.data.lights.new(f"Lamp.{i}", type="AREA")
        data.size = max(1e-6, radius * size_scale)
        scale_i = float(power_scale[i] if i < len(power_scale) else 1.0)
        data.energy = 2_400_000.0 * (distance / 425.1282) ** 2 * scale_i
        try:
            data.cycles.cast_shadow = True
            data.cycles.use_multiple_importance_sampling = True
        except Exception:  # noqa: BLE001
            pass
        col = spec["color"]
        data.color = (col[0], col[1], col[2])
        obj = bpy.data.objects.new(f"Lamp.{i}", data)
        direction = Vector(spec["direction"])
        direction = light_rotation @ direction
        direction.normalize()
        obj.location = center + direction * distance
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "X").to_euler()
        try:
            obj.visible_camera = False
        except Exception:  # noqa: BLE001
            pass
        bpy.context.scene.collection.objects.link(obj)


def _add_light(scene_spec: dict) -> None:
    """Add the selected preset's lights."""
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    rig = scene_spec.get("direction", {}).get("lighting", {})
    kind = rig.get("kind", "sun" if rig else "sun")
    if kind == "none":
        return
    if kind == "neuvid":
        _add_neuvid_lights(scene_spec, rig)
        return
    energy = scene_spec.get("lighting", {}).get("key_energy", 3000.0)
    base = rig.get("key_energy") or max(2.0, energy / 600.0)
    fill_mult, rim_mult = rig.get("fill_ratio", 0.45), rig.get("rim_ratio", 0.6)
    kick_mult = rig.get("kick_ratio", 0.0)
    colors = {"Key": rig.get("key_color", [1.0, 1.0, 1.0]),    # warm/cool studio split:
              "Fill": rig.get("fill_color", [1.0, 1.0, 1.0]),  # warm key + cool fill/rim,
              "Rim": rig.get("rim_color", [1.0, 1.0, 1.0]),    # opposing colored kicker
              "Kick": rig.get("kick_color", [1.0, 1.0, 1.0])}
    for name, rot, mult in [("Key", (0.6, 0.2, 0.4), 1.0),
                            ("Fill", (-0.5, -0.3, 2.4), fill_mult),
                            ("Rim", (1.2, 0.0, -1.8), rim_mult),
                            ("Kick", (1.2, 0.0, 1.8), kick_mult)]:
        data = bpy.data.lights.new(name, type="SUN")
        data.energy = base * mult
        col = colors[name]
        data.color = (col[0], col[1], col[2])
        obj = bpy.data.objects.new(name, data)
        obj.rotation_euler = rot
        bpy.context.scene.collection.objects.link(obj)
    # (even ambient fill is provided by the world's gray ambient in _setup_render)


def _setup_bloom(scene_spec: dict) -> None:
    """Soft bloom on bright/emissive structures via a compositor Glare (fog-glow)
    node — the 'publication glow' that makes the colored meshes read as illuminated
    against the dark background. Constant (no per-object flash), so it scales to any
    number of objects. From the director; absent => no compositor change."""
    b = scene_spec.get("direction", {}).get("bloom") or {}
    if not b.get("enabled"):
        return
    # Best-effort: the compositor API varies by Blender version (Scene.use_nodes /
    # node_tree is deprecated and is None here). If it isn't available, skip bloom —
    # the material edge-glow + emission still carry the look. Never fail the render.
    try:
        scene = bpy.context.scene
        scene.use_nodes = True
        tree = getattr(scene, "node_tree", None)
        if tree is None:
            print("[blender] bloom skipped: compositor node tree unavailable")
            return
        rl = next((n for n in tree.nodes if n.type == "R_LAYERS"), None)
        comp = next((n for n in tree.nodes if n.type == "COMPOSITE"), None)
        if rl is None or comp is None:
            return
        glare = tree.nodes.new("CompositorNodeGlare")
        glare.glare_type = "FOG_GLOW"
        glare.quality = "HIGH"
        glare.threshold = b.get("threshold", 0.6)
        glare.size = int(b.get("size", 7))
        glare.mix = b.get("mix", -0.55)
        tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
        tree.links.new(glare.outputs["Image"], comp.inputs["Image"])
    except Exception as e:  # noqa: BLE001
        print(f"[blender] bloom skipped: {e}")


def _update_lights(frame: dict, rig: dict) -> None:
    """Re-aim the key/fill/rim suns relative to the camera for this frame, so the
    rig (and the rim edge-light) stays consistent as the camera moves. Suns are
    directional, so only their orientation matters."""
    if not rig.get("camera_relative", True):
        return
    pos = Vector(frame["camera"]["position_bu"])
    fwd = (Vector(frame["camera"]["look_at_bu"]) - pos)
    if fwd.length < 1e-9:
        return
    fwd.normalize()
    up = Vector(frame["camera"].get("up", [0.0, 0.0, 1.0]))
    right = fwd.cross(up)
    right = right.normalized() if right.length > 1e-9 else Vector((1.0, 0.0, 0.0))
    tup = right.cross(fwd).normalized()                  # true up, orthogonal to fwd
    # Mimic neuroglancer: the key is a HEADLIGHT (travels ~along the view), so
    # camera-facing surfaces are lit and grazing edges/bumps darken (texture via the
    # normals), evenly across the frame — not a raking key that blows tops / crushes
    # undersides. A small off-axis fill adds a touch of dimension; ambient fills the rest.
    if rig.get("headlight"):
        # neuroglancer's exact model: light points straight down the view axis, so the
        # surfaces you see are the lit ones and any cast shadow falls BEHIND the geometry
        # (hidden) -> no visible shadow, evenly lit. Tiny offsets keep it from being dead-flat.
        dirs = {"Key":  (fwd + 0.05 * right - 0.05 * tup),
                "Fill": (fwd - 0.2 * right + 0.1 * tup),
                "Rim":  (fwd + 0.1 * tup),
                "Kick": (fwd - 0.1 * right)}
    else:
        # Hybrid: raking KEY gives the directional "rake" modeling (the liked style), while
        # the FILL is a CAMERA-FRONT headlight that lifts the shadow side so nothing the
        # viewer is looking at goes mysteriously dark (NG-like even visibility) and the
        # off-angle shadow patches are washed up. Rim/kick separate silhouettes.
        dirs = {"Key":  (0.4 * fwd + 0.85 * right - 0.7 * tup),   # off-axis raking key
                "Fill": (fwd + 0.05 * right),                     # camera-front fill (lift)
                "Rim":  (-0.8 * fwd + 0.7 * right + 0.4 * tup),   # back-right edge light
                "Kick": (-0.8 * fwd - 0.7 * right + 0.4 * tup)}   # back-left (opposing) kicker
    for name, d in dirs.items():
        obj = bpy.data.objects.get(name)
        if obj and d.length > 1e-9:
            obj.rotation_euler = d.normalized().to_track_quat("-Z", "Y").to_euler()


_CLIP_AXIS_W = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _build_clip_nodes(nt, surf):
    """Add cutaway nodes that make `surf` transparent on the hidden side of an
    axis-aligned plane. Returns the new surface socket (a Mix Shader). The plane is
    driven by value nodes cm_clip_{wx,wy,wz,pos,side,on} set per frame by _apply_clip:
    selected-axis world coord = X*wx + Y*wy + Z*wz; hide where (coord-pos)*side > 0."""
    def V(name, val):
        n = nt.nodes.new("ShaderNodeValue"); n.name = name
        n.outputs[0].default_value = val
        return n.outputs[0]

    def M(op, a, b):
        n = nt.nodes.new("ShaderNodeMath"); n.operation = op
        nt.links.new(a, n.inputs[0])
        if hasattr(b, "default_value") or not isinstance(b, (int, float)):
            nt.links.new(b, n.inputs[1])
        else:
            n.inputs[1].default_value = b
        return n.outputs[0]

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    wx, wy, wz = V("cm_clip_wx", 0.0), V("cm_clip_wy", 0.0), V("cm_clip_wz", 1.0)
    axc = M("ADD", M("ADD", M("MULTIPLY", sep.outputs["X"], wx),
                            M("MULTIPLY", sep.outputs["Y"], wy)),
                   M("MULTIPLY", sep.outputs["Z"], wz))   # selected-axis world coord (BU)
    pos_v, side_v, on_v = V("cm_clip_pos", 0.0), V("cm_clip_side", 1.0), V("cm_clip_on", 0.0)
    signed = M("MULTIPLY", M("SUBTRACT", axc, pos_v), side_v)   # >0 on the hidden side
    # Clean cut: remove ALL geometry beyond the plane (front AND back faces), so the cut
    # fully opens the mesh and reveals what's inside — no leftover inner wall veiling it.
    hide = M("MULTIPLY", M("GREATER_THAN", signed, 0.0), on_v)  # 1 -> transparent
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader"); mix.name = "cm_clip_mix"
    nt.links.new(hide, mix.inputs[0])
    nt.links.new(surf, mix.inputs[1])
    nt.links.new(transp.outputs[0], mix.inputs[2])
    return mix.outputs[0]


def _apply_clip(nt, clip, f=None) -> None:
    """Set (and optionally keyframe) the cm_clip_* value nodes from a clip dict
    {axis, position_bu, side} or None. No-op on materials without clip nodes."""
    if nt.nodes.get("cm_clip_on") is None:
        return
    nrm = (clip or {}).get("normal")
    if nrm:   # oblique: the dot-product weights ARE the (unit) plane normal
        mag = (nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2) ** 0.5 or 1.0
        wx, wy, wz = nrm[0] / mag, nrm[1] / mag, nrm[2] / mag
    else:
        wx, wy, wz = _CLIP_AXIS_W.get((clip or {}).get("axis", "z"), (0.0, 0.0, 1.0))
    vals = {"cm_clip_wx": wx, "cm_clip_wy": wy, "cm_clip_wz": wz,
            "cm_clip_pos": float((clip or {}).get("position_bu", 0.0)),
            "cm_clip_side": float((clip or {}).get("side", 1)),
            "cm_clip_on": 1.0 if clip else 0.0}
    for nm, val in vals.items():
        n = nt.nodes.get(nm)
        if n is not None:
            n.outputs[0].default_value = float(val)
            if f is not None:
                n.outputs[0].keyframe_insert("default_value", frame=f)


def _load_npz_mesh(path: str, name: str, calc_edges: bool = False, color_space: str = "linear"):
    """Fast direct-to-bpy mesh loader: read a `.npz` of verts/faces/colors and
    populate a `bpy.types.Mesh` via `foreach_set` (vectorized). Avoids
    `bpy.ops.wm.ply_import`, the undo stack, and operator selection state — the
    dominant cost of cold-starting a render with many large meshes."""
    import numpy as np

    arrs = np.load(path)
    v = np.ascontiguousarray(arrs["v"], dtype=np.float32)
    f = np.ascontiguousarray(arrs["f"], dtype=np.int32)
    Nv, Nf = int(len(v)), int(len(f))
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(Nv)
    mesh.vertices.foreach_set("co", v.ravel())
    mesh.loops.add(Nf * 3)
    mesh.loops.foreach_set("vertex_index", f.ravel())
    mesh.polygons.add(Nf)
    mesh.polygons.foreach_set("loop_start",
                              (np.arange(Nf, dtype=np.int32) * 3))
    mesh.polygons.foreach_set("loop_total", np.full(Nf, 3, dtype=np.int32))
    if "c" in arrs.files:                 # per-vertex (per-segment) colors
        c = np.ascontiguousarray(arrs["c"], dtype=np.uint8)
        if c.ndim == 2 and c.shape[1] == 3:    # add opaque alpha
            c = np.concatenate([c, np.full((len(c), 1), 255, dtype=np.uint8)], axis=1)
        # Trimesh creates a uniform [102,102,102,255] visual by default even when
        # CineMap did not intentionally bake colors. Treat that as uncolored geometry
        # so the explicit mesh material/NG color in scene.json drives the render.
        is_default_trimesh_gray = (
            len(c) > 0
            and np.all(c == np.array([102, 102, 102, 255], dtype=np.uint8))
        )
        if not is_default_trimesh_gray:
            cf = c.astype(np.float32) / 255.0
            if str(color_space).lower() == "srgb":
                rgb = cf[:, :3]
                cf[:, :3] = np.where(
                    rgb <= 0.04045,
                    rgb / 12.92,
                    ((rgb + 0.055) / 1.055) ** 2.4,
                )
            cf = cf.ravel()
            ca = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
            ca.data.foreach_set("color", cf)
    # calc_edges builds an explicit edge table from the loops. Only cutaway meshes need
    # that downstream for bmesh bisect/weld/cap; normal render-only meshes can render
    # straight from faces, and skipping edge construction saves cold-start time on big
    # scenes. We deliberately SKIP mesh.validate(): it's a full single-threaded pass over
    # every face. The verts/faces come from our own decode pipeline, so there is nothing
    # to repair here.
    mesh.update(calc_edges=calc_edges)
    if "n" in arrs.files:
        try:
            n = np.ascontiguousarray(arrs["n"], dtype=np.float32)
            if n.shape == (Nv, 3):
                mesh.normals_split_custom_set_from_vertices(n)
                mesh.update()
        except Exception as e:  # noqa: BLE001
            print(f"[blender] custom normals skipped for {name}: {e}", flush=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _import_meshes(scene_spec: dict) -> dict:
    """Import each mesh once; return name -> (object, material).

    `.npz` assets (verts + faces + optional per-vertex colors) load via the fast
    direct path; `.ply`/`.obj` legacy assets fall back to Blender's import operators.
    Per-vertex colors (distinct color per segment) drive Base Color from the color
    attribute; otherwise a solid color.
    """
    import numpy as np

    out = {}
    prof = scene_spec.get("direction", {}).get("material", {})
    flat_shading = prof.get("flat_shading", True)
    color_space = prof.get("color_space", "linear")
    t_all = time.perf_counter()
    for m in scene_spec["meshes"]:
        t_mesh = time.perf_counter()
        path = m["obj_path"]
        if path.lower().endswith(".npz"):
            obj = _load_npz_mesh(path, m["id"],
                                 calc_edges=bool(m.get("clip_cap", m.get("clip"))),
                                 color_space=color_space)
        else:
            # legacy fallback for any pre-existing .ply / .obj cached assets
            before = set(bpy.data.objects)
            if path.lower().endswith(".ply"):
                bpy.ops.wm.ply_import(filepath=path)
            else:
                bpy.ops.wm.obj_import(filepath=path)
            new = [o for o in bpy.data.objects if o not in before]
            if not new:
                continue
            obj = new[0]
            if len(new) > 1:
                with bpy.context.temp_override(active_object=obj, selected_editable_objects=new):
                    bpy.ops.object.join()
            obj.name = m["id"]
        bpy.context.view_layer.objects.active = obj
        # flat (per-face) shading by default — each face shades dark/light on its own,
        # giving the crisp faceted definition neuroglancer has; smooth blurs it to blobs.
        # Direct attribute write (no operator) — `bpy.ops.object.shade_flat()` goes
        # through selection/undo and is slow per-mesh.
        Np = len(obj.data.polygons)
        if Np:
            smooth = np.zeros(Np, dtype=bool) if flat_shading else np.ones(Np, dtype=bool)
            obj.data.polygons.foreach_set("use_smooth", smooth)
            obj.data.update()
        s = 1.0 / scene_spec["world"]["nm_per_bu"]  # nm -> BU
        obj.scale = (s, s, s)

        mat = bpy.data.materials.new(f"mat_{m['id']}")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        col = _color_for_blender(m["color"], color_space)
        alpha_v = nt.nodes.new("ShaderNodeValue")
        alpha_v.name = "cm_alpha"
        alpha_v.outputs[0].default_value = 1.0
        silh_v = nt.nodes.new("ShaderNodeValue")
        silh_v.name = "cm_silh"
        silh_v.outputs[0].default_value = 0.0

        def _set_in(node, key, val):
            sock = node.inputs.get(key)
            if sock is None:
                return
            try:
                cur = sock.default_value
                if hasattr(cur, "__len__") and not isinstance(val, (list, tuple)):
                    vals = [float(val)] * len(cur)
                    if len(vals) == 4:
                        vals[3] = 1.0
                    sock.default_value = vals
                else:
                    sock.default_value = val
            except Exception:
                sock.default_value = val

        def _link_specular(node, val):
            key = "Specular IOR Level" if "Specular IOR Level" in node.inputs else "Specular"
            if key not in node.inputs:
                return
            spec_v = nt.nodes.new("ShaderNodeValue")
            spec_v.name = "cm_specular"
            spec_v.outputs[0].default_value = float(val)
            out = spec_v.outputs[0]
            if prof.get("alpha_scaled_specular", False):
                mult = nt.nodes.new("ShaderNodeMath")
                mult.operation = "MULTIPLY"
                nt.links.new(alpha_v.outputs[0], mult.inputs[0])
                nt.links.new(spec_v.outputs[0], mult.inputs[1])
                out = mult.outputs[0]
            nt.links.new(out, node.inputs[key])

        _set_in(bsdf, "Roughness", prof.get("roughness", 0.35))
        _link_specular(bsdf, prof.get("specular", 0.5))
        _set_in(bsdf, "Metallic", prof.get("metallic", 0.0))
        # per-frame material override: value nodes drive Metallic/Roughness so a layer can
        # turn reflective over the movie (set per frame in _set_mesh_state; default = the
        # look/director base, so meshes that never set it look exactly as before).
        base_metal = float(prof.get("metallic", 0.0)); base_rough = float(prof.get("roughness", 0.35))
        metal_v = nt.nodes.new("ShaderNodeValue"); metal_v.name = "cm_metal"; metal_v.outputs[0].default_value = base_metal
        rough_v = nt.nodes.new("ShaderNodeValue"); rough_v.name = "cm_rough"; rough_v.outputs[0].default_value = base_rough
        if "Metallic" in bsdf.inputs:
            nt.links.new(metal_v.outputs[0], bsdf.inputs["Metallic"])
        if "Roughness" in bsdf.inputs:
            nt.links.new(rough_v.outputs[0], bsdf.inputs["Roughness"])
        _mat_base[obj.name] = (base_metal, base_rough)
        _set_in(bsdf, "Transmission Weight", prof.get("transmission", 0.0))
        _set_in(bsdf, "Subsurface Weight", prof.get("subsurface", 0.0))
        _set_in(bsdf, "Sheen Weight", prof.get("sheen", 0.0))
        _set_in(bsdf, "Coat Weight", prof.get("coat", 0.0))
        has_colors = bool(getattr(obj.data, "color_attributes", None)) and len(obj.data.color_attributes) > 0
        if has_colors:  # per-vertex (per-segment) colors
            csrc = nt.nodes.new("ShaderNodeVertexColor")
            csrc.layer_name = obj.data.color_attributes[0].name
            color_out = csrc.outputs["Color"]
        else:  # solid color
            csrc = nt.nodes.new("ShaderNodeRGB")
            csrc.name = "cm_color"
            csrc.outputs[0].default_value = col
            color_out = csrc.outputs[0]
        if "Specular Tint" in bsdf.inputs:
            tint = float(prof.get("specular_tint", 0.0) or 0.0)
            sock = bsdf.inputs["Specular Tint"]
            cur = getattr(sock, "default_value", None)
            if hasattr(cur, "__len__") and len(cur) >= 3:
                mix_tint = nt.nodes.new("ShaderNodeMixRGB")
                mix_tint.inputs["Fac"].default_value = tint
                mix_tint.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
                nt.links.new(color_out, mix_tint.inputs["Color2"])
                nt.links.new(mix_tint.outputs["Color"], sock)
            else:
                _set_in(bsdf, "Specular Tint", tint)
        # Ambient occlusion: darken crevices/concavities so bumpy surfaces read crisp
        # and defined (the "within-mesh shadows" that make NG meshes pop). The AO node
        # outputs the color attenuated by occlusion; blend it in by the `ao` amount.
        ao_amt = prof.get("ao", 0.0)
        if ao_amt > 0:
            ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
            ao.samples = 16
            # reach (BU) = AO distance in nm / nm_per_bu — long enough to catch where
            # separate tubes contact/overlap, giving the dark contact shadows NG shows.
            ao.inputs["Distance"].default_value = (
                prof.get("ao_distance_nm", 2000.0) / scene_spec["world"]["nm_per_bu"])
            nt.links.new(color_out, ao.inputs["Color"])
            mixao = nt.nodes.new("ShaderNodeMixRGB"); mixao.blend_type = "MIX"
            mixao.inputs[0].default_value = ao_amt
            nt.links.new(color_out, mixao.inputs[1])          # ao=0 -> original color
            nt.links.new(ao.outputs["Color"], mixao.inputs[2])  # ao=1 -> crevices darkened
            color_out = mixao.outputs[0]
        # Cavity / curvature shading via the geometry's Pointiness (0 concave .. 1 convex,
        # 0.5 flat): factor = 1 + cavity*2*(pointiness-0.5) -> concave creases darken,
        # convex ridges brighten. Sharper than AO and follows the surface (MeshLab-like),
        # making every fold/bump pop. Multiplies the running color.
        cav = prof.get("cavity", 0.0)
        if cav > 0:
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            csub = nt.nodes.new("ShaderNodeMath"); csub.operation = "SUBTRACT"
            csub.inputs[1].default_value = 0.5
            nt.links.new(geo.outputs["Pointiness"], csub.inputs[0])
            cmul = nt.nodes.new("ShaderNodeMath"); cmul.operation = "MULTIPLY"
            cmul.inputs[1].default_value = 2.0 * cav
            nt.links.new(csub.outputs[0], cmul.inputs[0])
            cadd = nt.nodes.new("ShaderNodeMath"); cadd.operation = "ADD"
            cadd.use_clamp = True                                 # factor >= 0
            cadd.inputs[1].default_value = 1.0
            nt.links.new(cmul.outputs[0], cadd.inputs[0])
            cavmix = nt.nodes.new("ShaderNodeMixRGB"); cavmix.blend_type = "MULTIPLY"
            cavmix.inputs[0].default_value = 1.0
            nt.links.new(color_out, cavmix.inputs[1])
            nt.links.new(cadd.outputs[0], cavmix.inputs[2])       # color * curvature factor
            color_out = cavmix.outputs[0]
        # Subtle Fresnel edge-darken: a soft dark rim at each object's silhouette so
        # overlapping/adjacent objects separate visually (the front one's grazing edge
        # darkens against whatever is behind). factor = 1 - edge_darken * facing^power,
        # with facing = 0 head-on, 1 at grazing -> only the rim darkens, interior intact.
        edk = prof.get("edge_darken", 0.0)
        if edk > 0:
            elw = nt.nodes.new("ShaderNodeLayerWeight")
            epw = nt.nodes.new("ShaderNodeMath"); epw.operation = "POWER"
            epw.inputs[1].default_value = prof.get("edge_power", 4.0)
            nt.links.new(elw.outputs["Facing"], epw.inputs[0])
            emul = nt.nodes.new("ShaderNodeMath"); emul.operation = "MULTIPLY"
            emul.inputs[1].default_value = edk
            nt.links.new(epw.outputs[0], emul.inputs[0])
            esub = nt.nodes.new("ShaderNodeMath"); esub.operation = "SUBTRACT"
            esub.use_clamp = True
            esub.inputs[0].default_value = 1.0
            nt.links.new(emul.outputs[0], esub.inputs[1])        # 1 - edge_darken*facing^p
            edmix = nt.nodes.new("ShaderNodeMixRGB"); edmix.blend_type = "MULTIPLY"
            edmix.inputs[0].default_value = 1.0
            nt.links.new(color_out, edmix.inputs[1])
            nt.links.new(esub.outputs[0], edmix.inputs[2])       # color * factor (broadcast)
            color_out = edmix.outputs[0]
        ng_absdot = None
        if prof.get("ng_shader"):
            # Faithful port of neuroglancer's mesh GLSL: per fragment,
            #   lightingFactor = abs(dot(normal, lightDirection)) * 0.8 + 0.2
            #   color = lightingFactor * baseColor
            # Neuroglancer uses one camera-oriented light vector for the whole frame, not
            # Blender's per-fragment Incoming vector. _set_ng_shader_light drives these nodes
            # from the current camera before every render/export keyframe.
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            lx = nt.nodes.new("ShaderNodeValue"); lx.name = "cm_ng_light_x"; lx.outputs[0].default_value = 0.0
            ly = nt.nodes.new("ShaderNodeValue"); ly.name = "cm_ng_light_y"; ly.outputs[0].default_value = 0.0
            lz = nt.nodes.new("ShaderNodeValue"); lz.name = "cm_ng_light_z"; lz.outputs[0].default_value = 1.0
            light_vec = nt.nodes.new("ShaderNodeCombineXYZ")
            nt.links.new(lx.outputs[0], light_vec.inputs["X"])
            nt.links.new(ly.outputs[0], light_vec.inputs["Y"])
            nt.links.new(lz.outputs[0], light_vec.inputs["Z"])
            dotp = nt.nodes.new("ShaderNodeVectorMath"); dotp.operation = "DOT_PRODUCT"
            nt.links.new(geo.outputs["Normal"], dotp.inputs[0])
            nt.links.new(light_vec.outputs["Vector"], dotp.inputs[1])
            absd = nt.nodes.new("ShaderNodeMath"); absd.operation = "ABSOLUTE"
            nt.links.new(dotp.outputs["Value"], absd.inputs[0])
            ng_absdot = absd.outputs[0]
            fac = nt.nodes.new("ShaderNodeMath"); fac.operation = "MULTIPLY_ADD"
            fac.inputs[1].default_value = 0.8                       # directionalLighting
            fac.inputs[2].default_value = 0.2                       # ambientLighting
            nt.links.new(absd.outputs[0], fac.inputs[0])
            emis = nt.nodes.new("ShaderNodeVectorMath"); emis.operation = "SCALE"
            nt.links.new(color_out, emis.inputs[0])
            nt.links.new(fac.outputs["Value"], emis.inputs["Scale"])
            bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            if "Emission Color" in bsdf.inputs:
                nt.links.new(emis.outputs["Vector"], bsdf.inputs["Emission Color"])
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 1.0
            _set_in(bsdf, "Roughness", 1.0); _set_in(bsdf, "Specular IOR Level", 0.0)
        else:
            nt.links.new(color_out, bsdf.inputs["Base Color"])
            if "Emission Color" in bsdf.inputs:
                nt.links.new(color_out, bsdf.inputs["Emission Color"])
            # Emission strength via a value node so the director can pulse it per frame
            # (the appear/highlight glow) by overriding cm_emit; base = the material floor.
            emit_v = nt.nodes.new("ShaderNodeValue"); emit_v.name = "cm_emit"
            emit_v.outputs[0].default_value = prof.get("emission_strength", 0.15)
            if "Emission Strength" in bsdf.inputs:
                nt.links.new(emit_v.outputs[0], bsdf.inputs["Emission Strength"])

        # NG 3D render state: alpha = object_alpha * silhouette_factor.  In NG shader mode
        # the factor is Neuroglancer's exact pow(1 - 0.8*abs(dot(N, light)), power);
        # the other presets keep the older Blender-facing approximation.
        # silhouette=0 -> factor 1 -> plain object_alpha everywhere.
        if prof.get("ng_shader") and ng_absdot is not None:
            # Neuroglancer applies silhouette after directional scaling:
            # vColor *= pow(1.0 - absCosAngle, uSilhouettePower), where
            # absCosAngle = abs(dot(normal, lightDirection * 0.8)).
            scaled = nt.nodes.new("ShaderNodeMath"); scaled.operation = "MULTIPLY"
            scaled.inputs[1].default_value = 0.8
            nt.links.new(ng_absdot, scaled.inputs[0])
            one_minus = nt.nodes.new("ShaderNodeMath"); one_minus.operation = "SUBTRACT"
            one_minus.inputs[0].default_value = 1.0
            nt.links.new(scaled.outputs[0], one_minus.inputs[1])
            powr = nt.nodes.new("ShaderNodeMath"); powr.operation = "POWER"
            nt.links.new(one_minus.outputs[0], powr.inputs[0])
        else:
            lw = nt.nodes.new("ShaderNodeLayerWeight")
            powr = nt.nodes.new("ShaderNodeMath"); powr.operation = "POWER"
            nt.links.new(lw.outputs["Facing"], powr.inputs[0])   # base = facing (0 head-on, 1 grazing)
        nt.links.new(silh_v.outputs[0], powr.inputs[1])      # exponent = silhouette power
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.use_clamp = True
        nt.links.new(alpha_v.outputs[0], mul.inputs[0])
        nt.links.new(powr.outputs[0], mul.inputs[1])
        if "Alpha" in bsdf.inputs:
            nt.links.new(mul.outputs[0], bsdf.inputs["Alpha"])

        # Fresnel edge-glow: grazing edges emit their own color (a soft rim glow that
        # makes structures read as 'lit' against the dark background, esp. with bloom).
        # Emission Strength = cm_emit (base/pulse) + Facing * edge_glow.
        edge = prof.get("edge_glow", 0.0)
        if edge > 0 and "Emission Strength" in bsdf.inputs and "emit_v" in locals():
            egw = nt.nodes.new("ShaderNodeMath"); egw.operation = "MULTIPLY"
            egw.inputs[1].default_value = edge
            nt.links.new(lw.outputs["Facing"], egw.inputs[0])    # 0 head-on, 1 grazing
            eadd = nt.nodes.new("ShaderNodeMath"); eadd.operation = "ADD"
            nt.links.new(emit_v.outputs[0], eadd.inputs[0])
            nt.links.new(egw.outputs[0], eadd.inputs[1])
            nt.links.new(eadd.outputs[0], bsdf.inputs["Emission Strength"])

        out_node = nt.nodes.get("Material Output") or next(
            (n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        surf = bsdf.outputs[0]
        # Optional backface culling (Cycles): mix in a Transparent BSDF on back faces, so
        # a tube/cell-body shows only its front surface instead of front+back stacking.
        # At low opacity that halves the alpha build-up -> glassier, see-through transparent
        # state closer to neuroglancer (instead of dense clusters piling up to opaque).
        if prof.get("backface_cull") and out_node is not None:
            geo_bf = nt.nodes.new("ShaderNodeNewGeometry")
            transp = nt.nodes.new("ShaderNodeBsdfTransparent")
            bfmix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(geo_bf.outputs["Backfacing"], bfmix.inputs[0])  # 1 on back
            nt.links.new(surf, bfmix.inputs[1])                          # front -> shaded
            nt.links.new(transp.outputs[0], bfmix.inputs[2])             # back -> clear
            surf = bfmix.outputs[0]
        # Optional cutaway clip plane (per layer, animatable): hide geometry on the chosen
        # side of an axis-aligned plane by mixing in a Transparent BSDF, revealing what's
        # inside/behind. Plane params are driven per frame by the cm_clip_* value nodes.
        if m.get("clip"):
            surf = _build_clip_nodes(nt, surf)
        if m.get("clip_cap", m.get("clip")):
            # geometric cutaway: keep a pristine copy of the geometry; each frame we
            # rebuild the object as this mesh sliced at the clip plane and capped (filled
            # cross-section) — a true solid cut, not a shader transparency trick.
            # The cap needs CLOSED cut loops, but draco fragments are concatenated unwelded
            # (process=False) -> ~1000s of open seam edges -> the slice produces open loops
            # and triangle_fill can't cap (looks like no cap). Weld the pristine copy ONCE
            # here (cheap, not per frame) so every slice yields watertight cross-sections.
            import bmesh
            orig = obj.data.copy()
            bmw = bmesh.new(); bmw.from_mesh(orig)
            # Weld coincident seam vertices at a tight tolerance — larger ones collapse
            # thin neurites and CREATE boundary edges, so this is intentionally small.
            bmesh.ops.remove_doubles(bmw, verts=bmw.verts[:], dist=1e-3)
            # Then close any tiny leftover artifact holes (a few stray boundary edges) so
            # the cross-section is fully watertight and caps solid even where pieces nearly
            # meet. Genuine large openings have many edges and are left alone.
            # Pass ONLY boundary edges (the open seams) — not bmw.edges[:]. holes_fill
            # over the full edge set forced a scan of all ~9M edges on the big meshes and
            # was a multi-minute single-threaded stall; boundary edges are a tiny subset.
            try:
                boundary = [e for e in bmw.edges if e.is_boundary]
                if boundary:
                    bmesh.ops.holes_fill(bmw, edges=boundary, sides=8)
            except Exception:  # noqa: BLE001
                pass
            bmw.to_mesh(orig); bmw.free()
            orig.name = f"orig_{m['id']}"   # stable + fake-user so the warm cache keeps it
            orig.use_fake_user = True
            _orig_mesh[obj.name] = orig
        if out_node is not None:
            nt.links.new(surf, out_node.inputs["Surface"])

        try:
            mat.blend_method = prof.get("blend_method", "BLEND")
        except Exception:  # noqa: BLE001
            pass
        try:
            mat.shadow_method = prof.get("shadow_method", "HASHED")
        except Exception:  # noqa: BLE001
            pass
        if prof.get("show_transparent_back") is not None:
            try:
                mat.show_transparent_back = bool(prof.get("show_transparent_back"))
            except Exception:  # noqa: BLE001
                pass
        try:
            if hasattr(mat, "surface_render_method"):
                desired = {
                    "HASHED": "DITHERED",
                    "DITHERED": "DITHERED",
                    "BLEND": "BLENDED",
                    "BLENDED": "BLENDED",
                }.get(prof.get("blend_method"))
                if desired:
                    try:
                        mat.surface_render_method = desired
                    except Exception:  # noqa: BLE001
                        mat.surface_render_method = "DITHERED"
        except Exception:  # noqa: BLE001
            pass
        try:
            if hasattr(mat, "cycles"):
                mat.cycles.use_transparent_shadow = bool(prof.get("transparent_shadows", True))
        except Exception:  # noqa: BLE001
            pass
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        out[m["id"]] = (obj, mat)
        print(f"[blender] imported {m['id']} faces={len(obj.data.polygons)} "
              f"clip={bool(m.get('clip'))} cap={bool(m.get('clip_cap', m.get('clip')))} "
              f"in {time.perf_counter() - t_mesh:.1f}s",
              flush=True)
    print(f"[blender] imported {len(out)}/{len(scene_spec['meshes'])} meshes "
          f"in {time.perf_counter() - t_all:.1f}s", flush=True)
    return out


_orig_mesh: dict = {}      # obj.name -> pristine (unclipped) mesh datablock
_clip_state: dict = {}     # obj.name -> last applied clip signature (skip redundant rebuilds)
_mat_base: dict = {}       # obj.name -> (base_metallic, base_roughness) for per-frame override


def _geometric_clip(obj, clip) -> None:
    """Rebuild obj's mesh as the pristine geometry sliced at the clip plane, with the
    cut capped — a solid cross-section cutaway. `clip`=None restores the full mesh.
    Operates in the object's local space (vertices are nm), so uses position_nm."""
    import bmesh
    orig = _orig_mesh.get(obj.name)
    if orig is None:
        return
    sig = None if not clip else (round(clip.get("position_nm", 0.0), 1), clip.get("side"),
                                 tuple(clip["normal"]) if clip.get("normal") else clip.get("axis"))
    if _clip_state.get(obj.name) == sig:
        return                                     # unchanged this frame -> no rebuild
    _clip_state[obj.name] = sig
    bm = bmesh.new()
    bm.from_mesh(orig)                             # always start from the pristine mesh
    if clip:
        nrm = clip.get("normal")
        if nrm:
            mag = (nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2) ** 0.5 or 1.0
            n = Vector((nrm[0] / mag, nrm[1] / mag, nrm[2] / mag))
        else:
            n = {"x": Vector((1, 0, 0)), "y": Vector((0, 1, 0)),
                 "z": Vector((0, 0, 1))}[clip.get("axis", "z")]
        side = 1.0 if float(clip.get("side", 1)) >= 0 else -1.0
        co = n * float(clip.get("position_nm", 0.0))
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-4,
                                     plane_co=co, plane_no=n * side, clear_outer=True)
        cut = [e for e in res.get("geom_cut", []) if isinstance(e, bmesh.types.BMEdge)]
        if cut:
            # Cap the cross-section by filling each cut boundary loop INDEPENDENTLY.
            # holes_fill closes every loop (one face per loop) -> a complete, watertight
            # cap even with many separate, concave cells. (triangle_fill does ONE global 2D
            # triangulation of all loops at once and leaves big gaps when there are many
            # complex cross-sections — the torn/partial caps.) Then triangulate the new cap
            # faces so concave n-gons shade cleanly.
            try:
                before = set(bm.faces)
                bmesh.ops.holes_fill(bm, edges=cut, sides=0)
                newf = [f for f in bm.faces if f not in before]
                if newf:
                    tri = bmesh.ops.triangulate(bm, faces=newf)
                    newf = [f for f in tri.get("faces", newf) if f.is_valid]
                # The cap is coplanar with the clip plane, but the rest of the cell is
                # SMOOTH-shaded — letting the cap inherit that averages its triangles'
                # normals into a wrinkled fan. Flat-shade the cap so it reads as one clean
                # solid cross-section; the cut + cap make a closed solid again, so recalc
                # gives every face (incl. the cap) a consistent outward normal.
                for f in newf:
                    f.smooth = False
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            except Exception:  # noqa: BLE001 (degenerate loop -> fall back to global fill)
                try:
                    bmesh.ops.triangle_fill(bm, edges=cut, use_beauty=True, normal=n * side)
                except Exception:  # noqa: BLE001
                    pass
        bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()


def _set_mesh_state(
    meshes: dict,
    overrides: dict,
    base_emit: float = 0.15,
    cast_shadows: bool = True,
    color_space: str = "linear",
) -> None:
    for mid, (obj, mat) in meshes.items():
        ov = overrides.get(mid)
        if ov is None:  # not referenced this frame -> hidden (belongs to another keyframe)
            obj.hide_render = True
            continue
        if obj.name in _orig_mesh:
            _geometric_clip(obj, ov.get("clip"))   # solid cutaway, rebuilt per frame
        opacity = ov.get("opacity", 1.0)            # effective alpha = fade * Opacity(3d)
        visible = ov.get("visible", True) and opacity > 0.001
        obj.hide_render = not visible
        # Always cast a shadow while visible. Cycles attenuates the shadow by the object's
        # alpha (transparent shadows), so as a layer fades in/out its shadow ramps smoothly
        # with opacity. A hard on/off threshold here instead caused a one-frame brightness
        # POP whenever a layer faded through it (e.g. the segmentation reveal at ~2s):
        # below the cutoff no shadow, above it the whole tangle self-shadowed at once.
        obj.visible_shadow = (
            visible
            and cast_shadows
            and not os.environ.get("CINEMAP_NO_CAST_SHADOWS")
        )
        nt = mat.node_tree
        av, sv = nt.nodes.get("cm_alpha"), nt.nodes.get("cm_silh")
        if av is not None:
            av.outputs[0].default_value = opacity
        if sv is not None:
            sv.outputs[0].default_value = ov.get("silhouette", 0.0)   # Silhouette (3d)
        # reset emission every frame (base + the director's transient glow, if any)
        ev = nt.nodes.get("cm_emit")
        if ev is not None:
            ev.outputs[0].default_value = base_emit + ov.get("emphasis", 0.0)
        cv = nt.nodes.get("cm_color")
        if cv is not None and ov.get("color") is not None:
            cv.outputs[0].default_value = _color_for_blender(ov["color"], color_space)
        if av is None and "Alpha" in nt.nodes["Principled BSDF"].inputs:
            nt.nodes["Principled BSDF"].inputs["Alpha"].default_value = opacity
        # per-frame material: override metallic/roughness when this frame sets them, else
        # fall back to the layer's base (the global look / director value).
        mv, rv = nt.nodes.get("cm_metal"), nt.nodes.get("cm_rough")
        if mv is not None:
            bm, br = _mat_base.get(obj.name, (0.0, 0.35))
            m_ov, r_ov = ov.get("metallic"), ov.get("roughness")
            mv.outputs[0].default_value = bm if m_ov is None else float(m_ov)
            if rv is not None:
                rv.outputs[0].default_value = br if r_ov is None else float(r_ov)
        _apply_clip(nt, ov.get("clip"))


def _ng_light_direction(frame: dict) -> Vector:
    """Neuroglancer's mesh light direction is camera orientation-derived and constant
    for the whole frame. The sign is immaterial because NG uses abs(dot(...))."""
    pos = Vector(frame["camera"]["position_bu"])
    look = Vector(frame["camera"]["look_at_bu"])
    direction = look - pos
    if direction.length <= 1e-9:
        return Vector((0.0, 0.0, 1.0))
    return direction.normalized()


def _set_ng_shader_light(meshes: dict, frame: dict, f: int | None = None) -> None:
    light = _ng_light_direction(frame)
    vals = {
        "cm_ng_light_x": light.x,
        "cm_ng_light_y": light.y,
        "cm_ng_light_z": light.z,
    }
    for _mid, (_obj, mat) in meshes.items():
        if mat is None or not getattr(mat, "use_nodes", False):
            continue
        for name, val in vals.items():
            node = mat.node_tree.nodes.get(name)
            if node is None:
                continue
            node.outputs[0].default_value = float(val)
            if f is not None:
                node.outputs[0].keyframe_insert("default_value", frame=f)


_slice_objs: list = []


def _make_slice(sl: dict, name: str):
    """Build one textured EM-slice quad at its world placement. Used both by the
    per-frame still renderer and the .blend exporter (one quad per frame there)."""
    origin = Vector(sl["origin_bu"])
    u = Vector(sl["u_bu"])
    v = Vector(sl["v_bu"])
    # build a quad from origin, +u, +u+v, +v
    verts = [origin, origin + u, origin + u + v, origin + v]
    mesh = bpy.data.meshes.new(f"slice_{name}")
    mesh.from_pydata([list(p) for p in verts], [], [[0, 1, 2, 3]])
    mesh.update()
    # UVs. The slice image has row 0 at the smallest-v world coord (the
    # `origin` corner), but Blender samples image row 0 at UV v=1 — so the v
    # axis must be flipped here, or the EM/seg texture renders mirrored along
    # v relative to the meshes (visible as a vertical misalignment).
    mesh.uv_layers.new(name="UVMap")
    uvs = [(0, 1), (1, 1), (1, 0), (0, 0)]  # origin->(0,1): row 0 maps to origin corner
    for li, _loop in enumerate(mesh.loops):
        mesh.uv_layers.active.data[li].uv = uvs[li % 4]
    obj = bpy.data.objects.new(f"slice_{name}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.visible_shadow = False  # don't shadow meshes that sit below the plane

    mat = bpy.data.materials.new(f"slice_mat_{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(sl["image_path"], check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    tex.image = img
    emit = nt.nodes.new("ShaderNodeEmission")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    op = float(sl.get("opacity", 1.0))
    if sl.get("occlude"):
        # opaque cross-section: the EM plane blocks geometry behind it (old behavior)
        emit.inputs["Strength"].default_value = 0.9
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = op
        nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
        nt.links.new(emit.outputs["Emission"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        # NON-occluding (default): the plane is fully transparent to camera rays, so it
        # NEVER hides the 3D meshes — the EM is ADDED as a glowing overlay where visible.
        # Meshes in front still cover it; meshes behind show through. ('both'/cutaway still
        # removes meshes on the cut side via the clip, independent of this.)
        emit.inputs["Strength"].default_value = 0.6 * op
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(emit.outputs["Emission"], add.inputs[0])
        nt.links.new(transp.outputs["BSDF"], add.inputs[1])
        nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    mat.blend_method = "BLEND"
    obj.data.materials.append(mat)
    return obj


def _build_slices(frame: dict) -> None:
    global _slice_objs
    for o in _slice_objs:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    _slice_objs = []
    for i, sl in enumerate(frame.get("slices", [])):
        if sl.get("opacity", 1.0) <= 0.001:
            continue
        _slice_objs.append(_make_slice(sl, str(i)))


def _set_camera(frame: dict) -> None:
    scene = bpy.context.scene
    cam = scene.objects.get("Camera")
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
    cam.location = Vector(frame["camera"]["position_bu"])
    if frame["camera"].get("type") == "ORTHO":  # straight-down validation views
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = frame["camera"].get("ortho_scale", 4.0)
    else:
        cam.data.type = "PERSP"
        # Pin the FOV to the VERTICAL axis so the vertical framing is constant across
        # aspect ratios. With Blender's default AUTO sensor fit, a wide (16:9) frame
        # applies the FOV to the horizontal axis, shrinking the vertical FOV and
        # cropping tall content top/bottom. The zoom is calibrated on vertical extent,
        # so vertical fit keeps the framing right (a wider frame just shows more sides).
        cam.data.sensor_fit = "VERTICAL"
        cam.data.lens_unit = "FOV"
        cam.data.angle = frame["camera"]["fov_rad"]  # FOV on the fit (vertical) axis
    # Orient from BOTH the look direction AND the camera up vector. Previously `up`
    # was ignored and the camera just tracked world +Y, which dropped all camera roll
    # and oriented the scene inconsistently with neuroglancer. The up vector is already
    # in the correct (neuroglancer Y-down) convention from ng_to_camera, so we use it
    # directly. Blender camera looks along -Z with +Y up, so we build the world
    # rotation from right/up/back columns.
    direction = (Vector(frame["camera"]["look_at_bu"]) - cam.location).normalized()
    up = Vector(frame["camera"].get("up", [0.0, 0.0, 1.0]))
    z = -direction                                  # camera local +Z (points back)
    y = up - up.dot(z) * z                           # up, orthogonalized to z
    y = y.normalized() if y.length > 1e-9 else Vector((0.0, 0.0, 1.0))
    x = y.cross(z)                                    # right (right-handed: x = y × z)
    if frame["camera"].get("flip_handed"):
        # z,y,x datasets: reordering the camera to xyz reflects chirality, so mirror the
        # camera (negated right -> det -1 matrix) to reproduce neuroglancer's view. A
        # reflection isn't a rotation, so set matrix_world directly (rotation_euler can't).
        basis = Matrix((-x, y, z)).transposed()
        m = basis.to_4x4(); m.translation = cam.location
        cam.matrix_world = m
    else:
        cam.rotation_euler = Matrix((x, y, z)).transposed().to_euler()
    # subtle depth-of-field on the framed subject: focus at the look-at (what
    # neuroglancer centered on). Faithful — only far/near context softens slightly.
    dof = frame["camera"].get("dof")
    if dof and cam.data.type == "PERSP":
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = (Vector(frame["camera"]["look_at_bu"]) - cam.location).length
        cam.data.dof.aperture_fstop = dof.get("fstop", 4.0)
    else:
        cam.data.dof.use_dof = False


_fade_overlay = None


def _ensure_fade_overlay():
    global _fade_overlay
    try:
        if _fade_overlay and _fade_overlay.name in bpy.data.objects:
            return _fade_overlay
    except ReferenceError:
        _fade_overlay = None
    mesh = bpy.data.meshes.new("cm_fade_overlay_mesh")
    mesh.from_pydata([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [], [[0, 1, 2, 3]])
    mesh.update()
    obj = bpy.data.objects.new("cm_fade_overlay", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.visible_shadow = False

    mat = bpy.data.materials.new("cm_fade_overlay_mat")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.show_transparent_back = False
    nt = mat.node_tree
    nt.nodes.clear()
    alpha = nt.nodes.new("ShaderNodeValue")
    alpha.name = "cm_fade_alpha"
    alpha.outputs[0].default_value = 0.0
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    black = nt.nodes.new("ShaderNodeBsdfDiffuse")
    black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(alpha.outputs[0], mix.inputs[0])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(black.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    _fade_overlay = obj
    return obj


def _set_fade_overlay(frame: dict, f: int | None = None) -> None:
    alpha = max(0.0, min(1.0, float(frame.get("fade_alpha", 0.0) or 0.0)))
    obj = _ensure_fade_overlay()
    cam = bpy.context.scene.camera
    obj.parent = cam
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = (0.0, 0.0, -1.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    aspect = max(1e-6, bpy.context.scene.render.resolution_x / max(1, bpy.context.scene.render.resolution_y))
    if cam and cam.data.type == "ORTHO":
        h = float(cam.data.ortho_scale)
    else:
        h = 2.0 * math.tan(float(frame["camera"]["fov_rad"]) / 2.0)
    obj.scale = (h * aspect * 0.5, h * 0.5, 1.0)
    obj.hide_render = obj.hide_viewport = alpha <= 0.001
    av = obj.active_material.node_tree.nodes.get("cm_fade_alpha")
    if av is not None:
        av.outputs[0].default_value = alpha
        if f is not None:
            av.outputs[0].keyframe_insert("default_value", frame=f)
    if f is not None:
        obj.keyframe_insert("hide_render", frame=f)
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("scale", frame=f)


def _recover_meshes(spec: dict):
    """Rebuild the {id: (obj, mat)} map (and the _orig_mesh / _mat_base globals) from a
    warm-cached .blend that was just opened — so we skip the expensive re-import + weld.
    Returns None if any expected object is missing (treat as a cache miss -> rebuild)."""
    out = {}
    for m in spec["meshes"]:
        obj = bpy.data.objects.get(m["id"])
        if obj is None:
            return None
        mat = obj.active_material
        out[m["id"]] = (obj, mat)
        if mat and mat.use_nodes:
            mv = mat.node_tree.nodes.get("cm_metal")
            rv = mat.node_tree.nodes.get("cm_rough")
            if mv is not None:
                _mat_base[obj.name] = (mv.outputs[0].default_value,
                                       rv.outputs[0].default_value if rv else 0.35)
        og = bpy.data.meshes.get(f"orig_{m['id']}")
        if og is not None:
            _orig_mesh[obj.name] = og
    return out


def main(scene_path: str) -> None:
    import os as _os
    with open(scene_path) as f:
        spec = json.load(f)
    if spec.get("export_blend"):
        export_blend(spec)
        return
    # Warm-scene reuse: the expensive part (import 1000s of .ply + weld for cutaways) is
    # cached as a .blend keyed by geometry signature. If it exists, open it and re-drive
    # the cheap per-frame state instead of rebuilding. Render settings / lights / camera
    # are (re)applied fresh below either way, so the fast toggle + resolution still take.
    warm = spec.get("warm_blend")
    meshes = None
    if warm and _os.path.exists(warm):
        try:
            bpy.ops.wm.open_mainfile(filepath=warm)
            meshes = _recover_meshes(spec)
            print(f"[blender] warm scene {'REUSED' if meshes else 'miss (rebuilding)'}: {warm}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[blender] warm open failed ({e}); rebuilding", flush=True)
            meshes = None
    if meshes is None:
        _clear()
        t_import = time.perf_counter()
        meshes = _import_meshes(spec)           # the expensive build
        print(f"[blender] cold scene build complete in {time.perf_counter() - t_import:.1f}s",
              flush=True)
        skip_warm_save = _os.environ.get("CINEMAP_SKIP_WARM_SAVE", "").lower() in (
            "1", "true", "yes", "on")
        if warm and skip_warm_save:
            print("[blender] warm cache save skipped (CINEMAP_SKIP_WARM_SAVE=1)",
                  flush=True)
        if warm and not skip_warm_save:         # cache it (a copy; current session untouched)
            try:
                _os.makedirs(_os.path.dirname(warm), exist_ok=True)
                t_save = time.perf_counter()
                bpy.ops.wm.save_as_mainfile(filepath=warm, copy=True)
                print(f"[blender] warm scene cached in {time.perf_counter() - t_save:.1f}s: "
                      f"{warm}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[blender] warm cache save failed: {e}", flush=True)
    # always (re)apply render settings + lights + bloom fresh (cheap; they change per render)
    _setup_render(spec)
    _add_light(spec)
    _setup_bloom(spec)
    scene = bpy.context.scene
    out_dir = spec["output_dir"]
    rig = spec.get("direction", {}).get("lighting", {})
    material_profile = spec.get("direction", {}).get("material", {})
    base_emit = material_profile.get("emission_strength", 0.15)
    cast_shadows = bool(material_profile.get("cast_shadows", True))
    color_space = material_profile.get("color_space", "linear")
    for fi, frame in enumerate(spec["frames"]):
        _set_camera(frame)
        if rig:
            _update_lights(frame, rig)
        _set_ng_shader_light(meshes, frame)
        _build_slices(frame)
        _set_mesh_state(
            meshes, frame.get("mesh_overrides", {}), base_emit, cast_shadows, color_space
        )
        _set_fade_overlay(frame)
        _set_world_bg(frame)
        idx = frame.get("index", fi)  # global frame index (for split cluster jobs)
        scene.render.filepath = f"{out_dir}/frame_{idx:05d}.png"
        print(f"[blender] frame {fi + 1}/{len(spec['frames'])}", flush=True)
        bpy.ops.render.render(write_still=True)


# --------------------------------------------------------------------------
# .blend export: same scene, but per-frame state is baked to F-curves so the
# saved file plays/renders the whole shot. EM slices, textures and mesh geometry
# are all packed/embedded, so the result is self-contained — open it anywhere,
# scrub the timeline, hit F12.
# --------------------------------------------------------------------------


def _build_export_slices(spec: dict) -> None:
    """One textured quad per (frame, slice), each shown only on its own timeline
    frame via keyframed visibility. This reproduces the changing EM cross-section
    as the timeline plays — and, unlike an image SEQUENCE, single-image textures
    CAN be packed into the .blend (pack_all refuses sequences/movies), so the file
    stays self-contained."""
    frames = spec["frames"]
    n = len(frames)
    planes: list[tuple] = []  # (object, scene_frame)
    for fi, fr in enumerate(frames):
        f = fi + 1
        for si, sl in enumerate(fr.get("slices", [])):
            if sl.get("opacity", 1.0) <= 0.001:
                continue
            planes.append((_make_slice(sl, f"f{f:05d}_{si}"), f))
    # Boolean F-curves interpolate as constant, so three keyframes per quad
    # (hidden / shown / hidden) make it visible on exactly its own frame; Blender
    # holds the first/last keyframe value beyond the ends.
    for obj, f in planes:
        for kf, hidden in ((f - 1, True), (f, False), (f + 1, True)):
            if kf < 1 or kf > n:
                continue
            obj.hide_render = obj.hide_viewport = hidden
            obj.keyframe_insert("hide_render", frame=kf)
            obj.keyframe_insert("hide_viewport", frame=kf)


def _keyframe_camera(frame: dict, f: int) -> None:
    _set_camera(frame)  # positions/orients exactly as the still renderer does
    cam = bpy.context.scene.camera
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    if cam.data.type == "ORTHO":
        cam.data.keyframe_insert("ortho_scale", frame=f)
    else:
        cam.data.keyframe_insert("lens", frame=f)  # lens_unit=FOV -> lens tracks fov


def _keyframe_lights(frame: dict, rig: dict, f: int) -> None:
    """Bake camera-relative light orientation exactly as the still renderer drives it."""
    if not rig:
        return
    _update_lights(frame, rig)
    for name in ("Key", "Fill", "Rim", "Kick"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.keyframe_insert("rotation_euler", frame=f)


def _keyframe_meshes(
    meshes: dict,
    overrides: dict,
    f: int,
    base_emit: float = 0.15,
    cast_shadows: bool = True,
    color_space: str = "linear",
) -> None:
    for _mid, (obj, mat) in meshes.items():
        ov = overrides.get(_mid)
        opacity = ov.get("opacity", 1.0) if ov else 0.0
        silh = ov.get("silhouette", 0.0) if ov else 0.0
        visible = bool(ov) and ov.get("visible", True) and opacity > 0.001
        obj.hide_render = not visible
        obj.hide_viewport = not visible
        obj.visible_shadow = visible and cast_shadows
        obj.keyframe_insert("hide_render", frame=f)
        obj.keyframe_insert("hide_viewport", frame=f)
        try:
            obj.keyframe_insert("visible_shadow", frame=f)
        except Exception:  # noqa: BLE001
            pass
        nt = mat.node_tree
        av, sv = nt.nodes.get("cm_alpha"), nt.nodes.get("cm_silh")
        if av is not None:
            av.outputs[0].default_value = opacity
            av.outputs[0].keyframe_insert("default_value", frame=f)
        if sv is not None:
            sv.outputs[0].default_value = silh
            sv.outputs[0].keyframe_insert("default_value", frame=f)
        cv = nt.nodes.get("cm_color")
        if cv is not None and ov and ov.get("color") is not None:
            cv.outputs[0].default_value = _color_for_blender(ov["color"], color_space)
            cv.outputs[0].keyframe_insert("default_value", frame=f)
        ev = nt.nodes.get("cm_emit")
        if ev is not None:
            ev.outputs[0].default_value = base_emit + (ov.get("emphasis", 0.0) if ov else 0.0)
            ev.outputs[0].keyframe_insert("default_value", frame=f)
        mv, rv = nt.nodes.get("cm_metal"), nt.nodes.get("cm_rough")
        if mv is not None:
            bm, br = _mat_base.get(obj.name, (0.0, 0.35))
            m_ov = ov.get("metallic") if ov else None
            r_ov = ov.get("roughness") if ov else None
            mv.outputs[0].default_value = bm if m_ov is None else float(m_ov)
            mv.outputs[0].keyframe_insert("default_value", frame=f)
            if rv is not None:
                rv.outputs[0].default_value = br if r_ov is None else float(r_ov)
                rv.outputs[0].keyframe_insert("default_value", frame=f)
        _apply_clip(nt, ov.get("clip") if ov else None, f)


def export_blend(spec: dict) -> None:
    blend_path = spec["export_blend"]
    _clear()
    meshes = _import_meshes(spec)
    _setup_render(spec)
    _add_light(spec)
    scene = bpy.context.scene
    frames = spec["frames"]
    material_profile = spec.get("direction", {}).get("material", {})
    rig = spec.get("direction", {}).get("lighting", {})
    base_emit = material_profile.get("emission_strength", 0.15)
    cast_shadows = bool(material_profile.get("cast_shadows", True))
    color_space = material_profile.get("color_space", "linear")
    n = len(frames)
    scene.frame_start = 1
    scene.frame_end = max(1, n)
    scene.render.fps = int(spec.get("fps", 30))

    _build_export_slices(spec)  # per-frame slice quads + visibility keyframes

    for fi, frame in enumerate(frames):
        f = fi + 1
        scene.frame_set(f)
        _keyframe_camera(frame, f)
        _keyframe_lights(frame, rig, f)
        _set_ng_shader_light(meshes, frame, f)
        _keyframe_meshes(
            meshes,
            frame.get("mesh_overrides", {}),
            f,
            base_emit,
            cast_shadows,
            color_space,
        )
        _set_fade_overlay(frame, f)
        _set_world_bg(frame, f)
        print(f"[blender] frame {fi + 1}/{n}", flush=True)

    scene.frame_set(1)
    try:
        bpy.ops.file.pack_all()  # embed all textures + EM slice images into the .blend
    except Exception as e:  # noqa: BLE001
        print(f"[blender] pack_all warning: {e}", flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[blender] saved {blend_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[-1])
