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
import os
import sys

import bpy
from mathutils import Matrix, Vector


def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _setup_render(scene_spec: dict) -> None:
    scene = bpy.context.scene
    r = scene_spec["render"]
    scene.render.engine = r.get("engine", "CYCLES")
    if scene.render.engine == "CYCLES":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type in ("OPTIX", "CPU")
            scene.cycles.device = "GPU"
        except Exception as e:  # noqa: BLE001
            print(f"[blender] GPU unavailable, CPU: {e}")
        scene.cycles.samples = r.get("samples", 64)
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
        except Exception as e:  # noqa: BLE001
            print(f"[blender] denoise/adaptive unavailable: {e}")
    scene.render.resolution_x = r["width"]
    scene.render.resolution_y = r["height"]
    scene.render.image_settings.file_format = "PNG"
    # View transform from the director (default AgX + "Punchy"). AgX rolls bright values
    # off smoothly instead of clipping them to neon -> no oversaturation/blowout, and it
    # gives soft, dimensional filmic shadows; the "Punchy" look restores color richness so
    # it isn't washed. (Standard/sRGB matches NG's flat look but clips -> oversaturation.)
    view = scene_spec.get("direction", {}).get("view", {})
    try:
        scene.view_settings.view_transform = view.get("transform", "AgX")
    except Exception as e:  # noqa: BLE001
        print(f"[blender] view transform: {e}")
    look = view.get("look", "AgX - Punchy")
    if look:
        try:
            scene.view_settings.look = look
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
    bg_amb = nt.nodes.new("ShaderNodeBackground")
    bg_amb.inputs[0].default_value = (amb * ac[0], amb * ac[1], amb * ac[2], 1.0)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])  # 0 -> ambient, 1 -> bg
    nt.links.new(bg_amb.outputs[0], mix.inputs[1])
    nt.links.new(bg_cam.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    scene.world = world

    _setup_freestyle(scene_spec)


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


def _add_light(scene_spec: dict) -> None:
    """Three-point key/fill/rim rig. The director (auto-direct) supplies energies and
    asks for a camera-relative rig — _update_lights then re-aims these per frame so
    the rim/key stay consistent as the camera orbits. Without it, the fixed world
    rotations below are the faithful fallback."""
    rig = scene_spec.get("direction", {}).get("lighting", {})
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


def _import_meshes(scene_spec: dict) -> dict:
    """Import each mesh once; return name -> (object, material).

    PLY assets may carry per-vertex colors (distinct color per segment); if so the
    material drives Base Color from the color attribute. Otherwise a solid color.
    """
    out = {}
    for m in scene_spec["meshes"]:
        path = m["obj_path"]
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
        bpy.context.view_layer.objects.active = obj
        # flat (per-face) shading by default — each face shades dark/light on its own,
        # giving the crisp faceted definition neuroglancer has; smooth blurs it to blobs.
        if scene_spec.get("direction", {}).get("material", {}).get("flat_shading", True):
            bpy.ops.object.shade_flat()
        else:
            bpy.ops.object.shade_smooth()
        s = 1.0 / scene_spec["world"]["nm_per_bu"]  # nm -> BU
        obj.scale = (s, s, s)

        mat = bpy.data.materials.new(f"mat_{m['id']}")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        col = m["color"]
        # publication-quality shading over the NG color: tuned roughness/specular +
        # a touch of sheen. From the director when auto-direct is on; sensible
        # defaults otherwise. Input names vary by Blender version, so set defensively.
        prof = scene_spec.get("direction", {}).get("material", {})
        def _set_in(node, key, val):
            if key in node.inputs:
                node.inputs[key].default_value = val
        _set_in(bsdf, "Roughness", prof.get("roughness", 0.35))
        _set_in(bsdf, "Specular IOR Level", prof.get("specular", 0.5))
        _set_in(bsdf, "Metallic", prof.get("metallic", 0.0))
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
            csrc.outputs[0].default_value = (col[0], col[1], col[2], 1.0)
            color_out = csrc.outputs[0]
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
        if prof.get("ng_shader"):
            # Faithful port of neuroglancer's mesh GLSL: per fragment,
            #   lightingFactor = abs(dot(normal, viewDir)) * 0.8 + 0.2
            #   color = lightingFactor * baseColor
            # A HEADLIGHT (viewDir) from geometry, emission-only (no external lights), so
            # it's exactly NG: vivid (factor<=1 -> never clips/oversaturates), camera-relative
            # shading, no cast shadows. Base Color black so lights/world don't add. NOTE: no
            # cm_emit node here -> _set_mesh_state can't reset Emission Strength (it stays 1).
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            dotp = nt.nodes.new("ShaderNodeVectorMath"); dotp.operation = "DOT_PRODUCT"
            nt.links.new(geo.outputs["Normal"], dotp.inputs[0])
            nt.links.new(geo.outputs["Incoming"], dotp.inputs[1])   # viewDir (toward camera)
            absd = nt.nodes.new("ShaderNodeMath"); absd.operation = "ABSOLUTE"
            nt.links.new(dotp.outputs["Value"], absd.inputs[0])
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

        # neuroglancer 3D render state: Alpha = object_alpha * facing^silhouette, where
        # `facing` is Blender's LayerWeight Facing output = 0 head-on, 1 at grazing
        # (== neuroglancer's 1 - |normal·view|). So with silhouette>0 the head-on faces
        # go transparent and only the rim stays opaque (NG's meshSilhouetteRendering, a
        # glassy shell); silhouette=0 -> facing^0 = 1 -> plain object_alpha everywhere.
        # Driven per frame by the cm_alpha / cm_silh value nodes.
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        powr = nt.nodes.new("ShaderNodeMath"); powr.operation = "POWER"
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.use_clamp = True
        alpha_v = nt.nodes.new("ShaderNodeValue"); alpha_v.name = "cm_alpha"; alpha_v.outputs[0].default_value = 1.0
        silh_v = nt.nodes.new("ShaderNodeValue"); silh_v.name = "cm_silh"; silh_v.outputs[0].default_value = 0.0
        nt.links.new(lw.outputs["Facing"], powr.inputs[0])   # base = facing (0 head-on, 1 grazing)
        nt.links.new(silh_v.outputs[0], powr.inputs[1])      # exponent = silhouette power
        nt.links.new(alpha_v.outputs[0], mul.inputs[0])
        nt.links.new(powr.outputs[0], mul.inputs[1])
        if "Alpha" in bsdf.inputs:
            nt.links.new(mul.outputs[0], bsdf.inputs["Alpha"])

        # Fresnel edge-glow: grazing edges emit their own color (a soft rim glow that
        # makes structures read as 'lit' against the dark background, esp. with bloom).
        # Emission Strength = cm_emit (base/pulse) + Facing * edge_glow.
        edge = prof.get("edge_glow", 0.0)
        if edge > 0 and "Emission Strength" in bsdf.inputs:
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
        if m.get("clip") and out_node is not None:
            surf = _build_clip_nodes(nt, surf)
        if out_node is not None:
            nt.links.new(surf, out_node.inputs["Surface"])

        mat.blend_method = "BLEND"
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        out[m["id"]] = (obj, mat)
    return out


def _set_mesh_state(meshes: dict, overrides: dict, base_emit: float = 0.15) -> None:
    for mid, (obj, mat) in meshes.items():
        ov = overrides.get(mid)
        if ov is None:  # not referenced this frame -> hidden (belongs to another keyframe)
            obj.hide_render = True
            continue
        opacity = ov.get("opacity", 1.0)            # effective alpha = fade * Opacity(3d)
        visible = ov.get("visible", True) and opacity > 0.001
        obj.hide_render = not visible
        # Always cast a shadow while visible. Cycles attenuates the shadow by the object's
        # alpha (transparent shadows), so as a layer fades in/out its shadow ramps smoothly
        # with opacity. A hard on/off threshold here instead caused a one-frame brightness
        # POP whenever a layer faded through it (e.g. the segmentation reveal at ~2s):
        # below the cutoff no shadow, above it the whole tangle self-shadowed at once.
        obj.visible_shadow = visible and not os.environ.get("CINEMAP_NO_CAST_SHADOWS")
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
        if av is None and "Alpha" in nt.nodes["Principled BSDF"].inputs:
            nt.nodes["Principled BSDF"].inputs["Alpha"].default_value = opacity
        _apply_clip(nt, ov.get("clip"))


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
    emit.inputs["Strength"].default_value = 0.9  # slightly tame the bright EM plane
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    mix.inputs[0].default_value = sl.get("opacity", 1.0)
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
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


def main(scene_path: str) -> None:
    with open(scene_path) as f:
        spec = json.load(f)
    if spec.get("export_blend"):
        export_blend(spec)
        return
    _clear()
    _setup_render(spec)
    _add_light(spec)
    _setup_bloom(spec)
    meshes = _import_meshes(spec)
    scene = bpy.context.scene
    out_dir = spec["output_dir"]
    rig = spec.get("direction", {}).get("lighting", {})
    base_emit = spec.get("direction", {}).get("material", {}).get("emission_strength", 0.15)
    for fi, frame in enumerate(spec["frames"]):
        _set_camera(frame)
        if rig:
            _update_lights(frame, rig)
        _build_slices(frame)
        _set_mesh_state(meshes, frame.get("mesh_overrides", {}), base_emit)
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


def _keyframe_meshes(meshes: dict, overrides: dict, f: int) -> None:
    for _mid, (obj, mat) in meshes.items():
        ov = overrides.get(_mid)
        opacity = ov.get("opacity", 1.0) if ov else 0.0
        silh = ov.get("silhouette", 0.0) if ov else 0.0
        visible = bool(ov) and ov.get("visible", True) and opacity > 0.001
        obj.hide_render = not visible
        obj.hide_viewport = not visible
        obj.keyframe_insert("hide_render", frame=f)
        obj.keyframe_insert("hide_viewport", frame=f)
        nt = mat.node_tree
        av, sv = nt.nodes.get("cm_alpha"), nt.nodes.get("cm_silh")
        if av is not None:
            av.outputs[0].default_value = opacity
            av.outputs[0].keyframe_insert("default_value", frame=f)
        if sv is not None:
            sv.outputs[0].default_value = silh
            sv.outputs[0].keyframe_insert("default_value", frame=f)
        _apply_clip(nt, ov.get("clip") if ov else None, f)


def export_blend(spec: dict) -> None:
    blend_path = spec["export_blend"]
    _clear()
    _setup_render(spec)
    _add_light(spec)
    meshes = _import_meshes(spec)
    scene = bpy.context.scene
    frames = spec["frames"]
    n = len(frames)
    scene.frame_start = 1
    scene.frame_end = max(1, n)
    scene.render.fps = int(spec.get("fps", 30))

    _build_export_slices(spec)  # per-frame slice quads + visibility keyframes

    for fi, frame in enumerate(frames):
        f = fi + 1
        scene.frame_set(f)
        _keyframe_camera(frame, f)
        _keyframe_meshes(meshes, frame.get("mesh_overrides", {}), f)
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
