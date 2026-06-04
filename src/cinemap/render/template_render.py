"""Render a CineMap mesh INTO a BlenderKit-style template scene.

The template .blend is an animation/lighting rig with stand-in "product" geometry
(e.g. AirPods) framed by an animated camera. We open it, delete the product meshes
(keeping the studio backdrop, lights, world, and animated camera), drop the CineMap
mesh in where the product was — scaled to match — and render the camera animation.

Invoke:  python -m cinemap.render.template_render <spec.json>
spec keys: template, mesh_ply, out_dir, width, height, samples,
           frame_start, frame_end, fill (mesh size vs product, default 1.1)
"""
from __future__ import annotations

import json
import math
import statistics
import sys

import bpy
from mathutils import Vector


def _world_bbox(objs):
    deps = bpy.context.evaluated_depsgraph_get()
    mn = [math.inf] * 3; mx = [-math.inf] * 3; found = False
    for o in objs:
        if o.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            continue
        ob = o.evaluated_get(deps)
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i]); mx[i] = max(mx[i], w[i])
            found = True
    return (mn, mx) if found else None


def _setup_gpu(scene, samples, w, h):
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type in ("OPTIX", "CPU")
        scene.cycles.device = "GPU"
    except Exception as e:  # noqa: BLE001
        print(f"[template] GPU unavailable, CPU: {e}")
    scene.cycles.samples = samples
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.image_settings.file_format = "PNG"


def main(spec_path: str) -> None:
    with open(spec_path) as f:
        spec = json.load(f)
    bpy.ops.wm.open_mainfile(filepath=spec["template"])   # camera + lights + backdrop + anim
    scene = bpy.context.scene
    fs = spec.get("frame_start", 1)
    fe = spec.get("frame_end", scene.frame_end)
    scene.frame_set(fs)

    # classify meshes: the big flat plane(s) are the studio backdrop (KEEP); every
    # other mesh is stand-in product geometry (the earbuds) -> DELETE.
    deps = bpy.context.evaluated_depsgraph_get()
    info = []
    for o in [o for o in scene.objects if o.type == "MESH"]:
        ob = o.evaluated_get(deps)
        mn = [math.inf] * 3; mx = [-math.inf] * 3
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], w[i]); mx[i] = max(mx[i], w[i])
        ext = [mx[i] - mn[i] for i in range(3)]
        info.append((o, max(ext), min(ext), mn, mx))
    med = statistics.median(sorted(m for _, m, _, _, _ in info)) if info else 1.0
    backdrops = {o for o, mxe, mne, _, _ in info if mxe > 4 * med and mne < 0.02 * mxe}
    product = [t for t in info if t[0] not in backdrops]
    if not product:
        raise RuntimeError("no product meshes found to replace")

    pmn = [min(t[3][i] for t in product) for i in range(3)]
    pmx = [max(t[4][i] for t in product) for i in range(3)]
    pcenter = [(pmn[i] + pmx[i]) / 2 for i in range(3)]
    psize = max(pmx[i] - pmn[i] for i in range(3)) or 1.0
    print(f"[template] product bbox center={[round(c,2) for c in pcenter]} size={psize:.3g}; "
          f"deleting {len(product)} product mesh(es), keeping {len(backdrops)} backdrop(s)", flush=True)
    for t in product:
        bpy.data.objects.remove(t[0], do_unlink=True)

    # inject the CineMap mesh
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=spec["mesh_ply"])
    nuc = [o for o in bpy.data.objects if o not in before][0]
    bpy.context.view_layer.objects.active = nuc
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("nuc_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.45
    if getattr(nuc.data, "color_attributes", None) and len(nuc.data.color_attributes):
        vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = nuc.data.color_attributes[0].name
        mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    nuc.data.materials.clear()
    nuc.data.materials.append(mat)

    # scale to the product's footprint, center where the product was
    nb = _world_bbox([nuc])
    nsz = max(nb[1][i] - nb[0][i] for i in range(3)) or 1.0
    s = (psize / nsz) * spec.get("fill", 1.1)
    nuc.scale = (s, s, s)
    bpy.context.view_layer.update()
    nb = _world_bbox([nuc])
    ncenter = [(nb[0][i] + nb[1][i]) / 2 for i in range(3)]
    nuc.location = Vector([pcenter[i] - ncenter[i] for i in range(3)])

    _setup_gpu(scene, spec.get("samples", 96), spec["width"], spec["height"])
    out = spec["out_dir"]
    for f in range(fs, fe + 1):
        scene.frame_set(f)
        scene.render.filepath = f"{out}/frame_{f:05d}.png"
        print(f"[template] frame {f}/{fe}", flush=True)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main(sys.argv[-1])
