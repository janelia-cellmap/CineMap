"""Spike #3 — Blender headless mesh render on the GPU (Cycles).

Goal: prove bpy can render a mesh headlessly with GPU Cycles on the RTX 5090,
producing a PNG with alpha (transparent background) so it can be composited over
a Neuroglancer frame later.

This stands in a generated icosphere for a real precomputed mesh; the
cloud-volume -> trimesh -> bpy import path is exercised separately once a real
mesh URL is wired in (see load_trimesh_into_blender, unused here but kept as the
template for the real pipeline).

Run:  conda run -n mv_env python spikes/spike3_blender_mesh.py
Out:  spikes/out/blender_frame.png  (RGBA, transparent bg)
"""
import os

import bpy

OUT = os.path.join(os.path.dirname(__file__), "out")
SIZE = (1280, 720)


def enable_gpu() -> str:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen = None
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type in (backend, "CPU")
                chosen = backend
                print(f"[spike3] GPU backend: {backend} -> {[d.name for d in gpus]}")
                break
        except Exception as e:  # noqa: BLE001
            print(f"[spike3] backend {backend} unavailable: {e}")
    if not chosen:
        print("[spike3] no GPU backend; falling back to CPU")
    return chosen or "CPU"


def build_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    backend = enable_gpu()
    scene.cycles.device = "GPU" if backend != "CPU" else "CPU"
    scene.cycles.samples = 64

    # subject mesh (stand-in for a precomputed segmentation mesh)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("mito")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.91, 0.45, 0.23, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.4
    obj.data.materials.append(mat)

    # camera (this is what mvideo animates along the keyframe path)
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0.0, -4.0, 1.5)
    cam.rotation_euler = (1.2, 0.0, 0.0)
    scene.camera = cam

    # light
    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.energy = 800
    light = bpy.data.objects.new("Key", light_data)
    light.location = (3, -3, 5)
    scene.collection.objects.link(light)

    # transparent film so the mesh composites over a Neuroglancer frame
    scene.render.film_transparent = True
    scene.render.resolution_x, scene.render.resolution_y = SIZE
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    build_scene()
    out_path = os.path.join(OUT, "blender_frame.png")
    bpy.context.scene.render.filepath = out_path
    print("[spike3] rendering…")
    bpy.ops.render.render(write_still=True)
    ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    print(f"[spike3] {'PASS' if ok else 'FAIL'} — wrote {out_path}")


if __name__ == "__main__":
    main()
