"""Spike #2 (new) — EM zarr slice + mesh, composited entirely in Blender.

This is the NEW core mechanism after the architecture simplification: there is
ONE render engine (Blender), ONE camera, ONE coordinate system. EM context comes
from sampling a cross-section slice out of a zarr volume and texturing it onto a
plane positioned in 3D; the segmentation mesh intersects that plane. No
Neuroglancer in the render path -> no camera alignment problem.

Steps:
  1. Write a synthetic EM-like zarr volume (stand-in for /nrs EM; proves the
     zarr read path with the real `zarr` library).
  2. Read one z cross-section as a 2D array -> save as an image.
  3. Blender: textured plane (the EM slice, emission/unlit so it reads like EM)
     at the slice's world Z, plus an icosphere "mesh" intersecting it, one camera,
     GPU Cycles (OPTIX). Plane and mesh share one voxel->world transform.

Run:  conda run -n mv_env python spikes/spike2_zarr_slice_in_blender.py
Out:  spikes/out/em_slice.png        (the raw EM cross-section)
      spikes/out/spike2_composite.png (mesh intersecting the EM slice, 1 engine)
"""
import os
import shutil

import numpy as np
import zarr

OUT = os.path.join(os.path.dirname(__file__), "out")
ZARR_PATH = os.path.join(OUT, "synthetic_em.zarr")
N = 256                      # volume is N^3 voxels
SLICE_Z = N // 2            # cross-section index
VOXEL_NM = 8.0             # voxel size (nm) — metadata we'd read from .zattrs
NM_PER_BU = 256.0         # world scale: nm per Blender unit (1 BU = 256 nm here)
SIZE = (1000, 750)


# ----------------------------- data side -----------------------------
def make_synthetic_em() -> None:
    """EM-ish grayscale: noisy 'membranes' (sinusoidal shells) + speckle."""
    if os.path.exists(ZARR_PATH):
        shutil.rmtree(ZARR_PATH)
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
    c = N / 2
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)
    membranes = (np.sin(r / 6.0) ** 2) * 160
    texture = np.sin(xx / 3.0) * np.sin(yy / 3.0) * 20
    vol = np.clip(membranes + texture + rng.normal(40, 18, r.shape), 0, 255).astype(np.uint8)

    z = zarr.open(
        ZARR_PATH, mode="w", shape=(N, N, N), chunks=(64, 64, 64), dtype="uint8"
    )
    z[:] = vol
    z.attrs["voxel_size_nm"] = [VOXEL_NM, VOXEL_NM, VOXEL_NM]
    print(f"[spike2] wrote synthetic zarr {ZARR_PATH} shape={z.shape}")


def read_slice_png(path: str) -> tuple[int, int]:
    z = zarr.open(ZARR_PATH, mode="r")
    sl = np.asarray(z[SLICE_Z, :, :])  # one cheap 2D read (the whole point)
    from PIL import Image

    Image.fromarray(sl, mode="L").save(path)
    print(f"[spike2] wrote EM slice {path} ({sl.shape}, z={SLICE_Z})")
    return sl.shape  # (height=y, width=x)


# ----------------------------- blender side -----------------------------
def render_composite(slice_png: str, dims: tuple[int, int], out_path: str) -> None:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # GPU (OPTIX) — same as spike3
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type in ("OPTIX", "CPU")
        scene.cycles.device = "GPU"
        print("[spike2] GPU: OPTIX")
    except Exception as e:  # noqa: BLE001
        scene.cycles.device = "CPU"
        print(f"[spike2] CPU fallback: {e}")
    scene.cycles.samples = 64

    # voxel -> world transform (shared by plane AND mesh)
    extent_bu = (N * VOXEL_NM) / NM_PER_BU       # full volume size in Blender units
    z_world = (SLICE_Z * VOXEL_NM) / NM_PER_BU - extent_bu / 2  # centre the volume

    # --- EM slice plane ---
    bpy.ops.mesh.primitive_plane_add(size=extent_bu, location=(0, 0, z_world))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("em_slice")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(slice_png)
    tex.image.colorspace_settings.name = "Non-Color"
    emit = nt.nodes.new("ShaderNodeEmission")  # unlit, so EM reads clearly
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    plane.data.materials.append(mat)

    # --- segmentation mesh (stand-in) intersecting the slice ---
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=4, radius=extent_bu * 0.28, location=(0, 0, z_world)
    )
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    smat = bpy.data.materials.new("mito")
    smat.use_nodes = True
    bsdf = smat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.91, 0.45, 0.23, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    obj.data.materials.append(smat)

    # camera looking at the origin from an angle (sees mesh poking through slice)
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (extent_bu * 1.1, -extent_bu * 1.4, extent_bu * 1.0)
    # aim at origin
    from mathutils import Vector

    direction = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    # lighting
    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.energy = 3000
    light = bpy.data.objects.new("Key", light_data)
    light.location = (extent_bu * 2, -extent_bu * 2, extent_bu * 3)
    scene.collection.objects.link(light)

    scene.render.resolution_x, scene.render.resolution_y = SIZE
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out_path
    print("[spike2] rendering composite…")
    bpy.ops.render.render(write_still=True)
    print(f"[spike2] {'PASS' if os.path.getsize(out_path) > 0 else 'FAIL'} — {out_path}")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    make_synthetic_em()
    slice_png = os.path.join(OUT, "em_slice.png")
    dims = read_slice_png(slice_png)
    render_composite(slice_png, dims, os.path.join(OUT, "spike2_composite.png"))


if __name__ == "__main__":
    main()
