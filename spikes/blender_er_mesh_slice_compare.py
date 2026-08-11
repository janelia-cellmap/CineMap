from __future__ import annotations

import json
import time
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector


ER_MESH = Path(
    "/nrs/cellmap/ackermand/cinemap_projects/"
    "liverzonmovie-5-a100-50m-surface/assets/mesh_er_c551581c.npz"
)
OUT_DIR = Path(
    "/nrs/cellmap/ackermand/cinemap_projects/"
    "liverzonmovie-5-a100-50m-surface/assets/er_slice_compare"
)


def _plane_for_bounds(lo: np.ndarray, hi: np.ndarray):
    center = (lo + hi) / 2.0
    normal = np.array([1.0, 0.65, 0.35], dtype=float)
    normal /= np.linalg.norm(normal)
    return center, normal


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(ER_MESH, allow_pickle=False) as z:
        verts = np.asarray(z["v"], dtype=np.float32)
        faces = np.asarray(z["f"], dtype=np.int32)
    load_s = time.perf_counter() - t0

    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    plane_point, normal = _plane_for_bounds(lo, hi)

    t_mesh = time.perf_counter()
    mesh = bpy.data.meshes.new("er_full")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("er_full", mesh)
    bpy.context.collection.objects.link(obj)
    build_s = time.perf_counter() - t_mesh

    t_slice = time.perf_counter()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    res = bmesh.ops.bisect_plane(
        bm,
        geom=geom,
        dist=1e-4,
        plane_co=Vector(tuple(float(v) for v in plane_point)),
        plane_no=Vector(tuple(float(-v) for v in normal)),
        clear_outer=True,
    )
    cut = [e for e in res.get("geom_cut", []) if isinstance(e, bmesh.types.BMEdge)]
    before_faces = len(bm.faces)
    if cut:
        before = set(bm.faces)
        bmesh.ops.holes_fill(bm, edges=cut, sides=0)
        newf = [f for f in bm.faces if f not in before]
        if newf:
            bmesh.ops.triangulate(bm, faces=newf)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    sliced = bpy.data.meshes.new("er_mesh_first_oblique_cap")
    bm.to_mesh(sliced)
    bm.free()
    slice_s = time.perf_counter() - t_slice

    out_obj = bpy.data.objects.new("er_mesh_first_oblique_cap", sliced)
    bpy.context.collection.objects.link(out_obj)
    bpy.context.view_layer.objects.active = out_obj
    out_obj.select_set(True)
    out_ply = OUT_DIR / "er_955_mesh_first_oblique_bmesh_cap.ply"
    t_export = time.perf_counter()
    bpy.ops.wm.ply_export(filepath=str(out_ply), export_selected_objects=True)
    export_s = time.perf_counter() - t_export

    result = {
        "path": str(out_ply),
        "method": "mesh_first_bmesh",
        "cap": True,
        "vertices": int(len(sliced.vertices)),
        "faces": int(len(sliced.polygons)),
        "faces_before_cap": int(before_faces),
        "cut_edges": int(len(cut)),
        "plane_point_nm": [float(v) for v in plane_point],
        "plane_normal_xyz": [float(v) for v in normal],
        "load_seconds": load_s,
        "mesh_build_seconds": build_s,
        "slice_seconds": slice_s,
        "export_seconds": export_s,
        "total_seconds": time.perf_counter() - t0,
    }
    summary = OUT_DIR / "mesh_oblique_bmesh_summary_cap.json"
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
