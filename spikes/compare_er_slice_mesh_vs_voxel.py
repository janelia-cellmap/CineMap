from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import zmesh

from cinemap.data.mesh_from_labels import (
    _bbox_for_ids,
    _choose_plan,
    _cleanup_mesh_lossless,
    _mesh_to_trimesh,
    _read_plan_array,
    seg_color,
)
from cinemap.data.slice_loader import get_volume


LABEL_URL = (
    "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-liver-zon-1/"
    "jrc_mus-liver-zon-1.zarr/recon-1/labels/inference/segmentations/er/"
)
SEG_ID = 955
ER_MESH = Path(
    "/nrs/cellmap/ackermand/cinemap_projects/"
    "liverzonmovie-5-a100-50m-surface/assets/mesh_er_c551581c.npz"
)
OUT_DIR = Path(
    "/nrs/cellmap/ackermand/cinemap_projects/"
    "liverzonmovie-5-a100-50m-surface/assets/er_slice_compare"
)


def _export(mesh: trimesh.Trimesh, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return {
        "path": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": np.asarray(mesh.bounds).tolist(),
    }


def _mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        v = z["v"]
        return v.min(axis=0), v.max(axis=0)


def _plane_for_bbox(lo: np.ndarray, hi: np.ndarray, mode: str):
    center = (lo + hi) / 2.0
    if mode == "axis":
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
        name = "xmid"
    elif mode == "oblique":
        normal = np.array([1.0, 0.65, 0.35], dtype=float)
        normal /= np.linalg.norm(normal)
        name = "oblique"
    else:
        raise ValueError(mode)
    return name, center, normal


def voxel_first(out_dir: Path, target_vertices: int, plane_mode: str) -> dict:
    t0 = time.perf_counter()
    vol = get_volume(LABEL_URL)
    bbox = _bbox_for_ids(LABEL_URL, [SEG_ID])
    if bbox is None:
        raise RuntimeError("could not find ER bbox")
    lo = np.asarray(bbox[0], dtype=float)
    hi = np.asarray(bbox[1], dtype=float)
    plane_name, plane_point, normal = _plane_for_bbox(lo, hi, plane_mode)
    if plane_mode == "axis":
        clipped_bbox = (tuple(lo), (float(plane_point[0]), float(hi[1]), float(hi[2])))
    else:
        clipped_bbox = bbox

    plan = _choose_plan(
        vol,
        clipped_bbox,
        target_voxels=1,
        target_vertices=target_vertices,
        seg_ids=[SEG_ID],
    )
    arr, (z0, y0, x0), sc, tr = _read_plan_array(vol, plan, pad=3)
    keep = arr == SEG_ID
    if plane_mode == "oblique":
        zz, yy, xx = np.indices(arr.shape, sparse=True)
        wx = (xx + x0) * sc[2] + tr[2]
        wy = (yy + y0) * sc[1] + tr[1]
        wz = (zz + z0) * sc[0] + tr[0]
        signed = (
            (wx - plane_point[0]) * normal[0]
            + (wy - plane_point[1]) * normal[1]
            + (wz - plane_point[2]) * normal[2]
        )
        keep &= signed <= 0
    labels = np.where(keep, arr, 0).astype(np.uint32, copy=False)
    labels = np.pad(labels, 1)

    t_mesh = time.perf_counter()
    mesher = zmesh.Mesher(tuple(float(s) for s in sc))
    mesher.mesh(labels)
    raw = mesher.get(SEG_ID, reduction_factor=0, max_error=None, voxel_centered=False)
    origin = ((z0 - 1) * sc[0] + tr[0], (y0 - 1) * sc[1] + tr[1], (x0 - 1) * sc[2] + tr[2])
    mesh = _mesh_to_trimesh(raw, SEG_ID, origin, colorize=lambda _sid: seg_color(SEG_ID, "er"))
    if mesh is None:
        raise RuntimeError("zmesh returned no ER mesh")
    mesh = _cleanup_mesh_lossless(mesh)
    out = _export(mesh, out_dir / f"er_955_voxel_first_{plane_name}.ply")
    out.update(
        {
            "method": "voxel_first",
            "clip": "negative plane half-space",
            "plane_mode": plane_mode,
            "plane_point_nm": [float(v) for v in plane_point],
            "plane_normal_xyz": [float(v) for v in normal],
            "level": int(plan.level),
            "scale_zyx_nm": [float(s) for s in sc],
            "read_shape_zyx": list(map(int, arr.shape)),
            "selected_voxels": int(keep.sum()),
            "read_seconds": float(t_mesh - t0),
            "mesh_seconds": float(time.perf_counter() - t_mesh),
            "total_seconds": float(time.perf_counter() - t0),
        }
    )
    return out


def mesh_first(out_dir: Path, cap: bool, plane_mode: str) -> dict:
    t0 = time.perf_counter()
    with np.load(ER_MESH, allow_pickle=False) as z:
        verts = np.asarray(z["v"], dtype=np.float32)
        faces = np.asarray(z["f"], dtype=np.int64)
        colors = np.asarray(z["c"]) if "c" in z.files else None
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    plane_name, plane_point, normal = _plane_for_bbox(lo, hi, plane_mode)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)
    t_slice = time.perf_counter()
    sliced = mesh.slice_plane(
        plane_origin=plane_point,
        plane_normal=-normal,
        cap=cap,
    )
    if sliced is None:
        raise RuntimeError("mesh slice returned no geometry")
    out = _export(sliced, out_dir / f"er_955_mesh_first_{plane_name}_cap{int(cap)}.ply")
    out.update(
        {
            "method": "mesh_first",
            "cap": bool(cap),
            "clip": "negative plane half-space",
            "plane_mode": plane_mode,
            "plane_point_nm": [float(v) for v in plane_point],
            "plane_normal_xyz": [float(v) for v in normal],
            "load_seconds": float(t_slice - t0),
            "slice_seconds": float(time.perf_counter() - t_slice),
            "total_seconds": float(time.perf_counter() - t0),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bounds", "voxel", "mesh"], required=True)
    ap.add_argument("--target-vertices", type=int, default=50_000_000)
    ap.add_argument("--plane", choices=["axis", "oblique"], default="axis")
    ap.add_argument("--cap", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.method == "bounds":
        lo, hi = _mesh_bounds(ER_MESH)
        name, point, normal = _plane_for_bbox(lo, hi, args.plane)
        result = {
            "mesh": str(ER_MESH),
            "bounds": [lo.tolist(), hi.tolist()],
            "plane_mode": args.plane,
            "plane_name": name,
            "plane_point_nm": point.tolist(),
            "plane_normal_xyz": normal.tolist(),
        }
    elif args.method == "voxel":
        result = voxel_first(args.out_dir, args.target_vertices, args.plane)
    else:
        result = mesh_first(args.out_dir, cap=args.cap, plane_mode=args.plane)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = args.out_dir / f"{args.method}_{args.plane}_summary{'_cap' if args.cap else ''}.json"
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
