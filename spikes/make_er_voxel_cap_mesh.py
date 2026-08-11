from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from cinemap.data.mesh_from_labels import _bbox_for_ids, _choose_plan, _read_plan_array, seg_color
from cinemap.data.slice_loader import get_volume
from spikes.compare_er_slice_mesh_vs_voxel import LABEL_URL, SEG_ID, _plane_for_bbox


OUT_DIR = Path(
    "/nrs/cellmap/ackermand/cinemap_projects/"
    "liverzonmovie-5-a100-50m-surface/assets/er_slice_compare/cap_only"
)


def _bbox_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=np.float64,
    )


def _basis(normal_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal_xyz, dtype=np.float64)
    n /= np.linalg.norm(n) or 1.0
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(n, u)
    return u, v, n


def make_cap(out_dir: Path, target_vertices: int, plane_mode: str, sample_nm: float | None) -> dict:
    t0 = time.perf_counter()
    vol = get_volume(LABEL_URL)
    bbox = _bbox_for_ids(LABEL_URL, [SEG_ID])
    if bbox is None:
        raise RuntimeError("could not find ER bbox")

    lo = np.asarray(bbox[0], dtype=np.float64)
    hi = np.asarray(bbox[1], dtype=np.float64)
    plane_name, plane_point, normal = _plane_for_bbox(lo, hi, plane_mode)
    if plane_mode != "oblique":
        raise ValueError("cap-only spike currently expects --plane oblique")

    plan = _choose_plan(
        vol,
        bbox,
        target_voxels=1,
        target_vertices=target_vertices,
        seg_ids=[SEG_ID],
    )
    arr, (z0, y0, x0), sc, tr = _read_plan_array(vol, plan, pad=3)
    t_read = time.perf_counter()

    u, v, n = _basis(normal)
    corners = _bbox_corners(lo, hi)
    rel = corners - plane_point
    uvals = rel @ u
    vvals = rel @ v
    step = float(sample_nm or min(sc))
    u0 = np.floor(uvals.min() / step - 1.0) * step
    u1 = np.ceil(uvals.max() / step + 1.0) * step
    v0 = np.floor(vvals.min() / step - 1.0) * step
    v1 = np.ceil(vvals.max() / step + 1.0) * step

    us = np.arange(u0 + 0.5 * step, u1, step, dtype=np.float64)
    vs = np.arange(v0 + 0.5 * step, v1, step, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    world = plane_point[None, None, :] + uu[..., None] * u + vv[..., None] * v

    fx = (world[..., 0] - tr[2]) / sc[2] - x0
    fy = (world[..., 1] - tr[1]) / sc[1] - y0
    fz = (world[..., 2] - tr[0]) / sc[0] - z0
    oob = (
        (fz < 0)
        | (fz > arr.shape[0] - 1)
        | (fy < 0)
        | (fy > arr.shape[1] - 1)
        | (fx < 0)
        | (fx > arr.shape[2] - 1)
    )
    iz = np.clip(np.round(fz).astype(np.int64), 0, arr.shape[0] - 1)
    iy = np.clip(np.round(fy).astype(np.int64), 0, arr.shape[1] - 1)
    ix = np.clip(np.round(fx).astype(np.int64), 0, arr.shape[2] - 1)
    mask = arr[iz, iy, ix] == SEG_ID
    mask[oob] = False
    t_sample = time.perf_counter()

    cell_y, cell_x = np.nonzero(mask)
    used = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=bool)
    used[cell_y, cell_x] = True
    used[cell_y, cell_x + 1] = True
    used[cell_y + 1, cell_x + 1] = True
    used[cell_y + 1, cell_x] = True

    vertex_ids = np.full(used.shape, -1, dtype=np.int64)
    used_y, used_x = np.nonzero(used)
    vertex_ids[used_y, used_x] = np.arange(len(used_y), dtype=np.int64)

    corner_us = u0 + used_x.astype(np.float64) * step
    corner_vs = v0 + used_y.astype(np.float64) * step
    vertices = (
        plane_point[None, :]
        + corner_us[:, None] * u[None, :]
        + corner_vs[:, None] * v[None, :]
    ).astype(np.float32)

    q0 = vertex_ids[cell_y, cell_x]
    q1 = vertex_ids[cell_y, cell_x + 1]
    q2 = vertex_ids[cell_y + 1, cell_x + 1]
    q3 = vertex_ids[cell_y + 1, cell_x]
    faces = np.empty((len(cell_y) * 2, 3), dtype=np.int64)
    faces[0::2] = np.stack([q0, q1, q2], axis=1)
    faces[1::2] = np.stack([q0, q2, q3], axis=1)

    color = np.array((*seg_color(SEG_ID, "er"), 1.0), dtype=np.float32)
    vertex_colors = np.tile((color * 255).astype(np.uint8), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / f"er_955_cap_only_{plane_name}_{int(round(step))}nm.ply"
    mesh.export(ply)
    t_export = time.perf_counter()

    result = {
        "path": str(ply),
        "method": "cap_only_voxel_plane_quads",
        "segment_id": SEG_ID,
        "plane_mode": plane_mode,
        "plane_point_nm": [float(x) for x in plane_point],
        "plane_normal_xyz": [float(x) for x in n],
        "level": int(plan.level),
        "scale_zyx_nm": [float(x) for x in sc],
        "sample_nm": step,
        "read_shape_zyx": [int(x) for x in arr.shape],
        "plane_grid_yx": [int(x) for x in mask.shape],
        "occupied_plane_cells": int(mask.sum()),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "read_seconds": float(t_read - t0),
        "sample_seconds": float(t_sample - t_read),
        "mesh_export_seconds": float(t_export - t_sample),
        "total_seconds": float(t_export - t0),
    }
    summary = out_dir / f"er_955_cap_only_{plane_name}_{int(round(step))}nm_summary.json"
    summary.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-vertices", type=int, default=50_000_000)
    ap.add_argument("--plane", choices=["oblique"], default="oblique")
    ap.add_argument("--sample-nm", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    print(json.dumps(make_cap(args.out_dir, args.target_vertices, args.plane, args.sample_nm), indent=2))


if __name__ == "__main__":
    main()
