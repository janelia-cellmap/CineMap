"""Fast mesh cutaway assets.

For label-derived meshes, cutting and capping inside Blender is expensive because
it rebuilds bmesh geometry per animated frame. This module builds ordinary mesh
assets ahead of Blender: an exactly clipped exterior mesh plus a cap sampled from
the source label volume on the same plane.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import trimesh

from .mesh_from_labels import _choose_plan, _read_plan_array
from .slice_loader import get_volume


def _bbox_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )


def _plane_basis(normal_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal_xyz, dtype=np.float64)
    n /= np.linalg.norm(n) or 1.0
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(n, u)
    return u, v, n


def _clip_arrays(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    normal_xyz: np.ndarray,
    position_nm: float,
    side: float,
) -> dict:
    n = np.asarray(normal_xyz, dtype=np.float64)
    n /= np.linalg.norm(n) or 1.0
    signed = ((verts.astype(np.float64) @ n) - float(position_nm)) * (1.0 if side >= 0 else -1.0)
    signed = signed.astype(np.float32, copy=False)

    out_faces: list[np.ndarray] = []
    out_verts: list[np.ndarray] = []
    out_color_idxs: list[np.ndarray] = []
    next_index = len(verts)
    chunk = 2_000_000

    def add_one_inside(tris: np.ndarray, slot: int) -> None:
        nonlocal next_index
        if len(tris) == 0:
            return
        i0 = tris[:, slot]
        i1 = tris[:, (slot + 1) % 3]
        i2 = tris[:, (slot + 2) % 3]
        s0 = signed[i0]
        s1 = signed[i1]
        s2 = signed[i2]
        t01 = s0 / (s0 - s1 + 1e-20)
        t02 = s0 / (s0 - s2 + 1e-20)
        p0 = verts[i0]
        q1 = p0 + (verts[i1] - p0) * t01[:, None]
        q2 = p0 + (verts[i2] - p0) * t02[:, None]
        ids = np.arange(next_index, next_index + 2 * len(tris), dtype=np.int64).reshape(len(tris), 2)
        extra = np.empty((2 * len(tris), 3), dtype=np.float32)
        extra[0::2] = q1.astype(np.float32, copy=False)
        extra[1::2] = q2.astype(np.float32, copy=False)
        out_verts.append(extra)
        out_faces.append(np.stack([i0.astype(np.int64), ids[:, 0], ids[:, 1]], axis=1))
        out_color_idxs.append(np.repeat(i0.astype(np.int64), 2))
        next_index += 2 * len(tris)

    def add_two_inside(tris: np.ndarray, outside_slot: int) -> None:
        nonlocal next_index
        if len(tris) == 0:
            return
        i_out = tris[:, outside_slot]
        i0 = tris[:, (outside_slot + 1) % 3]
        i1 = tris[:, (outside_slot + 2) % 3]
        so = signed[i_out]
        s0 = signed[i0]
        s1 = signed[i1]
        t0 = s0 / (s0 - so + 1e-20)
        t1 = s1 / (s1 - so + 1e-20)
        q0 = verts[i0] + (verts[i_out] - verts[i0]) * t0[:, None]
        q1 = verts[i1] + (verts[i_out] - verts[i1]) * t1[:, None]
        ids = np.arange(next_index, next_index + 2 * len(tris), dtype=np.int64).reshape(len(tris), 2)
        extra = np.empty((2 * len(tris), 3), dtype=np.float32)
        extra[0::2] = q0.astype(np.float32, copy=False)
        extra[1::2] = q1.astype(np.float32, copy=False)
        out_verts.append(extra)
        out_faces.append(np.stack([i0.astype(np.int64), i1.astype(np.int64), ids[:, 1]], axis=1))
        out_faces.append(np.stack([i0.astype(np.int64), ids[:, 1], ids[:, 0]], axis=1))
        out_color_idxs.append(np.repeat(i0.astype(np.int64), 2))
        next_index += 2 * len(tris)

    for start in range(0, len(faces), chunk):
        f = faces[start:start + chunk].astype(np.int64, copy=False)
        fs = signed[f]
        inside = fs <= 0.0
        ninside = inside.sum(axis=1)
        if np.any(ninside == 3):
            out_faces.append(f[ninside == 3])
        crossing_mask = (ninside > 0) & (ninside < 3)
        crossing = f[crossing_mask]
        if len(crossing) == 0:
            continue
        ci = inside[crossing_mask]
        for slot in range(3):
            add_one_inside(crossing[(ci.sum(axis=1) == 1) & ci[:, slot]], slot)
            add_two_inside(crossing[(ci.sum(axis=1) == 2) & ~ci[:, slot]], slot)

    if not out_faces:
        out = {
            "v": np.empty((0, 3), dtype=np.float32),
            "f": np.empty((0, 3), dtype=np.int32),
        }
        if colors is not None:
            out["c"] = np.empty((0, colors.shape[1]), dtype=np.uint8)
        return out

    all_faces = np.concatenate(out_faces, axis=0).astype(np.int64, copy=False)
    if out_verts:
        extra = np.concatenate(out_verts, axis=0).astype(np.float32, copy=False)
        all_verts = np.concatenate([verts, extra], axis=0)
    else:
        all_verts = verts
    used = np.unique(all_faces.reshape(-1))
    remap = np.full(len(all_verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    out = {
        "v": all_verts[used].astype(np.float32, copy=False),
        "f": remap[all_faces].astype(np.int32, copy=False),
    }
    if colors is not None and len(colors) == len(verts):
        if out_color_idxs:
            extra_colors = colors[np.concatenate(out_color_idxs)]
            all_colors = np.concatenate([colors, extra_colors], axis=0)
        else:
            all_colors = colors
        out["c"] = all_colors[used].astype(np.uint8, copy=False)
    return out


def write_exact_clipped_npz(
    source_npz: Path,
    out_npz: Path,
    *,
    normal_xyz,
    position_nm: float,
    side: float,
) -> dict:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if out_npz.exists():
        with np.load(out_npz, allow_pickle=False) as z:
            return {"path": str(out_npz), "vertices": int(len(z["v"])), "faces": int(len(z["f"])), "reused": True}
    t0 = time.perf_counter()
    with np.load(source_npz, allow_pickle=False) as z:
        verts = np.asarray(z["v"], dtype=np.float32)
        faces = np.asarray(z["f"], dtype=np.int32)
        colors = np.asarray(z["c"], dtype=np.uint8) if "c" in z.files else None
    out = _clip_arrays(verts, faces, colors, np.asarray(normal_xyz, dtype=np.float64), position_nm, side)
    np.savez(out_npz, **out)
    return {
        "path": str(out_npz),
        "vertices": int(len(out["v"])),
        "faces": int(len(out["f"])),
        "source_vertices": int(len(verts)),
        "source_faces": int(len(faces)),
        "build_seconds": float(time.perf_counter() - t0),
        "reused": False,
    }


def write_voxel_cap_npz(
    label_zarr_url: str,
    seg_ids,
    out_npz: Path,
    *,
    bbox_xyz_nm,
    normal_xyz,
    position_nm: float,
    target_vertices: int | None,
    colorize=None,
    sample_nm: float | None = None,
) -> dict:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_npz.with_suffix(".json")
    if out_npz.exists():
        with np.load(out_npz, allow_pickle=False) as z:
            out = {"path": str(out_npz), "vertices": int(len(z["v"])), "faces": int(len(z["f"])), "reused": True}
        if summary_path.exists():
            try:
                out.update(json.loads(summary_path.read_text()))
            except Exception:  # noqa: BLE001
                pass
        return out

    t0 = time.perf_counter()
    ids = [int(s) for s in seg_ids]
    id_set = set(ids)
    vol = get_volume(label_zarr_url)
    plan = _choose_plan(
        vol,
        bbox_xyz_nm,
        target_voxels=1,
        target_vertices=target_vertices,
        seg_ids=ids,
    )
    arr, (z0, y0, x0), sc, tr = _read_plan_array(vol, plan, pad=3)
    t_read = time.perf_counter()

    lo = np.asarray(bbox_xyz_nm[0], dtype=np.float64)
    hi = np.asarray(bbox_xyz_nm[1], dtype=np.float64)
    u, v, n = _plane_basis(np.asarray(normal_xyz, dtype=np.float64))
    plane_point = n * float(position_nm)
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
        (fz < 0) | (fz > arr.shape[0] - 1)
        | (fy < 0) | (fy > arr.shape[1] - 1)
        | (fx < 0) | (fx > arr.shape[2] - 1)
    )
    iz = np.clip(np.round(fz).astype(np.int64), 0, arr.shape[0] - 1)
    iy = np.clip(np.round(fy).astype(np.int64), 0, arr.shape[1] - 1)
    ix = np.clip(np.round(fx).astype(np.int64), 0, arr.shape[2] - 1)
    labels = arr[iz, iy, ix].astype(np.int64, copy=False)
    mask = np.isin(labels, ids)
    mask[oob] = False
    t_sample = time.perf_counter()

    cell_y, cell_x = np.nonzero(mask)
    if len(cell_y) == 0:
        arrs: dict[str, np.ndarray] = {
            "v": np.empty((0, 3), dtype=np.float32),
            "f": np.empty((0, 3), dtype=np.int32),
        }
        if colorize is not None:
            arrs["c"] = np.empty((0, 4), dtype=np.uint8)
        np.savez(out_npz, **arrs)
        t_export = time.perf_counter()
        result = {
            "path": str(out_npz),
            "method": "voxel_plane_cap",
            "level": int(plan.level),
            "scale_zyx_nm": [float(x) for x in sc],
            "sample_nm": float(step),
            "read_shape_zyx": [int(x) for x in arr.shape],
            "plane_grid_yx": [int(x) for x in mask.shape],
            "occupied_plane_cells": 0,
            "vertices": 0,
            "faces": 0,
            "read_seconds": float(t_read - t0),
            "sample_seconds": float(t_sample - t_read),
            "mesh_export_seconds": float(t_export - t_sample),
            "total_seconds": float(t_export - t0),
            "reused": False,
        }
        summary_path.write_text(json.dumps(result, indent=2))
        return result

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
    vertices = (plane_point[None, :] + corner_us[:, None] * u[None, :] + corner_vs[:, None] * v[None, :]).astype(np.float32)

    q0 = vertex_ids[cell_y, cell_x]
    q1 = vertex_ids[cell_y, cell_x + 1]
    q2 = vertex_ids[cell_y + 1, cell_x + 1]
    q3 = vertex_ids[cell_y + 1, cell_x]
    faces = np.empty((len(cell_y) * 2, 3), dtype=np.int32)
    faces[0::2] = np.stack([q0, q1, q2], axis=1)
    faces[1::2] = np.stack([q0, q2, q3], axis=1)

    arrs: dict[str, np.ndarray] = {"v": vertices, "f": faces}
    if colorize is not None and len(vertices):
        vertex_label = np.zeros(len(vertices), dtype=np.int64)
        for cy, cx, lab in zip(cell_y, cell_x, labels[cell_y, cell_x]):
            if int(lab) not in id_set:
                continue
            for vy, vx in ((cy, cx), (cy, cx + 1), (cy + 1, cx + 1), (cy + 1, cx)):
                vi = vertex_ids[vy, vx]
                if vi >= 0 and vertex_label[vi] == 0:
                    vertex_label[vi] = int(lab)
        default_id = ids[0] if ids else 0
        colors = np.empty((len(vertices), 4), dtype=np.uint8)
        for i, sid in enumerate(vertex_label):
            r, g, b = colorize(int(sid or default_id))
            colors[i] = np.asarray([r, g, b, 1.0], dtype=np.float32) * 255
        arrs["c"] = colors
    np.savez(out_npz, **arrs)
    t_export = time.perf_counter()

    result = {
        "path": str(out_npz),
        "method": "voxel_plane_cap",
        "level": int(plan.level),
        "scale_zyx_nm": [float(x) for x in sc],
        "sample_nm": float(step),
        "read_shape_zyx": [int(x) for x in arr.shape],
        "plane_grid_yx": [int(x) for x in mask.shape],
        "occupied_plane_cells": int(mask.sum()),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "read_seconds": float(t_read - t0),
        "sample_seconds": float(t_sample - t_read),
        "mesh_export_seconds": float(t_export - t_sample),
        "total_seconds": float(t_export - t0),
        "reused": False,
    }
    summary_path.write_text(json.dumps(result, indent=2))
    return result
