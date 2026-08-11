from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from spikes.compare_er_slice_mesh_vs_voxel import _plane_for_bbox


ROOT = Path(
    "/nrs/cellmap/ackermand/cinemap_projects/"
    "liverzonmovie-5-a100-50m-surface"
)
BASE_SCENE = ROOT / "renders/20260622-201706/scene.json"
ER_MESH = ROOT / "assets/mesh_er_c551581c.npz"
CAP_MESH = ROOT / "assets/er_slice_compare/cap_only/er_955_cap_only_oblique_64nm.ply"
OUT_DIR = ROOT / "assets/er_slice_compare/cap_only/render_proof_exact"
CLIPPED_MESH = OUT_DIR / "mesh_er_c551581c_oblique_exactclip.npz"


def _interp(a: np.ndarray, b: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    t = sa / (sa - sb)
    return a + t[:, None] * (b - a)


def _append_case_one_inside(
    out_faces: list[np.ndarray],
    out_verts: list[np.ndarray],
    face_subset: np.ndarray,
    verts: np.ndarray,
    signed: np.ndarray,
    inside_slot: int,
    next_index: int,
) -> int:
    if len(face_subset) == 0:
        return next_index
    slots = [0, 1, 2]
    slots.remove(inside_slot)
    i0 = face_subset[:, inside_slot]
    o1 = face_subset[:, slots[0]]
    o2 = face_subset[:, slots[1]]
    p0 = verts[i0]
    p1 = _interp(p0, verts[o1], signed[i0], signed[o1])
    p2 = _interp(p0, verts[o2], signed[i0], signed[o2])
    n = len(face_subset)
    new_ids = np.arange(next_index, next_index + 2 * n, dtype=np.int64).reshape(n, 2)
    out_verts.append(np.empty((2 * n, 3), dtype=np.float32))
    out_verts[-1][0::2] = p1.astype(np.float32, copy=False)
    out_verts[-1][1::2] = p2.astype(np.float32, copy=False)
    out_faces.append(np.stack([i0.astype(np.int64), new_ids[:, 0], new_ids[:, 1]], axis=1))
    return next_index + 2 * n


def _append_case_two_inside(
    out_faces: list[np.ndarray],
    out_verts: list[np.ndarray],
    face_subset: np.ndarray,
    verts: np.ndarray,
    signed: np.ndarray,
    outside_slot: int,
    next_index: int,
) -> int:
    if len(face_subset) == 0:
        return next_index
    slots = [0, 1, 2]
    slots.remove(outside_slot)
    i0 = face_subset[:, slots[0]]
    i1 = face_subset[:, slots[1]]
    o = face_subset[:, outside_slot]
    p0 = verts[i0]
    p1 = verts[i1]
    q0 = _interp(p0, verts[o], signed[i0], signed[o])
    q1 = _interp(p1, verts[o], signed[i1], signed[o])
    n = len(face_subset)
    new_ids = np.arange(next_index, next_index + 2 * n, dtype=np.int64).reshape(n, 2)
    out_verts.append(np.empty((2 * n, 3), dtype=np.float32))
    out_verts[-1][0::2] = q0.astype(np.float32, copy=False)
    out_verts[-1][1::2] = q1.astype(np.float32, copy=False)
    # Keep the inside quad as two triangles. Winding is inherited well enough for Cycles;
    # Blender can recalc if this ever becomes a production path.
    out_faces.append(np.stack([i0.astype(np.int64), i1.astype(np.int64), new_ids[:, 1]], axis=1))
    out_faces.append(np.stack([i0.astype(np.int64), new_ids[:, 1], new_ids[:, 0]], axis=1))
    return next_index + 2 * n


def _make_exact_clipped_mesh(plane_point: np.ndarray, normal: np.ndarray) -> dict:
    if CLIPPED_MESH.exists():
        with np.load(CLIPPED_MESH, allow_pickle=False) as z:
            return {
                "path": str(CLIPPED_MESH),
                "vertices": int(len(z["v"])),
                "faces": int(len(z["f"])),
                "reused": True,
            }

    t0 = time.perf_counter()
    with np.load(ER_MESH, allow_pickle=False) as z:
        verts = np.asarray(z["v"], dtype=np.float32)
        faces = np.asarray(z["f"], dtype=np.int32)
        colors = np.asarray(z["c"], dtype=np.uint8) if "c" in z.files else None

    signed = ((verts.astype(np.float64) - plane_point[None, :]) @ normal).astype(np.float32)
    out_faces: list[np.ndarray] = []
    out_verts: list[np.ndarray] = []
    next_index = len(verts)
    chunk = 2_000_000
    for start in range(0, len(faces), chunk):
        f = faces[start:start + chunk]
        fs = signed[f]
        inside = fs <= 0.0
        ninside = inside.sum(axis=1)
        if np.any(ninside == 3):
            out_faces.append(f[ninside == 3].astype(np.int64, copy=False))
        crossing = f[(ninside > 0) & (ninside < 3)]
        if len(crossing) == 0:
            continue
        ci = inside[(ninside > 0) & (ninside < 3)]
        for slot in range(3):
            one = crossing[(ci.sum(axis=1) == 1) & ci[:, slot]]
            next_index = _append_case_one_inside(
                out_faces, out_verts, one, verts, signed, slot, next_index
            )
            two = crossing[(ci.sum(axis=1) == 2) & ~ci[:, slot]]
            next_index = _append_case_two_inside(
                out_faces, out_verts, two, verts, signed, slot, next_index
            )

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
        default_color = colors[0] if len(colors) else np.array([255, 255, 255], dtype=np.uint8)
        if len(all_verts) > len(verts):
            extra_colors = np.tile(default_color, (len(all_verts) - len(verts), 1))
            all_colors = np.concatenate([colors, extra_colors], axis=0)
        else:
            all_colors = colors
        out["c"] = all_colors[used].astype(np.uint8, copy=False)

    np.savez(CLIPPED_MESH, **out)
    return {
        "path": str(CLIPPED_MESH),
        "vertices": int(len(out["v"])),
        "faces": int(len(out["f"])),
        "source_vertices": int(len(verts)),
        "source_faces": int(len(faces)),
        "build_seconds": float(time.perf_counter() - t0),
        "reused": False,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = json.loads(BASE_SCENE.read_text())

    with np.load(ER_MESH, allow_pickle=False) as z:
        verts = np.asarray(z["v"], dtype=np.float64)
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)

    _name, plane_point, normal = _plane_for_bbox(lo, hi, "oblique")
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal) or 1.0
    clipped = _make_exact_clipped_mesh(plane_point, normal)

    nm_per_bu = float(spec.get("world", {}).get("nm_per_bu", 1000.0))
    point_bu = plane_point / nm_per_bu
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(ref, normal)
    right /= np.linalg.norm(right) or 1.0
    up = np.cross(normal, right)
    span_bu = float(np.linalg.norm((hi - lo) / nm_per_bu))
    dist = max(85.0, span_bu * 0.9)
    camera = {
        "position_bu": (point_bu + normal * dist + up * dist * 0.12).tolist(),
        "look_at_bu": point_bu.tolist(),
        "fov_rad": math.radians(42.0),
        "up": up.tolist(),
        "flip_handed": False,
        "dof": {"fstop": 8.0},
    }

    render = dict(spec["render"])
    render.update({"width": 1280, "height": 720, "samples": 24, "noise_threshold": 0.08})
    spec.update(
        {
            "render": render,
            "output_dir": str(OUT_DIR),
            "export_blend": "",
            "warm_blend": str(OUT_DIR / "er_clip_cap_exact.blend"),
            "meshes": [
                {
                    "id": "er_exterior_exactclip",
                    "obj_path": str(CLIPPED_MESH),
                    "color": [0.91, 0.45, 0.23],
                    "clip": False,
                    "clip_cap": False,
                },
                {
                    "id": "er_voxel_cap",
                    "obj_path": str(CAP_MESH),
                    "color": [1.0, 0.74, 0.38],
                    "clip": False,
                    "clip_cap": False,
                },
            ],
            "frames": [
                {
                    "index": 0,
                    "camera": camera,
                    "slices": [],
                    "annotations": [],
                    "mesh_overrides": {
                        "er_exterior_exactclip": {
                            "opacity": 0.92,
                            "visible": True,
                            "silhouette": 0.0,
                        },
                        "er_voxel_cap": {
                            "opacity": 1.0,
                            "visible": True,
                            "silhouette": 0.0,
                        },
                    },
                }
            ],
        }
    )
    direction = dict(spec.get("direction") or {})
    mat = dict(direction.get("material") or {})
    mat.update({"flat_shading": False, "ao": 0.15, "edge_glow": 0.05, "roughness": 0.55})
    direction["material"] = mat
    spec["direction"] = direction

    scene_path = OUT_DIR / "er_clip_cap_exact_scene.json"
    scene_path.write_text(json.dumps(spec, indent=2))
    subprocess.run(
        [sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
        check=True,
    )
    print(json.dumps({
        "scene": str(scene_path),
        "image": str(OUT_DIR / "frame_00000.png"),
        "blend": str(OUT_DIR / "er_clip_cap_exact.blend"),
        "clipped_mesh": clipped,
        "plane_point_nm": plane_point.tolist(),
        "plane_normal_xyz": normal.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
