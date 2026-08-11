from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from cinemap.data.mesh_from_labels import generate_zmesh_auto, seg_color
from spikes.compare_er_slice_mesh_vs_voxel import LABEL_URL, SEG_ID, _plane_for_bbox
from spikes.make_er_voxel_cap_mesh import make_cap
from spikes.render_er_clip_cap_exact import _make_exact_clipped_mesh


ROOT = Path(
    "/nrs/cellmap/ackermand/cinemap_projects/"
    "liverzonmovie-5-a100-50m-surface"
)
BASE_SCENE = ROOT / "renders/20260622-201706/scene.json"
OUT_DIR = ROOT / "assets/er_slice_compare/compare_15m"
ER15 = OUT_DIR / "mesh_er_955_15m.npz"
CAP_DIR = OUT_DIR / "cap"


def _save_mesh_npz(mesh: trimesh.Trimesh, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    color = np.array((*seg_color(SEG_ID, "er"), 1.0), dtype=np.float32)
    colors = np.tile((color * 255).astype(np.uint8), (len(verts), 1))
    np.savez(path, v=verts, f=faces, c=colors)
    return {"path": str(path), "vertices": int(len(verts)), "faces": int(len(faces))}


def _ensure_er15() -> dict:
    if ER15.exists():
        with np.load(ER15, allow_pickle=False) as z:
            return {"path": str(ER15), "vertices": int(len(z["v"])), "faces": int(len(z["f"])), "reused": True}
    t0 = time.perf_counter()
    mesh = generate_zmesh_auto(
        LABEL_URL,
        [SEG_ID],
        target_voxels=1,
        target_vertices=15_000_000,
        smooth_iters=0,
        simplify_budget_factor=0.0,
        colorize=lambda _sid: seg_color(SEG_ID, "er"),
    )
    out = _save_mesh_npz(mesh, ER15)
    out["build_seconds"] = float(time.perf_counter() - t0)
    out["reused"] = False
    return out


def _camera_and_plane(mesh_path: Path, spec: dict) -> tuple[dict, np.ndarray, np.ndarray, float]:
    with np.load(mesh_path, allow_pickle=False) as z:
        verts = np.asarray(z["v"], dtype=np.float64)
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
    _name, plane_point, normal = _plane_for_bbox(lo, hi, "oblique")
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal) or 1.0
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
    return camera, plane_point, normal, float(np.dot(plane_point, normal))


def _base_spec(out_dir: Path, width: int = 1280, height: int = 720) -> dict:
    spec = json.loads(BASE_SCENE.read_text())
    render = dict(spec["render"])
    render.update({"width": width, "height": height, "samples": 24, "noise_threshold": 0.08})
    direction = dict(spec.get("direction") or {})
    mat = dict(direction.get("material") or {})
    mat.update({"flat_shading": False, "ao": 0.15, "edge_glow": 0.05, "roughness": 0.55})
    direction["material"] = mat
    spec["render"] = render
    spec["direction"] = direction
    spec["output_dir"] = str(out_dir)
    spec["export_blend"] = ""
    return spec


def _run_scene(spec: dict, scene_path: Path) -> dict:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(json.dumps(spec, indent=2))
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)], check=True)
    return {"scene": str(scene_path), "seconds": float(time.perf_counter() - t0)}


def render_current(mesh_info: dict, camera: dict, normal: np.ndarray, plane_dot_nm: float) -> dict:
    out = OUT_DIR / "current_bmesh_cap"
    spec = _base_spec(out)
    spec["warm_blend"] = str(out / "current_bmesh_cap.blend")
    spec["meshes"] = [
        {
            "id": "er_current_bmesh_cap",
            "obj_path": mesh_info["path"],
            "color": [0.91, 0.45, 0.23],
            "clip": True,
            "clip_cap": True,
        }
    ]
    spec["frames"] = [
        {
            "index": 0,
            "camera": camera,
            "slices": [],
            "annotations": [],
            "mesh_overrides": {
                "er_current_bmesh_cap": {
                    "opacity": 0.92,
                    "visible": True,
                    "silhouette": 0.0,
                    "clip": {
                        "normal": normal.tolist(),
                        "position_bu": plane_dot_nm / float(spec["world"]["nm_per_bu"]),
                        "position_nm": plane_dot_nm,
                        "side": 1,
                    },
                }
            },
        }
    ]
    result = _run_scene(spec, out / "current_bmesh_cap_scene.json")
    result["image"] = str(out / "frame_00000.png")
    result["blend"] = str(out / "current_bmesh_cap.blend")
    return result


def render_exact_hybrid(mesh_info: dict, camera: dict, plane_point: np.ndarray, normal: np.ndarray) -> dict:
    out = OUT_DIR / "exact_hybrid"
    out.mkdir(parents=True, exist_ok=True)
    cap = make_cap(CAP_DIR, 15_000_000, "oblique", None)
    # Reuse the exact clipping function by temporarily pointing its globals at the 15M paths.
    import spikes.render_er_clip_cap_exact as exact

    old_er, old_out, old_clip = exact.ER_MESH, exact.OUT_DIR, exact.CLIPPED_MESH
    try:
        exact.ER_MESH = Path(mesh_info["path"])
        exact.OUT_DIR = out
        exact.CLIPPED_MESH = out / "mesh_er_955_15m_exactclip.npz"
        clipped = _make_exact_clipped_mesh(plane_point, normal)
    finally:
        exact.ER_MESH, exact.OUT_DIR, exact.CLIPPED_MESH = old_er, old_out, old_clip

    spec = _base_spec(out)
    spec["warm_blend"] = str(out / "exact_hybrid.blend")
    spec["meshes"] = [
        {
            "id": "er_exactclip_15m",
            "obj_path": clipped["path"],
            "color": [0.91, 0.45, 0.23],
            "clip": False,
            "clip_cap": False,
        },
        {
            "id": "er_voxel_cap_15m",
            "obj_path": cap["path"],
            "color": [1.0, 0.74, 0.38],
            "clip": False,
            "clip_cap": False,
        },
    ]
    spec["frames"] = [
        {
            "index": 0,
            "camera": camera,
            "slices": [],
            "annotations": [],
            "mesh_overrides": {
                "er_exactclip_15m": {"opacity": 0.92, "visible": True, "silhouette": 0.0},
                "er_voxel_cap_15m": {"opacity": 1.0, "visible": True, "silhouette": 0.0},
            },
        }
    ]
    result = _run_scene(spec, out / "exact_hybrid_scene.json")
    result["image"] = str(out / "frame_00000.png")
    result["blend"] = str(out / "exact_hybrid.blend")
    result["cap"] = cap
    result["clipped"] = clipped
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_info = _ensure_er15()
    spec = json.loads(BASE_SCENE.read_text())
    camera, plane_point, normal, plane_dot_nm = _camera_and_plane(Path(mesh_info["path"]), spec)
    current_image = OUT_DIR / "current_bmesh_cap/frame_00000.png"
    if os.environ.get("CINEMAP_SKIP_CURRENT") == "1" and current_image.exists():
        current = {
            "image": str(current_image),
            "scene": str(OUT_DIR / "current_bmesh_cap/current_bmesh_cap_scene.json"),
            "blend": str(OUT_DIR / "current_bmesh_cap/current_bmesh_cap.blend"),
            "reused": True,
        }
    else:
        current = render_current(mesh_info, camera, normal, plane_dot_nm)
    hybrid = render_exact_hybrid(mesh_info, camera, plane_point, normal)
    result = {
        "mesh_15m": mesh_info,
        "plane_point_nm": plane_point.tolist(),
        "plane_normal_xyz": normal.tolist(),
        "current_bmesh_cap": current,
        "exact_hybrid": hybrid,
    }
    summary = OUT_DIR / "compare_15m_summary.json"
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
