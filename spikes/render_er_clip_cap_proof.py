from __future__ import annotations

import json
import math
import subprocess
import sys
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
OUT_DIR = ROOT / "assets/er_slice_compare/cap_only/render_proof"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = json.loads(BASE_SCENE.read_text())

    with np.load(ER_MESH, allow_pickle=False) as z:
        verts = np.asarray(z["v"], dtype=np.float64)
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)

    _name, plane_point, normal = _plane_for_bbox(lo, hi, "oblique")
    nm_per_bu = float(spec.get("world", {}).get("nm_per_bu", 1000.0))
    point_bu = plane_point / nm_per_bu
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal) or 1.0
    clip_position_bu = float(np.dot(point_bu, normal))

    # Camera from the clipped-away side, looking into the retained ER and cap.
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
            "export_blend": str(OUT_DIR / "er_clip_cap_proof.blend"),
            "warm_blend": "",
            "meshes": [
                {
                    "id": "er_full_shader_clip",
                    "obj_path": str(ER_MESH),
                    "color": [0.91, 0.45, 0.23],
                    "clip": True,
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
                        "er_full_shader_clip": {
                            "opacity": 0.92,
                            "visible": True,
                            "silhouette": 0.0,
                            "clip": {
                                "normal": normal.tolist(),
                                "position_bu": clip_position_bu,
                                "position_nm": float(np.dot(plane_point, normal)),
                                "side": 1,
                            },
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
    # Use a quiet, non-geometric material profile so the proof focuses on clip + cap.
    direction = dict(spec.get("direction") or {})
    mat = dict(direction.get("material") or {})
    mat.update({"flat_shading": False, "ao": 0.15, "edge_glow": 0.05, "roughness": 0.55})
    direction["material"] = mat
    spec["direction"] = direction

    scene_path = OUT_DIR / "er_clip_cap_scene.json"
    scene_path.write_text(json.dumps(spec, indent=2))
    subprocess.run(
        [sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
        check=True,
    )
    print(json.dumps({
        "scene": str(scene_path),
        "image": str(OUT_DIR / "frame_00000.png"),
        "blend": str(OUT_DIR / "er_clip_cap_proof.blend"),
        "plane_point_nm": plane_point.tolist(),
        "plane_normal_xyz": normal.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
