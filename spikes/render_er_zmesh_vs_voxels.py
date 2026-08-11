from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
import zmesh

from cinemap.config import NM_PER_BU
from cinemap.data.mesh_from_labels import _bbox_for_ids, _choose_plan, _mesh_to_trimesh, _read_plan_array
from cinemap.data.slice_loader import get_volume


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = Path("/groups/cellmap/cellmap/ackermand/cinemap_projects2/liverzonmovie-30/project.json")
KEYFRAME_ID = "kf_d241583d"
OUT = ROOT / "spikes" / "er_comparison"
LABEL_URL = (
    "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-liver-zon-1/"
    "jrc_mus-liver-zon-1.zarr/recon-1/labels/inference/segmentations/er/"
)
SEG_ID = 955
COLOR = (0.90, 0.86, 0.86)
TARGET_VOXELS = 12_000_000
TARGET_VERTICES = 1_500_000


def _colorize(_seg_id: int):
    return COLOR


def exposed_voxel_surface(mask: np.ndarray, origin_zyx_nm, scale_zyx_nm) -> trimesh.Trimesh:
    """Blocky surface from occupied voxels. Coordinates are xyz nm."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    occ = np.argwhere(mask)
    verts = []
    faces = []
    colors = []
    sc = np.asarray(scale_zyx_nm, dtype=float)
    origin = np.asarray(origin_zyx_nm, dtype=float)

    # face definitions in zyx local cube coords, one quad per exposed side
    sides = [
        ((-1, 0, 0), [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
        ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
        ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
        ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
        ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
    ]
    rgba = np.array([*COLOR, 1.0]) * 255
    for z, y, x in occ:
        pz, py, px = z + 1, y + 1, x + 1
        for (dz, dy, dx), corners in sides:
            if padded[pz + dz, py + dy, px + dx]:
                continue
            base = len(verts)
            for c in corners:
                zyx = np.array((z + c[0], y + c[1], x + c[2]), dtype=float)
                world_zyx = origin + zyx * sc
                verts.append(world_zyx[::-1])
                colors.append(rgba)
            faces.append((base, base + 1, base + 2, base + 3))
    return trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=np.asarray(colors, dtype=np.uint8),
        process=False,
    )


def zmesh_from_mask(mask: np.ndarray, origin_zyx_nm, scale_zyx_nm) -> trimesh.Trimesh:
    labels = np.pad(np.where(mask, SEG_ID, 0).astype(np.uint32, copy=False), 1)
    mesher = zmesh.Mesher(tuple(float(s) for s in scale_zyx_nm))
    mesher.mesh(labels)
    origin = (
        origin_zyx_nm[0] - scale_zyx_nm[0],
        origin_zyx_nm[1] - scale_zyx_nm[1],
        origin_zyx_nm[2] - scale_zyx_nm[2],
    )
    mesh = _mesh_to_trimesh(mesher.get(SEG_ID), SEG_ID, origin, colorize=_colorize)
    if mesh is None:
        raise RuntimeError("zmesh produced no geometry")
    return mesh


def _camera_from_project():
    if not PROJECT_PATH.exists():
        return None
    project = json.loads(PROJECT_PATH.read_text())
    for keyframe in project.get("keyframes", []):
        if keyframe.get("id") == KEYFRAME_ID:
            return keyframe.get("camera")
    return None


def _scene(mesh_path: Path, output_dir: Path, center_nm, radius_nm, label: str, camera: dict | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if camera:
        camera_spec = {
            "position_bu": (np.asarray(camera["position_nm"], dtype=float) / NM_PER_BU).tolist(),
            "look_at_bu": (np.asarray(camera["look_at_nm"], dtype=float) / NM_PER_BU).tolist(),
            "fov_rad": math.radians(float(camera.get("fov_deg", 45.0))),
            "up": camera.get("up", [0.0, 0.0, 1.0]),
        }
    else:
        distance = radius_nm * 3.0
        cam = np.asarray(center_nm, dtype=float) + np.array([distance * 0.95, -distance * 1.15, distance * 0.75])
        camera_spec = {
            "position_bu": (cam / NM_PER_BU).tolist(),
            "look_at_bu": (np.asarray(center_nm) / NM_PER_BU).tolist(),
            "fov_rad": math.radians(34.0),
            "up": [0.0, 0.0, 1.0],
        }
    spec = {
        "world": {"nm_per_bu": NM_PER_BU, "background": [0.015, 0.016, 0.020]},
        "lighting": {"key_energy": 4500.0},
        "render": {
            "width": 1280,
            "height": 720,
            "fps": 1,
            "samples": 48,
            "engine": "CYCLES",
            "export_blend": False,
        },
        "meshes": [{"id": label, "obj_path": str(mesh_path), "color": list(COLOR)}],
        "frames": [{
            "camera": camera_spec,
            "slices": [],
            "mesh_overrides": {label: {"opacity": 1.0, "visible": True, "silhouette": 0.0}},
            "index": 0,
        }],
        "output_dir": str(output_dir),
        "fps": 1,
        "export_blend": None,
    }
    scene_path = output_dir / "scene.json"
    scene_path.write_text(json.dumps(spec, indent=2))
    return scene_path


def _render(scene_path: Path):
    subprocess.run(
        [sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
        cwd=str(ROOT),
        check=True,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bbox = _bbox_for_ids(LABEL_URL, [SEG_ID])
    if bbox is None:
        raise RuntimeError(f"segment {SEG_ID} not found")
    lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0)

    vol = get_volume(LABEL_URL)
    plan = _choose_plan(
        vol,
        bbox,
        target_voxels=TARGET_VOXELS,
        target_vertices=TARGET_VERTICES,
        seg_ids=[SEG_ID],
    )
    arr, (z0, y0, x0), sc, tr = _read_plan_array(vol, plan, pad=3)
    mask = arr == SEG_ID
    if not mask.any():
        raise RuntimeError(f"segment {SEG_ID} not present in voxel read")
    origin_zyx_nm = (
        z0 * sc[0] + tr[0],
        y0 * sc[1] + tr[1],
        x0 * sc[2] + tr[2],
    )

    zmesh = zmesh_from_mask(mask, origin_zyx_nm, sc)
    zmesh_path = OUT / "er_955_zmesh.ply"
    zmesh.export(zmesh_path)

    voxel_mesh = exposed_voxel_surface(mask, origin_zyx_nm, sc)
    voxel_path = OUT / "er_955_voxel_surface.ply"
    voxel_mesh.export(voxel_path)

    camera = _camera_from_project()
    for label, mesh_path in (("zmesh", zmesh_path), ("voxels", voxel_path)):
        scene_path = _scene(mesh_path, OUT / label, center, radius, label, camera)
        _render(scene_path)

    summary = {
        "bbox_xyz_nm": [lo.tolist(), hi.tolist()],
        "zmesh_vertices": int(len(zmesh.vertices)),
        "zmesh_faces": int(len(zmesh.faces)),
        "shared_input_level": int(plan.level),
        "shared_input_stride": int(plan.stride),
        "shared_input_scale_zyx_nm": list(sc),
        "shared_input_shape_zyx": list(arr.shape),
        "voxel_occupied": int(mask.sum()),
        "voxel_surface_vertices": int(len(voxel_mesh.vertices)),
        "voxel_surface_faces": int(len(voxel_mesh.faces)),
        "project_path": str(PROJECT_PATH) if PROJECT_PATH.exists() else None,
        "keyframe_id": KEYFRAME_ID if camera else None,
        "camera": camera,
        "zmesh_png": str(OUT / "zmesh" / "frame_00000.png"),
        "voxels_png": str(OUT / "voxels" / "frame_00000.png"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
