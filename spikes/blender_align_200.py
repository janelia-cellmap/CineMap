"""Render EM slice + 3D mesh together in Blender for seg 200, one image per axis
(x / y / z) — the same three planes validated in validate_mesh_alignment.py.

Each frame shows the grayscale EM cross-section as a textured plane with the
segment's marching-cubes mesh passing through it, so the mesh surface can be
eyeballed against the EM structure at the cut. Reuses the real render pipeline's
scene-spec format + blender_script.

Run:  conda run -n mv_env python spikes/blender_align_200.py
Out:  spikes/out/blender_align_200_{x,y,z}.png
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinemap.data.mesh_loader import MeshLoader
from cinemap.data.slice_loader import EMVolume

EM = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8"
MESH_URL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/neuroglancer/mesh/inference/segmentations/mito"
LABEL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/labels/inference/segmentations/mito"

SEG = 200
NM_PER_BU = 1000.0
OUT = ROOT / "spikes" / "out"
_AXIS = {"x": 0, "y": 1, "z": 2}
# look direction per slice axis: mostly along the slice normal, tilted for a 3D feel
_VIEW_DIR = {"z": (0.35, 0.35, 1.0), "y": (0.35, 1.0, 0.35), "x": (1.0, 0.35, 0.35)}


def bu(p):
    return [c / NM_PER_BU for c in p]


def em_slice_png(vol, axis, pos, center, half, path):
    res = vol.read_slice(axis, pos, region=(tuple(center), half))
    Image.fromarray(np.asarray(res.image)).convert("RGB").save(path)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    ml = MeshLoader(MESH_URL, LABEL)
    bbox = ml._draco(SEG).bounds                       # xyz nm
    center = (bbox[0] + bbox[1]) / 2
    half = float(np.max(bbox[1] - bbox[0])) * 0.75 + 700

    # one solid, bright mesh so it reads against the grayscale EM
    mesh = ml.load(SEG, colorize=lambda s: (0.96, 0.55, 0.18))
    ply = OUT / f"mesh_{SEG}.ply"
    mesh.export(str(ply))
    print(f"mesh: {len(mesh.vertices)} verts -> {ply.name}")

    em_vol = EMVolume(EM)
    uid = f"seg{SEG}"
    cam_dist = 6.5  # BU

    frames = []
    for axis in ("x", "y", "z"):
        pos = float(center[_AXIS[axis]])
        png = OUT / f"em_{SEG}_{axis}.png"
        res = em_slice_png(em_vol, axis, pos, center, half, png)
        print(f"  EM {axis} @ {pos:.0f}nm  img={res.image.shape}  -> {png.name}")

        d = np.array(_VIEW_DIR[axis], float)
        d /= np.linalg.norm(d)
        cam_pos = np.array(bu(center)) + d * cam_dist
        frames.append({
            "index": len(frames),
            "camera": {
                "position_bu": cam_pos.tolist(),
                "look_at_bu": bu(center),
                "fov_rad": 0.62,
                "up": [0, 0, 1],
            },
            "slices": [{
                "image_path": str(png),
                "origin_bu": bu(res.origin_nm),
                "u_bu": bu(res.u_nm),
                "v_bu": bu(res.v_nm),
                "opacity": 1.0,
            }],
            "mesh_overrides": {uid: {"opacity": 0.85, "visible": True}},
            "_axis": axis,
        })

    spec = {
        "world": {"nm_per_bu": NM_PER_BU, "background": [0.02, 0.02, 0.03]},
        "lighting": {"key_energy": 3000.0},
        "render": {"engine": "CYCLES", "width": 1000, "height": 1000, "samples": 96},
        "meshes": [{"id": uid, "obj_path": str(ply), "color": [0.96, 0.55, 0.18]}],
        "frames": frames,
        "output_dir": str(OUT),
    }
    scene_path = OUT / f"scene_align_{SEG}.json"
    scene_path.write_text(json.dumps(spec, indent=2))

    print("rendering in Blender ...")
    subprocess.run(
        [sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
        cwd=str(ROOT / "src"), check=True,
    )

    # blender writes frame_{index:05d}.png; rename to per-axis names
    for fr in frames:
        src = OUT / f"frame_{fr['index']:05d}.png"
        dst = OUT / f"blender_align_{SEG}_{fr['_axis']}.png"
        if src.exists():
            src.replace(dst)
            print(f"  -> {dst.name}")


if __name__ == "__main__":
    main()
