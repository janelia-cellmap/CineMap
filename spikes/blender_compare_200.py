"""Side-by-side comparison: EM slice WITHOUT mesh vs WITH mesh, for seg 200,
one composite per axis (x / y / z). Same camera/plane as blender_align_200.py.

Run:  conda run -n mv_env python spikes/blender_compare_200.py
Out:  spikes/out/compare_200_{x,y,z}.png  (left = EM only, right = EM + mesh)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinemap.data.colors import LayerColors
from cinemap.data.mesh_loader import MeshLoader
from cinemap.data.slice_loader import EMVolume

COLORS = LayerColors(seed=0)  # neuroglancer hash coloring, keyed by seg_id

EM = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8"
MESH_URL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/neuroglancer/mesh/inference/segmentations/mito"
LABEL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/labels/inference/segmentations/mito"

SEG = 200
NM_PER_BU = 1000.0
OUT = ROOT / "spikes" / "out"
_AXIS = {"x": 0, "y": 1, "z": 2}
_VIEW_DIR = {"z": (0.35, 0.35, 1.0), "y": (0.35, 1.0, 0.35), "x": (1.0, 0.35, 0.35)}


def bu(p):
    return [c / NM_PER_BU for c in p]


def em_with_seg(em_vol, lab_vol, axis, pos, center, half, path, seg=SEG):
    """Grayscale EM crop with the segmentation label cross-section blended on
    (neuroglancer color for seg_id).

    The EM and label volumes snap their crops to *different* voxel grids (8 nm vs
    16 nm, different translations), so the two rectangles are offset by a few nm.
    Resampling the label by shape-ratio (as worker._slice_png does) assumes the
    rectangles coincide and shifts the overlay ~half a pixel off the mesh. Here we
    instead sample the label at each EM pixel's *true world coordinate*, so the
    painted seg lands exactly where the mesh cuts the plane."""
    res = em_vol.read_slice(axis, pos, region=(tuple(center), half))
    rgb = np.repeat(np.asarray(res.image)[:, :, None].astype(np.float64), 3, axis=2)
    H, W = rgb.shape[:2]
    lres = lab_vol.read_slice(axis, pos, region=(tuple(center), half))
    lab = np.asarray(lres.image)               # rows=v, cols=u
    Hl, Wl = lab.shape

    # world (xyz nm) of each EM pixel center: origin + s*u + t*v, rows=v cols=u
    eo, eu, ev = (np.array(res.origin_nm), np.array(res.u_nm), np.array(res.v_nm))
    s = (np.arange(W) + 0.5) / W
    t = (np.arange(H) + 0.5) / H
    world = eo[None, None, :] + t[:, None, None] * ev[None, None, :] + s[None, :, None] * eu[None, None, :]

    # project those world points into the label crop's (u,v) pixel grid
    lo, lu, lv = (np.array(lres.origin_nm), np.array(lres.u_nm), np.array(lres.v_nm))
    d = world - lo[None, None, :]
    sl = (d @ lu) / (lu @ lu)                  # 0..1 across label cols
    tl = (d @ lv) / (lv @ lv)                  # 0..1 across label rows
    ci = np.clip((sl * Wl).astype(int), 0, Wl - 1)
    ri = np.clip((tl * Hl).astype(int), 0, Hl - 1)
    mask = lab[ri, ci] == seg

    if mask.any():
        a = 0.5
        color = np.array(COLORS.rgb(seg), float) * 255  # neuroglancer color for seg_id
        rgb[mask] = rgb[mask] * (1 - a) + color * a
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(path)
    return res


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, len(text) * 11 + 16, 30], fill=(0, 0, 0))
    d.text((8, 8), text, fill=(255, 255, 255))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ml = MeshLoader(MESH_URL, LABEL)
    bbox = ml._draco(SEG).bounds
    center = (bbox[0] + bbox[1]) / 2
    half = float(np.max(bbox[1] - bbox[0])) * 0.75 + 700

    mesh = ml.load(SEG, colorize=COLORS.rgb)   # same seg_id-based color as the overlay
    ply = OUT / f"mesh_{SEG}.ply"
    mesh.export(str(ply))
    mcol = COLORS.rgb(SEG)

    em_vol = EMVolume(EM)
    lab_vol = EMVolume(LABEL)
    uid = f"seg{SEG}"
    frames = []
    meta = []  # (axis, kind) per frame index

    for axis in ("x", "y", "z"):
        pos = float(center[_AXIS[axis]])
        png = OUT / f"emseg_{SEG}_{axis}.png"
        res = em_with_seg(em_vol, lab_vol, axis, pos, center, half, png)

        d = np.array(_VIEW_DIR[axis], float); d /= np.linalg.norm(d)
        cam_pos = (np.array(bu(center)) + d * 6.5).tolist()
        cam = {"position_bu": cam_pos, "look_at_bu": bu(center), "fov_rad": 0.62, "up": [0, 0, 1]}
        sl = [{"image_path": str(png), "origin_bu": bu(res.origin_nm),
               "u_bu": bu(res.u_nm), "v_bu": bu(res.v_nm), "opacity": 1.0}]
        for kind, ov in (("em", {}), ("mesh", {uid: {"opacity": 0.85, "visible": True}})):
            frames.append({"index": len(frames), "camera": cam, "slices": sl,
                           "mesh_overrides": ov})
            meta.append((axis, kind))

    spec = {
        "world": {"nm_per_bu": NM_PER_BU, "background": [0.02, 0.02, 0.03]},
        "lighting": {"key_energy": 3000.0},
        "render": {"engine": "CYCLES", "width": 900, "height": 900, "samples": 96},
        "meshes": [{"id": uid, "obj_path": str(ply), "color": list(mcol)}],
        "frames": frames,
        "output_dir": str(OUT),
    }
    scene_path = OUT / f"scene_compare_{SEG}.json"
    scene_path.write_text(json.dumps(spec, indent=2))

    print(f"rendering {len(frames)} frames in Blender ...")
    subprocess.run([sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
                   cwd=str(ROOT / "src"), check=True)

    # group by axis -> stitch [em | mesh]
    by_axis: dict[str, dict[str, Image.Image]] = {}
    for fi, (axis, kind) in enumerate(meta):
        img = Image.open(OUT / f"frame_{fi:05d}.png").convert("RGB")
        by_axis.setdefault(axis, {})[kind] = img
    for axis, pair in by_axis.items():
        a = label(pair["em"].copy(), f"{axis}: EM + seg")
        b = label(pair["mesh"].copy(), f"{axis}: EM + seg + mesh")
        W, H = a.size
        comp = Image.new("RGB", (W * 2 + 6, H), (10, 10, 12))
        comp.paste(a, (0, 0)); comp.paste(b, (W + 6, 0))
        dst = OUT / f"compare_{SEG}_{axis}.png"
        comp.save(dst)
        print(f"  -> {dst.name}")


if __name__ == "__main__":
    main()
