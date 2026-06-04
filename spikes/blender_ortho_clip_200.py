"""Two parallax-free demonstrations that the mesh sits on the segmentation:

  A) ORTHO   — orthographic camera straight down the slice normal, full mesh.
               No perspective, so the mesh silhouette projects onto its own
               cross-section (and the seg).
  B) CLIP    — mesh clipped to a thin slab at the slice plane, viewed with the
               SAME angled perspective camera that showed the float. With the
               out-of-plane body removed, the slab sits exactly on the seg.

Per axis (x/y/z) -> spikes/out/ortho_200_{ax}.png and clip_200_{ax}.png.
Reuses the EM+seg slice texture (world-coord sampled) and blender_script.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinemap.data.colors import LayerColors
from cinemap.data.mesh_loader import MeshLoader
from cinemap.data.slice_loader import EMVolume

EM = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8"
MURL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/neuroglancer/mesh/inference/segmentations/mito"
LAB = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/labels/inference/segmentations/mito"
SEG = 200
NM_PER_BU = 1000.0
AX = {"x": 0, "y": 1, "z": 2}
OUT = ROOT / "spikes" / "out"
COLORS = LayerColors(seed=0)
SLAB_NM = 60.0                       # half-thickness of the clipped slab
VIEW_DIR = {"z": (0.35, 0.35, 1.0), "y": (0.35, 1.0, 0.35), "x": (1.0, 0.35, 0.35)}


def bu(p):
    return [c / NM_PER_BU for c in p]


def em_seg_png(em, lab, ax, pos, center, half, path):
    """EM crop + seg cross-section (label sampled at each EM pixel's world coord)."""
    e = em.read_slice(ax, pos, region=(tuple(center), half))
    rgb = np.repeat(np.asarray(e.image)[:, :, None].astype(np.float64), 3, axis=2)
    H, W = rgb.shape[:2]
    l = lab.read_slice(ax, pos, region=(tuple(center), half))
    limg = np.asarray(l.image); Hl, Wl = limg.shape
    lo, lu, lv = map(np.array, (l.origin_nm, l.u_nm, l.v_nm))
    eo, eu, ev = map(np.array, (e.origin_nm, e.u_nm, e.v_nm))
    s = (np.arange(W) + 0.5) / W; t = (np.arange(H) + 0.5) / H
    world = eo + t[:, None, None] * ev + s[None, :, None] * eu
    d = world - lo
    ci = np.clip(((d @ lu) / (lu @ lu) * Wl).astype(int), 0, Wl - 1)
    ri = np.clip(((d @ lv) / (lv @ lv) * Hl).astype(int), 0, Hl - 1)
    mask = limg[ri, ci] == SEG
    rgb[mask] = rgb[mask] * 0.5 + np.array(COLORS.rgb(SEG)) * 255 * 0.5
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(path)
    return e


def main():
    ml = MeshLoader(MURL, LAB)
    bb = ml._draco(SEG).bounds
    center = (bb[0] + bb[1]) / 2
    half = float(np.max(bb[1] - bb[0])) * 0.75 + 700
    mcol = list(COLORS.rgb(SEG))

    full = ml.load(SEG, colorize=COLORS.rgb)
    full_ply = OUT / f"mesh_{SEG}_full.ply"
    full.export(str(full_ply))

    em, lab = EMVolume(EM), EMVolume(LAB)
    meshes = [{"id": "full", "obj_path": str(full_ply), "color": mcol}]
    frames, meta = [], []

    for ax in ("x", "y", "z"):
        pos = float(center[AX[ax]])
        png = OUT / f"emseg_{SEG}_{ax}.png"
        e = em_seg_png(em, lab, ax, pos, center, half, png)
        slc = [{"image_path": str(png), "origin_bu": bu(e.origin_nm),
                "u_bu": bu(e.u_nm), "v_bu": bu(e.v_nm), "opacity": 1.0}]

        n = np.zeros(3); n[AX[ax]] = 1.0
        # slab-clipped mesh: keep only |dist to plane| <= SLAB_NM, cap the cuts
        slab = full.slice_plane(center - n * SLAB_NM, n, cap=True)
        slab = slab.slice_plane(center + n * SLAB_NM, -n, cap=True)
        slab.visual.vertex_colors = (np.array(mcol + [1.0]) * 255).astype(np.uint8)
        slab_ply = OUT / f"mesh_{SEG}_slab_{ax}.ply"
        slab.export(str(slab_ply))
        meshes.append({"id": f"slab_{ax}", "obj_path": str(slab_ply), "color": mcol})

        # A) orthographic straight down the normal
        cam_ortho = (np.array(bu(center)) + n * 12).tolist()
        ext = 2 * half / NM_PER_BU * 1.05
        frames.append({"index": len(frames),
                       "camera": {"type": "ORTHO", "position_bu": cam_ortho,
                                  "look_at_bu": bu(center), "ortho_scale": ext,
                                  "fov_rad": 0.6, "up": [0, 0, 1]},
                       "slices": slc, "mesh_overrides": {"full": {"opacity": 0.85}}})
        meta.append((ax, "ortho"))

        # B) clipped slab, same angled perspective camera as the original float view
        d = np.array(VIEW_DIR[ax], float); d /= np.linalg.norm(d)
        cam_persp = (np.array(bu(center)) + d * 6.5).tolist()
        frames.append({"index": len(frames),
                       "camera": {"position_bu": cam_persp, "look_at_bu": bu(center),
                                  "fov_rad": 0.62, "up": [0, 0, 1]},
                       "slices": slc, "mesh_overrides": {f"slab_{ax}": {"opacity": 0.95}}})
        meta.append((ax, "clip"))

    spec = {"world": {"nm_per_bu": NM_PER_BU, "background": [0.02, 0.02, 0.03]},
            "lighting": {"key_energy": 3000.0},
            "render": {"engine": "CYCLES", "width": 900, "height": 900, "samples": 96},
            "meshes": meshes, "frames": frames, "output_dir": str(OUT)}
    scene_path = OUT / f"scene_ortho_clip_{SEG}.json"
    scene_path.write_text(json.dumps(spec, indent=2))

    print(f"rendering {len(frames)} frames ...")
    subprocess.run([sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
                   cwd=str(ROOT / "src"), check=True)
    for fi, (ax, kind) in enumerate(meta):
        src = OUT / f"frame_{fi:05d}.png"
        dst = OUT / f"{kind}_{SEG}_{ax}.png"
        if src.exists():
            src.replace(dst); print(f"  -> {dst.name}")


if __name__ == "__main__":
    main()
