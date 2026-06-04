"""Flat (no 3D camera) proof of alignment: for each axis, draw the EM crop, the
segmentation cross-section (world-coord sampled), and the mesh's 2D cross-section
CONTOUR — all at the exact cut plane. Removes all perspective/parallax, so this
shows whether the mesh and seg actually coincide at the slice.

Out: spikes/out/flat_200_{x,y,z}.png   (yellow = mesh section contour)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import polygon as skpoly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinemap.data.colors import LayerColors
from cinemap.data.mesh_loader import MeshLoader
from cinemap.data.slice_loader import EMVolume

EM = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8"
MURL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/neuroglancer/mesh/inference/segmentations/mito"
LAB = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/labels/inference/segmentations/mito"
SEG = 200
AX = {"x": 0, "y": 1, "z": 2}
OUT = ROOT / "spikes" / "out"
COLORS = LayerColors(seed=0)


def main():
    ml = MeshLoader(MURL, LAB)
    b = ml._draco(SEG).bounds
    c = (b[0] + b[1]) / 2
    half = float(np.max(b[1] - b[0])) * 0.75 + 700
    mesh = ml.load(SEG)
    em, lab = EMVolume(EM), EMVolume(LAB)
    UP = 3  # upsample for a crisp contour

    for ax in ("x", "y", "z"):
        pos = float(c[AX[ax]])
        e = em.read_slice(ax, pos, region=(tuple(c), half))
        l = lab.read_slice(ax, pos, region=(tuple(c), half))
        H, W = e.image.shape
        eo, eu, ev = map(np.array, (e.origin_nm, e.u_nm, e.v_nm))
        labimg = np.asarray(l.image); Hl, Wl = labimg.shape
        lo, lu, lv = map(np.array, (l.origin_nm, l.u_nm, l.v_nm))

        # seg overlay: sample label at each EM pixel's true world coord
        s = (np.arange(W) + 0.5) / W; t = (np.arange(H) + 0.5) / H
        world = eo + t[:, None, None] * ev + s[None, :, None] * eu
        d = world - lo
        ci = np.clip(((d @ lu) / (lu @ lu) * Wl).astype(int), 0, Wl - 1)
        ri = np.clip(((d @ lv) / (lv @ lv) * Hl).astype(int), 0, Hl - 1)
        segmask = labimg[ri, ci] == SEG

        # base image (upsampled), EM grayscale + seg tint
        rgb = np.repeat(np.asarray(e.image)[:, :, None].astype(np.float64), 3, axis=2)
        col = np.array(COLORS.rgb(SEG)) * 255
        rgb[segmask] = rgb[segmask] * 0.5 + col * 0.5
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((W * UP, H * UP), Image.NEAREST)
        draw = ImageDraw.Draw(img)

        # mesh cross-section contour + fill (for IoU), in EM-pixel coords
        n = np.zeros(3); n[AX[ax]] = 1.0
        sec = mesh.section(plane_origin=eo + eu / 2 + ev / 2, plane_normal=n)
        fill = np.zeros((H, W), bool)
        if sec is not None:
            for ent in sec.entities:
                p = sec.vertices[ent.points]; dd = p - eo
                cc = (dd @ eu) / (eu @ eu) * (W - 1)
                rr = (dd @ ev) / (ev @ ev) * (H - 1)
                draw.line([(x * UP, y * UP) for x, y in zip(cc, rr)], fill=(255, 230, 40), width=2)
                pr, pc = skpoly(rr, cc, shape=(H, W)); mm = np.zeros((H, W), bool); mm[pr, pc] = True
                fill ^= mm
        iou = (segmask & fill).sum() / (segmask | fill).sum() if (segmask | fill).any() else float("nan")
        d2 = ImageDraw.Draw(img)
        d2.rectangle([0, 0, 230, 24], fill=(0, 0, 0))
        d2.text((6, 6), f"{ax}: seg vs mesh-section  IoU={iou:.3f}", fill=(255, 255, 255))
        out = OUT / f"flat_200_{ax}.png"
        img.save(out)
        print(f"{ax}: IoU={iou:.3f}  -> {out.name}")


if __name__ == "__main__":
    main()
