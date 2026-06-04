"""Validate that a generated mesh aligns with its SEGMENTATION labels.

The mesh is produced by marching cubes on the label volume, so the labels — not
the EM — are its ground truth. The EM and the labels can sit on different
coordinate grids, so checking a mesh against the EM image conflates a mesh-
transform bug with an EM/label offset. Here we instead:

  1. generate the mesh for one segment from the labels,
  2. read the segmentation label slice at a plane through the segment,
  3. cut the mesh with that exact plane to get its cross-section contour,
  4. rasterize both into the same world rectangle, overlay, and report IoU.

A correct mesh transform => the mesh contour hugs the label-mask boundary
(IoU ~1.0). A drift => the contour is shifted/scaled off the mask.

Usage:
    python spikes/validate_mesh_alignment.py [seg_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cinemap.data.mesh_loader import MeshLoader
from cinemap.data.slice_loader import EMVolume

MESH_URL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/neuroglancer/mesh/inference/segmentations/mito"
LABEL = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/labels/inference/segmentations/mito"

_AXIS = {"x": 0, "y": 1, "z": 2}


def rasterize_mask(label_img, seg_id, out_h, out_w):
    """Binary mask of seg_id in the label slice, resized (nearest) to out_h x out_w."""
    lab = np.asarray(label_img)
    yi = (np.arange(out_h) * lab.shape[0] / out_h).astype(int).clip(0, lab.shape[0] - 1)
    xi = (np.arange(out_w) * lab.shape[1] / out_w).astype(int).clip(0, lab.shape[1] - 1)
    return lab[yi][:, xi] == seg_id


def mesh_section_to_pixels(mesh, axis, position_nm, origin_nm, u_nm, v_nm, out_h, out_w):
    """Cut `mesh` with the plane `axis = position_nm`; return (line_segments_px,
    filled_mask) in the slice's pixel grid.

    The slice rectangle spans  origin + s*u + t*v,  s,t in [0,1]; rows=v, cols=u
    (matching SliceResult / worker._slice_png)."""
    import trimesh

    normal = np.zeros(3)
    normal[_AXIS[axis]] = 1.0
    section = mesh.section(plane_origin=np.array(origin_nm) + np.array(u_nm) / 2 + np.array(v_nm) / 2,
                           plane_normal=normal)
    if section is None:
        return np.empty((0, 2, 2)), np.zeros((out_h, out_w), bool)

    o = np.array(origin_nm, float)
    u = np.array(u_nm, float)
    v = np.array(v_nm, float)
    ulen2 = float(u @ u)
    vlen2 = float(v @ v)

    def to_px(pts3):
        d = pts3 - o
        s = (d @ u) / ulen2        # 0..1 across columns
        t = (d @ v) / vlen2        # 0..1 across rows
        return np.column_stack([s * (out_w - 1), t * (out_h - 1)])  # (col, row)

    # line segments for drawing the contour
    segs_px = []
    fill = np.zeros((out_h, out_w), bool)
    for ent in section.entities:
        verts = section.vertices[ent.points]          # ordered loop, 3D nm
        px = to_px(verts)
        for i in range(len(px) - 1):
            segs_px.append([px[i], px[i + 1]])
        # filled polygon (for IoU); xor handles nested loops / holes
        if len(px) >= 3:
            from skimage.draw import polygon as sk_polygon
            rr, cc = sk_polygon(px[:, 1], px[:, 0], shape=(out_h, out_w))  # (rows, cols)
            poly_mask = np.zeros((out_h, out_w), bool)
            poly_mask[rr, cc] = True
            fill ^= poly_mask
    return np.array(segs_px) if segs_px else np.empty((0, 2, 2)), fill


def draw(out_path, label_mask, mesh_fill, segs_px):
    from PIL import Image, ImageDraw

    h, w = label_mask.shape
    img = np.zeros((h, w, 3), np.uint8)
    img[label_mask] = (60, 120, 220)            # segmentation label: blue fill
    only_mesh = mesh_fill & ~label_mask
    img[only_mesh] = (220, 80, 80)              # mesh-only (over-coverage): red
    pim = Image.fromarray(img)
    d = ImageDraw.Draw(pim)
    for (c0, r0), (c1, r1) in segs_px:          # mesh contour: yellow line
        d.line([(c0, r0), (c1, r1)], fill=(255, 230, 40), width=1)
    pim.save(out_path)


def main():
    seg_id = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_h = out_w = 600

    ml = MeshLoader(MESH_URL, LABEL)
    bbox = ml._draco(seg_id).bounds                 # xyz nm
    mesh = ml.load(seg_id)                           # marching cubes from labels
    print(f"seg {seg_id}: bbox {bbox[0].round()} -> {bbox[1].round()} nm, "
          f"{len(mesh.vertices)} verts")

    vol = EMVolume(LABEL)
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)

    # slice through the segment center along each axis
    center = (bbox[0] + bbox[1]) / 2
    for axis in ("z", "y", "x"):
        pos = float(center[_AXIS[axis]])
        half = float(np.max(bbox[1] - bbox[0])) * 0.75 + 500
        region = (tuple(center), half)
        sl = vol.read_slice(axis, pos, region=region)
        mask = rasterize_mask(sl.image, seg_id, out_h, out_w)
        segs, fill = mesh_section_to_pixels(
            mesh, axis, pos, sl.origin_nm, sl.u_nm, sl.v_nm, out_h, out_w)

        inter = (mask & fill).sum()
        union = (mask | fill).sum()
        iou = inter / union if union else float("nan")
        # centroid offset (label vs mesh fill), in pixels and nm
        off_px = off_nm = None
        if mask.any() and fill.any():
            my, mx = np.argwhere(mask).mean(0)
            fy, fx = np.argwhere(fill).mean(0)
            off_px = (fx - mx, fy - my)
            unm = np.array(sl.u_nm); vnm = np.array(sl.v_nm)
            off_nm = off_px[0] / out_w * np.linalg.norm(unm), off_px[1] / out_h * np.linalg.norm(vnm)

        p = out / f"align_{seg_id}_{axis}.png"
        draw(p, mask, fill, segs)
        msg = f"  {axis} @ {pos:8.0f}nm  IoU={iou:.3f}  label_px={int(mask.sum())} mesh_px={int(fill.sum())}"
        if off_nm:
            msg += f"  centroid_off=({off_nm[0]:+.0f},{off_nm[1]:+.0f})nm"
        print(msg, "->", p.name)


if __name__ == "__main__":
    main()
