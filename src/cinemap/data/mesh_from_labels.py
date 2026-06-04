"""Generate a clean mesh for a segment directly from its OME-Zarr label volume.

This bypasses the precomputed multilod-draco meshes (which cloud-volume decodes
with per-chunk gaps, producing a 'stippled' look). Marching cubes on the label
mask yields a single watertight surface.
"""
from __future__ import annotations

import colorsys

import numpy as np
import trimesh
from skimage import measure

from .slice_loader import get_volume


def _layer_offset(layer: str) -> float:
    """Stable hue offset per layer name, so different layers (nuc vs mito) sit in
    different color families even when they share segment ids."""
    if not layer:
        return 0.0
    h = 0
    for c in layer:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    # golden-ratio spread so similar names (nuc/mito) land far apart on the hue wheel
    return (h * 0.6180339887498949) % 1.0


def seg_color(seg_id: int, layer: str = "") -> tuple[float, float, float]:
    """Stable, well-spread color per segment id (golden-ratio hue), shifted by a
    per-layer offset so segments of different layers are distinguishable."""
    h = (_layer_offset(layer) + int(seg_id) * 0.6180339887498949) % 1.0
    return colorsys.hsv_to_rgb(h, 0.62, 0.95)


def _vertex_colors(verts_world_zyx, normals, sc, tr, arr, colorize=None) -> np.ndarray:
    """Per-vertex RGBA by segment id.

    A marching-cubes vertex sits on the segment/background boundary, so sampling
    the label at the rounded voxel straddles into background (id 0) for ~half the
    vertices — which would tint each mesh with two colors. Instead we sample one
    voxel *inside* the surface (along the inward normal) so every vertex gets its
    real segment id.
    """
    sc = np.array(sc, dtype=float)
    tr = np.array(tr, dtype=float)
    shape = np.array(arr.shape)

    def sample(points):  # world (z,y,x) nm -> voxel index
        v = np.clip(np.round((points - tr) / sc).astype(int), 0, shape - 1)
        return arr[v[:, 0], v[:, 1], v[:, 2]]

    segs = sample(verts_world_zyx - normals * sc)      # one voxel inside (normals point out)
    zero = segs == 0
    if zero.any():                                     # fall back to the other side, then the vertex
        segs[zero] = sample(verts_world_zyx + normals * sc)[zero]
    zero = segs == 0
    if zero.any():
        segs[zero] = sample(verts_world_zyx)[zero]

    cf = colorize or (lambda s: seg_color(s))
    lut = {int(s): (cf(int(s)) if s != 0 else (0.6, 0.6, 0.6)) for s in np.unique(segs)}
    rgba = np.ones((len(segs), 4), dtype=np.float64)
    for s, c in lut.items():
        rgba[segs == s, :3] = c
    return (rgba * 255).astype(np.uint8)


def generate(
    label_zarr_url: str,
    seg_id: int,
    bbox_xyz_nm,
    target_voxels: int = 8_000_000,
    smooth_iters: int = 10,
    colorize=None,
) -> trimesh.Trimesh:
    """Marching-cubes mesh (nm, x/y/z) for `seg_id` within `bbox_xyz_nm`.

    bbox_xyz_nm = ((x0,y0,z0), (x1,y1,z1)). The bbox typically comes from the
    precomputed mesh (its global placement is correct); geometry comes from labels.
    """
    vol = get_volume(label_zarr_url)
    level = vol.pick_level_for_box(bbox_xyz_nm, target_voxels)
    sub, (z0, y0, x0), sc, tr = vol.read_box(bbox_xyz_nm, level, pad=2)

    mask = (sub == seg_id).astype(np.uint8)
    if not mask.any():
        raise ValueError(f"segment {seg_id} not present in label box")
    mask = np.pad(mask, 1)  # seal the surface at the box border

    # world_zyx = (voxel_index * scale) + translation; verts are in (z,y,x) nm
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=sc)
    origin_zyx_nm = np.array([(z0 - 1) * sc[0] + tr[0], (y0 - 1) * sc[1] + tr[1],
                              (x0 - 1) * sc[2] + tr[2]])
    verts_world_zyx = verts + origin_zyx_nm
    verts_xyz = verts_world_zyx[:, ::-1]  # z,y,x -> x,y,z

    vcolors = None
    if colorize is not None:  # uniform color for this single segment (NG color)
        r, g, b = colorize(int(seg_id))
        vcolors = np.tile(np.array([r, g, b, 1.0]) * 255, (len(verts_xyz), 1)).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, vertex_colors=vcolors, process=False)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    return mesh


def generate_union(
    label_zarr_url: str,
    seg_ids,
    target_voxels: int = 20_000_000,
    smooth_iters: int = 6,
    colorize=None,
) -> trimesh.Trimesh:
    """One mesh for ALL `seg_ids` at once — a single coarse whole-volume label
    read + marching cubes on the union mask. Cheap for hundreds/thousands of
    segments (e.g. "show every nucleus") vs. per-segment reads.
    """
    vol = get_volume(label_zarr_url)
    # finest whole-volume level whose total voxels fit the budget
    level = len(vol.level_scale_nm) - 1
    for lvl in range(len(vol.level_scale_nm)):
        shp = vol.level_shape_zyx(lvl)
        if shp[0] * shp[1] * shp[2] <= target_voxels:
            level = lvl
            break
    # Even the coarsest available level can blow the budget (e.g. a single-scale
    # volume with no downsampled levels). Subsample on read with an integer stride
    # so the materialized array — and thus peak memory — stays bounded regardless.
    shp = vol.level_shape_zyx(level)
    total = shp[0] * shp[1] * shp[2]
    stride = max(1, int(np.ceil((total / max(1, target_voxels)) ** (1 / 3))))
    arr = np.asarray(vol._open_level(level)[::stride, ::stride, ::stride].read().result())
    sc = [s * stride for s in vol.level_scale_nm[level]]   # nm/voxel after striding (z,y,x)
    tr = vol.level_translation_nm[level]    # z,y,x nm; world = voxel*scale + translation
    mask = np.isin(arr, list(seg_ids)).astype(np.uint8)
    if not mask.any():
        raise ValueError("none of the selected segments present in labels")
    mask = np.pad(mask, 1)
    verts, faces, normals, _ = measure.marching_cubes(mask, level=0.5, spacing=sc)
    verts_world_zyx = verts + np.array([tr[0] - sc[0], tr[1] - sc[1], tr[2] - sc[2]])  # undo pad + translation
    colors = _vertex_colors(verts_world_zyx, normals, sc, tr, arr, colorize)  # per-segment color
    mesh = trimesh.Trimesh(vertices=verts_world_zyx[:, ::-1], faces=faces,
                           vertex_colors=colors, process=False)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    return mesh


def selected_region(label_zarr_url: str, seg_ids, target_voxels: int = 12_000_000):
    """Robust center + radius (xyz nm) of where the selected segments actually are,
    ignoring sparse outliers — so framing/slicing land on the dense cluster."""
    vol = get_volume(label_zarr_url)
    level = len(vol.level_scale_nm) - 1
    for lvl in range(len(vol.level_scale_nm)):
        shp = vol.level_shape_zyx(lvl)
        if shp[0] * shp[1] * shp[2] <= target_voxels:
            level = lvl
            break
    arr = np.asarray(vol._open_level(level)[:, :, :].read().result())
    sc = vol.level_scale_nm[level]          # z,y,x
    tr = vol.level_translation_nm[level]    # z,y,x
    zz, yy, xx = np.where(np.isin(arr, list(seg_ids)))
    if len(zz) == 0:
        raise ValueError("selected segments not found")
    # 5th–95th percentile bounds in nm, per axis (x,y,z); world = voxel*scale + translation
    lo = np.array([np.percentile(xx, 5) * sc[2] + tr[2], np.percentile(yy, 5) * sc[1] + tr[1],
                   np.percentile(zz, 5) * sc[0] + tr[0]])
    hi = np.array([np.percentile(xx, 95) * sc[2] + tr[2], np.percentile(yy, 95) * sc[1] + tr[1],
                   np.percentile(zz, 95) * sc[0] + tr[0]])
    center = (lo + hi) / 2
    radius = 0.5 * float(np.max(hi - lo))
    return center.tolist(), max(radius, 1000.0)
