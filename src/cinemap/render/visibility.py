"""View-dependent visibility + on-screen sizing — the math behind the per-chunk
(view-aware) LOD mode, mirroring how neuroglancer decides what to draw.

Given a camera and an axis-aligned bounding box (a segment, or later an octree
fragment), we compute (a) whether it's inside the view frustum at all, and (b) the
on-screen scale (nm per pixel) at its nearest point — so far/small things load
coarse and are culled when off-screen, and near/large things load fine. This is the
faithful, frustum-culled equivalent of NG's per-node test, just at segment grain.
"""
from __future__ import annotations

import math

import numpy as np


def camera_basis(pos, look_at, up):
    """Right/true-up/forward unit vectors for a look-at camera (forward = toward
    look_at). Mirrors the render camera so culling matches what's actually framed."""
    pos = np.asarray(pos, float); look_at = np.asarray(look_at, float)
    fwd = look_at - pos
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    up = np.asarray(up, float)
    right = np.cross(fwd, up)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
    true_up = np.cross(right, fwd)
    return right, true_up, fwd


def bbox_corners(bbox) -> np.ndarray:
    """8 corners of an AABB given as ((x0,y0,z0),(x1,y1,z1))."""
    (x0, y0, z0), (x1, y1, z1) = bbox
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def project_bbox(corners, pos, right, true_up, fwd, fov_rad, aspect, height, margin=1.15):
    """Return (visible, nm_per_px) for an AABB.

    `visible` is False when the box is entirely behind the camera or its projected
    extent falls outside the frustum (NG uses a fixed VERTICAL fov; `aspect` =
    width/height widens the horizontal bound). `nm_per_px` is taken at the box's
    NEAREST in-front point (the finest detail it needs); inf when not visible."""
    pos = np.asarray(pos, float)
    rel = corners - pos
    depth = rel @ fwd                      # signed distance along view direction
    front = depth > 1e-6
    if not front.any():
        return False, math.inf
    tan_v = math.tan(fov_rad / 2.0)
    x = rel @ right
    y = rel @ true_up
    d = np.where(front, depth, np.nan)
    # normalized screen coords in [-1,1]; clamp depth to the front points
    ndc_x = x / (d * tan_v * aspect)
    ndc_y = y / (d * tan_v)
    with np.errstate(invalid="ignore"):
        in_x = np.nanmax(ndc_x) >= -margin and np.nanmin(ndc_x) <= margin
        in_y = np.nanmax(ndc_y) >= -margin and np.nanmin(ndc_y) <= margin
    visible = bool(in_x and in_y)
    near = float(np.nanmin(d))             # nearest in-front depth
    nmpp = 2.0 * near * tan_v / max(1, height)
    return visible, nmpp


def select_fragments(per_lod, lod_scales_nm, pos, look_at, up, fov_rad, aspect, height,
                     tol=1.0):
    """Neuroglancer's octree traversal, offline. Descend the mesh octree top-down from
    the coarsest LOD: frustum-cull a node's whole subtree if its box is off-screen;
    render a node at its LOD once it's fine enough for its on-screen size
    (`lodScale <= pixelSize*tol`) or it's a leaf; otherwise recurse into its children
    (the finer fragments that exist). Because the finer LODs tile the same surface,
    recursing covers every surface region exactly once — no gaps, no overlap. (My
    earlier per-fragment rule compared each fragment to its parent using DIFFERENT
    pixel sizes, which dropped regions — the 'broken up' look.)

    Returns ({lod: [frag_index, ...]}, total_bytes)."""
    import itertools

    right, true_up, fwd = camera_basis(pos, look_at, up)
    maxlod = len(per_lod) - 1
    by_pos = []                                  # per lod: {grid_pos: (idx, lo, hi, nbytes)}
    for frags in per_lod:
        by_pos.append({tuple(gp): (idx, lo, hi, nb) for (idx, lo, hi, gp, nb) in frags})

    sel: dict[int, list] = {}
    total = 0

    def children(lod, gpos):
        if lod == 0:
            return []
        base = tuple(2 * c for c in gpos)
        return [(lod - 1, q) for d in itertools.product((0, 1), repeat=3)
                if (q := (base[0] + d[0], base[1] + d[1], base[2] + d[2])) in by_pos[lod - 1]]

    def visit(lod, gpos):
        nonlocal total
        node = by_pos[lod].get(gpos)
        kids = children(lod, gpos)
        if node is None:                         # missing parent -> descend to children
            for cl, cq in kids:
                visit(cl, cq)
            return
        idx, lo, hi, nb = node
        vis, pxnm = project_bbox(bbox_corners((lo, hi)), pos, right, true_up, fwd,
                                 fov_rad, aspect, height)
        if not vis:                              # subtree is inside this box -> all off-screen
            return
        if lod == 0 or not kids or lod_scales_nm[lod] <= pxnm * tol:
            sel.setdefault(lod, []).append(idx)  # fine enough (or leaf) -> render here
            total += nb
        else:
            for cl, cq in kids:                  # too coarse -> refine
                visit(cl, cq)

    roots = set()                                # coarsest-LOD ancestor of every fragment
    for L, frags in enumerate(per_lod):
        shift = maxlod - L
        for (_idx, _lo, _hi, gp, _nb) in frags:
            roots.add(tuple(c >> shift for c in gp))
    for r in roots:
        visit(maxlod, r)
    return sel, total


def visible_segments(seg_bboxes, pos, look_at, up, fov_rad, aspect, height):
    """{seg_id: nm_per_px} for the segments whose bbox is in the frustum. Off-screen
    segments are dropped (not built/rendered); each kept segment carries its own
    on-screen scale for LOD selection."""
    right, true_up, fwd = camera_basis(pos, look_at, up)
    out: dict[int, float] = {}
    for seg, bbox in seg_bboxes.items():
        vis, nmpp = project_bbox(bbox_corners(bbox), pos, right, true_up, fwd,
                                 fov_rad, aspect, height)
        if vis:
            out[seg] = nmpp
    return out
