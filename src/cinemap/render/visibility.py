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
