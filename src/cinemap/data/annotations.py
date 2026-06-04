"""Turn neuroglancer annotations (points / lines / boxes / ellipsoids) into
renderable geometry.

Neuroglancer draws annotations as screen-space overlays; for a 3D render we give
each one real geometry the camera can orbit: points -> spheres, lines -> tubes,
axis-aligned bounding boxes -> wireframe tubes, ellipsoids -> scaled spheres.

This module is source-agnostic: it works on a *normalized* primitive dict (all
coordinates already in nm, x/y/z). `parse_inline` builds that dict from the
annotations embedded in a neuroglancer state layer; a precomputed-source reader
can produce the same dict and reuse `annotations_to_mesh`.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .skeleton import edges_to_tubes

DEFAULT_POINT_RADIUS_NM = 80.0
DEFAULT_LINE_RADIUS_NM = 40.0

# the 12 edges of a box given its 8 corners (order matches _box_corners below)
_BOX_EDGES = np.array([
    [0, 1], [1, 3], [3, 2], [2, 0],   # bottom face (z=lo)
    [4, 5], [5, 7], [7, 6], [6, 4],   # top face (z=hi)
    [0, 4], [1, 5], [2, 6], [3, 7],   # verticals
])


def _box_corners(lo, hi) -> np.ndarray:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
    ], dtype=np.float64)


def _sphere(center, radii, rgba, subdivisions: int = 2) -> trimesh.Trimesh:
    """Icosphere at `center` scaled by `radii` (scalar or [rx,ry,rz]), tinted rgba."""
    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    s.vertices = s.vertices * np.asarray(radii, dtype=np.float64) + np.asarray(center, dtype=np.float64)
    s.visual.vertex_colors = np.tile(rgba, (len(s.vertices), 1))
    return s


def annotations_to_mesh(prims: dict, color_rgb, point_radius_nm: float = DEFAULT_POINT_RADIUS_NM,
                        line_radius_nm: float = DEFAULT_LINE_RADIUS_NM) -> trimesh.Trimesh | None:
    """Combine all primitives in `prims` (nm) into one tinted Trimesh, or None if
    there's nothing to draw."""
    r, g, b = color_rgb
    rgba = (np.array([r, g, b, 1.0]) * 255).astype(np.uint8)
    parts: list[trimesh.Trimesh] = []

    for p in prims.get("points", []):
        parts.append(_sphere(p, point_radius_nm, rgba))

    for c in prims.get("ellipsoids", []):
        parts.append(_sphere(c["center"], c["radii"], rgba))

    # lines and box edges are both tube sweeps
    line_verts, line_edges = [], []
    for a, b_ in prims.get("lines", []):
        base = len(line_verts)
        line_verts.extend([a, b_])
        line_edges.append([base, base + 1])
    for lo, hi in prims.get("boxes", []):
        base = len(line_verts)
        line_verts.extend(_box_corners(lo, hi).tolist())
        line_edges.extend((_BOX_EDGES + base).tolist())
    if line_edges:
        tube = edges_to_tubes(np.asarray(line_verts, dtype=np.float64),
                              np.asarray(line_edges, dtype=np.int64),
                              line_radius_nm,
                              rgba=np.tile(rgba, (len(line_edges), 1)))
        if tube is not None:
            parts.append(tube)

    if not parts:
        return None
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def parse_inline(layer: dict, voxel_size_nm) -> dict:
    """Normalized primitive dict (nm) from a state layer's inline `annotations`.
    Annotation coords are in the layer's voxel space, so we scale by voxel size."""
    vx, vy, vz = (float(v) for v in voxel_size_nm)

    def to_nm(p):
        return [float(p[0]) * vx, float(p[1]) * vy, float(p[2]) * vz]

    out: dict = {"points": [], "lines": [], "boxes": [], "ellipsoids": []}
    for a in layer.get("annotations") or []:
        t = a.get("type")
        if t == "point" and a.get("point"):
            out["points"].append(to_nm(a["point"]))
        elif t == "line" and a.get("pointA") and a.get("pointB"):
            out["lines"].append([to_nm(a["pointA"]), to_nm(a["pointB"])])
        elif t == "axis_aligned_bounding_box" and a.get("pointA") and a.get("pointB"):
            A, B = to_nm(a["pointA"]), to_nm(a["pointB"])
            lo = [min(A[i], B[i]) for i in range(3)]
            hi = [max(A[i], B[i]) for i in range(3)]
            out["boxes"].append([lo, hi])
        elif t == "ellipsoid" and a.get("center") and a.get("radii"):
            out["ellipsoids"].append({
                "center": to_nm(a["center"]),
                "radii": [abs(float(a["radii"][0])) * vx, abs(float(a["radii"][1])) * vy,
                          abs(float(a["radii"][2])) * vz],
            })
    return out


def has_geometry(prims: dict) -> bool:
    return any(prims.get(k) for k in ("points", "lines", "boxes", "ellipsoids"))
