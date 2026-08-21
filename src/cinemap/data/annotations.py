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
                        line_radius_nm: float = DEFAULT_LINE_RADIUS_NM,
                        styles: dict | None = None) -> trimesh.Trimesh | None:
    """Combine all primitives in `prims` (nm) into one Trimesh, or None if nothing to draw.

    `styles` optionally carries the annotation shader's result: {kind: (N, 4) uint8 RGBA}
    for kind in point/line/box/ellipsoid, one row per primitive of that kind, in the same
    order as the geometry lists. Without it every primitive takes the flat `color_rgb`,
    which is what neuroglancer's default shader (`setColor(defaultColor())`) produces.
    """
    r, g, b = color_rgb
    # +0.5 then truncate, exactly as ng_shader quantizes: plain truncation lost a step
    # (0.30 -> 76 instead of 77), and np.rint rounds half to EVEN, which disagrees with
    # the shader path at exactly .5. An identical colour must not depend on which path
    # produced it.
    flat = np.clip(np.array([r, g, b, 1.0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    styles = styles or {}

    def color_of(kind: str, i: int) -> np.ndarray:
        arr = styles.get(kind)
        if arr is None or i >= len(arr):
            return flat
        return np.asarray(arr[i], dtype=np.uint8)

    parts: list[trimesh.Trimesh] = []

    for i, p in enumerate(prims.get("points", [])):
        parts.append(_sphere(p, point_radius_nm, color_of("point", i)))

    for i, c in enumerate(prims.get("ellipsoids", [])):
        parts.append(_sphere(c["center"], c["radii"], color_of("ellipsoid", i)))

    # lines and box edges are both tube sweeps; a box contributes 12 edges, all of which
    # take that box's colour, so the per-edge colour array has to be expanded to match.
    line_verts, line_edges, edge_rgba = [], [], []
    for i, (a, b_) in enumerate(prims.get("lines", [])):
        base = len(line_verts)
        line_verts.extend([a, b_])
        line_edges.append([base, base + 1])
        edge_rgba.append(color_of("line", i))
    for i, (lo, hi) in enumerate(prims.get("boxes", [])):
        base = len(line_verts)
        line_verts.extend(_box_corners(lo, hi).tolist())
        line_edges.extend((_BOX_EDGES + base).tolist())
        edge_rgba.extend([color_of("box", i)] * len(_BOX_EDGES))
    if line_edges:
        tube = edges_to_tubes(np.asarray(line_verts, dtype=np.float64),
                              np.asarray(line_edges, dtype=np.int64),
                              line_radius_nm,
                              rgba=np.asarray(edge_rgba, dtype=np.uint8))
        if tube is not None:
            parts.append(tube)

    if not parts:
        return None
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def _prop_specs(layer: dict) -> list[dict]:
    """The layer's `annotationProperties`, which give each entry of an annotation's
    positional `props` array its id and type (annotation/index.ts:1349)."""
    out = []
    for spec in layer.get("annotationProperties") or []:
        pid = spec.get("id")
        if pid:
            out.append({"id": pid, "type": spec.get("type", "float32"),
                        "default": spec.get("default", 0)})
    return out


def _prop_value(raw, ptype: str):
    """One property value as the shader sees it.

    `rgb`/`rgba` are JSON colour strings that neuroglancer uploads as NORMALIZED bytes
    (annotation/type_handler.ts:174-185), so they become floats in 0-1; every other type
    is passed through at its raw magnitude, which is what the integer and float32
    handlers do.
    """
    if ptype in ("rgb", "rgba"):
        n = 3 if ptype == "rgb" else 4
        if isinstance(raw, str):
            h = raw.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            vals = [int(h[i:i + 2], 16) / 255.0 for i in range(0, min(len(h), 2 * n), 2)]
        else:
            vals = [float(x) for x in (raw or [])]
        return (vals + [1.0] * n)[:n]
    if ptype == "bool":
        return 1.0 if raw else 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def parse_inline(layer: dict, voxel_size_nm, perm=(0, 1, 2)) -> dict:
    """Normalized primitive dict (nm, xyz) from a state layer's inline `annotations`.
    Annotation coords are in the NG dimension order/units, so `voxel_size_nm` is the
    per-dimension scale (from the NG state, not the manifest) and `perm` reorders a
    dimension-ordered vector to (x, y, z) — same convention as the camera path.

    Also collects each annotation's `props` under `out["props"][kind][property_id]`, in
    the same order as the geometry lists, so the layer's annotation shader can be
    evaluated per annotation the way neuroglancer evaluates it per vertex.
    """
    vox = [float(v) for v in voxel_size_nm]
    specs = _prop_specs(layer)

    def to_nm(p):
        scaled = [float(p[i]) * vox[i] for i in range(3)]   # NG dimension order
        return [scaled[perm[0]], scaled[perm[1]], scaled[perm[2]]]  # -> x, y, z

    def radii_nm(r):
        scaled = [abs(float(r[i])) * vox[i] for i in range(3)]
        return [scaled[perm[0]], scaled[perm[1]], scaled[perm[2]]]

    out: dict = {"points": [], "lines": [], "boxes": [], "ellipsoids": []}
    props: dict = {k: {sp["id"]: [] for sp in specs}
                   for k in ("points", "lines", "boxes", "ellipsoids")}

    def record(kind: str, a: dict) -> None:
        raw = a.get("props")
        for i, sp in enumerate(specs):
            v = raw[i] if isinstance(raw, list) and i < len(raw) else sp["default"]
            props[kind][sp["id"]].append(_prop_value(v, sp["type"]))

    for a in layer.get("annotations") or []:
        t = a.get("type")
        if t == "point" and a.get("point"):
            out["points"].append(to_nm(a["point"]))
            record("points", a)
        elif t == "line" and a.get("pointA") and a.get("pointB"):
            out["lines"].append([to_nm(a["pointA"]), to_nm(a["pointB"])])
            record("lines", a)
        elif t == "axis_aligned_bounding_box" and a.get("pointA") and a.get("pointB"):
            A, B = to_nm(a["pointA"]), to_nm(a["pointB"])
            lo = [min(A[i], B[i]) for i in range(3)]
            hi = [max(A[i], B[i]) for i in range(3)]
            out["boxes"].append([lo, hi])
            record("boxes", a)
        elif t == "ellipsoid" and a.get("center") and a.get("radii"):
            out["ellipsoids"].append({
                "center": to_nm(a["center"]),
                "radii": radii_nm(a["radii"]),
            })
            record("ellipsoids", a)
    out["props"] = props
    return out


def has_geometry(prims: dict) -> bool:
    return any(prims.get(k) for k in ("points", "lines", "boxes", "ellipsoids"))
