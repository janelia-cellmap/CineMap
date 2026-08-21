"""Load precomputed neuroglancer skeletons and turn them into renderable tubes.

A skeleton source's own `info` is a `neuroglancer_skeletons` info (no volume
`scales`), so — exactly like the mesh loader — we point CloudVolume at the parent
dir with a fabricated *volume* info whose `skeletons` key names the subdir, then
read per-segment skeletons via `cv.skeleton.get`.

Skeletons are vertices (nm, x/y/z) + edges. Neuroglancer draws them as screen-space
lines; for a 3D render we sweep each edge into a thin cylinder so the skeleton has
real geometry the camera can orbit. Tubes are built vectorized (one combined mesh
for a whole layer's selected segments) to stay fast across thousands of segments.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh
from cloudvolume import CloudVolume

# concurrent per-segment fetches; logical CPU count (incl. hyperthreading) by default
_FETCH_WORKERS = int(os.environ.get("CINEMAP_FETCH_WORKERS") or (os.cpu_count() or 8))

# Default tube radius (nm). Skeletons are 1D, so this is a render choice, not data;
# tuned to read as a visible strand at EM/organelle scale. Override per call.
DEFAULT_RADIUS_NM = 60.0
# Sides per tube cross-section. 6 is a clean low-poly tube; bump for hero closeups.
_SIDES = 6


class ShaderColormap:
    """A piecewise-smoothstep colormap recovered from a neuroglancer skeleton
    `skeletonRendering.shader`. Matches the common cellmap pattern:

        emitRGB(<map>(min(1.0, <attr>/<norm>)))

    where <map> is built from `const float eN = ...; const vec3 vN = vec3(...)`
    control points and consecutive stops are blended with `smoothstep`. So skeleton
    tubes get the same per-vertex coloring the user sees in neuroglancer instead of
    a flat segment color."""

    def __init__(self, attr: str, norm: float, edges: list[float], colors: list[list[float]]):
        self.attr = attr
        self.norm = norm or 1.0
        self.edges = np.asarray(edges, dtype=np.float64)
        self.colors = np.asarray(colors, dtype=np.float64)  # (S,3) in 0..1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Map normalized values `x` (already divided by norm) to (N,3) rgb."""
        x = np.clip(np.asarray(x, dtype=np.float64), self.edges[0], self.edges[-1])
        out = np.tile(self.colors[0], (len(x), 1))
        for i in range(len(self.edges) - 1):
            e0, e1 = self.edges[i], self.edges[i + 1]
            seg = (x >= e0) & (x <= e1)
            if not seg.any():
                continue
            t = np.clip((x[seg] - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
            a = t * t * (3 - 2 * t)  # smoothstep
            out[seg] = self.colors[i] * (1 - a)[:, None] + self.colors[i + 1] * a[:, None]
        return out


def parse_shader_colormap(shader: str) -> ShaderColormap | None:
    """Recover a ShaderColormap from a skeleton shader, or None if it doesn't match
    the recognized colormap pattern (then we fall back to flat segment colors)."""
    if not shader:
        return None
    floats = dict(re.findall(r"float\s+(\w+)\s*=\s*([\d.eE+f-]+)", shader))
    edges, colors = [], []
    i = 0
    while f"e{i}" in floats:
        m = re.search(rf"vec3\s+v{i}\s*=\s*vec3\(([^)]+)\)", shader)
        if not m:
            break
        edges.append(float(floats[f"e{i}"].rstrip("f")))
        colors.append([float(c) for c in m.group(1).split(",")[:3]])
        i += 1
    if len(edges) < 2:
        return None
    # attribute and its normalization, from e.g. `lsp_nm/norm` with `float norm = 50000.0f`
    m = re.search(r"(\w+)\s*/\s*(\w+|[\d.eE+f-]+)", shader)
    if not m:
        return None
    attr, denom = m.group(1), m.group(2)
    norm = float(floats.get(denom, denom).rstrip("f")) if denom else 1.0
    return ShaderColormap(attr, norm, edges, colors)


def _perp_frame(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each unit direction `d` (E,3)."""
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(d), 1))
    # where d is ~parallel to z, use x as the reference instead
    nearly_z = np.abs(d[:, 2]) > 0.9
    ref[nearly_z] = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    v = np.cross(d, u)
    return u, v


def edges_to_tubes(verts: np.ndarray, edges: np.ndarray, radius: float,
                   rgba: np.ndarray | None = None, sides: int = _SIDES) -> trimesh.Trimesh | None:
    """Sweep each (p0,p1) edge into a `sides`-gon cylinder. `rgba` (E,4 uint8) tints
    each edge. Returns one combined Trimesh (vertices in the same nm frame), or None
    when there is no non-degenerate edge."""
    if len(edges) == 0:
        return None
    p0 = verts[edges[:, 0]].astype(np.float64)
    p1 = verts[edges[:, 1]].astype(np.float64)
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    keep = length > 1e-6
    if not keep.any():
        return None
    p0, p1, d, length = p0[keep], p1[keep], d[keep], length[keep]
    if rgba is not None:
        rgba = rgba[keep]
    d /= length[:, None]
    u, v = _perp_frame(d)  # (E,3) each

    e = len(p0)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = (np.cos(theta)[:, None] * u[:, None, :] +
            np.sin(theta)[:, None] * v[:, None, :])  # (E, sides, 3)
    ring0 = p0[:, None, :] + radius * ring           # (E, sides, 3)
    ring1 = p1[:, None, :] + radius * ring
    verts_out = np.concatenate([ring0, ring1], axis=1).reshape(-1, 3)  # (E*2*sides, 3)

    # faces for one edge's cylinder wall (two triangles per side quad)
    k = np.arange(sides)
    kn = (k + 1) % sides
    quad = np.stack([k, kn, sides + kn, sides + k], axis=1)            # (sides, 4)
    tri = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]], axis=0)  # (2*sides, 3)
    offsets = (np.arange(e) * 2 * sides)[:, None, None]
    faces_out = (tri[None] + offsets).reshape(-1, 3)

    colors = None
    if rgba is not None:
        colors = np.repeat(rgba, 2 * sides, axis=0)  # per-vertex from per-edge color
    return trimesh.Trimesh(vertices=verts_out, faces=faces_out, vertex_colors=colors,
                           process=False)


class SkeletonLoader:
    def __init__(self, skeleton_url: str = "", radius_nm: float = DEFAULT_RADIUS_NM,
                 shader: str = "", shader_controls: dict | None = None):
        self.skeleton_url = (skeleton_url or "").rstrip("/")
        self.radius_nm = radius_nm
        self.parent, self.subdir = self.skeleton_url.rsplit("/", 1) if self.skeleton_url else ("", "")
        self.colormap = parse_shader_colormap(shader)  # None if shader has no colormap
        self.shader_src = shader or ""
        self.shader_controls: dict = dict(shader_controls or {})
        self._warned = False
        self._cv = None

    @property
    def cv(self) -> CloudVolume:
        if self._cv is None:
            info = {
                "@type": "neuroglancer_multiscale_volume",
                "type": "segmentation",
                "data_type": "uint64",
                "num_channels": 1,
                "skeletons": self.subdir,
                "scales": [{
                    "key": "s0", "size": [1, 1, 1], "resolution": [1, 1, 1],
                    "chunk_sizes": [[64, 64, 64]], "encoding": "raw", "voxel_offset": [0, 0, 0],
                }],
            }
            self._cv = CloudVolume(
                f"precomputed://{self.parent}", info=info, use_https=True, progress=False
            )
        return self._cv

    def _edge_colors(self, skel, edges, seg_id, colorize) -> np.ndarray | None:
        """Per-edge rgba (E,4 uint8), matching what neuroglancer paints on the skeleton.

        Preference order:
          1. Interpret the layer's `skeletonRendering.shader` as GLSL over the skeleton's
             own vertex properties (`prop_radius()` and friends). This handles any shader,
             including the cellmap `turbo(log(prop_radius()))` ones.
          2. The legacy piecewise-smoothstep colormap matcher, kept because it needs no
             property lookup and covers the shaders it was written for.
          3. The flat per-segment colour.

        The attribute is averaged per EDGE and shaded once, rather than shading each
        vertex: we sweep one tube per edge, so the tube gets a single colour either way.
        """
        rgba = self._shader_edge_colors(skel, edges)
        if rgba is not None:
            return rgba
        cm = self.colormap
        if cm is not None and hasattr(skel, cm.attr):
            attr = np.asarray(getattr(skel, cm.attr), dtype=np.float64).reshape(-1)
            ev = (attr[edges[:, 0]] + attr[edges[:, 1]]) * 0.5 / cm.norm  # per-edge, normalized
            rgb = cm(ev)
            rgba = np.empty((len(edges), 4), dtype=np.uint8)
            rgba[:, :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            rgba[:, 3] = 255
            return rgba
        if colorize is not None:
            r, g, b = colorize(int(seg_id))
            return np.tile((np.array([r, g, b, 1.0]) * 255).astype(np.uint8), (len(edges), 1))
        return None

    def _shader_edge_colors(self, skel, edges) -> np.ndarray | None:
        """Run the layer's GLSL over the skeleton's vertex properties; None if we can't.

        Returns None (rather than raising) whenever the shader declares no properties, the
        skeleton is missing one it reads, or the GLSL is outside the supported subset --
        the caller then falls back. The reason is logged ONCE per loader so a shader we
        cannot interpret is visible in the log instead of silently rendering flat.
        """
        from .ng_shader import shade_properties, shader_property_names

        if not self.shader_src:
            return None
        names = shader_property_names(self.shader_src)
        if not names:
            return None
        props = {}
        for name in names:
            if not hasattr(skel, name):
                self._warn_shader(f"skeleton has no vertex property {name!r}")
                return None
            vals = np.asarray(getattr(skel, name), dtype=np.float64).reshape(-1)
            props[name] = (vals[edges[:, 0]] + vals[edges[:, 1]]) * 0.5   # per-edge
        rgb, warn = shade_properties(self.shader_src, props, self.shader_controls)
        if warn:
            self._warn_shader(warn)
            return None
        rgba = np.empty((len(edges), 4), dtype=np.uint8)
        rgba[:, :3] = rgb
        rgba[:, 3] = 255
        return rgba

    def _warn_shader(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"[skeleton] {self.subdir}: shader not applied ({msg}); "
                  f"falling back to the flat segment color")

    def load_many(self, seg_ids, colorize=None, radius_nm: float | None = None) -> trimesh.Trimesh:
        """One combined tube mesh for all `seg_ids`. Colors each segment via the
        shader colormap (per-vertex attribute) when the layer has one, else the flat
        `colorize(seg_id)->rgb`. Skips segments with no skeleton."""
        seg_ids = list(seg_ids)
        if not seg_ids:
            raise ValueError("no segment ids")
        radius = self.radius_nm if radius_nm is None else radius_nm

        def _fetch(s):  # network + tube build per segment, run concurrently
            try:
                skel = self.cv.skeleton.get(int(s))
            except Exception as e:  # noqa: BLE001
                print(f"[skeleton] {s} failed: {e}")
                return None
            verts = np.asarray(skel.vertices, dtype=np.float64)
            edges = np.asarray(skel.edges, dtype=np.int64)
            if len(verts) == 0 or len(edges) == 0:
                return None
            rgba = self._edge_colors(skel, edges, s, colorize)
            return edges_to_tubes(verts, edges, radius, rgba=rgba)

        if len(seg_ids) > 1:
            with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(seg_ids))) as ex:
                parts = list(ex.map(_fetch, seg_ids))
        else:
            parts = [_fetch(seg_ids[0])]
        parts = [p for p in parts if p is not None]
        if not parts:
            raise ValueError("no skeleton geometry for the selected segments")
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
