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

import numpy as np
import trimesh
from cloudvolume import CloudVolume

# Default tube radius (nm). Skeletons are 1D, so this is a render choice, not data;
# tuned to read as a visible strand at EM/organelle scale. Override per call.
DEFAULT_RADIUS_NM = 60.0
# Sides per tube cross-section. 6 is a clean low-poly tube; bump for hero closeups.
_SIDES = 6


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
    def __init__(self, skeleton_url: str = "", radius_nm: float = DEFAULT_RADIUS_NM):
        self.skeleton_url = (skeleton_url or "").rstrip("/")
        self.radius_nm = radius_nm
        self.parent, self.subdir = self.skeleton_url.rsplit("/", 1) if self.skeleton_url else ("", "")
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

    def load_many(self, seg_ids, colorize=None, radius_nm: float | None = None) -> trimesh.Trimesh:
        """One combined tube mesh for all `seg_ids`. `colorize(seg_id)->rgb` tints
        each segment's tubes (neuroglancer-matched). Skips segments with no skeleton."""
        seg_ids = list(seg_ids)
        if not seg_ids:
            raise ValueError("no segment ids")
        radius = self.radius_nm if radius_nm is None else radius_nm
        parts: list[trimesh.Trimesh] = []
        for s in seg_ids:
            try:
                skel = self.cv.skeleton.get(int(s))
            except Exception as e:  # noqa: BLE001
                print(f"[skeleton] {s} failed: {e}")
                continue
            verts = np.asarray(skel.vertices, dtype=np.float64)
            edges = np.asarray(skel.edges, dtype=np.int64)
            if len(verts) == 0 or len(edges) == 0:
                continue
            rgba = None
            if colorize is not None:
                r, g, b = colorize(int(s))
                rgba = np.tile((np.array([r, g, b, 1.0]) * 255).astype(np.uint8), (len(edges), 1))
            tube = edges_to_tubes(verts, edges, radius, rgba=rgba)
            if tube is not None:
                parts.append(tube)
        if not parts:
            raise ValueError("no skeleton geometry for the selected segments")
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
