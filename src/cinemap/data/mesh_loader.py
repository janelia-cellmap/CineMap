"""Load precomputed (neuroglancer multilod-draco) meshes via cloud-volume.

The mesh dir's own `info` is a *mesh* info, so we point CloudVolume at its parent
with a fabricated *volume* info whose `mesh` key names the subdir — the validated
approach from the spike. Meshes come back in nm; we export per-segment OBJ files
for Blender import.
"""
from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache

import numpy as np
import trimesh
from cloudvolume import CloudVolume


def _segment_ids(mesh_url: str) -> list[int]:
    url = f"{mesh_url.rstrip('/')}/segment_properties/info"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.load(r)
        return [int(x) for x in d.get("inline", {}).get("ids", [])]
    except Exception:
        return []


class MeshLoader:
    def __init__(self, mesh_url: str = "", label_zarr: str = ""):
        self.mesh_url = (mesh_url or "").rstrip("/")
        self.label_zarr = (label_zarr or "").rstrip("/")
        if self.mesh_url:
            self.parent, self.subdir = self.mesh_url.rsplit("/", 1)
        self._cv = None

    @property
    def cv(self) -> CloudVolume:
        if self._cv is None:
            info = {
                "@type": "neuroglancer_multiscale_volume",
                "type": "segmentation",
                "data_type": "uint64",
                "num_channels": 1,
                "mesh": self.subdir,
                "scales": [{
                    "key": "s0", "size": [1, 1, 1], "resolution": [8, 8, 8],
                    "chunk_sizes": [[64, 64, 64]], "encoding": "raw", "voxel_offset": [0, 0, 0],
                }],
            }
            self._cv = CloudVolume(
                f"precomputed://{self.parent}", info=info, use_https=True, progress=False
            )
        return self._cv

    @lru_cache(maxsize=1)
    def list_segments(self) -> tuple[int, ...]:
        return tuple(_segment_ids(self.mesh_url)) if self.mesh_url else ()

    def _draco(self, seg_id: int) -> trimesh.Trimesh:
        m = self.cv.mesh.get(seg_id)
        mesh = m[seg_id] if isinstance(m, dict) else m
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False,
        )

    def load(self, seg_id: int, colorize=None, target_voxels: int = 8_000_000) -> trimesh.Trimesh:
        """Clean mesh for a segment.

        Prefers marching cubes on the label volume (gap-free, watertight); uses
        the precomputed draco mesh only for the bounding box, or as a fallback
        when no label volume is available. `target_voxels` caps the marching-cubes
        resolution (lower = faster/coarser, e.g. for draft previews)."""
        if self.label_zarr:
            from .mesh_from_labels import generate

            bbox = self._draco(seg_id).bounds if self.mesh_url else self._label_bbox(seg_id)
            return generate(self.label_zarr, seg_id, (tuple(bbox[0]), tuple(bbox[1])),
                            target_voxels=target_voxels, colorize=colorize)
        return self._draco(seg_id)

    def _label_union_fits(self, target_voxels: int) -> bool:
        """True if the label volume has a whole-volume multiscale level small
        enough to union in memory. Single-scale volumes (no downsampled pyramid)
        return False, so we avoid reading tens of GB and fall back to the
        precomputed meshes instead."""
        if not self.label_zarr:
            return False
        from .slice_loader import get_volume

        try:
            vol = get_volume(self.label_zarr)
        except Exception:  # noqa: BLE001
            return False
        return any(
            s0 * s1 * s2 <= target_voxels
            for (s0, s1, s2) in (vol.level_shape_zyx(l) for l in range(len(vol.level_scale_nm)))
        )

    def _draco_concat(self, seg_ids, colorize=None) -> trimesh.Trimesh | None:
        """Combine the precomputed draco meshes for `seg_ids`, tinting each with
        its neuroglancer color. Fast and memory-bounded (no label volume read)."""
        parts = []
        for s in seg_ids:
            try:
                mesh = self._draco(s)
            except Exception as e:  # noqa: BLE001
                print(f"[mesh] {s} failed: {e}")
                continue
            if colorize is not None:
                r, g, b = colorize(int(s))
                rgba = (np.array([r, g, b, 1.0]) * 255).astype(np.uint8)
                mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
            parts.append(mesh)
        if not parts:
            return None
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    def load_many(self, seg_ids, colorize=None, target_voxels_single: int = 8_000_000,
                  target_voxels_union: int = 20_000_000) -> trimesh.Trimesh:
        """One mesh for a set of segments. A single segment gets the fine
        per-segment surface; many segments use the cheap whole-volume label union
        when the volume has a coarse-enough pyramid level, else the precomputed
        draco meshes (so single-scale volumes don't read tens of GB).
        `colorize(seg_id)->rgb` provides the (neuroglancer-matched) colors. The
        `target_voxels_*` budgets cap resolution (lower = faster draft meshes)."""
        seg_ids = list(seg_ids)
        if not seg_ids:
            raise ValueError("no segment ids")
        if len(seg_ids) == 1:
            return self.load(seg_ids[0], colorize=colorize, target_voxels=target_voxels_single)
        # Preferred: watertight label union, but only when a whole-volume level
        # fits the budget (a real multiscale pyramid).
        if self._label_union_fits(target_voxels_union):
            from .mesh_from_labels import generate_union

            return generate_union(self.label_zarr, seg_ids, target_voxels=target_voxels_union,
                                  colorize=colorize)
        # Otherwise use the precomputed meshes (e.g. single-scale label volumes).
        combined = self._draco_concat(seg_ids, colorize=colorize) if self.mesh_url else None
        if combined is not None:
            return combined
        # Last resort: a label-only layer with no usable pyramid — strided union.
        from .mesh_from_labels import generate_union

        return generate_union(self.label_zarr, seg_ids, target_voxels=target_voxels_union,
                              colorize=colorize)

    def _label_bbox(self, seg_id: int):
        """Bounding box (xyz nm) of a segment found by scanning a coarse label level."""
        from .slice_loader import get_volume

        vol = get_volume(self.label_zarr)
        level = max(0, len(vol.level_scale_nm) - 4)  # a coarse-but-not-tiny level
        arr = np.asarray(vol._open_level(level)[:, :, :].read().result())
        zz, yy, xx = np.where(arr == seg_id)
        if len(zz) == 0:
            raise ValueError(f"segment {seg_id} not found in labels")
        sc = vol.level_scale_nm[level]          # z,y,x
        tr = vol.level_translation_nm[level]    # z,y,x; world = voxel*scale + translation
        lo = (xx.min() * sc[2] + tr[2], yy.min() * sc[1] + tr[1], zz.min() * sc[0] + tr[0])
        hi = ((xx.max() + 1) * sc[2] + tr[2], (yy.max() + 1) * sc[1] + tr[1], (zz.max() + 1) * sc[0] + tr[0])
        return np.array([lo, hi])

    def export_obj(self, seg_id: int, out_path: str) -> str:
        """Load seg_id and write an OBJ (vertices in nm). Returns out_path."""
        mesh = self.load(seg_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh.export(out_path)
        return out_path
