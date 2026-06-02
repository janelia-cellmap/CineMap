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

    def load(self, seg_id: int) -> trimesh.Trimesh:
        """Clean mesh for a segment.

        Prefers marching cubes on the label volume (gap-free, watertight); uses
        the precomputed draco mesh only for the bounding box, or as a fallback
        when no label volume is available.
        """
        if self.label_zarr:
            from .mesh_from_labels import generate

            bbox = self._draco(seg_id).bounds if self.mesh_url else self._label_bbox(seg_id)
            return generate(self.label_zarr, seg_id, (tuple(bbox[0]), tuple(bbox[1])))
        return self._draco(seg_id)

    def load_many(self, seg_ids) -> trimesh.Trimesh:
        """One mesh for a set of segments. A single segment gets the fine
        per-segment surface; many segments use the cheap whole-volume union."""
        seg_ids = list(seg_ids)
        if not seg_ids:
            raise ValueError("no segment ids")
        if len(seg_ids) == 1:
            return self.load(seg_ids[0])
        if self.label_zarr:
            from .mesh_from_labels import generate_union

            return generate_union(self.label_zarr, seg_ids)
        parts = []
        for s in seg_ids:
            try:
                parts.append(self._draco(s))
            except Exception as e:  # noqa: BLE001
                print(f"[mesh] {s} failed: {e}")
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    def _label_bbox(self, seg_id: int):
        """Bounding box (xyz nm) of a segment found by scanning a coarse label level."""
        from .slice_loader import EMVolume

        vol = EMVolume(self.label_zarr)
        level = max(0, len(vol.level_scale_nm) - 4)  # a coarse-but-not-tiny level
        arr = np.asarray(vol._open_level(level)[:, :, :].read().result())
        zz, yy, xx = np.where(arr == seg_id)
        if len(zz) == 0:
            raise ValueError(f"segment {seg_id} not found in labels")
        sc = vol.level_scale_nm[level]  # z,y,x
        lo = (xx.min() * sc[2], yy.min() * sc[1], zz.min() * sc[0])
        hi = ((xx.max() + 1) * sc[2], (yy.max() + 1) * sc[1], (zz.max() + 1) * sc[0])
        return np.array([lo, hi])

    def export_obj(self, seg_id: int, out_path: str) -> str:
        """Load seg_id and write an OBJ (vertices in nm). Returns out_path."""
        mesh = self.load(seg_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh.export(out_path)
        return out_path
