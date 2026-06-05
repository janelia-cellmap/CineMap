"""Load precomputed (neuroglancer multilod-draco) meshes via cloud-volume.

The mesh dir's own `info` is a *mesh* info, so we point CloudVolume at its parent
with a fabricated *volume* info whose `mesh` key names the subdir — the validated
approach from the spike. Meshes come back in nm; we export per-segment OBJ files
for Blender import.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from functools import lru_cache

import numpy as np
import trimesh
from cloudvolume import CloudVolume


def _http(url: str) -> str:
    """urllib can't open gs://; map cloud URLs to their https equivalent."""
    if url.startswith("gs://"):
        return "https://storage.googleapis.com/" + url[len("gs://"):]
    if url.startswith("s3://"):
        return "https://s3.amazonaws.com/" + url[len("s3://"):]
    return url


def _segment_ids(mesh_url: str) -> list[int]:
    url = _http(f"{mesh_url.rstrip('/')}/segment_properties/info")
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
            # Two source shapes both expose meshes:
            #  - a precomputed *segmentation* volume whose own info has `scales` and a
            #    `mesh` key (e.g. flyem hemibrain `.../segmentation`) -> read directly.
            #  - a bare neuroglancer *mesh* dir (mesh info, no scales; e.g. cellmap
            #    `.../mesh/.../nuc/`) -> point at the parent with a fabricated volume
            #    info that names this subdir as `mesh`.
            # CloudVolume reads gs://, s3://, https:// info itself; opening a bare mesh
            # dir as a volume raises (no `scales`), which sends us to the fabricate path.
            try:
                direct = CloudVolume(
                    f"precomputed://{self.mesh_url}", use_https=True, progress=False
                )
                if "scales" in direct.info and direct.info.get("mesh"):
                    self._cv = direct
            except Exception:  # noqa: BLE001
                pass
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

    def _draco(self, seg_id: int, lod: int = 0) -> trimesh.Trimesh:
        """Precomputed mesh for `seg_id` at level-of-detail `lod` (0 = finest).
        Falls back to the finest mesh if the source isn't multi-resolution."""
        try:
            m = self.cv.mesh.get(int(seg_id), lod=lod) if lod else self.cv.mesh.get(int(seg_id))
        except TypeError:  # source has no LOD support -> finest only
            m = self.cv.mesh.get(int(seg_id))
        mesh = m[seg_id] if isinstance(m, dict) else m
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False,
        )

    def _max_lod(self, seg_id: int) -> int:
        """Coarsest available LOD index for a multi-resolution mesh (0 if single
        resolution). Reads only the mesh manifest, not geometry."""
        try:
            self.cv.mesh.get(int(seg_id), lod=999)
            return 0
        except TypeError:
            return 0
        except Exception as e:  # noqa: BLE001 — error names the valid range
            m = re.search(r"-?\d+\s*-\s*(\d+)\)", str(e))
            return int(m.group(1)) if m else 0

    def _draco_lod_for_screen(self, seg_id: int, nm_per_px: float, draft: bool,
                              max_verts: float | None = None) -> trimesh.Trimesh | None:
        """Fetch the precomputed mesh at the coarsest LOD that still looks sharp at
        the given on-screen scale (`nm_per_px`). Like neuroglancer: a mesh that's
        small on screen loads coarse, a close-up loads fine. Fetches coarse->fine and
        stops once vertex spacing is finer than ~1-2 px, so little data is wasted."""
        # target world-space vertex spacing that projects to ~px_spacing pixels.
        # The (extent/spacing)^2 budget below treats the mesh as a full sheet, which
        # over-counts for thin neurites, so px_spacing is set generously — large for
        # draft previews (coarse, fast), tighter for the final video.
        px_spacing = 8.0 if draft else 2.0
        spacing_nm = max(px_spacing * nm_per_px, 1e-6)
        target = None
        chosen = None
        for lod in range(self._max_lod(seg_id), -1, -1):  # coarse -> fine
            try:
                mesh = self._draco(seg_id, lod=lod)
            except Exception as e:  # noqa: BLE001
                print(f"[mesh] {seg_id} lod{lod} failed: {e}")
                continue
            if target is None:  # size the screen budget from the (cheap) coarsest mesh
                extent = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
                target = (extent / spacing_nm) ** 2  # ~verts for a surface at that spacing
            # hard budget ceiling: if going this fine would exceed max_verts, keep the
            # previous (coarser, in-budget) LOD instead.
            if max_verts is not None and len(mesh.vertices) > max_verts and chosen is not None:
                break
            chosen = mesh
            if len(mesh.vertices) >= target:  # enough on-screen detail
                break
        return chosen

    def _precomputed(self, seg_id, colorize, nm_per_px, draft, max_verts=None) -> trimesh.Trimesh | None:
        """A single segment's precomputed mesh (LOD-adaptive when `nm_per_px` set,
        capped at `max_verts`), tinted with its neuroglancer color."""
        mesh = (self._draco_lod_for_screen(seg_id, nm_per_px, draft, max_verts=max_verts)
                if nm_per_px else self._draco(seg_id))
        if mesh is None:
            return None
        if colorize is not None:
            r, g, b = colorize(int(seg_id))
            rgba = (np.array([r, g, b, 1.0]) * 255).astype(np.uint8)
            mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        return mesh

    def load(self, seg_id: int, colorize=None, target_voxels: int = 8_000_000,
             nm_per_px: float | None = None, draft: bool = False,
             prefer_labels: bool = False) -> trimesh.Trimesh:
        """Mesh for a segment. By default uses the precomputed mesh (LOD-adaptive,
        matches neuroglancer); with `prefer_labels` (and a label volume available)
        it regenerates a watertight mesh via marching cubes instead. `target_voxels`
        caps the marching-cubes resolution."""
        if prefer_labels and self.label_zarr:
            from .mesh_from_labels import generate

            bbox = self._draco(seg_id).bounds if self.mesh_url else self._label_bbox(seg_id)
            return generate(self.label_zarr, seg_id, (tuple(bbox[0]), tuple(bbox[1])),
                            target_voxels=target_voxels, colorize=colorize)
        if self.mesh_url:
            mesh = self._precomputed(seg_id, colorize, nm_per_px, draft)
            if mesh is not None:
                return mesh
        # no precomputed mesh (or it failed) -> generate from labels if we can
        if self.label_zarr:
            from .mesh_from_labels import generate

            bbox = self._label_bbox(seg_id)
            return generate(self.label_zarr, seg_id, (tuple(bbox[0]), tuple(bbox[1])),
                            target_voxels=target_voxels, colorize=colorize)
        raise ValueError(f"no mesh source for segment {seg_id}")

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

    def _draco_concat(self, seg_ids, colorize=None, nm_per_px=None, draft=False,
                      total_budget=None) -> trimesh.Trimesh | None:
        """Combine the precomputed meshes for `seg_ids` (LOD-adaptive when `nm_per_px`
        is set), each tinted with its neuroglancer color. `total_budget` caps the
        combined vertex count — split evenly across segments — so a layer with many
        segments (e.g. 50 neurons) stays bounded even when the camera zooms in on one
        frame; a single segment can still use the whole budget for a sharp close-up."""
        per_seg = (total_budget / max(1, len(seg_ids))) if total_budget else None
        parts = [self._precomputed(s, colorize, nm_per_px, draft, max_verts=per_seg)
                 for s in seg_ids]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    def load_many(self, seg_ids, colorize=None, target_voxels_single: int = 8_000_000,
                  target_voxels_union: int = 20_000_000, nm_per_px: float | None = None,
                  draft: bool = False, prefer_labels: bool = False) -> trimesh.Trimesh:
        """One mesh for a set of segments. By default downloads the precomputed
        meshes (LOD picked from on-screen scale `nm_per_px`); with `prefer_labels`
        it regenerates watertight meshes from the label volume (a cheap whole-volume
        union when a coarse pyramid level fits, else per-segment). `colorize(seg_id)`
        gives the neuroglancer-matched colors."""
        seg_ids = list(seg_ids)
        if not seg_ids:
            raise ValueError("no segment ids")
        if prefer_labels and self.label_zarr:
            if len(seg_ids) > 1 and self._label_union_fits(target_voxels_union):
                from .mesh_from_labels import generate_union

                return generate_union(self.label_zarr, seg_ids,
                                      target_voxels=target_voxels_union, colorize=colorize)
            # per-segment watertight (single seg, or no coarse pyramid level)
            parts = [self.load(s, colorize=colorize, target_voxels=target_voxels_single,
                               prefer_labels=True) for s in seg_ids]
            return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        # default: precomputed meshes (LOD-adaptive, total vertex budget per layer so
        # a many-segment layer can't balloon when one frame zooms in)
        if self.mesh_url:
            total_budget = 1_200_000 if draft else 5_000_000
            combined = self._draco_concat(seg_ids, colorize=colorize, nm_per_px=nm_per_px,
                                          draft=draft, total_budget=total_budget)
            if combined is not None:
                return combined
        # fallback: no precomputed source -> generate from labels
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
