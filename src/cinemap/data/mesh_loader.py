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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# segments are fetched concurrently (cloud-volume .get is network + draco decode =
# I/O-bound, GIL released), a big win for many-segment layers (e.g. thousands of mitos).
# Default to logical CPU count (includes hyperthreading); CINEMAP_FETCH_WORKERS overrides.
_FETCH_WORKERS = int(os.environ.get("CINEMAP_FETCH_WORKERS") or (os.cpu_count() or 8))

import numpy as np
import trimesh
from cloudvolume import CloudVolume


def _tune_http_pool():
    """Size cloudfiles' shared HTTP connection pool to our parallel fetching, so it
    doesn't churn connections under many workers (the 'Connection pool is full'
    urllib3 warnings) — both a perf fix and silences the noise. Best-effort."""
    pool = max(32, _FETCH_WORKERS * 2)
    try:
        import requests
        from cloudfiles.interfaces import HttpInterface
        HttpInterface.adaptor = requests.adapters.HTTPAdapter(
            pool_connections=pool, pool_maxsize=pool)
    except Exception:  # noqa: BLE001
        pass
    import logging  # also drop just this one message if anything else still emits it

    class _PoolFilter(logging.Filter):
        def filter(self, record):
            return "Connection pool is full" not in record.getMessage()

    logging.getLogger("urllib3.connectionpool").addFilter(_PoolFilter())


_tune_http_pool()


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
    def __init__(self, mesh_url: str = "", label_zarr: str = "", cache_dir=None):
        self.mesh_url = (mesh_url or "").rstrip("/")
        self.label_zarr = (label_zarr or "").rstrip("/")
        self.parent, self.subdir = (
            self.mesh_url.rsplit("/", 1) if "/" in self.mesh_url else ("", self.mesh_url))
        self._cv = None
        self._manual = None   # lazily: does this source need our model-space draco decode?
        # persistent on-disk cache of RAW (uncolored) per-(segment, LOD) geometry,
        # keyed by (mesh_url, seg, lod) — so re-rendering at a different quality/zoom
        # only downloads the genuinely-new finer LODs and reuses the rest.
        self._cache_dir = str(cache_dir) if cache_dir else None

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

    def _manifest_bits(self) -> int:
        try:
            return int(self.cv.mesh.meta.info.get("vertex_quantization_bits", 16))
        except Exception:  # noqa: BLE001
            return 16

    def _needs_manual_decode(self) -> bool:
        """Some multilod exports bake the dequantization transform into the draco stream,
        so the decoded points are already in model space (range ~chunk_shape) instead of
        the standard integer range [0, 2^bits). cloud-volume assumes integers and re-scales
        by chunk_shape -> double-scaling that balloons coarse LODs. Detect it by sampling
        one fragment: points beyond the quantization range mean it's the model-space kind."""
        if self._manual is None:
            self._manual = False
            try:
                seg = int(self.list_segments()[0])
                ml = self._max_lod(seg)
                if ml > 0:
                    # a well-formed multi-LOD mesh has the SAME extent at every LOD.
                    # This encoding makes cloud-volume scale each coarser LOD up, so the
                    # coarsest is far bigger than the finest -> our cue to decode manually.
                    v0 = np.asarray(self._cv_mesh(seg, 0).vertices); e0 = v0.max(0) - v0.min(0)
                    vN = np.asarray(self._cv_mesh(seg, ml).vertices); eN = vN.max(0) - vN.min(0)
                    self._manual = bool(np.any(eN > 1.5 * e0 + 1.0))
            except Exception:  # noqa: BLE001  (sharded / unreachable -> trust cloud-volume)
                self._manual = False
        return self._manual

    def _cv_mesh(self, seg_id: int, lod: int = 0):
        """Raw cloud-volume mesh for a segment at a LOD (no manual-decode dispatch)."""
        try:
            m = self.cv.mesh.get(int(seg_id), lod=lod) if lod else self.cv.mesh.get(int(seg_id))
        except TypeError:  # source has no LOD support
            m = self.cv.mesh.get(int(seg_id))
        return m[seg_id] if isinstance(m, dict) else m

    def _draco_manual(self, seg_id: int) -> trimesh.Trimesh:
        """Decode an unsharded multilod mesh whose draco points are already in model
        space: vertex = grid_origin + fragment_position*chunk_shape + points (finest LOD).
        Avoids cloud-volume's double-scaling for this encoding."""
        import struct

        import DracoPy
        idx = urllib.request.urlopen(_http(f"{self.mesh_url}/{int(seg_id)}.index"), timeout=30).read()
        o = 0
        cs = np.array(struct.unpack("<3f", idx[o:o + 12])); o += 12
        go = np.array(struct.unpack("<3f", idx[o:o + 12])); o += 12
        nl = struct.unpack("<I", idx[o:o + 4])[0]; o += 4
        o += 4 * nl + 12 * nl                                  # lod_scales + vertex_offsets
        nfrag = struct.unpack(f"<{nl}I", idx[o:o + 4 * nl]); o += 4 * nl
        data = urllib.request.urlopen(_http(f"{self.mesh_url}/{int(seg_id)}"), timeout=120).read()
        n = nfrag[0]                                           # finest LOD = first in the file
        fpos = np.array(struct.unpack(f"<{3 * n}I", idx[o:o + 12 * n])).reshape(3, n).T; o += 12 * n
        fsz = np.array(struct.unpack(f"<{n}I", idx[o:o + 4 * n])); o += 4 * n
        dp = 0; V = []; F = []; nv = 0
        for i in range(n):
            b = data[dp:dp + fsz[i]]; dp += fsz[i]
            if not fsz[i]:
                continue
            mm = DracoPy.decode(b)
            v = go + fpos[i] * cs + np.asarray(mm.points, float)   # points already model-space
            f = np.asarray(mm.faces, np.int64) + nv
            V.append(v); F.append(f); nv += len(v)
        return trimesh.Trimesh(vertices=np.vstack(V), faces=np.vstack(F), process=False)

    def _draco(self, seg_id: int, lod: int = 0) -> trimesh.Trimesh:
        """Precomputed mesh for `seg_id` at level-of-detail `lod` (0 = finest).
        Falls back to the finest mesh if the source isn't multi-resolution. Raw
        geometry is disk-cached (when a cache dir is set) so it's downloaded once."""
        cache = None
        if self._cache_dir:
            import hashlib
            key = hashlib.md5(f"{self.mesh_url}|{int(seg_id)}|{int(lod)}".encode()).hexdigest()
            cache = os.path.join(self._cache_dir, f"{key}.ply")
            if os.path.exists(cache):
                try:
                    return trimesh.load(cache, process=False)
                except Exception:  # noqa: BLE001  (corrupt cache entry -> re-fetch)
                    pass
        if self._needs_manual_decode():
            out = self._draco_manual(seg_id)   # always finest LOD (correct + consistent)
            if cache:
                try:
                    os.makedirs(self._cache_dir, exist_ok=True); out.export(cache)
                except Exception:  # noqa: BLE001
                    pass
            return out
        mesh = self._cv_mesh(seg_id, lod)
        out = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False,
        )
        if cache:
            try:
                os.makedirs(self._cache_dir, exist_ok=True)
                out.export(cache)
            except Exception:  # noqa: BLE001  (cache write best-effort)
                pass
        return out

    @lru_cache(maxsize=8192)
    def _max_lod(self, seg_id: int) -> int:
        """Coarsest available LOD index for a multi-resolution mesh (0 if single
        resolution). Network-probes the manifest, so it's cached per segment (it was
        being re-probed for every LOD pick — a big chunk-mode slowdown)."""
        try:
            self.cv.mesh.get(int(seg_id), lod=999)
            return 0
        except TypeError:
            return 0
        except Exception as e:  # noqa: BLE001 — error names the valid range
            m = re.search(r"-?\d+\s*-\s*(\d+)\)", str(e))
            return int(m.group(1)) if m else 0

    def seg_bbox(self, seg_id: int):
        """World-space ((x0,y0,z0),(x1,y1,z1)) bbox in nm from the coarsest LOD (a
        cheap fetch) — used by the view-aware ('chunk') LOD mode to frustum-cull and
        size each segment on screen. None if the segment can't be read."""
        try:
            b = self._draco(int(seg_id), lod=self._max_lod(seg_id)).bounds
            return ((float(b[0][0]), float(b[0][1]), float(b[0][2])),
                    (float(b[1][0]), float(b[1][1]), float(b[1][2])))
        except Exception as e:  # noqa: BLE001
            print(f"[mesh] bbox {seg_id} failed: {e}")
            return None

    @lru_cache(maxsize=4096)
    def fragment_boxes(self, seg_id: int):
        """The mesh's octree as per-LOD fragments with exact WORLD (nm) bounding
        boxes — the basis for true per-chunk LOD (frustum-cull + per-fragment screen
        size, like neuroglancer). Uses cloud-volume's own decode (multilod.py):

            model = grid_origin + vertex_offsets[lod] + chunk_shape·2^lod·[pos, pos+1]
            nm    = info `transform` · model        (transform carries the resolution)

        No calibration — the resolution is read from the mesh info. EMPTY fragments
        (byte size 0) are skipped, matching cloud-volume's decoder, and the index is
        the position in the NON-EMPTY sequence so it lines up with get(...,concat=
        False). Returns (per_lod, lod_scales_nm): per_lod[lod] is a list of
        (frag_index, lo_nm(3,), hi_nm(3,), grid_pos, n_bytes); lod_scales_nm is each
        LOD's spatial resolution in nm."""
        m = self.cv.mesh
        T = np.asarray(m.transform, float)               # 4x4, resolution baked in
        man = m.get_manifest(int(seg_id))
        go = np.asarray(man.grid_origin, float)
        cs = np.asarray(man.chunk_shape, float)
        vo = np.asarray(man.vertex_offsets, float)
        scale = float(abs(np.linalg.det(T[:3, :3])) ** (1.0 / 3.0))  # nm per model unit
        per_lod = []
        for lod in range(man.num_lods):
            cell = cs * (2 ** lod)
            base = go + vo[lod]
            offs = np.asarray(man.fragment_offsets[lod])
            positions = np.asarray(man.fragment_positions[lod], float)
            frags = []
            ni = 0                                        # index among NON-empty fragments
            for idx, p in enumerate(positions):
                if offs[idx] == 0:                        # empty -> decoder skips it
                    continue
                lo_m, hi_m = base + cell * p, base + cell * (p + 1)
                corners = np.array([[x, y, z] for x in (lo_m[0], hi_m[0])
                                    for y in (lo_m[1], hi_m[1]) for z in (lo_m[2], hi_m[2])])
                w = (T[:3, :3] @ corners.T).T + T[:3, 3]
                frags.append((ni, w.min(0), w.max(0), tuple(int(v) for v in p), int(offs[idx])))
                ni += 1
            per_lod.append(frags)
        lod_scales_nm = np.asarray(man.lod_scales, float) * scale
        return per_lod, lod_scales_nm

    def _frag_cache_path(self, seg_id: int, lod: int, idx: int):
        if not self._cache_dir:
            return None
        import hashlib
        key = hashlib.md5(f"{self.mesh_url}|{int(seg_id)}|f|{int(lod)}|{int(idx)}".encode()).hexdigest()
        return os.path.join(self._cache_dir, f"{key}.ply")

    def get_fragments(self, seg_id: int, selection: dict, colorize=None) -> trimesh.Trimesh | None:
        """Fetch only the SELECTED octree fragments and combine them. `selection` is
        {lod: [frag_index, ...]} (indices from fragment_boxes). Each RAW fragment is
        disk-cached individually (per mesh_url/seg/lod/index), so a re-render or a
        different zoom reuses fragments already fetched; a whole-LOD download happens
        only when some needed fragment isn't cached yet. Fine where near, coarse where
        far — like neuroglancer."""
        parts = []
        for lod, idxs in selection.items():
            idxs = sorted({int(i) for i in (idxs or [])})
            if not idxs:
                continue
            meshes: dict[int, trimesh.Trimesh] = {}      # idx -> raw (uncolored)
            need_fetch = False
            for i in idxs:
                cp = self._frag_cache_path(seg_id, lod, i)
                if cp and os.path.exists(cp):
                    try:
                        meshes[i] = trimesh.load(cp, process=False)
                    except Exception:  # noqa: BLE001
                        need_fetch = True
                else:
                    need_fetch = True
            if need_fetch:
                try:
                    got = self.cv.mesh.get(int(seg_id), lod=int(lod), concat=False)
                except Exception as e:  # noqa: BLE001
                    print(f"[chunk] {seg_id} lod{lod} frags failed: {e}")
                    got = None
                frags = (got.get(int(seg_id)) if isinstance(got, dict) else got) or []
                for j, fm in enumerate(frags):           # cache the WHOLE lod's fragments
                    raw = trimesh.Trimesh(vertices=np.asarray(fm.vertices, dtype=np.float64),
                                          faces=np.asarray(fm.faces, dtype=np.int64), process=False)
                    cp = self._frag_cache_path(seg_id, lod, j)
                    if cp:
                        try:
                            os.makedirs(self._cache_dir, exist_ok=True)
                            raw.export(cp)
                        except Exception:  # noqa: BLE001
                            pass
                    if j in idxs and j not in meshes:
                        meshes[j] = raw
            for i in idxs:
                mesh = meshes.get(i)
                if mesh is None:
                    continue
                if colorize is not None:
                    r, g, b = colorize(int(seg_id))
                    mesh.visual.vertex_colors = np.tile(
                        (np.array([r, g, b, 1.0]) * 255).astype(np.uint8), (len(mesh.vertices), 1))
                parts.append(mesh)
        if not parts:
            return None
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    @staticmethod
    def _mesh_resolution_nm(mesh: trimesh.Trimesh) -> float:
        """A LOD's spatial resolution (nm) ~ its mean triangle edge length. A robust
        stand-in for the multilod manifest's lodScale, measured from the geometry."""
        try:
            el = mesh.edges_unique_length
            if len(el):
                return float(el.mean())
        except Exception:  # noqa: BLE001
            pass
        ext = mesh.bounds[1] - mesh.bounds[0]   # fallback: cube-root volume per vertex
        return float((float(np.prod(ext)) / max(1, len(mesh.vertices))) ** (1 / 3))

    def _draco_lod_for_screen(self, seg_id: int, nm_per_px: float, draft: bool,
                              max_verts: float | None = None) -> trimesh.Trimesh | None:
        """Pick the coarsest LOD that still looks sharp at the given on-screen scale
        (`nm_per_px`) — neuroglancer's criterion: render a LOD once its spatial
        resolution (lodScale) is finer than one screen pixel times a tolerance
        (`lodScale <= nm_per_px * detailCutoff`; NG's default cutoff is ~1). We
        estimate each LOD's resolution from its mean edge length, which — unlike a
        sheet-area vertex count — doesn't over-refine thin neurites. Capped at
        `max_verts` (the offline budget NG doesn't need, since it streams)."""
        tol = 6.0 if draft else 1.0      # px of mesh resolution to allow (matches NG's ~1)
        target_nm = max(tol * nm_per_px, 1e-6)
        chosen = None
        for lod in range(self._max_lod(seg_id), -1, -1):  # coarse -> fine
            try:
                mesh = self._draco(seg_id, lod=lod)
            except Exception as e:  # noqa: BLE001
                print(f"[mesh] {seg_id} lod{lod} failed: {e}")
                continue
            # hard budget ceiling: if going this fine would exceed max_verts, keep the
            # previous (coarser, in-budget) LOD instead.
            if max_verts is not None and len(mesh.vertices) > max_verts and chosen is not None:
                break
            chosen = mesh
            if self._mesh_resolution_nm(mesh) <= target_nm:  # fine enough on screen
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
        seg_ids = list(seg_ids)
        per_seg = (total_budget / max(1, len(seg_ids))) if total_budget else None

        def _fetch(s):
            try:
                return self._precomputed(s, colorize, nm_per_px, draft, max_verts=per_seg)
            except Exception as e:  # noqa: BLE001  (one bad segment shouldn't sink the layer)
                print(f"[mesh] segment {s} failed: {e}")
                return None

        if len(seg_ids) > 1:
            with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(seg_ids))) as ex:
                parts = list(ex.map(_fetch, seg_ids))
        else:
            parts = [_fetch(seg_ids[0])]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    def load_many(self, seg_ids, colorize=None, target_voxels_single: int = 8_000_000,
                  target_voxels_union: int = 20_000_000, nm_per_px: float | None = None,
                  draft: bool = False, prefer_labels: bool = False,
                  total_budget: int | None = None) -> trimesh.Trimesh:
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
            budget = total_budget if total_budget is not None else (1_200_000 if draft else 5_000_000)
            combined = self._draco_concat(seg_ids, colorize=colorize, nm_per_px=nm_per_px,
                                          draft=draft, total_budget=budget)
            if combined is not None:
                return combined
        # fallback: no precomputed source -> generate from labels
        if not self.label_zarr:
            raise ValueError(f"no mesh source for segments {seg_ids}")
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
