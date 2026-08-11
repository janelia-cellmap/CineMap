"""Render worker: keyframes -> per-frame assets -> Blender subprocess -> mp4.

Prepares assets on the Python side (EM slice PNGs via the slice loader, mesh OBJs
via the mesh loader), writes a scene spec with everything in Blender units, runs
the isolated `bpy` subprocess, then encodes frames with ffmpeg.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from ..config import NM_PER_BU, PROJECTS_DIR
from ..models import Manifest, Project, RenderJob
from ..data.mesh_loader import MeshLoader
from ..data.mesh_loader import _FETCH_WORKERS
from ..data.slice_loader import EMVolume, get_volume
from .interpolate import FrameAnnotation, FrameState, build_frames, state_at_time

Progress = Callable[[float, str], None]
# Slice reads already use TensorStore internally, which can parallelize chunk IO and
# decompression for a single read. Keep CineMap's outer slice-read fanout serial by
# default so oblique slices do not stack multiple large NumPy grids/subvolumes in RAM.
_SLICE_FETCH_WORKERS = int(os.environ.get("CINEMAP_SLICE_FETCH_WORKERS") or 1)
_OBLIQUE_BATCH_MAX_BYTES = int(os.environ.get("CINEMAP_OBLIQUE_BATCH_MAX_BYTES") or 768_000_000)
_OBLIQUE_BATCH_MAX_SLICES = int(os.environ.get("CINEMAP_OBLIQUE_BATCH_MAX_SLICES") or 32)
_OBLIQUE_BATCH_MAX_OVERREAD = float(os.environ.get("CINEMAP_OBLIQUE_BATCH_MAX_OVERREAD") or 1.25)
_OBLIQUE_TILE_THRESHOLD_BYTES = int(os.environ.get("CINEMAP_OBLIQUE_TILE_THRESHOLD_BYTES") or 512_000_000)
_OBLIQUE_TILE_PX = int(os.environ.get("CINEMAP_OBLIQUE_TILE_PX") or 192)


class RenderCancelled(Exception):
    """Raised when a render is cancelled via the worker's cancel event."""


def _bu(p, nm_per_bu):
    return [c / nm_per_bu for c in p]


def _contrast_key(limits) -> tuple[float, float] | None:
    if not limits or len(limits) != 2:
        return None
    try:
        lo, hi = float(limits[0]), float(limits[1])
    except (TypeError, ValueError):
        return None
    if hi <= lo:
        return None
    return (round(lo, 6), round(hi, 6))


def _apply_contrast_window(image, limits):
    import numpy as np

    src = np.asarray(image)
    arr = src.astype(np.float64, copy=False)
    key = _contrast_key(limits)
    if key is None:
        return arr
    lo, hi = key
    # Neuroglancer shader controls are often normalized 0..1. Scale from the source
    # dtype instead of the crop's observed max so a dark/bright crop still uses the
    # same display window as NG.
    if hi <= 1.0:
        if np.issubdtype(src.dtype, np.integer):
            scale = float(np.iinfo(src.dtype).max)
        else:
            scale = 255.0 if arr.size and np.nanmax(arr) > 1.0 else 1.0
        lo *= scale
        hi *= scale
    return (arr - lo) * (255.0 / (hi - lo))


def _director_settings_for_render(project, auto_direct: bool):
    """Resolve the render look for a scene spec.

    `auto_direct` controls optional camera smoothing and per-frame emphasis. It
    must not suppress an explicitly selected Look preset; otherwise the Blender
    side falls back to legacy lights/materials and bypasses NG/neuVid matching.
    """
    from . import director

    look = getattr(project, "look", None) or {}
    if look:
        return director.make_settings(look)
    if auto_direct:
        return director.make_settings(None)
    return director.make_settings({"preset": "ng"})


def _ng_quantize_decode_normals(normals):
    """Match Neuroglancer's mesh normal path.

    Neuroglancer computes float vertex normals, encodes them to 2x snorm8
    octahedral coordinates in the worker, then decodes them in the WebGL shader.
    Store the decoded values in Blender so the NG preset uses the same quantized
    normals for its headlight contrast.
    """
    import numpy as np

    n = np.ascontiguousarray(normals, dtype=np.float32)
    if n.size == 0:
        return n.reshape((-1, 3))
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    l1 = np.abs(x) + np.abs(y) + np.abs(z)
    inv = np.zeros_like(l1, dtype=np.float32)
    ok = l1 > 0
    inv[ok] = 1.0 / l1[ok]

    def sign_not_zero(a):
        return np.where(a < 0, -1.0, 1.0).astype(np.float32)

    ex = x * inv
    ey = y * inv
    neg_z = z < 0
    ex = np.where(neg_z, (1.0 - np.abs(y * inv)) * sign_not_zero(x), ex)
    ey = np.where(neg_z, (1.0 - np.abs(x * inv)) * sign_not_zero(y), ey)

    def snorm8(a):
        # Port Neuroglancer's `Math.min(Math.max(-127, x * 127 + 0.5), 127) >>> 0`.
        q = np.minimum(np.maximum(-127.0, a * 127.0 + 0.5), 127.0)
        q = np.nan_to_num(q, nan=0.0).astype(np.int64)
        return (q & 0xFF).astype(np.uint8)

    encoded = np.stack([snorm8(ex), snorm8(ey)], axis=1)

    signed = encoded.astype(np.int16)
    signed = np.where(signed >= 128, signed - 256, signed).astype(np.float32)
    e = np.maximum(-1.0, signed / 127.0)
    out = np.empty((len(n), 3), dtype=np.float32)
    out[:, 0] = e[:, 0]
    out[:, 1] = e[:, 1]
    out[:, 2] = 1.0 - np.abs(e[:, 0]) - np.abs(e[:, 1])
    folded = out[:, 2] < 0.0
    if np.any(folded):
        ox = out[folded, 0].copy()
        oy = out[folded, 1].copy()
        out[folded, 0] = (1.0 - np.abs(oy)) * sign_not_zero(ox)
        out[folded, 1] = (1.0 - np.abs(ox)) * sign_not_zero(oy)
    lens = np.linalg.norm(out, axis=1)
    ok = lens > 0
    out[ok] /= lens[ok, None]
    out[~ok] = (0.0, 0.0, 1.0)
    return np.ascontiguousarray(out, dtype=np.float32)


def _export_mesh_npz(mesh, out: Path) -> None:
    """Write a trimesh-like object as a compact `.npz` (vertices float32, faces
    int32, Neuroglancer-style vertex normals, and optional uint8 vertex colors).
    Blender loads this ~5–10× faster than going through `bpy.ops.wm.ply_import`
    (which routes through the operator system + undo stack)."""
    import numpy as np

    v = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    f = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    arrs: dict = {"v": v, "f": f}
    if len(v) and len(f):
        face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 1]])
        lens = np.linalg.norm(face_normals, axis=1)
        ok = lens > 0
        face_normals[ok] /= lens[ok, None]
        face_normals[~ok] = 0.0
        normals = np.zeros_like(v, dtype=np.float32)
        np.add.at(normals, f[:, 0], face_normals)
        np.add.at(normals, f[:, 1], face_normals)
        np.add.at(normals, f[:, 2], face_normals)
        lens = np.linalg.norm(normals, axis=1)
        ok = lens > 0
        normals[ok] /= lens[ok, None]
        arrs["n"] = _ng_quantize_decode_normals(normals)
    vc = None
    try:
        vc = mesh.visual.vertex_colors      # Nx4 uint8 (trimesh)
    except Exception:  # noqa: BLE001
        vc = None
    if vc is not None and len(vc) == len(v):
        vc = np.ascontiguousarray(vc, dtype=np.uint8)
        # Trimesh attaches a uniform [102,102,102,255] visual by default even when
        # geometry is intentionally uncolored. Do not bake that into the cache; Blender
        # should use the explicit material color from the scene spec instead.
        default_gray = (
            len(vc) > 0
            and np.all(vc == np.array([102, 102, 102, 255], dtype=np.uint8))
        )
        if not default_gray:
            arrs["c"] = vc
    # raw save (no compression) — load speed in Blender matters more than disk
    np.savez(str(out), **arrs)


def _ease(t: float, mode: str) -> float:
    if mode == "ease-in-out":
        return t * t * (3 - 2 * t)
    if mode == "ease-in":
        return t * t
    if mode == "ease-out":
        return t * (2 - t)
    return t


def _sweep_progress(sw, t: float) -> float:
    """Eased progress (0..1) of a sweep at global time `t`. With mirror=True it ping-pongs
    (0->1 over the first half, 1->0 over the second) so the cut/slice goes forward THEN back
    and ends where it started."""
    u = min(1.0, max(0.0, (t - sw.start_s) / (sw.duration_s or 1e-9)))
    if getattr(sw, "mirror", False):
        u = u * 2 if u < 0.5 else (1 - u) * 2
    return _ease(u, sw.easing)


def _clip_params(cl) -> dict | None:
    """Normalize a layer's cutaway plane to a plain dict, or None if absent/disabled.
    `cl` may be a ClipPlane model OR a plain dict (legacy), so read with getattr/get;
    attribute access is required because a ClipPlane isn't subscriptable."""
    if cl is None:
        return None
    g = cl.get if isinstance(cl, dict) else (lambda k, d=None: getattr(cl, k, d))
    if not g("enabled", True):          # an explicitly-disabled plane -> no cutaway
        return None
    return {"axis": g("axis", "z"), "side": g("side", 1),
            "normal": g("normal"), "position_nm": float(g("position_nm", 0.0))}


class RenderWorker:
    # Hard ceiling on a layer's combined vertex count, regardless of mesh_detail.
    # 50M is intended for high-memory GPUs/workstations; the OOM retry path backs
    # off if Blender cannot fit the resulting scene.
    MESH_BUDGET_CEILING = 50_000_000

    def __init__(self, project: Project, job: RenderJob, nm_per_bu: float = NM_PER_BU):
        self.project = project
        self.job = job
        self.nm_per_bu = nm_per_bu
        self.manifest: Manifest = project.manifest
        self.workdir = PROJECTS_DIR / project.id / "renders" / job.id
        self.frames_dir = self.workdir / "frames"
        # asset cache is PROJECT-level (not per-job) so re-renders, A/B (director on/off),
        # and changes to fps/samples/detail all reuse already-downloaded geometry. The
        # uid keys geometry+LOD+color, so sharing across jobs is safe.
        self.assets_dir = PROJECTS_DIR / project.id / "assets"
        # raw per-(segment, LOD) mesh download cache, shared across ALL projects (keyed
        # by mesh_url+seg+lod) so changing quality/zoom only fetches new finer LODs.
        self._mesh_cache_dir = PROJECTS_DIR / ".mesh_cache"
        self.blend_path = self.workdir / "scene.blend"
        self.cancel = threading.Event()   # set to request cancellation
        self._proc: subprocess.Popen | None = None
        self._em: EMVolume | None = None
        self._label_vols: dict[str, EMVolume] = {}
        self._slice_cache: dict[tuple, dict] = {}
        self._slice_cache_lock = threading.Lock()
        # Resolution budgets. Draft (bake/update/preview thumbnails) trades detail
        # for speed: a coarse EM level and low-voxel meshes. EM scale is selected
        # per frame from physical nm/pixel, matching Neuroglancer's multiscale choice;
        # meshes use zmesh voxel/vertex budgets for label-derived geometry.
        draft = bool(getattr(job.settings, "draft", False))
        self._draft = draft
        # Fallback/cap for paths that still request an explicit resampled image size
        # (notably oblique slices). Axis-aligned slices use per-frame nm/pixel instead.
        self._em_target_px = 768 if draft else min(2560, max(1280, int(job.settings.width * 1.25)))
        self._mesh_voxels_single = 1_500_000 if draft else 8_000_000
        self._mesh_voxels_union = 3_000_000 if draft else 20_000_000
        # mesh sourcing: precomputed (LOD-adaptive) by default; opt in to zmesh
        # meshing from labels via the render setting.
        self._prefer_labels = bool(getattr(job.settings, "mesh_from_labels", False))
        # Per-layer vertex budget = base * mesh_detail, hard-capped so a too-high
        # setting can't recreate the multi-GB mesh that stalled asset prep / OOM'd the
        # GPU. The OOM-retry loop in run() halves this and rebuilds if Cycles runs out.
        detail = max(0.25, min(float(getattr(job.settings, "mesh_detail", 1.0) or 1.0), 10.0))
        base_budget = 3_000_000 if draft else 5_000_000
        self._mesh_budget = min(int(base_budget * detail), self.MESH_BUDGET_CEILING)
        self._label_smooth_iters = max(
            0,
            min(8, int(getattr(job.settings, "label_mesh_smooth_iters", 0) or 0)),
        )
        # Decimation keep-fraction (0 disables): after loading up to the vertex budget,
        # decimate each segment to ~this fraction of its faces, so the final mesh lands
        # below the budget (e.g. 0.25 keeps ~a quarter). Quality knob, not a read driver.
        self._label_decimate_fraction = max(
            0.0,
            min(1.0, float(getattr(job.settings, "label_mesh_decimate_fraction", 0.0) or 0.0)),
        )
        # Blockwise label meshing (read+mesh per cubic block, then weld) — bounds peak
        # memory so sparse-but-huge bboxes don't OOM. "auto" (default) decides per layer
        # from the planned read size; "on"/"off" force it.
        _bw = str(getattr(job.settings, "label_mesh_blockwise", "auto") or "auto").lower()
        self._label_blockwise = {"on": True, "off": False}.get(_bw, "auto")
        self._nm_per_px = None  # finest on-screen scale across frames (set per build)
        # mesh LOD strategy: "single" (one build), "frame" (per-frame adaptive, like
        # neuroglancer; free on orbits), or "chunk" (precomputed-mesh fragments).
        self._lod_mode = getattr(job.settings, "lod_mode", "frame") or "frame"
        if self._prefer_labels and self._lod_mode == "chunk":
            self._lod_mode = "frame"
        # A cutaway needs a watertight cross-section to cap, which requires the CUT LAYER
        # at ONE consistent LOD across the frames where it's clipped (mixed LODs leave
        # non-welding fragment seams -> the cap cracks/can't close). We DON'T force that on
        # the whole movie anymore — _build_scene_spec pins only the cut layer, and only in
        # its cutaway window, to a single LOD; every other layer/frame keeps adaptive LOD
        # (coarse when far) so the rest of the shot stays light.
        # non-destructive presentation pass (lighting rig / materials / DOF)
        self._auto_direct = bool(getattr(job.settings, "auto_direct", True))
        # draw a wireframe box around each data source's extent (neuroglancer-style)
        self._show_bbox = bool(getattr(job.settings, "show_bbox", False))
        self._bbox_color = list(getattr(job.settings, "bbox_color", None) or [0.62, 0.66, 0.74])
        self._bbox_source = getattr(job.settings, "bbox_source", "") or ""   # "" = auto

    # ---- asset preparation ----
    def _em_vol(self) -> EMVolume:
        if self._em is None:
            assert self.manifest.em, "no EM source in manifest"
            self._em = get_volume(self.manifest.em.zarr_url)
        return self._em

    def _label_vol(self, url: str) -> EMVolume:
        if url not in self._label_vols:
            self._label_vols[url] = get_volume(url)
        return self._label_vols[url]

    def _vol_for(self, em_name: str):
        """Resolve a slice layer NAME to (zarr_url, is_label) — so a slice can point at
        any layer, not just the default EM. An EM/image layer -> grayscale cross-section;
        a segmentation layer (its label volume) -> colored labels. Empty/unknown name
        falls back to the manifest's EM image."""
        em = self.manifest.em
        if em and (not em_name or em_name == em.name):
            return em.zarr_url, False
        src = next((s for s in self.manifest.meshes
                    if s.name == em_name and s.label_zarr), None)
        if src:
            return src.label_zarr, True
        return (em.zarr_url, False) if em else (None, False)

    @staticmethod
    def _slice_read_plane(sl) -> tuple[str, float, list[float] | None]:
        """Normalize a FrameSlice for reading.

        A normal like [0, -1, 0] is still an axis-aligned plane. Treat it as a cheap
        2D read_slice, converting dot(x, normal)=position back to the axis coordinate
        (for [0,-1,0], y = -position). Only truly tilted planes use read_oblique_slice.
        """
        axis = getattr(sl, "axis", "z")
        pos = float(getattr(sl, "position_nm", 0.0))
        normal = getattr(sl, "normal", None)
        if not normal:
            return axis, pos, None
        n = [float(v) for v in normal]
        mag = math.sqrt(sum(v * v for v in n)) or 1.0
        n = [v / mag for v in n]
        ai = max(range(3), key=lambda i: abs(n[i]))
        if abs(n[ai]) >= 0.999:
            axis = ("x", "y", "z")[ai]
            return axis, pos / n[ai], None
        return axis, pos, n

    def _slice_cache_key(self, sl, region, seg_overlays, slice_seg=None,
                         target_nm_per_px: float | None = None) -> tuple:
        center, half = region
        axis, position_nm, normal = self._slice_read_plane(sl)
        zurl, is_label = self._vol_for(sl.em_name)
        seg_key = (tuple(sorted(int(i) for i in slice_seg[0])), slice_seg[1].cache_key()) \
            if (is_label and slice_seg) else None
        return (
            "slice-v4",
            sl.em_name,
            zurl,
            bool(is_label),
            seg_key,
            axis,
            round(float(position_nm)),
            getattr(sl, "scale_level", None),
            tuple(round(float(x), 6) for x in normal) if normal else None,
            tuple(round(float(c)) for c in center),
            round(float(half)),
            self._em_target_px,
            None if target_nm_per_px is None else round(float(target_nm_per_px), 3),
            _contrast_key(getattr(sl, "contrast_limits", None)),
            tuple((u, tuple(sorted(int(i) for i in ids)), lc.cache_key())
                  for u, ids, lc in seg_overlays),
        )

    def _slice_cache_paths(self, key: tuple, axis: str) -> tuple[Path, Path]:
        import hashlib

        sig = json.dumps(key, sort_keys=True, default=str).encode()
        digest = hashlib.md5(sig).hexdigest()[:16]
        path = self.assets_dir / f"slice_{axis}_{digest}.png"
        return path, path.with_suffix(".json")

    def _slice_cache_get(self, key: tuple, path: Path, meta_path: Path) -> dict | None:
        with self._slice_cache_lock:
            cached = self._slice_cache.get(key)
        if cached is not None:
            return cached
        if path.exists() and meta_path.exists():
            try:
                out = json.loads(meta_path.read_text())
                out["image_path"] = str(path)  # project dirs can move; the PNG beside us wins
                with self._slice_cache_lock:
                    self._slice_cache[key] = out
                return out
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _atomic_write_json(data, path: Path) -> None:
        tmp = f"{path}.{threading.get_ident()}.tmp"
        Path(tmp).write_text(json.dumps(data, separators=(",", ":")))
        os.replace(tmp, path)

    def _slice_cache_put(self, key: tuple, out: dict, meta_path: Path) -> None:
        self._atomic_write_json(out, meta_path)
        with self._slice_cache_lock:
            self._slice_cache[key] = out

    def _slice_png_from_result(
        self,
        sl,
        region,
        seg_overlays,
        slice_seg,
        target_nm_per_px: float | None,
        axis: str,
        position_nm: float,
        normal,
        zurl: str | None,
        is_label: bool,
        key: tuple,
        path: Path,
        meta_path: Path,
        res,
    ) -> dict:
        """Write a SliceResult to the exact PNG/meta cache entry _slice_png uses."""
        import numpy as np
        from PIL import Image

        if is_label:   # segmentation layer: color the labels directly (no EM grayscale)
            lab = np.asarray(res.image)
            H, W = lab.shape[:2]
            rgb = np.zeros((H, W, 3), dtype=np.float64)
            ids, lc = slice_seg if slice_seg else (None, None)
            idset = set(int(i) for i in ids) if ids else None
            for u in np.unique(lab):
                iu = int(u)
                if iu == 0 or (idset is not None and iu not in idset):
                    continue
                rgb[lab == u] = (np.array(lc.rgb(iu)) * 255 if lc else np.array([230.0, 180.0, 90.0]))
            self._atomic_save(Image.fromarray(rgb.clip(0, 255).astype(np.uint8)), path)
            out = {"image_path": str(path), "origin_bu": _bu(res.origin_nm, self.nm_per_bu),
                   "u_bu": _bu(res.u_nm, self.nm_per_bu), "v_bu": _bu(res.v_nm, self.nm_per_bu)}
            self._slice_cache_put(key, out, meta_path)
            return out

        rgb = np.repeat(_apply_contrast_window(res.image, getattr(sl, "contrast_limits", None))[:, :, None],
                        3, axis=2)  # grayscale EM
        H, W = rgb.shape[:2]

        for label_zarr, ids, lc in ([] if normal else seg_overlays):  # seg overlay: axis-aligned only
            if not ids:
                continue
            lres = self._label_vol(label_zarr).read_slice(axis, position_nm,
                                                          target_px=self._em_target_px,
                                                          region=region,
                                                          target_nm_per_px=target_nm_per_px,
                                                          raw=True)
            lab = np.asarray(lres.image)
            yi = (np.arange(H) * lab.shape[0] / H).astype(int).clip(0, lab.shape[0] - 1)
            xi = (np.arange(W) * lab.shape[1] / W).astype(int).clip(0, lab.shape[1] - 1)
            lab_rs = lab[yi][:, xi]                         # nearest-resample to EM size
            mask = np.isin(lab_rs, np.asarray(sorted(ids)))
            if not mask.any():
                continue
            color = np.zeros((H, W, 3))
            for u in np.unique(lab_rs[mask]):
                color[lab_rs == u] = lc.rgb(int(u))         # neuroglancer color
            a = 0.6                                          # overlay opacity
            m = mask[:, :, None]
            rgb = np.where(m, rgb * (1 - a) + color * 255 * a, rgb)

        self._atomic_save(Image.fromarray(rgb.clip(0, 255).astype(np.uint8)), path)
        out = {
            "image_path": str(path),
            "origin_bu": _bu(res.origin_nm, self.nm_per_bu),
            "u_bu": _bu(res.u_nm, self.nm_per_bu),
            "v_bu": _bu(res.v_nm, self.nm_per_bu),
        }
        self._slice_cache_put(key, out, meta_path)
        return out

    def _slice_png(self, sl, region, seg_overlays, slice_seg=None,
                   target_nm_per_px: float | None = None) -> dict:
        """Render a cross-section of the slice's chosen layer. For an EM/image layer:
        the grayscale EM with `seg_overlays` [(label_zarr, ids, lc), …] colored on top
        (like neuroglancer). For a SEGMENTATION layer (resolved via _vol_for): the
        layer's labels rendered in color directly (`slice_seg=(ids, lc)`). Cached per
        (slice, region, overlay)."""
        import numpy as np

        center, half = region
        axis, position_nm, normal = self._slice_read_plane(sl)
        zurl, is_label = self._vol_for(sl.em_name)
        key = self._slice_cache_key(sl, region, seg_overlays, slice_seg=slice_seg,
                                    target_nm_per_px=target_nm_per_px)
        path, meta_path = self._slice_cache_paths(key, axis)
        cached = self._slice_cache_get(key, path, meta_path)
        if cached is not None:
            return cached

        vol = self._label_vol(zurl) if (is_label and zurl) else self._em_vol()
        if normal:   # oblique plane: resample the tilted plane through the projected focus
            n = np.asarray(normal, float); n = n / (np.linalg.norm(n) or 1.0)
            c = np.asarray(center, float)
            cproj = c + (position_nm - float(np.dot(c, n))) * n
            target_px = self._em_target_px
            if target_nm_per_px:
                target_px = min(target_px, max(8, int(round(2 * half / target_nm_per_px))))
            res = vol.read_oblique_slice(
                normal,
                cproj,
                half,
                target_px=target_px,
                target_nm_per_px=target_nm_per_px,
            )
        else:
            res = vol.read_slice(axis, position_nm, level=sl.scale_level,
                                 target_px=self._em_target_px, region=region,
                                 target_nm_per_px=target_nm_per_px, raw=is_label)
        return self._slice_png_from_result(
            sl,
            region,
            seg_overlays,
            slice_seg,
            target_nm_per_px,
            axis,
            position_nm,
            normal,
            zurl,
            is_label,
            key,
            path,
            meta_path,
            res,
        )

    def _oblique_warm_context(self, job) -> dict | None:
        """Prepared batch context for an oblique EM slice warm job, or None fallback."""
        import numpy as np

        sl, region, seg_overlays, slice_seg, target_nm_per_px = job
        center, half = region
        axis, position_nm, normal = self._slice_read_plane(sl)
        if not normal:
            return None
        zurl, is_label = self._vol_for(sl.em_name)
        if is_label or not zurl:
            return None
        key = self._slice_cache_key(sl, region, seg_overlays, slice_seg=slice_seg,
                                    target_nm_per_px=target_nm_per_px)
        path, meta_path = self._slice_cache_paths(key, axis)
        cached = self._slice_cache_get(key, path, meta_path)
        if cached is not None:
            return {"cached": True, "job": job}

        n = np.asarray(normal, float)
        n = n / (np.linalg.norm(n) or 1.0)
        c = np.asarray(center, float)
        cproj = c + (position_nm - float(np.dot(c, n))) * n
        target_px = self._em_target_px
        if target_nm_per_px:
            target_px = min(target_px, max(8, int(round(2 * half / target_nm_per_px))))
        vol = get_volume(zurl)
        spec = vol.oblique_slice_spec(
            normal,
            cproj,
            half,
            target_px=target_px,
            target_nm_per_px=target_nm_per_px,
        )
        single_bytes = vol.estimate_box_bytes(spec.bbox_xyz_nm, spec.level)
        return {
            "cached": False,
            "job": job,
            "sl": sl,
            "region": region,
            "seg_overlays": seg_overlays,
            "slice_seg": slice_seg,
            "target_nm_per_px": target_nm_per_px,
            "axis": axis,
            "position_nm": position_nm,
            "normal": normal,
            "zurl": zurl,
            "is_label": is_label,
            "key": key,
            "path": path,
            "meta_path": meta_path,
            "vol": vol,
            "spec": spec,
            "single_bytes": single_bytes,
        }

    def _warm_oblique_batches(self, warm_jobs, mark_done) -> list:
        """Fill cache for compatible oblique EM jobs; return jobs needing fallback."""
        fallback = []
        groups: dict[tuple, list[dict]] = {}
        for job in warm_jobs:
            if self.cancel.is_set():
                fallback.append(job)
                continue
            try:
                ctx = self._oblique_warm_context(job)
            except Exception:  # noqa: BLE001
                ctx = None
            if ctx is None:
                fallback.append(job)
                continue
            if ctx.get("cached"):
                mark_done()
                continue
            spec = ctx["spec"]
            group_key = (
                ctx["zurl"],
                int(spec.level),
                tuple(round(float(x), 6) for x in spec.n_xyz),
            )
            groups.setdefault(group_key, []).append(ctx)

        def flush_batch(items: list[dict]) -> None:
            if not items:
                return
            if self.cancel.is_set():
                fallback.extend(ctx["job"] for ctx in items)
                return
            vol = items[0]["vol"]
            specs = [ctx["spec"] for ctx in items]
            try:
                bbox = vol.union_bbox(specs)
                level = int(specs[0].level)
                est_bytes = vol.estimate_box_bytes(bbox, level)
                if est_bytes > _OBLIQUE_TILE_THRESHOLD_BYTES:
                    print(
                        f"[worker] oblique batch tiled n={len(specs)} level={level} "
                        f"tile_px={_OBLIQUE_TILE_PX} full_bbox_est={est_bytes / 1e9:.2f}GB",
                        flush=True,
                    )
                    results = vol.read_oblique_specs_tiled(specs, tile_px=_OBLIQUE_TILE_PX)
                else:
                    print(
                        f"[worker] oblique batch full n={len(specs)} level={level} "
                        f"bbox_est={est_bytes / 1e9:.2f}GB",
                        flush=True,
                    )
                    results = vol.read_oblique_specs(specs)
                for ctx, res in zip(items, results):
                    self._slice_png_from_result(
                        ctx["sl"],
                        ctx["region"],
                        ctx["seg_overlays"],
                        ctx["slice_seg"],
                        ctx["target_nm_per_px"],
                        ctx["axis"],
                        ctx["position_nm"],
                        ctx["normal"],
                        ctx["zurl"],
                        ctx["is_label"],
                        ctx["key"],
                        ctx["path"],
                        ctx["meta_path"],
                        res,
                    )
                    mark_done()
            except Exception:  # noqa: BLE001
                fallback.extend(ctx["job"] for ctx in items)

        for items in groups.values():
            cur: list[dict] = []
            for ctx in items:
                candidate = cur + [ctx]
                too_many = len(candidate) > max(1, _OBLIQUE_BATCH_MAX_SLICES)
                too_big = False
                too_sparse = False
                estimate_failed = False
                if not too_many:
                    specs = [x["spec"] for x in candidate]
                    vol = ctx["vol"]
                    try:
                        bbox = vol.union_bbox(specs)
                        level = int(specs[0].level)
                        union_bytes = vol.estimate_box_bytes(bbox, level)
                        single_bytes = sum(max(1, int(x.get("single_bytes", 1))) for x in candidate)
                        largest_single = max(max(1, int(x.get("single_bytes", 1))) for x in candidate)
                        byte_cap = max(
                            _OBLIQUE_BATCH_MAX_BYTES,
                            int(largest_single * _OBLIQUE_BATCH_MAX_OVERREAD),
                        )
                        too_big = union_bytes > byte_cap
                        too_sparse = (
                            len(candidate) > 1
                            and union_bytes > single_bytes * _OBLIQUE_BATCH_MAX_OVERREAD
                        )
                    except Exception:  # noqa: BLE001
                        too_big = True
                        estimate_failed = True
                if cur and (too_many or too_big or too_sparse):
                    flush_batch(cur)
                    single_too_big = False
                    try:
                        bbox = ctx["vol"].union_bbox([ctx["spec"]])
                        level = int(ctx["spec"].level)
                        single_too_big = (
                            ctx["vol"].estimate_box_bytes(bbox, level)
                            > max(
                                _OBLIQUE_BATCH_MAX_BYTES,
                                int(max(1, int(ctx.get("single_bytes", 1))) * _OBLIQUE_BATCH_MAX_OVERREAD),
                            )
                        )
                    except Exception:  # noqa: BLE001
                        single_too_big = True
                    if single_too_big:
                        fallback.append(ctx["job"])
                        cur = []
                    else:
                        cur = [ctx]
                    continue
                if not cur and (too_big or estimate_failed):
                    fallback.append(ctx["job"])
                    continue
                cur = candidate
            flush_batch(cur)
        return fallback

    def _lod_tag_for(self, nmpp) -> str:
        """Cache-key component for a mesh built at on-screen scale `nmpp` (nm/px):
        re-framing, draft, source, budget, or a different LOD bucket each rebuild."""
        source_tag = (
            f"lab-zmesh-clean-v14-s{self._label_smooth_iters}"
            f"-d{self._label_decimate_fraction:.3g}-b{self._label_blockwise}"
            if self._prefer_labels else "pre"
        )
        quality_tag = "draft" if self._draft else "full"
        if self._prefer_labels:
            # Label-derived meshes are selected by the per-layer vertex budget.
            # Camera nm/px buckets are useful for precomputed NG LODs, but they
            # only create redundant zmesh base assets for the same IDs/budget.
            return f"{quality_tag}|{source_tag}|b{self._mesh_budget}|label-budget"
        return (
            f"npp{nmpp:.3g}|{quality_tag}|{source_tag}|b{self._mesh_budget}"
            if nmpp
            else f"{quality_tag}|{source_tag}|b{self._mesh_budget}"
        )

    def _lod_bucket_nmpp(self, nmpps, max_buckets: int = 4) -> list:
        """Per-frame nm/px to BUILD the mesh at. With dynamic LOD, far frames (large
        nm/px = layer small on screen) build a coarser mesh and near frames a finer
        one — like neuroglancer streaming higher-LOD chunks when you zoom in. The
        bucket value is the FINEST nm/px in its band, so no frame is under-detailed.
        A ~constant-distance shot collapses to one bucket (= the prior single-LOD
        build), so extra asset prep only happens for shots that change zoom."""
        if not nmpps:
            return []
        lo, hi = min(nmpps), max(nmpps)
        # "single" => one finest build; adaptive modes bucket by zoom (a ~constant-
        # distance shot still collapses to one bucket).
        if self._lod_mode == "single" or max_buckets <= 1 or hi <= lo * 1.6:
            return [lo] * len(nmpps)
        span = math.log(hi / lo)
        bands = [min(max_buckets - 1, int(math.log(x / lo) / span * max_buckets))
                 for x in nmpps]
        band_finest: dict[int, float] = {}
        for b, x in zip(bands, nmpps):
            band_finest[b] = min(band_finest.get(b, x), x)
        return [band_finest[b] for b in bands]

    def _mesh_source(self, mesh_name):
        return next((m for m in self.manifest.meshes if m.name == mesh_name), None)

    def _mesh_uid(self, mesh_name, ids, color_key=(), nmpp=None) -> str:
        """Stable id per geometry asset.

        Geometry normally depends on the layer, exact segment set, and chosen LOD scale.
        Color/material are per-frame Blender state for fixed-color layers.  For NG
        hash-colored multi-ID layers, however, the combined mesh needs baked per-segment
        vertex colors, so the color resolver key is part of the geometry cache key.
        Skeleton shader colors are also baked into tube vertex colors, so shader text
        must participate in the geometry key.
        """
        import hashlib

        src = self._mesh_source(mesh_name)
        shader_sig = ""
        if src and src.skeleton_url and not src.mesh_url and not src.label_zarr:
            shader_sig = hashlib.md5((src.skeleton_shader or "").encode()).hexdigest()[:8]
        # `geom10` versions the mesh decoder/cache key: this generation stores
        # Neuroglancer's octahedral-quantized/decoded vertex normals alongside base
        # geometry without baked colors for fixed-color layers, but keeps a color-keyed
        # variant when NG hash coloring or skeleton shader coloring must be baked.
        color_sig = str(color_key) if color_key else "solid"
        sig = (
            ",".join(map(str, sorted(ids)))
            + "|"
            + color_sig
            + "|"
            + self._lod_tag_for(nmpp)
            + "|"
            + shader_sig
            + "|geom10"
        )
        return f"{mesh_name}_{hashlib.md5(sig.encode()).hexdigest()[:8]}"

    def _clip_from_sweeps(self, layer_name: str, t: float) -> dict | None:
        """The cutaway clip for `layer_name` at global time `t` (seconds), from any active
        Sweep — evaluated on the SWEEP's own timeline, independent of the camera keyframes.
        Returns a clip dict (same shape as _clip_params) or None. First active sweep wins."""
        if not self._include_timeline_sweeps():
            return None
        for sw in getattr(self.project, "sweeps", []) or []:
            if (not getattr(sw, "enabled", True) or getattr(sw, "kind", "cutaway") != "cutaway"
                    or sw.layer != layer_name):
                continue
            if t < sw.start_s:
                continue                          # before it starts -> mesh is whole
            # holds open after the end (progress clamps to 1); mirror ping-pongs back.
            pos = sw.from_nm + (sw.to_nm - sw.from_nm) * _sweep_progress(sw, t)
            return {"axis": sw.axis, "side": sw.side, "normal": sw.normal,
                    "position_nm": float(pos), "cap": bool(getattr(sw, "cap", True))}
        return None

    def _slices_from_sweeps(self, t: float) -> list:
        """EM slice planes from active 'slice' sweeps at global time `t` — swept on the
        sweep's OWN timeline (independent of the camera keyframes). Unlike a cutaway,
        a slice sweep is a timeline clip: it is visible only during its own span."""
        from .interpolate import FrameSlice
        out = []
        if not self._include_timeline_sweeps():
            return out
        default_em = self.manifest.em.name if self.manifest.em else "em"
        for sw in getattr(self.project, "sweeps", []) or []:
            if not getattr(sw, "enabled", True) or getattr(sw, "kind", "") != "slice":
                continue
            if t < sw.start_s:
                continue
            if t > sw.start_s + (sw.duration_s or 0.0):
                continue
            pos = sw.from_nm + (sw.to_nm - sw.from_nm) * _sweep_progress(sw, t)
            # None lets read_slice auto-pick the EM pyramid level from the camera region.
            # 0 forces full-resolution s0, which turns full-plane sweeps into hundreds-MB
            # PNGs and makes asset prep look like "downloading EM frames" forever.
            sl = FrameSlice(sw.em_name or default_em, sw.axis, float(pos), None,
                            float(sw.opacity), normal=sw.normal)
            out.append((sl, list(getattr(sw, "overlay_layers", []) or [])))
        return out

    @staticmethod
    def _frame_colors(m):
        """LayerColors (neuroglancer seed / fixed colors) from a FrameMesh."""
        from ..data.colors import LayerColors

        return LayerColors(
            seed=getattr(m, "color_seed", 0),
            default=getattr(m, "default_color", None),
            overrides={int(k): v for k, v in (getattr(m, "segment_colors", {}) or {}).items()},
            saturation=getattr(m, "saturation", 1.0),
        )

    @staticmethod
    def _mesh_material_color(m, lc) -> list[float]:
        """Representative NG color for geometry cached without vertex colors.

        Base geometry is intentionally independent of color/material.  For those
        uncolored geometry assets, the Blender material should still come from the
        neuroglancer color resolver rather than MeshInstance.color, which is a generic
        fallback and is not necessarily the layer's NG color.
        """
        if getattr(lc, "default", None) is not None:
            return list(lc.rgb(0))
        ids = list(getattr(m, "segment_ids", None) or [])
        if len(ids) == 1:
            return list(lc.rgb(int(ids[0])))
        return list(getattr(m, "color", None) or [0.91, 0.45, 0.23])

    @staticmethod
    def _mesh_needs_vertex_colors(m, lc) -> bool:
        """True when one combined mesh must carry different colors per segment."""
        ids = list(getattr(m, "segment_ids", None) or [])
        if len(ids) <= 1:
            return False
        if getattr(lc, "default", None) is not None:
            return False
        return True

    def _mesh_uses_shader_vertex_colors(self, mesh_name: str) -> bool:
        src = self._mesh_source(mesh_name)
        return bool(
            src
            and src.skeleton_url
            and not src.mesh_url
            and not src.label_zarr
            and src.skeleton_shader
        )

    @staticmethod
    def _layer_visible(m) -> bool:
        return bool(getattr(m, "visible", True)) and float(getattr(m, "opacity", 1.0)) > 0.001

    def _mesh_render_alpha(self, m) -> float:
        if not self._layer_visible(m) or not getattr(m, "render_3d", True):
            return 0.0
        layer_opacity = max(0.0, min(1.0, float(getattr(m, "opacity", 1.0))))
        object_alpha = max(0.0, min(1.0, float(getattr(m, "object_alpha", 1.0))))
        return layer_opacity * object_alpha

    # --- slice-read derivation (shared by the parallel warm pass and the build loop, so
    # the two can never drift) -------------------------------------------------------
    def _frame_region(self, fr):
        """EM crop around the camera target, sized to what's on screen this frame."""
        if getattr(fr, "projection", "PERSP") == "ORTHO" and getattr(fr, "ortho_scale_nm", None):
            aspect = max(1e-6, float(self.job.settings.width) / max(1, self.job.settings.height))
            visible_half = 0.5 * float(fr.ortho_scale_nm) * max(1.0, aspect)
            half = max(500.0, visible_half * 1.05)
            return (tuple(fr.look_at_nm), half)
        dist = math.dist(fr.position_nm, fr.look_at_nm)
        half = max(500.0, dist * math.tan(math.radians(fr.fov_deg) / 2) * 1.25)
        return (tuple(fr.look_at_nm), half)

    def _frame_nm_per_px(self, fr) -> float:
        """Physical size of one rendered screen pixel at the camera target."""
        height = max(1, self.job.settings.height)
        if getattr(fr, "projection", "PERSP") == "ORTHO" and getattr(fr, "ortho_scale_nm", None):
            return float(fr.ortho_scale_nm) / height
        dist = math.dist(fr.position_nm, fr.look_at_nm)
        return 2.0 * dist * math.tan(math.radians(fr.fov_deg) / 2) / height

    def _frame_seg_overlays(self, fr):
        """Visible segmentation layers in this frame to overlay on the EM slice."""
        out = []
        for m in fr.meshes:
            if not self._layer_visible(m):
                continue
            src = next((s for s in self.manifest.meshes if s.name == m.mesh_name), None)
            if src and src.label_zarr and m.segment_ids:
                out.append((src.label_zarr, m.segment_ids, self._frame_colors(m)))
        return out

    def _frame_seg_overlays_for_layers(self, fr, layer_names):
        wanted = {str(x) for x in (layer_names or []) if str(x)}
        if not wanted:
            return []
        out = []
        for m in fr.meshes:
            if m.mesh_name not in wanted or not self._layer_visible(m):
                continue
            src = next((s for s in self.manifest.meshes if s.name == m.mesh_name), None)
            if src and src.label_zarr and m.segment_ids:
                out.append((src.label_zarr, m.segment_ids, self._frame_colors(m)))
        return out

    def _frame_slice_reads(self, fr, t_global):
        """Visible slices in this frame as (FrameSlice, slice_seg, include_seg_overlays).

        Keyframe EM slices keep the neuroglancer-like visible segmentation overlays.
        Timeline plane-scan sweeps are EM-only unless the sweep explicitly targets a
        segmentation layer; otherwise an EM scan unexpectedly drags every visible label
        source through the slice cache and paints labels onto the scan.

        A timeline sweep is an animation clip for a plane slot.  If it is active for the
        same EM/axis slot as a keyframe slice, it replaces the keyframe slice for that
        frame; otherwise the static keyframe plane hangs around as a second EM image while
        the sweep passes through it.
        """
        out = []
        keyframe_items = [(sl, True, None) for sl in list(fr.slices)]
        sweep_items = [
            (sl, False, overlay_layers)
            for sl, overlay_layers in self._slices_from_sweeps(t_global)
        ]
        sweep_slots = {(sl.em_name, sl.axis) for sl, _, _ in sweep_items}
        slice_items = [
            item for item in keyframe_items
            if (item[0].em_name, item[0].axis) not in sweep_slots
        ]
        slice_items.extend(sweep_items)
        for sl, allow_seg_overlays, overlay_layers in slice_items:
            if not getattr(sl, "visible", True):
                continue
            if sl.opacity <= 0.001:
                continue
            _, is_label = self._vol_for(sl.em_name)
            slice_seg = None
            if is_label:
                sm = next((mm for mm in fr.meshes
                           if mm.mesh_name == sl.em_name and mm.segment_ids
                           and self._layer_visible(mm)), None)
                if sm:
                    slice_seg = (list(sm.segment_ids), self._frame_colors(sm))
            out.append((sl, slice_seg, bool(allow_seg_overlays and not is_label), list(overlay_layers or [])))
        return out

    def _include_timeline_sweeps(self) -> bool:
        """Keyframe thumbnails are representative stills of the keyframe row.

        Timeline sweeps have their own preview lane; including them in a still thumbnail
        lets the first active EM plane/cutaway hide unrelated keyframe meshes.
        """
        return not bool(getattr(self.job.settings, "still", False))

    def _t_global(self, fi, frame_times, index_offset):
        return (frame_times[fi] if frame_times
                else (index_offset + fi) / max(1, self.job.settings.fps))

    @staticmethod
    def _atomic_save(img, path) -> None:
        """Save a PNG so parallel writers of the SAME path can't leave/observe a partial
        file: write a per-thread temp, then atomically rename into place (last wins).
        Identical-key slices (e.g. a camera hold fetched concurrently) hit the same path."""
        tmp = f"{path}.{threading.get_ident()}.tmp"
        img.save(tmp, format="PNG")
        os.replace(tmp, path)

    def _mesh_obj(self, mesh_name, segment_ids, lc, nmpp=None, colorize_segments: bool = False) -> str | None:
        src = next((m for m in self.manifest.meshes if m.name == mesh_name), None)
        if not src:
            return None
        ids = segment_ids or src.segment_ids
        if not ids:
            return None
        color_key = lc.cache_key() if colorize_segments else ()
        uid = self._mesh_uid(mesh_name, ids, color_key=color_key, nmpp=nmpp)
        # `.npz` (verts + faces + optional vertex colors) instead of `.ply`: Blender's
        # `bpy.ops.wm.ply_import` routes through the operator/undo system and is the
        # dominant cost of cold-starting a render; a numpy → `foreach_set` load is
        # ~5–10× faster on big meshes.
        out = self.assets_dir / f"mesh_{uid}.npz"
        if out.exists():
            return str(out)
        try:
            if src.skeleton_url and not src.mesh_url and not src.label_zarr:
                # skeleton-only layer -> sweep skeletons into tubes
                from ..data.skeleton import SkeletonLoader

                combined = SkeletonLoader(src.skeleton_url, shader=src.skeleton_shader).load_many(
                    ids, colorize=lc.rgb)
            else:
                combined = MeshLoader(src.mesh_url, src.label_zarr,
                                      cache_dir=self._mesh_cache_dir).load_many(
                    ids, colorize=lc.rgb if colorize_segments else None,
                    target_voxels_single=self._mesh_voxels_single,
                    target_voxels_union=self._mesh_voxels_union,
                    nm_per_px=nmpp, draft=self._draft,
                    prefer_labels=self._prefer_labels, total_budget=self._mesh_budget,
                    label_smooth_iters=self._label_smooth_iters,
                    label_decimate_fraction=self._label_decimate_fraction,
                    label_blockwise=self._label_blockwise)
        except Exception as e:  # noqa: BLE001
            print(f"[worker] mesh {mesh_name} ({len(ids)} segs) failed: {e}")
            return None
        os.makedirs(out.parent, exist_ok=True)
        _export_mesh_npz(combined, out)
        return str(out)

    @staticmethod
    def _clip_normal(cl: dict) -> list[float]:
        nrm = cl.get("normal")
        if nrm:
            mag = math.sqrt(sum(float(v) * float(v) for v in nrm)) or 1.0
            return [float(v) / mag for v in nrm]
        return {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }.get(cl.get("axis", "z"), [0.0, 0.0, 1.0])

    def _fast_cutaway_assets(
        self,
        *,
        mesh_name: str,
        segment_ids,
        lc,
        base_uid: str,
        base_path: str,
        clip: dict,
        color,
    ) -> tuple[str, str] | None:
        """Build cached pre-clipped geometry + voxel cap for label-derived cutaways.

        This replaces Blender's per-frame bmesh cap path for zmesh/label renders.
        It is intentionally conservative: precomputed mesh renders still use the old
        Blender path because their surface may not match the label volume exactly.
        """
        if not self._prefer_labels:
            return None
        src = next((m for m in self.manifest.meshes if m.name == mesh_name), None)
        if not src or not src.label_zarr:
            return None
        ids = list(segment_ids or src.segment_ids or [])
        if not ids:
            return None

        import hashlib
        import numpy as np

        normal = self._clip_normal(clip)
        side = 1.0 if float(clip.get("side", 1)) >= 0 else -1.0
        position_nm = float(clip.get("position_nm", 0.0))
        sig_src = json.dumps(
            [
                "fast-cutaway-v1",
                base_uid,
                src.label_zarr,
                sorted(int(s) for s in ids),
                lc.cache_key(),
                [round(float(x), 6) for x in normal],
                round(position_nm, 3),
                side,
                self._mesh_budget,
            ],
            sort_keys=True,
            default=str,
        )
        sig = hashlib.md5(sig_src.encode()).hexdigest()[:10]
        cut_uid = f"{base_uid}_cut_{sig}"
        cap_uid = f"{base_uid}_cap_{sig}"
        cut_path = self.assets_dir / f"mesh_{cut_uid}.npz"
        cap_path = self.assets_dir / f"mesh_{cap_uid}.npz"

        try:
            if not cut_path.exists() or not cap_path.exists():
                with np.load(base_path, allow_pickle=False) as z:
                    verts = np.asarray(z["v"], dtype=np.float32)
                    lo = verts.min(axis=0)
                    hi = verts.max(axis=0)
                bbox = (tuple(float(x) for x in lo), tuple(float(x) for x in hi))
                from ..data.cutaway import write_exact_clipped_npz, write_voxel_cap_npz

                cut_info = write_exact_clipped_npz(
                    Path(base_path),
                    cut_path,
                    normal_xyz=normal,
                    position_nm=position_nm,
                    side=side,
                )
                cap_info = write_voxel_cap_npz(
                    src.label_zarr,
                    ids,
                    cap_path,
                    bbox_xyz_nm=bbox,
                    normal_xyz=normal,
                    position_nm=position_nm,
                    target_vertices=self._mesh_budget,
                    colorize=lc.rgb,
                )
                print(
                    "[worker] fast cutaway "
                    f"{mesh_name}: cut_faces={cut_info.get('faces')} "
                    f"cap_faces={cap_info.get('faces')}"
                )
        except Exception as e:  # noqa: BLE001
            print(f"[worker] fast cutaway {mesh_name} failed; falling back to bmesh cap: {e}")
            return None

        # Treat both assets as plain meshes: no Blender clip nodes, no bmesh cap.
        # The cap carries vertex colors when possible; color remains as material fallback.
        return cut_uid, cap_uid

    @staticmethod
    def _ann_uid(an) -> str:
        """Stable id per (layer, geometry, color) so an edited annotation layer
        becomes a distinct asset."""
        import hashlib

        sig = json.dumps(["ann-v2", an.name, an.color, an.points, an.lines, an.boxes, an.ellipsoids,
                          an.point_radius_nm, an.line_radius_nm], sort_keys=True)
        return f"ann_{hashlib.md5(sig.encode()).hexdigest()[:10]}"

    def _extent_for_layer(self, name: str, kfs):
        """(lo_xyz, hi_xyz) nm extent of one named layer for the bounding box — the EM
        image's volume extent, a seg layer's label-volume extent, or (precomputed-mesh
        layer) its mesh AABB. Works even if the layer is hidden in the keyframes."""
        em = self.manifest.em
        if em and name == em.name:
            try:
                return self._em_vol().extent_nm()
            except Exception as e:  # noqa: BLE001
                print(f"[worker] bbox: EM extent failed: {e}")
                return None
        src = next((s for s in self.manifest.meshes if s.name == name), None)
        if src is None:
            return None
        if src.label_zarr:
            try:
                return self._label_vol(src.label_zarr).extent_nm()
            except Exception as e:  # noqa: BLE001
                print(f"[worker] bbox: {name} label extent failed: {e}")
        if src.mesh_url:
            from ..operations import mesh_aabb_nm
            ids = sorted({i for kf in kfs for m in kf.meshes
                          if m.mesh_name == name for i in m.segment_ids})
            try:
                return mesh_aabb_nm(src.mesh_url, ids)   # ids=[] -> samples the layer
            except Exception as e:  # noqa: BLE001
                print(f"[worker] bbox: {name} mesh AABB failed: {e}")
        return None

    def _bbox_boxes(self) -> list[tuple[list[float], list[float]]]:
        """Data-source extent boxes to outline (neuroglancer-style): the EM volume's
        box plus each rendered layer's label-volume box. When there's NO volume source
        (precomputed-mesh-only layers), falls back to a SINGLE union box around all the
        rendered meshes (not one per layer). (lo_xyz, hi_xyz) nm."""
        boxes: list[tuple[list[float], list[float]]] = []
        seen: set = set()

        def add(lo, hi):
            if lo is None or hi is None or not all(hi[i] > lo[i] for i in range(3)):
                return
            key = tuple(round(float(v), 1) for v in (*lo, *hi))
            if key not in seen:
                seen.add(key)
                boxes.append(([float(v) for v in lo], [float(v) for v in hi]))

        kfs = getattr(self, "_kfs", None) or self.project.keyframes
        used = {m.mesh_name for kf in kfs for m in kf.meshes}
        # explicit source: box THAT layer's extent, even if it's currently hidden
        if self._bbox_source:
            add(*( self._extent_for_layer(self._bbox_source, kfs) or (None, None) ))
            return boxes
        if self.manifest.em:
            try:
                add(*self._em_vol().extent_nm())
            except Exception as e:  # noqa: BLE001
                print(f"[worker] bbox: EM extent failed: {e}")
        for src in self.manifest.meshes:
            if src.name in used and src.label_zarr:
                try:
                    add(*self._label_vol(src.label_zarr).extent_nm())
                except Exception as e:  # noqa: BLE001
                    print(f"[worker] bbox: {src.name} label extent failed: {e}")
        if not boxes:   # no volume source -> ONE box around all the rendered meshes
            from ..operations import mesh_aabb_nm
            ulo = uhi = None
            for src in self.manifest.meshes:
                if src.name not in used or not src.mesh_url:
                    continue
                ids = sorted({i for kf in kfs for m in kf.meshes
                              if m.mesh_name == src.name for i in m.segment_ids})
                try:
                    bb = mesh_aabb_nm(src.mesh_url, ids)
                except Exception as e:  # noqa: BLE001
                    print(f"[worker] bbox: {src.name} mesh AABB failed: {e}")
                    continue
                if not bb:
                    continue
                blo, bhi = bb
                ulo = list(blo) if ulo is None else [min(a, b) for a, b in zip(ulo, blo)]
                uhi = list(bhi) if uhi is None else [max(a, b) for a, b in zip(uhi, bhi)]
            add(ulo, uhi)   # a single union box, not one per layer
        return boxes

    def _bbox_annotation(self) -> FrameAnnotation | None:
        """A synthetic annotation layer holding the data-source bounding box(es) as
        boxes, drawn as thin wireframe tubes through the normal annotation path."""
        boxes = self._bbox_boxes()
        if not boxes:
            return None
        # tube radius scaled to the box so it's a visible (not hairline) edge at any
        # dataset size — ~0.3% of the largest span, floored so small boxes still show
        span = max((hi[i] - lo[i]) for lo, hi in boxes for i in range(3))
        radius = max(60.0, span * 0.003)
        return FrameAnnotation(
            name="__bbox__", color=self._bbox_color, opacity=1.0,
            boxes=[[lo, hi] for lo, hi in boxes], line_radius_nm=radius)

    def _ann_obj(self, an) -> str | None:
        from ..data.annotations import annotations_to_mesh

        uid = self._ann_uid(an)
        out = self.assets_dir / f"{uid}.npz"
        if out.exists():
            return str(out)
        prims = {"points": an.points, "lines": an.lines, "boxes": an.boxes,
                 "ellipsoids": an.ellipsoids}
        mesh = annotations_to_mesh(prims, an.color, an.point_radius_nm, an.line_radius_nm)
        if mesh is None:
            return None
        os.makedirs(out.parent, exist_ok=True)
        _export_mesh_npz(mesh, out)
        return str(out)

    # rough draco bytes per vertex, only used to turn the layer vertex budget into a
    # byte ceiling for the per-chunk selection (raise tolerance if a frame is over).
    _CHUNK_BYTES_PER_VERT = 12

    def _build_chunk_assets(self, frames, mesh_specs) -> list[dict]:
        """True per-chunk LOD — mimics neuroglancer's loading. Per frame, per layer,
        per segment: walk the mesh octree, frustum-cull off-screen fragments, and keep
        each fragment at the coarsest LOD that's still sharp for ITS on-screen size
        (`lodScale <= pixel*tol`). So a single object renders fine only where it's near
        the camera and coarse/absent elsewhere. The layer vertex budget acts as a
        ceiling (tolerance is raised until a frame fits). Only the selected fragments
        are fetched + assembled, cached by selection so identical frames reuse it.
        Returns per-frame {layer name: asset uid}."""
        import hashlib
        import math as _m
        from concurrent.futures import ThreadPoolExecutor

        import trimesh

        from ..data.mesh_loader import MeshLoader, _FETCH_WORKERS
        from .visibility import select_fragments

        H = max(1, self.job.settings.height)
        aspect = self.job.settings.width / H
        ceiling = self._mesh_budget * self._CHUNK_BYTES_PER_VERT   # layer byte budget

        # one loader per 3D mesh layer; warm each segment's octree (fragment_boxes is
        # cached on the loader) in parallel so per-frame selection is pure CPU.
        layers: dict[str, tuple] = {}
        for fr in frames:
            for m in fr.meshes:
                if self._mesh_render_alpha(m) <= 0.001 or not m.segment_ids or m.mesh_name in layers:
                    continue
                src = next((s for s in self.manifest.meshes
                            if s.name == m.mesh_name and s.mesh_url), None)
                if not src:
                    continue
                ld = MeshLoader(src.mesh_url, src.label_zarr, cache_dir=self._mesh_cache_dir)
                with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(m.segment_ids))) as ex:
                    list(ex.map(ld.fragment_boxes, m.segment_ids))   # warm octree cache
                layers[m.mesh_name] = (ld, list(m.segment_ids))

        frame_layer_uid: list[dict] = []
        for fi, fr in enumerate(frames):
            if self.cancel.is_set():
                raise RenderCancelled()
            per: dict[str, str] = {}
            for m in fr.meshes:
                if self._mesh_render_alpha(m) <= 0.001 or m.mesh_name not in layers:
                    continue
                ld, seg_ids = layers[m.mesh_name]
                lc = self._frame_colors(m)
                colorize_segments = (
                    self._mesh_needs_vertex_colors(m, lc)
                    or self._mesh_uses_shader_vertex_colors(m.mesh_name)
                )
                ckey = str(lc.cache_key()) if colorize_segments else "solid"
                fov = _m.radians(fr.fov_deg)
                # select per-fragment LODs; raise tolerance until the frame fits budget
                tol = 1.0
                for _ in range(5):
                    seg_sel: dict[int, dict] = {}
                    total = 0
                    for s in seg_ids:
                        per_lod, lod_nm = ld.fragment_boxes(s)
                        sel, nbytes = select_fragments(per_lod, lod_nm, fr.position_nm,
                                                       fr.look_at_nm, fr.up, fov, aspect, H, tol=tol)
                        if sel:
                            seg_sel[s] = sel
                            total += nbytes
                    if total <= ceiling or not seg_sel:
                        break
                    tol *= 1.6
                if not seg_sel:
                    continue
                sig = tuple(sorted(
                    (s, tuple(sorted((L, tuple(sorted(ix))) for L, ix in sel.items())))
                    for s, sel in seg_sel.items()))
                # `frag2` versions the per-chunk decoder: bump to invalidate combined chunk
                # assets built before the manual-fragment fix (cloud-volume scattered them).
                uid = f"{m.mesh_name}_{hashlib.md5((ckey + '|' + str(sig) + '|frag4').encode()).hexdigest()[:10]}"
                per[m.mesh_name] = uid
                if uid not in mesh_specs:
                    def _one(s, _ld=ld, _lc=lc, _sel=seg_sel, _colorize=colorize_segments):
                        return _ld.get_fragments(s, _sel[s], colorize=_lc.rgb if _colorize else None)

                    with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(seg_sel))) as ex:
                        parts = [p for p in ex.map(_one, list(seg_sel)) if p is not None]
                    if not parts:
                        per.pop(m.mesh_name, None)
                        continue
                    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
                    out = self.assets_dir / f"mesh_{uid}.npz"
                    os.makedirs(out.parent, exist_ok=True)
                    _export_mesh_npz(combined, out)
                    mesh_specs[uid] = {
                        "id": uid,
                        "obj_path": str(out),
                        "color": self._mesh_material_color(m, lc),
                    }
            frame_layer_uid.append(per)
            self._progress(0.1 + 0.35 * (fi + 1) / len(frames),
                           f"chunk assets {fi + 1}/{len(frames)} ({len(mesh_specs)} built)")
        return frame_layer_uid

    # ---- scene spec ----
    def _build_scene_spec(self, frames: list[FrameState], index_offset: int = 0,
                          frame_times: list[float] | None = None) -> dict:
        # On-screen scale (nm per pixel) per frame, for picking precomputed-mesh LOD
        # like neuroglancer. With dynamic LOD this varies per frame (coarser when the
        # layer is far/small on screen); otherwise every frame uses the finest.
        frame_nmpp = [self._frame_nm_per_px(fr) for fr in frames]
        self._nm_per_px = min(frame_nmpp, default=None)
        frame_lod_nmpp = self._lod_bucket_nmpp(frame_nmpp)  # nm/px to build each frame at

        # neuroglancer-style bounding box: a static wireframe layer on every frame
        if self._show_bbox:
            bbox_an = self._bbox_annotation()
            if bbox_an is not None:
                for fr in frames:
                    fr.annotations = list(fr.annotations) + [bbox_an]

        # One Blender object per distinct (layer, segment set, LOD bucket): a frame
        # references only its bucket's variant and the others auto-hide, so far frames
        # render a coarse mesh and close-ups a fine one.
        mesh_specs: dict[str, dict] = {}
        # Cutaway LOD pinning: a clipped layer must be ONE consistent LOD across the frames
        # where it's cut (else the cross-section cap cracks). Find, per layer, the frames it
        # is clipped in, and pin those to the finest nmpp among them — so ONLY the cut layer,
        # and ONLY in its cutaway window, is forced to a single LOD; everything else keeps
        # adaptive (coarse-when-far) LOD and stays light.
        def _ft(i):
            return frame_times[i] if frame_times else (index_offset + i) / max(1, self.job.settings.fps)
        clip_frames: dict[str, set] = {}
        for fi, fr in enumerate(frames):
            t = _ft(fi)
            for m in fr.meshes:
                if self._mesh_render_alpha(m) <= 0.001:
                    continue
                cl = self._clip_from_sweeps(m.mesh_name, t) or _clip_params(getattr(m, "clip", None))
                if cl and cl.get("cap", True):
                    clip_frames.setdefault(m.mesh_name, set()).add(fi)
        clip_nmpp = {layer: min(frame_lod_nmpp[i] for i in fis) for layer, fis in clip_frames.items()}

        def _eff_nmpp(layer, fi):   # pinned LOD inside a layer's cutaway window, else adaptive
            return clip_nmpp[layer] if (layer in clip_nmpp and fi in clip_frames[layer]) else frame_lod_nmpp[fi]

        # "chunk" mode: per-frame frustum cull + per-segment on-screen LOD (see
        # _build_chunk_assets); other modes: one combined mesh per (layer, segset, bucket).
        frame_layer_uid = (self._build_chunk_assets(frames, mesh_specs)
                           if self._lod_mode == "chunk" else None)
        for fi, fr in enumerate(frames):
            if self.cancel.is_set():
                raise RenderCancelled()
            if frame_layer_uid is None:
                for m in fr.meshes:
                    if self._mesh_render_alpha(m) <= 0.001:
                        continue
                    nmpp = _eff_nmpp(m.mesh_name, fi)
                    lc = self._frame_colors(m)
                    colorize_segments = self._mesh_needs_vertex_colors(m, lc)
                    color_key = lc.cache_key() if colorize_segments else ()
                    uid = self._mesh_uid(m.mesh_name, m.segment_ids, color_key=color_key, nmpp=nmpp)
                    if uid not in mesh_specs:
                        obj = self._mesh_obj(
                            m.mesh_name, m.segment_ids, lc, nmpp,
                            colorize_segments=colorize_segments,
                        )
                        if obj:
                            mesh_specs[uid] = {
                                "id": uid,
                                "obj_path": obj,
                                "color": self._mesh_material_color(m, lc),
                            }
            # annotation layers -> geometry assets via the same import path
            for an in fr.annotations:
                uid = self._ann_uid(an)
                if uid not in mesh_specs:
                    obj = self._ann_obj(an)
                    if obj:
                        mesh_specs[uid] = {"id": uid, "obj_path": obj, "color": an.color}
            if frame_layer_uid is None:
                self._progress(0.1 + 0.35 * (fi + 1) / len(frames),
                               f"loading meshes {fi + 1}/{len(frames)}")
        # director Phase 2: per-frame (hero, glow, spotlight) emphasis for appear/
        # highlight events — a brief glow on the new structure + a context dip.
        emph_track = None
        if self._auto_direct:
            from . import director
            emph_track = director.emphasis_track(getattr(self, "_kfs", self.project.keyframes),
                                                 self.job.settings.fps)
        # Warm the slice-image cache before the frame-spec build loop. TensorStore handles
        # chunk-level IO/decompression concurrency inside each read; keep CineMap's outer
        # slice-read fanout low so oblique slices do not stack large temporary subvolumes.
        # _slice_png memoizes into self._slice_cache, so the build loop below becomes
        # pure-CPU cache hits.
        from concurrent.futures import ThreadPoolExecutor
        warm_jobs = []
        warm_seen: set[tuple] = set()
        for fi, fr in enumerate(frames):
            region = self._frame_region(fr)
            target_nm_per_px = self._frame_nm_per_px(fr)
            seg_overlays = self._frame_seg_overlays(fr)
            for sl, slice_seg, include_seg_overlays, overlay_layers in self._frame_slice_reads(
                fr, self._t_global(fi, frame_times, index_offset)
            ):
                slice_overlays = (
                    seg_overlays if include_seg_overlays
                    else self._frame_seg_overlays_for_layers(fr, overlay_layers)
                )
                key = self._slice_cache_key(sl, region, slice_overlays, slice_seg=slice_seg,
                                            target_nm_per_px=target_nm_per_px)
                if key in warm_seen:
                    continue
                warm_seen.add(key)
                warm_jobs.append((sl, region, slice_overlays, slice_seg, target_nm_per_px))

        def _warm(job):
            sl, region, seg_overlays, slice_seg, target_nm_per_px = job
            try:
                self._slice_png(sl, region, seg_overlays, slice_seg=slice_seg,
                                target_nm_per_px=target_nm_per_px)
            except Exception:  # noqa: BLE001 (failures re-surface in the build loop below)
                pass

        if warm_jobs and not self.cancel.is_set():
            self._progress(0.45, f"fetching {len(warm_jobs)} unique slice images")
            done = 0

            def _mark_done():
                nonlocal done
                done += 1
                if done % 8 == 0 or done == len(warm_jobs):
                    self._progress(0.45 + 0.10 * done / len(warm_jobs),
                                   f"fetching slice images {done}/{len(warm_jobs)}")

            fallback_jobs = self._warm_oblique_batches(warm_jobs, _mark_done)
            if fallback_jobs and not self.cancel.is_set():
                workers = min(_SLICE_FETCH_WORKERS, len(fallback_jobs))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for _ in ex.map(_warm, fallback_jobs):
                        _mark_done()
            elif done < len(warm_jobs):
                self._progress(0.45 + 0.10 * done / len(warm_jobs),
                               f"fetching slice images {done}/{len(warm_jobs)}")

        frame_specs = []
        for fi, fr in enumerate(frames):
            if self.cancel.is_set():
                raise RenderCancelled()
            emph_hero, emph_glow, emph_spot = (
                emph_track[fi] if emph_track and fi < len(emph_track) else (None, 0.0, 1.0))
            region = self._frame_region(fr)
            target_nm_per_px = self._frame_nm_per_px(fr)
            # Visible segmentation layers in this frame -> optional colored overlays on EM.
            # Hidden layers stay hidden so slice-only shots do not reintroduce meshes/labels.
            seg_overlays = self._frame_seg_overlays(fr)
            slices = []
            # keyframe slices PLUS any 'slice' sweeps evaluated on the global timeline
            t_global = self._t_global(fi, frame_times, index_offset)
            for sl, slice_seg, include_seg_overlays, overlay_layers in self._frame_slice_reads(fr, t_global):
                # slot is stable across frames (matches interpolate's slice identity)
                # so the blend exporter can group a slice's per-frame images into one
                # animated image-sequence plane. A slice that can't be read (e.g. a
                # non-OME-Zarr EM source) is skipped, not fatal to the whole render.
                # _slice_png is already cached from the parallel warm pass above.
                try:
                    slice_overlays = (
                        seg_overlays if include_seg_overlays
                        else self._frame_seg_overlays_for_layers(fr, overlay_layers)
                    )
                    png = self._slice_png(sl, region, slice_overlays, slice_seg=slice_seg,
                                          target_nm_per_px=target_nm_per_px)
                except Exception as e:  # noqa: BLE001
                    print(f"[worker] slice {sl.em_name}:{sl.axis} failed: {e}")
                    continue
                # occlude=True -> the EM plane renders as a SOLID cross-section (blocks what's
                # behind it) instead of a see-through additive overlay. opacity still fades it.
                slices.append({**png, "opacity": sl.opacity, "occlude": True,
                               "slot": f"{sl.em_name}:{sl.axis}"})
            overrides = {}
            for m in fr.meshes:
                lc = self._frame_colors(m)
                if frame_layer_uid is not None:           # chunk mode: per-frame selection
                    uid = frame_layer_uid[fi].get(m.mesh_name)
                else:
                    colorize_segments = self._mesh_needs_vertex_colors(m, lc)
                    color_key = lc.cache_key() if colorize_segments else ()
                    uid = self._mesh_uid(
                        m.mesh_name, m.segment_ids, color_key=color_key,
                        nmpp=_eff_nmpp(m.mesh_name, fi),
                    )
                if uid and uid in mesh_specs:
                    material_color = self._mesh_material_color(m, lc)
                    # effective 3D alpha = layer visibility * cinematic fade * NG "Opacity (3d)"
                    eff = self._mesh_render_alpha(m)
                    is_hero = (m.mesh_name == emph_hero)
                    if emph_track and not is_hero and emph_spot < 1.0:
                        eff *= emph_spot                 # spotlight: briefly dim context
                    ov = {
                        "opacity": eff,
                        "visible": eff > 0.001,
                        "silhouette": getattr(m, "silhouette", 0.0),
                        "color": material_color,
                    }
                    if getattr(m, "metallic", None) is not None:    # per-frame material override
                        ov["metallic"] = m.metallic
                    if getattr(m, "roughness", None) is not None:
                        ov["roughness"] = m.roughness
                    if is_hero and emph_glow > 0.0:
                        ov["emphasis"] = emph_glow       # brief emission glow on the hero
                    # a sweep (independent timeline) overrides the keyframe's own clip
                    t_global = (frame_times[fi] if frame_times else (index_offset + fi) / max(1, self.job.settings.fps))
                    cl = self._clip_from_sweeps(m.mesh_name, t_global) or _clip_params(
                        getattr(m, "clip", None))
                    if cl:
                        cl = {**cl, "position_bu": cl["position_nm"] / self.nm_per_bu}
                        fast = None
                        if cl.get("cap", True) and uid in mesh_specs:
                            fast = self._fast_cutaway_assets(
                                mesh_name=m.mesh_name,
                                segment_ids=m.segment_ids,
                                lc=lc,
                                base_uid=uid,
                                base_path=mesh_specs[uid]["obj_path"],
                                clip=cl,
                                color=material_color,
                            )
                        if fast:
                            cut_uid, cap_uid = fast
                            mesh_specs[cut_uid] = {
                                "id": cut_uid,
                                "obj_path": str(self.assets_dir / f"mesh_{cut_uid}.npz"),
                                "color": material_color,
                                "clip": False,
                                "clip_cap": False,
                            }
                            mesh_specs[cap_uid] = {
                                "id": cap_uid,
                                "obj_path": str(self.assets_dir / f"mesh_{cap_uid}.npz"),
                                "color": material_color,
                                "clip": False,
                                "clip_cap": False,
                            }
                            overrides[cut_uid] = dict(ov)
                            cap_ov = dict(ov)
                            cap_ov["silhouette"] = 0.0
                            overrides[cap_uid] = cap_ov
                            continue
                        ov["clip"] = cl
                        # clip=True means the material needs animated clip nodes. cap=True
                        # additionally requests the expensive geometric cut + filled face.
                        mesh_specs[uid]["clip"] = True
                        mesh_specs[uid]["clip_cap"] = bool(mesh_specs[uid].get("clip_cap")) or bool(
                            cl.get("cap", True)
                        )
                    overrides[uid] = ov
            for an in fr.annotations:
                uid = self._ann_uid(an)
                if uid in mesh_specs:
                    overrides[uid] = {"opacity": an.opacity, "visible": an.opacity > 0.001,
                                      "silhouette": 0.0}
            frame_specs.append({
                "camera": {
                    "position_bu": _bu(fr.position_nm, self.nm_per_bu),
                    "look_at_bu": _bu(fr.look_at_nm, self.nm_per_bu),
                    "fov_rad": math.radians(fr.fov_deg),
                    "up": fr.up,
                    "flip_handed": self._flip_handed,
                    "type": "ORTHO" if getattr(fr, "projection", "PERSP") == "ORTHO" else "PERSP",
                    "ortho_scale": (
                        float(fr.ortho_scale_nm) / self.nm_per_bu
                        if getattr(fr, "ortho_scale_nm", None) else 4.0
                    ),
                },
                "slices": slices,
                "mesh_overrides": overrides,
                "fade_alpha": max(0.0, min(1.0, float(getattr(fr, "fade_alpha", 0.0) or 0.0))),
                "background": list(getattr(fr, "background", None) or self.project.lighting.background),
                "index": index_offset + fi,   # global frame index (split cluster jobs)
            })
            self._progress(0.45 + 0.15 * (fi + 1) / len(frames),
                           f"building frame specs {fi + 1}/{len(frames)}")
        used_mesh_ids = {mid for fr in frame_specs for mid in fr.get("mesh_overrides", {})}
        mesh_specs = {mid: spec for mid, spec in mesh_specs.items() if mid in used_mesh_ids}
        spec = {
            "world": {"nm_per_bu": self.nm_per_bu,
                      "background": self.project.lighting.background},
            "lighting": {"key_energy": self.project.lighting.key_energy},
            "render": self.job.settings.model_dump(),
            "meshes": list(mesh_specs.values()),
            "frames": frame_specs,
            "output_dir": str(self.frames_dir),
            "fps": self.job.settings.fps,
            "export_blend": str(self.blend_path) if self.job.settings.export_blend else None,
        }
        # Material/light/view directives are always present. `auto_direct` only
        # controls optional smoothing and emphasis; selected Look presets must still
        # drive the Blender scene when those cinematic behaviors are disabled.
        from . import director
        plan = director.plan(
            self.project.keyframes,
            _director_settings_for_render(self.project, self._auto_direct),
        )
        spec["direction"] = plan
        if plan["dof"]["enabled"]:
            for fr in spec["frames"]:
                fr["camera"]["dof"] = {"fstop": plan["dof"]["fstop"]}
        # Warm-scene cache: full-quality renders reuse a built .blend (skip re-import +
        # weld) keyed by the geometry+look signature. Previews/thumbnails (draft) skip it —
        # they're already fast and the multi-GB .blend save would only slow them. Cache
        # lives in fast local /tmp (not NFS), and is purely a speedup (safe to evict).
        # Apply to BOTH draft (thumbnails / sweep previews) and full renders — the cache is
        # keyed by GEOMETRY, not project, so re-importing the same movie or re-baking the
        # same keyframe/sweep reuses the built scene instead of re-importing every time.
        # (draft and full get separate keys: the mesh ids already encode draft + budget.)
        if not self.job.settings.export_blend:
            import hashlib
            import tempfile
            # The key must include EVERY input _import_meshes uses to BUILD geometry+materials,
            # so a stale cache can never be reused: per-mesh (id=geometry+color+LOD, clip=weld,
            # color=tint), the whole material profile (direction.material -> roughness/metallic/
            # ao/cavity/edge/ng_shader/flat_shading/backface_cull/…), engine, nm_per_bu scale,
            # and auto_direct. Include an explicit material-graph version so shader-node fixes
            # (for example the NG light vector) cannot reuse stale warm .blends. Per-frame
            # state (camera/opacity/metallic OVERRIDES/clip position/slices/lights) is
            # re-driven on the cached geometry, so it correctly does NOT key.
            geom = sorted((mm["id"], bool(mm.get("clip")),
                           bool(mm.get("clip_cap", mm.get("clip"))),
                           tuple(mm.get("color") or ()))
                          for mm in mesh_specs.values())
            sig_src = json.dumps(["blender-material-v4",
                                  geom, spec.get("direction", {}).get("material", {}),
                                  self.job.settings.engine, round(self.nm_per_bu, 6),
                                  bool(self._auto_direct)],
                                 sort_keys=True, default=str)
            sig = hashlib.md5(sig_src.encode()).hexdigest()[:16]
            warm_dir = Path(tempfile.gettempdir()) / "cinemap_warm"
            spec["warm_blend"] = str(warm_dir / f"{sig}.blend")
            self._evict_warm_cache(warm_dir)
        return spec

    @staticmethod
    def _evict_warm_cache(warm_dir: Path, max_total_gb: float = 20.0) -> None:
        """Keep the most-recently-used warm .blend files up to a total size budget (they
        can be GBs each for full renders, much smaller for draft) — newest first, evict the
        rest. Size-based so it works whether there are a few huge or many small scenes."""
        try:
            blends = sorted(warm_dir.glob("*.blend"), key=lambda p: p.stat().st_mtime, reverse=True)
            total = 0
            for p in blends:
                total += p.stat().st_size
                if total > max_total_gb * 1e9:
                    p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    # ---- run ----
    def _progress(self, p, msg):
        self.job.progress = round(p, 3)
        self.job.message = msg
        if self._cb:
            self._cb(p, msg)

    def _kill_proc(self) -> None:
        """SIGKILL the Blender subprocess AND its whole process group (it's launched in its
        own session), so OptiX/denoiser/helper children die too — no orphans on Stop."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        import signal as _signal
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def terminate(self) -> None:
        """Stop the render: set the cancel flag (the asset/frame loops bail at their next
        check) and hard-kill the Blender subprocess group so nothing keeps running."""
        self.cancel.set()
        self._kill_proc()

    def _run_blender(self, scene_path, nframes: int) -> bool:
        """Launch the Blender render subprocess and stream progress. Returns True if it
        failed specifically with a GPU out-of-memory error (so the caller can retry at
        lower mesh detail); raises RuntimeError on any other non-zero exit."""
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,   # own process group -> terminate() can kill the whole tree
        )
        proc = self._proc
        done, oom = 0, False
        for line in proc.stdout:  # type: ignore
            if self.cancel.is_set():
                self._kill_proc()
                break
            low = line.lower()
            if ("out of memory" in low or "out of gpu memory" in low
                    or "cuda_error_out_of_memory" in low or "failed to allocate" in low):
                oom = True
            if line.startswith("[blender] frame"):
                done += 1
                self._progress(0.6 + 0.3 * done / nframes, line.strip())
            elif line.startswith("[blender]"):
                # surface Blender's own status (GPU device, sampling, warm-cache reuse) in
                # the server log so it's visible without reading the subprocess directly.
                print(line.rstrip(), flush=True)
        proc.wait()
        if self.cancel.is_set():
            return False
        if proc.returncode != 0 and not oom:
            self.job.status = "error"
            raise RuntimeError(f"blender exited {proc.returncode}")
        return oom

    def _render_keyframes(self, kfs=None):
        """Keyframes as they should be rendered, with old saved 2D NG states upgraded
        to orthographic cameras without mutating/saving the project file."""
        from ..data.ng_camera import cross_section_plane, ng_to_cross_section_camera
        from ..models import SlicePlane
        from ..scouting import _image_contrast_from_layer

        out = []
        em_name = self.manifest.em.name if self.manifest.em else "em"
        for k in (kfs if kfs is not None else self.project.keyframes):
            st = getattr(k, "ng_state", None) or {}
            if st.get("layout") == "3d" or not st:
                out.append(k)
                continue
            cam = ng_to_cross_section_camera(st, self.manifest.voxel_size_nm)
            layers = {l.get("name"): l for l in st.get("layers", [])}
            image_layer = layers.get(em_name)
            visible = (image_layer or {}).get("visible", True) is not False
            slices = []
            if visible:
                axis, pos_nm, normal = cross_section_plane(st, self.manifest.voxel_size_nm)
                old = next((s for s in k.slices if s.em_name == em_name), None)
                slices = [SlicePlane(
                    em_name=em_name,
                    axis=axis,
                    position_nm=pos_nm,
                    normal=normal,
                    contrast_limits=(
                        _image_contrast_from_layer(image_layer)
                        or (old.contrast_limits if old else None)
                    ),
                )]
            out.append(k.model_copy(update={"camera": cam, "slices": slices}))
        return out

    def _state_at_time(self, t: float):
        """The interpolated FrameState (camera/slices/meshes) at a GLOBAL time t (seconds),
        for one-off snapshot frames. Holds the last keyframe past the end."""
        kfs = getattr(self, "_kfs", None) or self._render_keyframes()
        from . import director as _director
        smooth = self._auto_direct and _director.DirectorSettings().smooth_camera
        return state_at_time(kfs, t, smooth_ends=smooth)

    def render_snapshots(self, times: list[float], out_dir) -> list[str]:
        """Render single still frames at the given GLOBAL times (seconds) into out_dir —
        used for clip preview thumbnails. The camera is interpolated at each time and any
        sweeps are applied at that time (via _build_scene_spec's frame_times)."""
        import json
        from pathlib import Path

        from ..data.ng_camera import handedness_flipped
        self._cb = None
        self._kfs = self._render_keyframes()
        st0 = next((k.ng_state for k in self.project.keyframes if k.ng_state), None)
        self._flip_handed = bool(st0 and handedness_flipped(st0))
        frames = [self._state_at_time(t) for t in times]
        if any(f is None for f in frames):
            return []
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        self.frames_dir = out
        spec = self._build_scene_spec(frames, frame_times=list(times))
        spec["output_dir"] = str(out)
        sp = out / "snap_scene.json"; sp.write_text(json.dumps(spec))
        self._run_blender(sp, len(frames))
        return sorted(str(p) for p in out.glob("frame_*.png"))

    def run(self, progress: Progress | None = None) -> str:
        self._cb = progress
        self.job.status = "running"
        for d in (self.frames_dir, self.assets_dir):
            d.mkdir(parents=True, exist_ok=True)

        kfs = self._render_keyframes()
        if self.job.kf_range:
            a, b = self.job.kf_range
            kfs = kfs[a : b + 1]
        self._kfs = kfs   # the exact keyframes these frames came from (for the director)
        # Whether this dataset's NG axis order is a reflection (e.g. z,y,x): the camera
        # is reordered to xyz, which flips image chirality, so the render mirrors the
        # camera to match neuroglancer. Constant per dataset -> read once from any kf.
        from ..data.ng_camera import handedness_flipped
        st0 = next((k.ng_state for k in self.project.keyframes if k.ng_state), None)
        self._flip_handed = bool(st0 and handedness_flipped(st0))
        self._progress(0.05, "interpolating keyframes")
        # Timing matches neuroglancer's video_tool exactly (per-transition frame counts
        # and t-values). NG interpolates LINEARLY (constant velocity), so we default to
        # linear too — video_tool-faithful. The director's optional cinematic ease of the
        # first/last transition (smooth_camera) is opt-in, since it changes the velocity
        # profile of the opening/closing glide and so reads as different timing vs NG.
        from . import director as _director
        smooth = self._auto_direct and _director.DirectorSettings().smooth_camera
        frames = build_frames(kfs, self.job.settings.fps, smooth_ends=smooth)
        if not frames:
            raise ValueError("no keyframes to render")
        if getattr(self.job.settings, "still", False):
            frames = frames[:1]   # a thumbnail: one frame only (no hold / no sweep extension)
        else:
            # A cutaway sweep runs on the GLOBAL timeline, so the movie must be at least as
            # long as the furthest sweep — otherwise a sweep over a single (static) keyframe
            # gets just one frame. Hold the last camera pose out to the latest sweep end.
            fps = max(1, self.job.settings.fps)
            sweep_end = max([0.0] + [float(s.start_s) + float(s.duration_s)
                                     for s in (getattr(self.project, "sweeps", []) or [])
                                     if getattr(s, "enabled", True)])
            need = int(round(sweep_end * fps))
            if need > len(frames):
                frames = frames + [frames[-1]] * (need - len(frames))

        exporting = self.job.settings.export_blend
        scene_path = self.workdir / "scene.json"
        # Build assets + render, retrying at half the mesh budget if Cycles runs out
        # of GPU memory (a too-high mesh_detail, or many dense layers in one frame).
        for attempt in range(3):
            self._progress(0.1, "preparing assets" if attempt == 0 else
                           f"GPU out of memory — retrying at lower detail "
                           f"({self._mesh_budget // 1000}k verts/layer)")
            spec = self._build_scene_spec(frames)  # raises RenderCancelled if cancelled
            if self.cancel.is_set():
                raise RenderCancelled()
            scene_path.write_text(json.dumps(spec, indent=2))
            self._progress(0.6, f"{'baking .blend' if exporting else 'rendering'} "
                                f"({len(frames)} frames) in Blender")
            oom = self._run_blender(scene_path, len(frames))
            if self.cancel.is_set():
                self.job.status = "cancelled"
                raise RenderCancelled()
            if not oom:
                break
            if attempt == 2:  # 3 tries (full -> half -> quarter budget) exhausted
                self.job.status = "error"
                raise RuntimeError("GPU out of memory even at reduced mesh detail")
            self._mesh_budget = max(300_000, self._mesh_budget // 2)

        if exporting:
            if not self.blend_path.exists():
                self.job.status = "error"
                raise RuntimeError("blender finished but no .blend was written")
            out = str(self.blend_path)
        else:
            out = self._encode(len(frames))
        self.job.status = "done"
        self.job.output_path = out
        self._progress(1.0, "done")
        return out

    def _encode(self, n_frames: int) -> str:
        self._progress(0.92, "encoding mp4")
        out = self.workdir / "output.mp4"
        if n_frames == 1:
            still = self.workdir / "output.png"
            shutil.copy(self.frames_dir / "frame_00000.png", still)
            return str(still)
        # Prefer the bundled ffmpeg (imageio-ffmpeg) so we don't depend on a system
        # ffmpeg being on the server's PATH; fall back to one on PATH if present.
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        # Two outputs from the lossless PNG master (the standard viz pipeline):
        #  - output.mp4     H.264 yuv420p — universal DELIVERY copy (QuickTime/Safari/web).
        #  - output_hq.mp4  H.264 yuv444p — near-lossless quality (full chroma; ~+9 dB PSNR
        #                   vs 420 on fine colored detail). Plays in Chrome/VLC, NOT
        #                   QuickTime. 4:2:0 throws away the fine colored detail (sparkly
        #                   meshes on black); no crf recovers it, only full chroma does.
        # (The retained PNG frames are the true lossless master for publication/ProRes.)
        common = ["-c:v", "libx264", "-preset", "slow", "-crf", "12", "-movflags", "+faststart"]
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(self.job.settings.fps),
            "-i", str(self.frames_dir / "frame_%05d.png"),
            *common, "-pix_fmt", "yuv420p", str(out),
        ], check=True, capture_output=True)
        hq = self.workdir / "output_hq.mp4"
        try:
            subprocess.run([
                ffmpeg, "-y", "-framerate", str(self.job.settings.fps),
                "-i", str(self.frames_dir / "frame_%05d.png"),
                *common, "-pix_fmt", "yuv444p", str(hq),
            ], check=True, capture_output=True)
        except Exception as e:  # noqa: BLE001
            print(f"[worker] hq (yuv444p) encode skipped: {e}")
        return str(out)
