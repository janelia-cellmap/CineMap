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
from ..data.slice_loader import EMVolume, get_volume
from .interpolate import FrameAnnotation, FrameState, build_frames, state_at_time

Progress = Callable[[float, str], None]


class RenderCancelled(Exception):
    """Raised when a render is cancelled via the worker's cancel event."""


def _bu(p, nm_per_bu):
    return [c / nm_per_bu for c in p]


def _export_mesh_npz(mesh, out: Path) -> None:
    """Write a trimesh-like object as a compact `.npz` (vertices float32, faces
    int32, optional uint8 vertex colors). Blender loads this ~5–10× faster than
    going through `bpy.ops.wm.ply_import` (which routes through the operator
    system + undo stack)."""
    import numpy as np

    v = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    f = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    arrs: dict = {"v": v, "f": f}
    vc = None
    try:
        vc = mesh.visual.vertex_colors      # Nx4 uint8 (trimesh)
    except Exception:  # noqa: BLE001
        vc = None
    if vc is not None and len(vc) == len(v):
        arrs["c"] = np.ascontiguousarray(vc, dtype=np.uint8)
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
    # Hard ceiling on a layer's combined vertex count, regardless of mesh_detail —
    # keeps the worst case well under the GPUs' VRAM (~9 GB free on an 11 GB card).
    MESH_BUDGET_CEILING = 20_000_000

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
        # meshes use marching-cubes voxel budgets (per-segment / union).
        draft = bool(getattr(job.settings, "draft", False))
        self._draft = draft
        # Fallback/cap for paths that still request an explicit resampled image size
        # (notably oblique slices). Axis-aligned slices use per-frame nm/pixel instead.
        self._em_target_px = 768 if draft else min(2560, max(1280, int(job.settings.width * 1.25)))
        self._mesh_voxels_single = 1_500_000 if draft else 8_000_000
        self._mesh_voxels_union = 3_000_000 if draft else 20_000_000
        # mesh sourcing: precomputed (LOD-adaptive) by default; opt in to watertight
        # marching-cubes-from-labels via the render setting.
        self._prefer_labels = bool(getattr(job.settings, "mesh_from_labels", False))
        # Per-layer vertex budget = base * mesh_detail, hard-capped so a too-high
        # setting can't recreate the multi-GB mesh that stalled asset prep / OOM'd the
        # GPU. The OOM-retry loop in run() halves this and rebuilds if Cycles runs out.
        detail = max(0.25, min(float(getattr(job.settings, "mesh_detail", 1.0) or 1.0), 8.0))
        base_budget = 3_000_000 if draft else 5_000_000
        self._mesh_budget = min(int(base_budget * detail), self.MESH_BUDGET_CEILING)
        self._nm_per_px = None  # finest on-screen scale across frames (set per build)
        # mesh LOD strategy: "single" (one build), "frame" (per-frame adaptive, like
        # neuroglancer; free on orbits), or "chunk" (per-chunk spatial — not yet
        # implemented, treated as "frame").
        self._lod_mode = getattr(job.settings, "lod_mode", "frame") or "frame"
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
            "slice-v2",
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

    def _slice_png(self, sl, region, seg_overlays, slice_seg=None,
                   target_nm_per_px: float | None = None) -> dict:
        """Render a cross-section of the slice's chosen layer. For an EM/image layer:
        the grayscale EM with `seg_overlays` [(label_zarr, ids, lc), …] colored on top
        (like neuroglancer). For a SEGMENTATION layer (resolved via _vol_for): the
        layer's labels rendered in color directly (`slice_seg=(ids, lc)`). Cached per
        (slice, region, overlay)."""
        import numpy as np
        from PIL import Image

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
            res = vol.read_oblique_slice(normal, cproj, half, target_px=target_px)
        else:
            res = vol.read_slice(axis, position_nm, level=sl.scale_level,
                                 target_px=self._em_target_px, region=region,
                                 target_nm_per_px=target_nm_per_px, raw=is_label)

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

        rgb = np.repeat(res.image[:, :, None].astype(np.float64), 3, axis=2)  # grayscale EM
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

    def _lod_tag_for(self, nmpp) -> str:
        """Cache-key component for a mesh built at on-screen scale `nmpp` (nm/px):
        re-framing, draft, source, budget, or a different LOD bucket each rebuild."""
        return (f"npp{nmpp:.3g}|{'draft' if self._draft else 'full'}"
                f"|{'lab' if self._prefer_labels else 'pre'}|b{self._mesh_budget}"
                if nmpp else f"b{self._mesh_budget}")

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

    def _mesh_uid(self, mesh_name, ids, color_key=(), nmpp=None) -> str:
        """Stable id per (layer, exact segment set, coloring, LOD scale) so a
        different segment set, color, OR on-screen resolution becomes a distinct
        cached asset."""
        import hashlib

        # `decode2` versions the mesh decoder: bump it to invalidate combined-layer assets
        # cached from an older (buggy) decode so a re-render can't reuse stale geometry.
        sig = (",".join(map(str, sorted(ids))) + "|" + str(color_key) + "|"
               + self._lod_tag_for(nmpp) + "|decode4")
        return f"{mesh_name}_{hashlib.md5(sig.encode()).hexdigest()[:8]}"

    def _clip_from_sweeps(self, layer_name: str, t: float) -> dict | None:
        """The cutaway clip for `layer_name` at global time `t` (seconds), from any active
        Sweep — evaluated on the SWEEP's own timeline, independent of the camera keyframes.
        Returns a clip dict (same shape as _clip_params) or None. First active sweep wins."""
        for sw in getattr(self.project, "sweeps", []) or []:
            if (not getattr(sw, "enabled", True) or getattr(sw, "kind", "cutaway") != "cutaway"
                    or sw.layer != layer_name):
                continue
            if t < sw.start_s:
                continue                          # before it starts -> mesh is whole
            # holds open after the end (progress clamps to 1); mirror ping-pongs back.
            pos = sw.from_nm + (sw.to_nm - sw.from_nm) * _sweep_progress(sw, t)
            return {"axis": sw.axis, "side": sw.side, "normal": sw.normal,
                    "position_nm": float(pos)}
        return None

    def _slices_from_sweeps(self, t: float) -> list:
        """EM slice planes from active 'slice' sweeps at global time `t` — swept on the
        sweep's OWN timeline (independent of the camera keyframes). Unlike a cutaway,
        a slice sweep is a timeline clip: it is visible only during its own span."""
        from .interpolate import FrameSlice
        out = []
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
            out.append(FrameSlice(sw.em_name or default_em, sw.axis, float(pos), None,
                                  float(sw.opacity), normal=sw.normal))
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

    # --- slice-read derivation (shared by the parallel warm pass and the build loop, so
    # the two can never drift) -------------------------------------------------------
    def _frame_region(self, fr):
        """EM crop around the camera target, sized to what's on screen this frame."""
        dist = math.dist(fr.position_nm, fr.look_at_nm)
        half = max(500.0, dist * math.tan(math.radians(fr.fov_deg) / 2) * 1.25)
        return (tuple(fr.look_at_nm), half)

    def _frame_nm_per_px(self, fr) -> float:
        """Physical size of one rendered screen pixel at the camera target."""
        height = max(1, self.job.settings.height)
        dist = math.dist(fr.position_nm, fr.look_at_nm)
        return 2.0 * dist * math.tan(math.radians(fr.fov_deg) / 2) / height

    def _frame_seg_overlays(self, fr):
        """Segmentation layers in this frame to overlay on the EM slice."""
        out = []
        for m in fr.meshes:
            src = next((s for s in self.manifest.meshes if s.name == m.mesh_name), None)
            if src and src.label_zarr and m.segment_ids:
                out.append((src.label_zarr, m.segment_ids, self._frame_colors(m)))
        return out

    def _frame_slice_reads(self, fr, t_global):
        """Visible slices in this frame as (FrameSlice, slice_seg) — keyframe slices plus
        any 'slice' sweeps on the global timeline. slice_seg=(ids, colors) when the slice
        points at a SEGMENTATION layer (so it renders colored labels), else None."""
        out = []
        for sl in list(fr.slices) + self._slices_from_sweeps(t_global):
            if sl.opacity <= 0.001:
                continue
            _, is_label = self._vol_for(sl.em_name)
            slice_seg = None
            if is_label:
                sm = next((mm for mm in fr.meshes
                           if mm.mesh_name == sl.em_name and mm.segment_ids), None)
                if sm:
                    slice_seg = (list(sm.segment_ids), self._frame_colors(sm))
            out.append((sl, slice_seg))
        return out

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

    def _mesh_obj(self, mesh_name, segment_ids, lc, nmpp=None) -> str | None:
        src = next((m for m in self.manifest.meshes if m.name == mesh_name), None)
        if not src:
            return None
        ids = segment_ids or src.segment_ids
        if not ids:
            return None
        uid = self._mesh_uid(mesh_name, ids, lc.cache_key(), nmpp)
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
                    ids, colorize=lc.rgb,
                    target_voxels_single=self._mesh_voxels_single,
                    target_voxels_union=self._mesh_voxels_union,
                    nm_per_px=nmpp, draft=self._draft,
                    prefer_labels=self._prefer_labels, total_budget=self._mesh_budget)
        except Exception as e:  # noqa: BLE001
            print(f"[worker] mesh {mesh_name} ({len(ids)} segs) failed: {e}")
            return None
        os.makedirs(out.parent, exist_ok=True)
        _export_mesh_npz(combined, out)
        return str(out)

    @staticmethod
    def _ann_uid(an) -> str:
        """Stable id per (layer, geometry, color) so an edited annotation layer
        becomes a distinct asset."""
        import hashlib

        sig = json.dumps([an.name, an.color, an.points, an.lines, an.boxes, an.ellipsoids,
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
                if not m.render_3d or not m.segment_ids or m.mesh_name in layers:
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
                if m.mesh_name not in layers:
                    continue
                ld, seg_ids = layers[m.mesh_name]
                lc = self._frame_colors(m)
                ckey = str(lc.cache_key())
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
                    def _one(s, _ld=ld, _lc=lc, _sel=seg_sel):
                        return _ld.get_fragments(s, _sel[s], colorize=_lc.rgb)

                    with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(seg_sel))) as ex:
                        parts = [p for p in ex.map(_one, list(seg_sel)) if p is not None]
                    if not parts:
                        per.pop(m.mesh_name, None)
                        continue
                    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
                    out = self.assets_dir / f"mesh_{uid}.npz"
                    os.makedirs(out.parent, exist_ok=True)
                    _export_mesh_npz(combined, out)
                    mesh_specs[uid] = {"id": uid, "obj_path": str(out), "color": m.color}
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
                if self._clip_from_sweeps(m.mesh_name, t) or _clip_params(getattr(m, "clip", None)):
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
                    if not m.render_3d:       # label-only layer -> slice overlay only
                        continue
                    nmpp = _eff_nmpp(m.mesh_name, fi)
                    lc = self._frame_colors(m)
                    uid = self._mesh_uid(m.mesh_name, m.segment_ids, lc.cache_key(), nmpp)
                    if uid not in mesh_specs:
                        obj = self._mesh_obj(m.mesh_name, m.segment_ids, lc, nmpp)
                        if obj:
                            mesh_specs[uid] = {"id": uid, "obj_path": obj, "color": m.color}
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
        # Warm the slice-image cache in PARALLEL before the (serial) build loop. Slice
        # reads are network-bound (zarr chunks over HTTP) and frames sweep through heavily
        # overlapping regions, so fetching them one-at-a-time was the dominant pre-render
        # cost (~50s per cold slice, ~40min total on big shots). Fan the reads out across
        # threads (tensorstore releases the GIL during IO); the shared cache pool in
        # slice_loader dedups the overlapping chunks. _slice_png memoizes into
        # self._slice_cache, so the build loop below becomes pure-CPU cache hits.
        from concurrent.futures import ThreadPoolExecutor
        from ..data.mesh_loader import _FETCH_WORKERS
        warm_jobs = []
        warm_seen: set[tuple] = set()
        for fi, fr in enumerate(frames):
            region = self._frame_region(fr)
            target_nm_per_px = self._frame_nm_per_px(fr)
            seg_overlays = self._frame_seg_overlays(fr)
            for sl, slice_seg in self._frame_slice_reads(fr, self._t_global(fi, frame_times, index_offset)):
                key = self._slice_cache_key(sl, region, seg_overlays, slice_seg=slice_seg,
                                            target_nm_per_px=target_nm_per_px)
                if key in warm_seen:
                    continue
                warm_seen.add(key)
                warm_jobs.append((sl, region, seg_overlays, slice_seg, target_nm_per_px))

        def _warm(job):
            sl, region, seg_overlays, slice_seg, target_nm_per_px = job
            try:
                self._slice_png(sl, region, seg_overlays, slice_seg=slice_seg,
                                target_nm_per_px=target_nm_per_px)
            except Exception:  # noqa: BLE001 (failures re-surface in the build loop below)
                pass

        if warm_jobs and not self.cancel.is_set():
            self._progress(0.45, f"fetching {len(warm_jobs)} unique slice images")
            with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as ex:
                done = 0
                for _ in ex.map(_warm, warm_jobs):
                    done += 1
                    if done % 8 == 0:
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
            # segmentation layers in this frame -> overlaid on the EM slice. Decoupled from
            # the 3D mesh opacity: the slice shows the cross-section even when the 3D meshes
            # are faded/hidden (so they don't occlude it).
            seg_overlays = self._frame_seg_overlays(fr)
            slices = []
            # keyframe slices PLUS any 'slice' sweeps evaluated on the global timeline
            t_global = self._t_global(fi, frame_times, index_offset)
            for sl, slice_seg in self._frame_slice_reads(fr, t_global):
                # slot is stable across frames (matches interpolate's slice identity)
                # so the blend exporter can group a slice's per-frame images into one
                # animated image-sequence plane. A slice that can't be read (e.g. a
                # non-OME-Zarr EM source) is skipped, not fatal to the whole render.
                # _slice_png is already cached from the parallel warm pass above.
                try:
                    png = self._slice_png(sl, region, seg_overlays, slice_seg=slice_seg,
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
                if frame_layer_uid is not None:           # chunk mode: per-frame selection
                    uid = frame_layer_uid[fi].get(m.mesh_name)
                else:
                    uid = self._mesh_uid(m.mesh_name, m.segment_ids,
                                         self._frame_colors(m).cache_key(), _eff_nmpp(m.mesh_name, fi))
                if uid and uid in mesh_specs:
                    # effective 3D alpha = cinematic fade (opacity) * NG "Opacity (3d)"
                    oa = getattr(m, "object_alpha", 1.0)
                    eff = m.opacity * oa
                    is_hero = (m.mesh_name == emph_hero)
                    if emph_track and not is_hero and emph_spot < 1.0:
                        eff *= emph_spot                 # spotlight: briefly dim context
                    ov = {
                        "opacity": eff,
                        "visible": eff > 0.001,
                        "silhouette": getattr(m, "silhouette", 0.0),
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
                        ov["clip"] = {**cl, "position_bu": cl["position_nm"] / self.nm_per_bu}
                        mesh_specs[uid]["clip"] = True   # geometric cutaway (slice + cap)
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
                },
                "slices": slices,
                "mesh_overrides": overrides,
                "fade_alpha": max(0.0, min(1.0, float(getattr(fr, "fade_alpha", 0.0) or 0.0))),
                "index": index_offset + fi,   # global frame index (split cluster jobs)
            })
            self._progress(0.45 + 0.15 * (fi + 1) / len(frames),
                           f"building frame specs {fi + 1}/{len(frames)}")
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
        if self._auto_direct:
            # Non-destructive presentation directives (optional fields; absent => the
            # plain neuroglancer-faithful look). DOF focuses on the framed subject —
            # the camera's look-at, which is exactly what neuroglancer centered on.
            from . import director
            plan = director.plan(self.project.keyframes,
                                 director.make_settings(getattr(self.project, "look", None)))
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
            # and auto_direct. Per-frame state (camera/opacity/metallic OVERRIDES/clip position/
            # slices/lights) is re-driven on the cached geometry, so it correctly does NOT key.
            geom = sorted((mm["id"], bool(mm.get("clip")), tuple(mm.get("color") or ()))
                          for mm in mesh_specs.values())
            sig_src = json.dumps([geom, spec.get("direction", {}).get("material", {}),
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

    def _state_at_time(self, t: float):
        """The interpolated FrameState (camera/slices/meshes) at a GLOBAL time t (seconds),
        for one-off snapshot frames. Holds the last keyframe past the end."""
        kfs = self.project.keyframes
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
        self._kfs = self.project.keyframes
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

        kfs = self.project.keyframes
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
