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
from .interpolate import FrameState, build_frames

Progress = Callable[[float, str], None]


class RenderCancelled(Exception):
    """Raised when a render is cancelled via the worker's cancel event."""


def _bu(p, nm_per_bu):
    return [c / nm_per_bu for c in p]


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
        # Resolution budgets. Draft (bake/update/preview thumbnails) trades detail
        # for speed: a coarse EM level and low-voxel meshes. The final video uses
        # full resolution. EM level is chosen by target px across the on-screen
        # crop; meshes by marching-cubes voxel budget (per-segment / union).
        draft = bool(getattr(job.settings, "draft", False))
        self._draft = draft
        self._em_target_px = 768 if draft else 1600
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
        # A cutaway needs a watertight cross-section to cap. Mixed LODs (per-chunk, and
        # per-frame's zoom buckets) place non-matching fragment boundaries next to each
        # other, so the seams don't weld and the cap can't close (plus the surface cracks).
        # Whenever any layer has a clip plane, force ONE consistent LOD for the whole shot
        # so fragment boundaries line up and the cap fills cleanly.
        if any(getattr(m, "clip", None) for kf in project.keyframes for m in kf.meshes):
            self._lod_mode = "single"
        # non-destructive presentation pass (lighting rig / materials / DOF)
        self._auto_direct = bool(getattr(job.settings, "auto_direct", True))

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

    def _slice_png(self, sl, region, seg_overlays) -> dict:
        """Render the EM cross-section, with the segmentation labels of
        `seg_overlays` [(label_zarr, segment_ids), …] colored on top (like the
        neuroglancer cross-section). Cached per (slice, region, overlay)."""
        import numpy as np
        from PIL import Image

        center, half = region
        normal = getattr(sl, "normal", None)
        key = (sl.em_name, sl.axis, round(sl.position_nm), tuple(normal) if normal else None,
               tuple(round(c) for c in center), round(half), self._em_target_px,
               tuple((u, tuple(sorted(ids)), lc.cache_key()) for u, ids, lc in seg_overlays))
        if key in self._slice_cache:
            return self._slice_cache[key]

        if normal:   # oblique plane: resample the tilted plane through the projected focus
            n = np.asarray(normal, float); n = n / (np.linalg.norm(n) or 1.0)
            c = np.asarray(center, float)
            cproj = c + (sl.position_nm - float(np.dot(c, n))) * n
            res = self._em_vol().read_oblique_slice(normal, cproj, half, target_px=self._em_target_px)
        else:
            res = self._em_vol().read_slice(sl.axis, sl.position_nm, level=sl.scale_level,
                                            target_px=self._em_target_px, region=region)
        rgb = np.repeat(res.image[:, :, None].astype(np.float64), 3, axis=2)  # grayscale EM
        H, W = rgb.shape[:2]

        for label_zarr, ids, lc in ([] if normal else seg_overlays):  # seg overlay: axis-aligned only
            if not ids:
                continue
            lres = self._label_vol(label_zarr).read_slice(sl.axis, sl.position_nm,
                                                          target_px=self._em_target_px, region=region)
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

        path = self.assets_dir / f"slice_{sl.axis}_{abs(hash(key)) % 10**8}.png"
        Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(path)
        out = {
            "image_path": str(path),
            "origin_bu": _bu(res.origin_nm, self.nm_per_bu),
            "u_bu": _bu(res.u_nm, self.nm_per_bu),
            "v_bu": _bu(res.v_nm, self.nm_per_bu),
        }
        self._slice_cache[key] = out
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
               + self._lod_tag_for(nmpp) + "|decode2")
        return f"{mesh_name}_{hashlib.md5(sig.encode()).hexdigest()[:8]}"

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

    def _mesh_obj(self, mesh_name, segment_ids, lc, nmpp=None) -> str | None:
        src = next((m for m in self.manifest.meshes if m.name == mesh_name), None)
        if not src:
            return None
        ids = segment_ids or src.segment_ids
        if not ids:
            return None
        uid = self._mesh_uid(mesh_name, ids, lc.cache_key(), nmpp)
        out = self.assets_dir / f"mesh_{uid}.ply"  # PLY keeps vertex colors
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
        combined.export(str(out))
        return str(out)

    @staticmethod
    def _ann_uid(an) -> str:
        """Stable id per (layer, geometry, color) so an edited annotation layer
        becomes a distinct asset."""
        import hashlib

        sig = json.dumps([an.name, an.color, an.points, an.lines, an.boxes, an.ellipsoids,
                          an.point_radius_nm, an.line_radius_nm], sort_keys=True)
        return f"ann_{hashlib.md5(sig.encode()).hexdigest()[:10]}"

    def _ann_obj(self, an) -> str | None:
        from ..data.annotations import annotations_to_mesh

        uid = self._ann_uid(an)
        out = self.assets_dir / f"{uid}.ply"
        if out.exists():
            return str(out)
        prims = {"points": an.points, "lines": an.lines, "boxes": an.boxes,
                 "ellipsoids": an.ellipsoids}
        mesh = annotations_to_mesh(prims, an.color, an.point_radius_nm, an.line_radius_nm)
        if mesh is None:
            return None
        os.makedirs(out.parent, exist_ok=True)
        mesh.export(str(out))
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
                uid = f"{m.mesh_name}_{hashlib.md5((ckey + '|' + str(sig) + '|frag2').encode()).hexdigest()[:10]}"
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
                    out = self.assets_dir / f"mesh_{uid}.ply"
                    os.makedirs(out.parent, exist_ok=True)
                    combined.export(str(out))
                    mesh_specs[uid] = {"id": uid, "obj_path": str(out), "color": m.color}
            frame_layer_uid.append(per)
            self._progress(0.1 + 0.35 * (fi + 1) / len(frames),
                           f"chunk assets {fi + 1}/{len(frames)} ({len(mesh_specs)} built)")
        return frame_layer_uid

    # ---- scene spec ----
    def _build_scene_spec(self, frames: list[FrameState], index_offset: int = 0) -> dict:
        # On-screen scale (nm per pixel) per frame, for picking precomputed-mesh LOD
        # like neuroglancer. With dynamic LOD this varies per frame (coarser when the
        # layer is far/small on screen); otherwise every frame uses the finest.
        height = max(1, self.job.settings.height)

        def _nmpp(fr):
            d = math.dist(fr.position_nm, fr.look_at_nm)
            return 2.0 * d * math.tan(math.radians(fr.fov_deg) / 2) / height

        frame_nmpp = [_nmpp(fr) for fr in frames]
        self._nm_per_px = min(frame_nmpp, default=None)
        frame_lod_nmpp = self._lod_bucket_nmpp(frame_nmpp)  # nm/px to build each frame at

        # One Blender object per distinct (layer, segment set, LOD bucket): a frame
        # references only its bucket's variant and the others auto-hide, so far frames
        # render a coarse mesh and close-ups a fine one.
        mesh_specs: dict[str, dict] = {}
        # "chunk" mode: per-frame frustum cull + per-segment on-screen LOD (see
        # _build_chunk_assets); other modes: one combined mesh per (layer, segset, bucket).
        frame_layer_uid = (self._build_chunk_assets(frames, mesh_specs)
                           if self._lod_mode == "chunk" else None)
        for fi, fr in enumerate(frames):
            if self.cancel.is_set():
                raise RenderCancelled()
            nmpp = frame_lod_nmpp[fi]
            if frame_layer_uid is None:
                for m in fr.meshes:
                    if not m.render_3d:       # label-only layer -> slice overlay only
                        continue
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
        frame_specs = []
        for fi, fr in enumerate(frames):
            if self.cancel.is_set():
                raise RenderCancelled()
            emph_hero, emph_glow, emph_spot = (
                emph_track[fi] if emph_track and fi < len(emph_track) else (None, 0.0, 1.0))
            # region to crop EM around the camera target, sized to what's on screen
            dist = math.dist(fr.position_nm, fr.look_at_nm)
            half = max(500.0, dist * math.tan(math.radians(fr.fov_deg) / 2) * 1.25)
            region = (tuple(fr.look_at_nm), half)
            # segmentation layers in this frame -> overlaid on the EM slice.
            # Decoupled from the 3D mesh opacity: the slice shows the cross-section
            # even when the 3D meshes are faded/hidden (so they don't occlude it).
            seg_overlays = []
            for m in fr.meshes:
                src = next((s for s in self.manifest.meshes if s.name == m.mesh_name), None)
                if src and src.label_zarr and m.segment_ids:
                    seg_overlays.append((src.label_zarr, m.segment_ids, self._frame_colors(m)))
            slices = []
            for sl in fr.slices:
                if sl.opacity <= 0.001:
                    continue
                # slot is stable across frames (matches interpolate's slice identity)
                # so the blend exporter can group a slice's per-frame images into one
                # animated image-sequence plane. A slice that can't be read (e.g. a
                # non-OME-Zarr EM source) is skipped, not fatal to the whole render.
                try:
                    png = self._slice_png(sl, region, seg_overlays)
                except Exception as e:  # noqa: BLE001
                    print(f"[worker] slice {sl.em_name}:{sl.axis} failed: {e}")
                    continue
                slices.append({**png, "opacity": sl.opacity, "slot": f"{sl.em_name}:{sl.axis}"})
            overrides = {}
            for m in fr.meshes:
                if frame_layer_uid is not None:           # chunk mode: per-frame selection
                    uid = frame_layer_uid[fi].get(m.mesh_name)
                else:
                    uid = self._mesh_uid(m.mesh_name, m.segment_ids,
                                         self._frame_colors(m).cache_key(), frame_lod_nmpp[fi])
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
                    if is_hero and emph_glow > 0.0:
                        ov["emphasis"] = emph_glow       # brief emission glow on the hero
                    cl = getattr(m, "clip", None)
                    if cl:
                        ov["clip"] = {"axis": cl["axis"], "side": cl["side"],
                                      "normal": cl.get("normal"),
                                      "position_nm": cl["position_nm"],
                                      "position_bu": cl["position_nm"] / self.nm_per_bu}
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
                "index": index_offset + fi,   # global frame index (split cluster jobs)
            })
            self._progress(0.45 + 0.15 * (fi + 1) / len(frames), f"slices {fi + 1}/{len(frames)}")
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
        return spec

    # ---- run ----
    def _progress(self, p, msg):
        self.job.progress = round(p, 3)
        self.job.message = msg
        if self._cb:
            self._cb(p, msg)

    def terminate(self) -> None:
        """Request cancellation; kills the Blender subprocess if it is running."""
        self.cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _run_blender(self, scene_path, nframes: int) -> bool:
        """Launch the Blender render subprocess and stream progress. Returns True if it
        failed specifically with a GPU out-of-memory error (so the caller can retry at
        lower mesh detail); raises RuntimeError on any other non-zero exit."""
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        proc = self._proc
        done, oom = 0, False
        for line in proc.stdout:  # type: ignore
            if self.cancel.is_set():
                proc.terminate()
                break
            low = line.lower()
            if ("out of memory" in low or "out of gpu memory" in low
                    or "cuda_error_out_of_memory" in low or "failed to allocate" in low):
                oom = True
            if line.startswith("[blender] frame"):
                done += 1
                self._progress(0.6 + 0.3 * done / nframes, line.strip())
        proc.wait()
        if self.cancel.is_set():
            return False
        if proc.returncode != 0 and not oom:
            self.job.status = "error"
            raise RuntimeError(f"blender exited {proc.returncode}")
        return oom

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
