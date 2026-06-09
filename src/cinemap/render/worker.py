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
        self.assets_dir = self.workdir / "assets"
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
        base_budget = 1_200_000 if draft else 5_000_000
        self._mesh_budget = min(int(base_budget * detail), self.MESH_BUDGET_CEILING)
        self._nm_per_px = None  # finest on-screen scale across frames (set per build)
        self._lod_tag = ""      # cache-key component so re-framing rebuilds LOD assets
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
        key = (sl.em_name, sl.axis, round(sl.position_nm),
               tuple(round(c) for c in center), round(half), self._em_target_px,
               tuple((u, tuple(sorted(ids)), lc.cache_key()) for u, ids, lc in seg_overlays))
        if key in self._slice_cache:
            return self._slice_cache[key]

        res = self._em_vol().read_slice(sl.axis, sl.position_nm, level=sl.scale_level,
                                        target_px=self._em_target_px, region=region)
        rgb = np.repeat(res.image[:, :, None].astype(np.float64), 3, axis=2)  # grayscale EM
        H, W = rgb.shape[:2]

        for label_zarr, ids, lc in seg_overlays:
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

    def _mesh_uid(self, mesh_name, ids, color_key=()) -> str:
        """Stable id per (layer, exact segment set, coloring, LOD scale) so a
        different segment set, color, OR on-screen resolution becomes a distinct
        cached asset."""
        import hashlib

        sig = ",".join(map(str, sorted(ids))) + "|" + str(color_key) + "|" + self._lod_tag
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

    def _mesh_obj(self, mesh_name, segment_ids, lc) -> str | None:
        src = next((m for m in self.manifest.meshes if m.name == mesh_name), None)
        if not src:
            return None
        ids = segment_ids or src.segment_ids
        if not ids:
            return None
        uid = self._mesh_uid(mesh_name, ids, lc.cache_key())
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
                combined = MeshLoader(src.mesh_url, src.label_zarr).load_many(
                    ids, colorize=lc.rgb,
                    target_voxels_single=self._mesh_voxels_single,
                    target_voxels_union=self._mesh_voxels_union,
                    nm_per_px=self._nm_per_px, draft=self._draft,
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

    # ---- scene spec ----
    def _build_scene_spec(self, frames: list[FrameState], index_offset: int = 0) -> dict:
        # On-screen scale (nm per pixel) for picking precomputed-mesh LOD, like
        # neuroglancer: take the FINEST requirement across frames (the closest the
        # camera ever gets) so the cached mesh is sharp enough for every shot.
        height = max(1, self.job.settings.height)

        def _nmpp(fr):
            d = math.dist(fr.position_nm, fr.look_at_nm)
            return 2.0 * d * math.tan(math.radians(fr.fov_deg) / 2) / height

        self._nm_per_px = min((_nmpp(fr) for fr in frames), default=None)
        # Asset cache key: nm/px (framing), draft/full, label/precomputed source, and
        # the vertex budget — so re-framing or a lower OOM-retry budget rebuilds LODs.
        self._lod_tag = (f"npp{self._nm_per_px:.3g}|{'draft' if self._draft else 'full'}"
                         f"|{'lab' if self._prefer_labels else 'pre'}|b{self._mesh_budget}"
                         if self._nm_per_px else f"b{self._mesh_budget}")

        # one Blender object per distinct (layer, segment set) across all frames
        mesh_specs: dict[str, dict] = {}
        for fr in frames:
            for m in fr.meshes:
                if not m.render_3d:           # label-only layer -> slice overlay only
                    continue
                lc = self._frame_colors(m)
                uid = self._mesh_uid(m.mesh_name, m.segment_ids, lc.cache_key())
                if uid not in mesh_specs:
                    obj = self._mesh_obj(m.mesh_name, m.segment_ids, lc)
                    if obj:
                        mesh_specs[uid] = {"id": uid, "obj_path": obj, "color": m.color}
            # annotation layers -> geometry assets via the same import path
            for an in fr.annotations:
                uid = self._ann_uid(an)
                if uid not in mesh_specs:
                    obj = self._ann_obj(an)
                    if obj:
                        mesh_specs[uid] = {"id": uid, "obj_path": obj, "color": an.color}
        frame_specs = []
        for fi, fr in enumerate(frames):
            if self.cancel.is_set():
                raise RenderCancelled()
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
                uid = self._mesh_uid(m.mesh_name, m.segment_ids, self._frame_colors(m).cache_key())
                if uid in mesh_specs:
                    # effective 3D alpha = cinematic fade (opacity) * NG "Opacity (3d)"
                    oa = getattr(m, "object_alpha", 1.0)
                    eff = m.opacity * oa
                    overrides[uid] = {
                        "opacity": eff,
                        "visible": eff > 0.001,
                        "silhouette": getattr(m, "silhouette", 0.0),
                    }
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
                },
                "slices": slices,
                "mesh_overrides": overrides,
                "index": index_offset + fi,   # global frame index (split cluster jobs)
            })
            self._progress(0.1 + 0.5 * (fi + 1) / len(frames), f"assets {fi + 1}/{len(frames)}")
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
            plan = director.plan(self.project.keyframes)
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
        self._progress(0.05, "interpolating keyframes")
        frames = build_frames(kfs, self.job.settings.fps)
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
        # yuv444p keeps full-resolution color (no 4:2:0 chroma subsampling) and crf 14
        # (near visually lossless) preserves the thin saturated colored structures, so
        # the mp4 closely matches the PNG frames. Trade-off: yuv444p is H.264 High
        # 4:4:4 — Chrome/VLC/ffmpeg play it, but Safari/QuickTime may not.
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(self.job.settings.fps),
            "-i", str(self.frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "14", str(out),
        ], check=True, capture_output=True)
        return str(out)
