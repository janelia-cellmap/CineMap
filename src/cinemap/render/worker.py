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
from ..data.slice_loader import EMVolume
from .interpolate import FrameState, build_frames

Progress = Callable[[float, str], None]


class RenderCancelled(Exception):
    """Raised when a render is cancelled via the worker's cancel event."""


def _bu(p, nm_per_bu):
    return [c / nm_per_bu for c in p]


class RenderWorker:
    def __init__(self, project: Project, job: RenderJob, nm_per_bu: float = NM_PER_BU):
        self.project = project
        self.job = job
        self.nm_per_bu = nm_per_bu
        self.manifest: Manifest = project.manifest
        self.workdir = PROJECTS_DIR / project.id / "renders" / job.id
        self.frames_dir = self.workdir / "frames"
        self.assets_dir = self.workdir / "assets"
        self.cancel = threading.Event()   # set to request cancellation
        self._proc: subprocess.Popen | None = None
        self._em: EMVolume | None = None
        self._slice_cache: dict[tuple, str] = {}

    # ---- asset preparation ----
    def _em_vol(self) -> EMVolume:
        if self._em is None:
            assert self.manifest.em, "no EM source in manifest"
            self._em = EMVolume(self.manifest.em.zarr_url)
        return self._em

    def _slice_png(self, em_name, axis, position_nm, level, region) -> dict:
        vol = self._em_vol()
        res = vol.read_slice(axis, position_nm, level=level, region=region)
        key = (axis, res.scale_level, round(res.position_nm),
               round(res.origin_nm[0]), round(res.origin_nm[1]), round(res.origin_nm[2]),
               round(res.u_nm[0] + res.u_nm[1] + res.u_nm[2]))
        if key not in self._slice_cache:
            from PIL import Image

            path = self.assets_dir / f"slice_{axis}_{res.scale_level}_{abs(hash(key)) % 10**8}.png"
            Image.fromarray(res.image).save(path)
            self._slice_cache[key] = str(path)
        return {
            "image_path": self._slice_cache[key],
            "origin_bu": _bu(res.origin_nm, self.nm_per_bu),
            "u_bu": _bu(res.u_nm, self.nm_per_bu),
            "v_bu": _bu(res.v_nm, self.nm_per_bu),
        }

    @staticmethod
    def _mesh_uid(mesh_name, ids) -> str:
        """Stable id per (layer, exact segment set) so different sets are different
        objects (a keyframe showing few segments != one showing all)."""
        import hashlib

        h = hashlib.md5((",".join(map(str, sorted(ids)))).encode()).hexdigest()[:8]
        return f"{mesh_name}_{h}"

    def _mesh_obj(self, mesh_name, segment_ids) -> str | None:
        src = next((m for m in self.manifest.meshes if m.name == mesh_name), None)
        if not src:
            return None
        ids = segment_ids or src.segment_ids
        if not ids:
            return None
        out = self.assets_dir / f"mesh_{self._mesh_uid(mesh_name, ids)}.ply"  # PLY keeps vertex colors
        if out.exists():
            return str(out)
        loader = MeshLoader(src.mesh_url, src.label_zarr)
        try:
            combined = loader.load_many(ids)
        except Exception as e:  # noqa: BLE001
            print(f"[worker] mesh {mesh_name} ({len(ids)} segs) failed: {e}")
            return None
        os.makedirs(out.parent, exist_ok=True)
        combined.export(str(out))
        return str(out)

    # ---- scene spec ----
    def _build_scene_spec(self, frames: list[FrameState]) -> dict:
        # one Blender object per distinct (layer, segment set) across all frames
        mesh_specs: dict[str, dict] = {}
        for fr in frames:
            for m in fr.meshes:
                uid = self._mesh_uid(m.mesh_name, m.segment_ids)
                if uid not in mesh_specs:
                    obj = self._mesh_obj(m.mesh_name, m.segment_ids)
                    if obj:
                        mesh_specs[uid] = {"id": uid, "obj_path": obj, "color": m.color}
        frame_specs = []
        for fi, fr in enumerate(frames):
            if self.cancel.is_set():
                raise RenderCancelled()
            # region to crop EM around the camera target, sized to what's on screen
            dist = math.dist(fr.position_nm, fr.look_at_nm)
            half = max(500.0, dist * math.tan(math.radians(fr.fov_deg) / 2) * 1.25)
            region = (tuple(fr.look_at_nm), half)
            slices = []
            for sl in fr.slices:
                if sl.opacity <= 0.001:
                    continue
                s = self._slice_png(sl.em_name, sl.axis, sl.position_nm, sl.scale_level, region)
                s["opacity"] = sl.opacity
                slices.append(s)
            overrides = {}
            for m in fr.meshes:
                uid = self._mesh_uid(m.mesh_name, m.segment_ids)
                if uid in mesh_specs:
                    overrides[uid] = {"opacity": m.opacity, "visible": m.opacity > 0.001}
            frame_specs.append({
                "camera": {
                    "position_bu": _bu(fr.position_nm, self.nm_per_bu),
                    "look_at_bu": _bu(fr.look_at_nm, self.nm_per_bu),
                    "fov_rad": math.radians(fr.fov_deg),
                    "up": fr.up,
                },
                "slices": slices,
                "mesh_overrides": overrides,
            })
            self._progress(0.1 + 0.5 * (fi + 1) / len(frames), f"assets {fi + 1}/{len(frames)}")
        return {
            "world": {"nm_per_bu": self.nm_per_bu,
                      "background": self.project.lighting.background},
            "lighting": {"key_energy": self.project.lighting.key_energy},
            "render": self.job.settings.model_dump(),
            "meshes": list(mesh_specs.values()),
            "frames": frame_specs,
            "output_dir": str(self.frames_dir),
        }

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

        self._progress(0.1, "preparing assets")
        spec = self._build_scene_spec(frames)  # raises RenderCancelled if cancelled
        if self.cancel.is_set():
            raise RenderCancelled()
        scene_path = self.workdir / "scene.json"
        scene_path.write_text(json.dumps(spec, indent=2))

        self._progress(0.6, f"rendering {len(frames)} frames in Blender")
        py = sys.executable
        self._proc = subprocess.Popen(
            [py, "-m", "cinemap.render.blender_script", str(scene_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        proc = self._proc
        done = 0
        for line in proc.stdout:  # type: ignore
            if self.cancel.is_set():
                proc.terminate()
                break
            if line.startswith("[blender] frame"):
                done += 1
                self._progress(0.6 + 0.3 * done / len(frames), line.strip())
        proc.wait()
        if self.cancel.is_set():
            self.job.status = "cancelled"
            raise RenderCancelled()
        if proc.returncode != 0:
            self.job.status = "error"
            raise RuntimeError(f"blender exited {proc.returncode}")

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
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(self.job.settings.fps),
            "-i", str(self.frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out),
        ], check=True, capture_output=True)
        return str(out)
