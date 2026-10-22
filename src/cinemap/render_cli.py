"""Headless render CLI for cluster jobs — render a frame range, then combine.

Splitting a long video across GPU jobs: each job renders a contiguous frame
range to a SHARED frames dir using GLOBAL frame indices (so there are no
collisions), then one final job concatenates them into the mp4.

For mesh-from-labels projects, run `prep` ONCE first (CPU job, no GPU needed)
to build every LOD-bucket mesh asset into a SHARED --assets-dir. Point every
`render` job at that same --assets-dir so they hit the on-disk mesh cache
instead of each rebuilding the same meshes in their own private assets dir.

Rather than hand-retyping every setting (and drifting from what the UI
actually submitted), pass --from-job JOB_ID to reuse an existing render job's
full persisted RenderSettings verbatim.

  prep mesh assets once, shared across all render jobs (one CPU job):
    python -m cinemap.render_cli prep --project P.cinemap.json --out DIR \
        --assets-dir DIR/assets --from-job 20260831-174238

  render a frame range (one GPU job per range):
    python -m cinemap.render_cli render --project P.cinemap.json --out DIR \
        --assets-dir DIR/assets --from-job 20260831-174238 --start A --end B

  combine frames -> mp4 (one CPU job, after all renders finish):
    python -m cinemap.render_cli combine --out DIR --fps 60 --output FINAL.mp4

  count total frames (to plan the split):
    python -m cinemap.render_cli count --project P.cinemap.json --fps 60
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .models import Project, RenderJob, RenderSettings
from .render import director as _director
from .render.interpolate import build_frames
from .render.worker import RenderWorker


def _load(path: str) -> Project:
    return Project.model_validate_json(Path(path).read_text())


def _settings(a, project: Project) -> RenderSettings:
    if a.from_job:
        # Reuse a job's exact persisted RenderSettings verbatim (noise_threshold,
        # show_scale_bar, scale_bar_scale, auto_direct, lod_mode, ... — everything,
        # not just the mesh/from-labels flags below) instead of hand-retyping every
        # field on the command line, which drifts silently as new settings are added.
        job = next((j for j in project.renders if j.id == a.from_job), None)
        if not job:
            sys.exit(f"[cli] no render job {a.from_job!r} found in {a.project}")
        print(f"[cli] using settings from job {a.from_job}: {job.settings.model_dump()}",
              flush=True)
        return job.settings
    return RenderSettings(
        width=a.width, height=a.height, fps=a.fps, samples=a.samples,
        export_blend=a.export_blend,
        mesh_from_labels=a.mesh_from_labels, mesh_detail=a.mesh_detail,
        label_mesh_smooth_iters=a.label_smooth_iters,
        label_mesh_decimate_fraction=a.label_decimate_fraction,
        label_mesh_blockwise=a.label_blockwise,
    )


def _worker(project: Project, a, out: Path,
            default_assets_dirname: str | None = None) -> tuple[RenderWorker, RenderSettings]:
    settings = _settings(a, project)
    worker = RenderWorker(project, RenderJob(id="cli", settings=settings))
    worker._cb = None  # not using run()'s progress callback
    # RenderWorker.run() sets these before ever calling _build_scene_spec; the CLI
    # calls _build_scene_spec directly (no run()), so it has to replicate that setup.
    worker._kfs = project.keyframes
    worker._flip_handed = worker._resolve_flip_handed()
    worker.workdir = out
    worker.frames_dir = out / "frames"
    if a.assets_dir:
        worker.assets_dir = Path(a.assets_dir)
    elif default_assets_dirname:
        worker.assets_dir = out / default_assets_dirname
    else:
        worker.assets_dir = out / f"assets_{a.start}_{a.end}"
    worker.blend_path = out / "scene.blend"
    worker.frames_dir.mkdir(parents=True, exist_ok=True)
    worker.assets_dir.mkdir(parents=True, exist_ok=True)
    return worker, settings


def _frames_for(project: Project, settings: RenderSettings) -> list:
    """Mirror RenderWorker.run()'s frame construction exactly (smooth start/end
    easing, still-frame truncation, sweep-hold extension) so a CLI render matches
    what the UI would have produced for the same settings."""
    smooth = settings.auto_direct and _director.DirectorSettings().smooth_camera
    frames = build_frames(project.keyframes, settings.fps, smooth_ends=smooth)
    if not frames:
        return frames
    if settings.still:
        return frames[:1]
    fps = max(1, settings.fps)
    sweep_end = max([0.0] + [float(s.start_s) + float(s.duration_s)
                             for s in (getattr(project, "sweeps", []) or [])
                             if getattr(s, "enabled", True)])
    need = int(round(sweep_end * fps))
    if need > len(frames):
        frames = frames + [frames[-1]] * (need - len(frames))
    return frames


def cmd_count(a) -> None:
    p = _load(a.project)
    print(len(build_frames(p.keyframes, a.fps)))


def cmd_prep(a) -> None:
    """Build every mesh/slice asset a frame range will need into --assets-dir, with
    no GPU/blender involved. Point every `render` job's --assets-dir (and, when
    sharding this across an array job, every other `prep` task's) at the same
    directory so they all hit this shared cache instead of rebuilding assets
    already produced by another task or job. --start/--end let this be split
    across an array job the same way `render` is — assets are content-addressed,
    so overlapping shards just redundantly rebuild (not corrupt) a shared asset."""
    project = _load(a.project)
    out = Path(a.out)
    worker, settings = _worker(project, a, out, default_assets_dirname="assets")
    frames = _frames_for(project, settings)
    start = a.start if a.start is not None else 0
    end = a.end if a.end is not None else len(frames)
    subset = frames[start:end]
    print(f"[cli] prep: total {len(frames)} frames; this task builds assets for "
          f"[{start}:{end}] = {len(subset)} frames into {worker.assets_dir}", flush=True)
    worker._build_scene_spec(subset, index_offset=start)
    print("[cli] prep done", flush=True)


def cmd_render(a) -> None:
    project = _load(a.project)
    out = Path(a.out)
    worker, settings = _worker(project, a, out)

    frames = _frames_for(project, settings)
    start = a.start if a.start is not None else 0
    end = a.end if a.end is not None else len(frames)
    subset = frames[start:end]
    print(f"[cli] total {len(frames)} frames; this job renders [{start}:{end}] "
          f"= {len(subset)} frames", flush=True)
    spec = worker._build_scene_spec(subset, index_offset=start)
    scene_path = out / f"scene_{start}_{end}.json"
    scene_path.write_text(json.dumps(spec))
    proc = subprocess.run(
        [sys.executable, "-m", "cinemap.render.blender_script", str(scene_path)],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if proc.returncode != 0:
        sys.exit(f"blender exited {proc.returncode}")
    if a.export_blend:
        print(f"[cli] done; WROTE {worker.blend_path}", flush=True)
    else:
        have = len(list(worker.frames_dir.glob("frame_*.png")))
        print(f"[cli] done; frames on disk: {have}", flush=True)


# Encode RGB frames with the BT.709 matrix and say so in the file. Untagged, ffmpeg uses
# BT.601 while QuickTime/Safari/Chrome/VLC decode HD as BT.709, so saturated colours
# shifted by viewer (a saturated magenta played back noticeably pinker and brighter).
BT709_ARGS = ["-vf", "scale=out_color_matrix=bt709:out_range=tv",
              "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


def cmd_combine(a) -> None:
    out = Path(a.out)
    frames_dir = out / "frames"
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        sys.exit("no frames to combine")
    # detect gaps (a failed split job) before encoding. ffmpeg's image2 demuxer
    # stops at the FIRST missing index, so a single failed split job would yield a
    # silently truncated video. Fail hard by default; --allow-gaps recovers what's
    # present (via a glob input that skips the holes) instead of truncating.
    idxs = [int(p.stem.split("_")[1]) for p in pngs]
    missing = sorted(set(range(idxs[0], idxs[-1] + 1)) - set(idxs))
    if missing and not getattr(a, "allow_gaps", False):
        sys.exit(f"[cli] ERROR: {len(missing)} missing frames (e.g. {missing[:5]}); "
                 f"re-run the failed render range, or pass --allow-gaps to encode anyway")
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        ffmpeg = "ffmpeg"
    output = a.output or str(out / "output.mp4")
    print(f"[cli] combining {len(pngs)} frames @ {a.fps}fps -> {output}"
          + (f"  ({len(missing)} gaps skipped)" if missing else ""), flush=True)
    if missing:  # glob includes every present frame in order, ignoring the holes
        in_args = ["-pattern_type", "glob", "-i", str(frames_dir / "frame_*.png")]
    else:        # contiguous: the cheap sequential reader from the first index
        in_args = ["-start_number", str(idxs[0]), "-i", str(frames_dir / "frame_%05d.png")]
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(a.fps), *in_args,
         "-c:v", "libx264", *BT709_ARGS, "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "+faststart", output],
        check=True,
    )
    print(f"[cli] WROTE {output}", flush=True)


def _add_mesh_args(sp) -> None:
    sp.add_argument("--from-job",
                     help="reuse an existing render job's exact settings (from --project's "
                          "renders list) verbatim — width/height/fps/samples/noise_threshold/"
                          "scale bar/auto_direct/lod_mode/mesh_* etc, not just the mesh flags "
                          "below. Overrides --width/--height/--fps/--samples/--mesh-* entirely.")
    sp.add_argument("--assets-dir",
                     help="shared mesh/asset cache dir (default: OUT/assets_START_END, "
                          "private per job). Point multiple render jobs and `prep` at the "
                          "same path to share the mesh cache and avoid rebuilding it per job.")
    sp.add_argument("--mesh-from-labels", action="store_true",
                     help="build per-segment meshes from label volumes (zmesh) instead of "
                          "loading precomputed mesh_url meshes")
    sp.add_argument("--mesh-detail", type=float, default=1.0)
    sp.add_argument("--label-smooth-iters", type=int, default=0)
    sp.add_argument("--label-decimate-fraction", type=float, default=0.0)
    sp.add_argument("--label-blockwise", choices=["auto", "on", "off"], default="auto")


def main() -> None:
    p = argparse.ArgumentParser(prog="cinemap.render_cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render"); r.set_defaults(fn=cmd_render)
    r.add_argument("--project", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--samples", type=int, default=48)
    r.add_argument("--width", type=int, default=1280)
    r.add_argument("--height", type=int, default=720)
    r.add_argument("--start", type=int)
    r.add_argument("--end", type=int)
    r.add_argument("--export-blend", action="store_true",
                   help="write a self-contained animated .blend instead of rendering frames")
    _add_mesh_args(r)

    pr = sub.add_parser("prep"); pr.set_defaults(fn=cmd_prep)
    pr.add_argument("--project", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--fps", type=int, default=30)
    pr.add_argument("--samples", type=int, default=48)  # unused by prep; kept for a shared _settings()
    pr.add_argument("--width", type=int, default=1280)
    pr.add_argument("--height", type=int, default=720)
    pr.add_argument("--start", type=int)
    pr.add_argument("--end", type=int)
    pr.add_argument("--export-blend", action="store_true", help=argparse.SUPPRESS)
    _add_mesh_args(pr)

    c = sub.add_parser("combine"); c.set_defaults(fn=cmd_combine)
    c.add_argument("--out", required=True)
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--output")
    c.add_argument("--allow-gaps", action="store_true",
                   help="encode whatever frames exist instead of failing on missing ones")

    n = sub.add_parser("count"); n.set_defaults(fn=cmd_count)
    n.add_argument("--project", required=True)
    n.add_argument("--fps", type=int, default=30)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
