"""Headless render CLI for cluster jobs — render a frame range, then combine.

Splitting a long video across GPU jobs: each job renders a contiguous frame
range to a SHARED frames dir using GLOBAL frame indices (so there are no
collisions), then one final job concatenates them into the mp4.

  render a frame range (one GPU job):
    python -m cinemap.render_cli render --project P.cinemap.json --out DIR \
        --fps 60 --samples 128 --width 1920 --height 1080 --start A --end B

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
from .render.interpolate import build_frames
from .render.worker import RenderWorker


def _load(path: str) -> Project:
    return Project.model_validate_json(Path(path).read_text())


def cmd_count(a) -> None:
    p = _load(a.project)
    print(len(build_frames(p.keyframes, a.fps)))


def cmd_render(a) -> None:
    project = _load(a.project)
    settings = RenderSettings(width=a.width, height=a.height, fps=a.fps, samples=a.samples,
                              export_blend=a.export_blend)
    worker = RenderWorker(project, RenderJob(id="cli", settings=settings))
    worker._cb = None  # not using run()'s progress callback
    out = Path(a.out)
    worker.workdir = out
    worker.frames_dir = out / "frames"
    worker.assets_dir = out / f"assets_{a.start}_{a.end}"
    worker.blend_path = out / "scene.blend"
    worker.frames_dir.mkdir(parents=True, exist_ok=True)
    worker.assets_dir.mkdir(parents=True, exist_ok=True)

    frames = build_frames(project.keyframes, a.fps)
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
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output],
        check=True,
    )
    print(f"[cli] WROTE {output}", flush=True)


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
