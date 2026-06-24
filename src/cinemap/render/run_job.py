"""Subprocess entrypoint for a full video render.

Invoked by the server as `python -m cinemap.render.run_job <pid> <job_id>`. Loads
the project, finds the named render job (already appended to `project.renders`),
runs it, and streams progress as newline-delimited JSON on stdout:

    {"p": <float 0..1>, "m": "<status message>"}
    ...
    {"final": "done"|"error"|"cancelled", "output": "<path>" | "message": "<err>"}

Why a subprocess: the render does CPU-heavy in-process work (mesh decode, draco,
trimesh) plus spawns its own helper subprocesses (Blender, cloud-volume mp pool).
Running everything under a fresh PID with `start_new_session=True` means the
server can SIGKILL the whole process group on Stop — guaranteed teardown, no
orphans, no memory leaks accumulating in the server.
"""
from __future__ import annotations

import json
import signal
import sys
import traceback

from .. import store
from .worker import RenderCancelled, RenderWorker


def _emit(obj: dict) -> None:
    """Write one JSON line to stdout (flushed) — the parent's reader thread parses
    line-by-line. Keep keys short to make the wire format dense in the log."""
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        _emit({"final": "error", "message": "usage: run_job <pid> <job_id>"})
        return 2
    pid, job_id = argv[1], argv[2]

    # Catch SIGTERM so an external termination is visible to the parent instead of
    # being mislabeled as a user cancel.
    def _sigterm(_sig, _frm):
        _emit({"final": "error", "message": "render process received SIGTERM"})
        sys.exit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    try:
        project = store.load(pid)
    except Exception as e:  # noqa: BLE001
        _emit({"final": "error", "message": f"load project failed: {e}"})
        return 1

    job = next((j for j in project.renders if j.id == job_id), None)
    if job is None:
        _emit({"final": "error", "message": f"job {job_id} not in project.renders"})
        return 1

    worker = RenderWorker(project, job)

    def progress(pr: float, msg: str) -> None:
        _emit({"p": float(pr), "m": str(msg)})

    try:
        out = worker.run(progress=progress)
    except RenderCancelled:
        # Mutations to job.status are already persisted by the worker's `finally`
        # path (or will be by the parent on the next render write); emit a final
        # marker so the parent can flip its state machine cleanly.
        _emit({"final": "cancelled"})
        return 0
    except Exception as e:  # noqa: BLE001
        # Surface the traceback to the parent log so we can debug, but the wire
        # message stays short for the UI.
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        _emit({"final": "error", "message": str(e)})
        return 1

    # Persist the job's final state on disk so the parent doesn't need to mirror
    # job mutations across the process boundary (it would race the subprocess if it
    # tried). The next /api/projects/<pid> read will see status=done + output_path.
    try:
        store.save(worker.project)
    except Exception as e:  # noqa: BLE001
        _emit({"final": "error", "message": f"save project failed: {e}"})
        return 1
    _emit({"final": "done", "output": out})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
