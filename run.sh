#!/usr/bin/env bash
# Launch the CineMap app (FastAPI + neuroglancer scouting viewer).
# The Blender render worker runs headless on the GPU as a subprocess (no display
# needed); the neuroglancer viewer is served for the browser to embed.
set -euo pipefail
cd "$(dirname "$0")"

ENV="${CINEMAP_ENV:-cinemap}"
HOST="${CINEMAP_HOST:-0.0.0.0}"          # uvicorn bind (0.0.0.0 = all interfaces)
PORT="${CINEMAP_PORT:-8000}"

# Where project state + per-project render assets (meshes, slice PNGs, frames, the
# .mesh_cache fragment cache) live. Defaults to ./projects (next to this repo). Set
# this to a path OUTSIDE the source tree so thousands of generated files don't sit
# on NFS / get walked by your editor / land in `git status`. Examples:
#   export CINEMAP_PROJECTS_DIR=/scratch/$USER/cinemap_projects
#   export CINEMAP_PROJECTS_DIR="$(realpath ..)/cinemap_projects"
export CINEMAP_PROJECTS_DIR="${CINEMAP_PROJECTS_DIR:-$(pwd)/projects}"

# Bind the neuroglancer viewer to all interfaces (incl. loopback) by default; the app
# rewrites the iframe URL to whatever host the browser used (see server._ng_url_for), so
# this works for both local (localhost) and remote access. Override with CINEMAP_NG_BIND
# (e.g. a specific hostname) if needed.
export CINEMAP_NG_BIND="${CINEMAP_NG_BIND:-0.0.0.0}"

echo "CineMap  →  http://$(hostname -f):${PORT}  (or http://localhost:${PORT}; server bound on ${HOST})"
# --reload auto-restarts the server when a source file under src/ changes, so code
# fixes take effect without a manual restart (the render worker runs in-process, so
# edits would otherwise stay stale until restart).
# The repo lives on NFS, where inotify file-watching does NOT fire — so plain --reload
# never triggers. WATCHFILES_FORCE_POLLING makes watchfiles poll instead, so reload
# works on the network filesystem.
export WATCHFILES_FORCE_POLLING="${WATCHFILES_FORCE_POLLING:-1}"
exec conda run --no-capture-output -n "$ENV" \
  uvicorn cinemap.server:app --host "$HOST" --port "$PORT" --reload --reload-dir src
