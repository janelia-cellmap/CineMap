#!/usr/bin/env bash
# Launch the CineMap app (FastAPI + neuroglancer scouting viewer).
# The Blender render worker runs headless on the GPU as a subprocess (no display
# needed); the neuroglancer viewer is served for the browser to embed.
set -euo pipefail
cd "$(dirname "$0")"

ENV="${CINEMAP_ENV:-cinemap}"
HOST="${CINEMAP_HOST:-0.0.0.0}"          # uvicorn bind (0.0.0.0 = all interfaces)
PORT="${CINEMAP_PORT:-8000}"

# Bind the neuroglancer viewer to all interfaces (incl. loopback) by default; the app
# rewrites the iframe URL to whatever host the browser used (see server._ng_url_for), so
# this works for both local (localhost) and remote access. Override with CINEMAP_NG_BIND
# (e.g. a specific hostname) if needed.
export CINEMAP_NG_BIND="${CINEMAP_NG_BIND:-0.0.0.0}"

echo "CineMap  →  http://$(hostname -f):${PORT}  (or http://localhost:${PORT}; server bound on ${HOST})"
exec conda run --no-capture-output -n "$ENV" \
  uvicorn cinemap.server:app --host "$HOST" --port "$PORT"
