#!/usr/bin/env bash
# Launch the CineMap app (FastAPI + neuroglancer scouting viewer).
# The Blender render worker runs headless on the GPU as a subprocess (no display
# needed); the neuroglancer viewer is served for the browser to embed.
set -euo pipefail
cd "$(dirname "$0")"

ENV=mv_env
HOST="${CINEMAP_HOST:-0.0.0.0}"          # uvicorn bind (0.0.0.0 = all interfaces)
PORT="${CINEMAP_PORT:-8000}"

# The neuroglancer iframe URL must be reachable from the *browser*, so it can't be
# 0.0.0.0 — advertise the machine's resolvable hostname. Override with CINEMAP_NG_BIND.
export CINEMAP_NG_BIND="${CINEMAP_NG_BIND:-$(hostname -f)}"

echo "CineMap  →  http://${CINEMAP_NG_BIND}:${PORT}   (server bound on ${HOST})"
exec conda run --no-capture-output -n "$ENV" \
  uvicorn cinemap.server:app --host "$HOST" --port "$PORT"
