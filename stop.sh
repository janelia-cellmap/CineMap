#!/usr/bin/env bash
# Stop CineMap cleanly so no render gets orphaned.
#   1) SIGTERM the server -> uvicorn runs the app's shutdown hook, which cancels in-flight
#      renders: kills each Blender process group AND the server's own child processes
#      (multiprocessing decode workers) BY PARENT PID, before they can be orphaned.
#   2) wait for exit, then SIGKILL any cinemap-specific stragglers as a backstop.
# (We deliberately do NOT pkill generic "multiprocessing" by name — that could hit
#  unrelated Python on the machine. The shutdown hook handles those by parentage.)
set -u

pkill -TERM -f "uvicorn cinemap.server:app" 2>/dev/null || true
for _ in $(seq 1 12); do
  pgrep -f "uvicorn cinemap.server:app" >/dev/null 2>&1 || break
  sleep 1
done

pkill -9 -f "uvicorn cinemap.server:app"    2>/dev/null || true
pkill -9 -f "cinemap.render.blender_script" 2>/dev/null || true
pkill -9 -f "Programming/CineMap.*run.sh"   2>/dev/null || true

echo "cinemap stopped"
