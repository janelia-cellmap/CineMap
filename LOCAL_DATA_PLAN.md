# Local-data fast path (not yet implemented)

When the underlying data is on a local mount (typical for cellmap work — meshes
+ zarr live under `/groups/cellmap/...`), we host it over HTTP only so the
**browser** (neuroglancer) can read it. The Python renderer running on the same
machine has no such constraint — it could read the bytes directly from disk and
skip the entire HTTP round-trip.

This plan covers two related changes that share one config knob.

## The two pieces

### A. Renderer prefers local when possible
Add a `localize(url)` helper that maps a remote/HTTP URL to its local-disk
equivalent when the URL falls under a configured mapping (or returns the URL
unchanged when it doesn't). The mesh loader and slice loader call it once at
open time; the rest of the pipeline is unchanged.

- **Meshes (cloud-volume → cloudfiles):** swap `CloudVolume(http_url)` →
  `CloudVolume("file:///abs/path")`. Cloudfiles speaks `file://` natively via
  its `FileInterface`.
- **EM / labels (tensorstore zarr):** swap the kvstore config from
  `{"driver": "http", "base_url": …}` to `{"driver": "file", "path": …}`.
  Everything downstream of `_open_level` is unchanged.

Config:
```sh
# alias = absolute on-disk path. Multiple via colon-separated list.
export CINEMAP_DATA_ROOTS="http://my-host:8000/data/=/groups/cellmap/cellmap/data/"
```

`localize()` checks each prefix and rewrites if matched.

### B. CineMap can host the data itself
Drop the need for a separate `python -m http.server` / nginx by mounting a
`StaticFiles` route on FastAPI:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                   allow_headers=["*"], expose_headers=["Content-Range"])
for alias, root in _parse_data_roots():
    app.mount(f"/data/{alias}", StaticFiles(directory=root, check_dir=False))
```

Same env var (`CINEMAP_DATA_ROOTS`) drives both. NG iframe loads from
`http://localhost:8000/data/...`; renderer's `localize()` resolves to disk.

## Behavior matrix

| Setup | NG | Renderer |
|---|---|---|
| Remote dataset (S3/GCS/external HTTP) | as today | as today (HTTP, no shortcut) |
| Local data, no env var set | as today | as today |
| Local data, env var set, separate static server | as today | **file:// shortcut** |
| Local data, env var set, no separate server | **CineMap serves it** | **file:// shortcut** |

Exporting/sharing a project to another machine still works: the manifest stores
HTTP URLs; only the renderer's `localize()` opportunistically takes the shortcut
when the mapping resolves on the local box.

## Expected speedup

Yes, faster — but **only on the network-bound parts** of asset prep:

- **HTTP fetch** of mesh fragments + zarr chunks → **gone**. Local FS read is
  sub-millisecond per call; localhost HTTP is 1–10 ms (TCP setup, parsing,
  serialization). For zarr's many-small-chunks pattern this stacks up.
- **TLS/keep-alive overhead** → gone.
- **urllib3 connection-pool contention** under heavy parallel fetch → gone.
  (Currently we get "Connection pool is full" warnings during big mesh phases.)
- **Draco decode, trimesh concatenate, PNG encode, marching-cubes** → unchanged
  (CPU-bound, same cost).

Rough estimate for the asset-prep phase specifically:
- **First render** of a new segment set (no `.mesh_cache` hits): **5–20× faster**
  fetch phase. The 1.94 GB / 23-mesh case we hit earlier would drop from minutes
  to seconds for the fetch portion. Draco decode + concat is then the new
  bottleneck.
- **Subsequent renders** (mesh fragments already cached on disk under
  `.mesh_cache`): smaller win, maybe **2–4×**, because the cache is already
  saving the HTTP cost — the win is mostly on zarr slice reads which usually
  aren't cached.
- **Slice-PNG phase** (worker.py: `_slice_png` for moving cameras): clear win on
  every frame that wasn't a cache hit, since each new camera region issues fresh
  zarr reads. Could collapse a multi-minute phase into seconds.
- **Blender phase**: unchanged. This plan does nothing for Cycles render time.

Net: the *whole* render isn't 10× faster, but the gap between "click Render" and
"first frame appears" should shrink noticeably, especially when the warm-cache
misses (different look settings, new mesh budget, fresh segment selection).

## What this does NOT solve

- Neuroglancer in the browser still needs HTTP. No way around that — it's a
  browser sandbox rule.
- Remote datasets get no benefit at all. By design.
- Doesn't address the threading-on-threading issue (`cloudfiles` ships
  `DEFAULT_THREADS = 20` per op, our outer pool also threads — combined 100+
  concurrent threads under heavy parallel fetch). That's a separate, ~30-min
  unification.

## Sequencing when we do this

1. `localize()` helper + env-var parsing in `config.py`.
2. Mesh loader: call `localize()` on `mesh_url` + `label_zarr` in `MeshLoader.__init__`.
3. Slice loader: call `localize()` in `EMVolume.__init__`, switch kvstore in
   `_open_level` when the resolved URL is local.
4. (B) `StaticFiles` mount + CORS in `server.py`. Optional — works even with an
   external static server.
5. Smoke-test: open an existing project whose data is on `/groups/...`, set the
   env var, render. Confirm with `strace -e network -p <pid>` that the fetch
   phase issues no `connect()` syscalls.

## Safety notes (for whenever this lands)

- `localize()` should `os.path.realpath` and reject results that escape the
  configured root (defense-in-depth against malformed mappings or symlink
  shenanigans).
- StaticFiles mount inherits the server's `0.0.0.0:8000` bind. On Janelia's
  internal network this is fine; for any setup where the host is exposed beyond
  the local network, gate it behind a token or restrict the bind.
- `check_dir=False` on the mount so startup isn't O(zarr tree size).
