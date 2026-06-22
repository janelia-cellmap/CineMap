"""Optional renderer-side shortcut from HTTP data URLs to local filesystem paths.

Project files keep portable HTTP URLs for Neuroglancer/browser use. When CineMap runs
on a machine that can see the same data under /nrs or /groups, Python loaders can skip
HTTP and read bytes directly from disk.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


_JANELIA_DATA_HOSTS = {
    "cellmap-vm1",
    "cellmap-vm1.int.janelia.org",
}


def _configured_mappings() -> list[tuple[str, str]]:
    """Mappings from CINEMAP_DATA_ROOTS.

    Format: `url_prefix=/local/root;url_prefix2=/local/root2`. Semicolon is used
    because URL prefixes themselves contain ':'.
    """
    out = []
    raw = os.environ.get("CINEMAP_DATA_ROOTS", "")
    for item in raw.replace("\n", ";").split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        prefix, root = item.split("=", 1)
        prefix, root = prefix.rstrip("/"), root.rstrip("/")
        if prefix and root:
            out.append((prefix, root))
    return out


def _join_under(root: str, rel: str) -> str | None:
    root_real = os.path.realpath(root)
    rel = urllib.parse.unquote(rel).lstrip("/")
    cand = os.path.realpath(os.path.join(root_real, rel))
    try:
        if os.path.commonpath([root_real, cand]) != root_real:
            return None
    except ValueError:
        return None
    return cand


def local_path(url: str, *, require_exists: bool = True) -> str | None:
    """Return a local path for `url` when a safe mapping applies, else None."""
    if not url:
        return None
    if url.startswith("file://"):
        path = urllib.parse.urlparse(url).path
        return path if (not require_exists or os.path.exists(path)) else None

    clean = url.rstrip("/")
    for prefix, root in _configured_mappings():
        pfx = prefix.rstrip("/")
        if clean == pfx or clean.startswith(pfx + "/"):
            rel = clean[len(pfx):].lstrip("/")
            path = _join_under(root, rel)
            if path and (not require_exists or os.path.exists(path)):
                return path
            return None

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.hostname in _JANELIA_DATA_HOSTS:
        # Common CellMap deployment: browser URLs expose mounted data through cellmap-vm1,
        # while cluster nodes can read the same tree directly.
        if parsed.path == "/nrs/data" or parsed.path.startswith("/nrs/data/"):
            path = _join_under("/nrs/cellmap/data", parsed.path[len("/nrs/data/"):])
            if path and (not require_exists or os.path.exists(path)):
                return path
        for root in ("/nrs", "/groups"):
            marker = root + "/"
            if parsed.path == root or parsed.path.startswith(marker):
                path = _join_under(root, parsed.path[len(marker):])
                if path and (not require_exists or os.path.exists(path)):
                    return path
    return None


def file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def localized_url(url: str) -> str:
    """Return a `file://` URL if local data is available, otherwise the original URL."""
    path = local_path(url)
    if path:
        return file_uri(path)
    return url


def read_bytes(url: str, timeout: float | None = None) -> bytes:
    path = local_path(url)
    if path:
        with open(path, "rb") as f:
            return f.read()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def read_json(url: str, timeout: float | None = None) -> dict:
    path = local_path(url)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def tensorstore_kvstore(base_url: str) -> dict:
    """TensorStore kvstore config for a zarr level base URL."""
    path = local_path(base_url)
    if path:
        return {"driver": "file", "path": path.rstrip("/") + "/"}
    return {"driver": "http", "base_url": base_url}
