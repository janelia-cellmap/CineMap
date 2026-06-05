"""Analyze a dataset (a neuroglancer state URL) into an cinemap Manifest.

Splits each neuroglancer layer source into its volume/mesh role, picks the EM
image layer as the slice source, and collects precomputed mesh layers.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..models import EMSource, Manifest, MeshSource

_MESH_PREFIXES = ("zarr://", "zarr2://", "n5://", "precomputed://")


def _clean_url(u: str) -> tuple[str, str | None, str]:
    """Return (url, format, role) for one neuroglancer source string."""
    fmt = None
    for pre in _MESH_PREFIXES:
        if u.startswith(pre):
            fmt = pre[:-3]
            u = u[len(pre) :]
            break
    if "|" in u:
        u, tail = u.rsplit("|", 1)
        tail = tail.rstrip(":")
        if tail:
            fmt = tail
    role = "mesh" if "/mesh/" in u else "skeleton" if "/skeleton/" in u else "volume"
    return u, fmt, role


def _sources(layer: dict) -> list[tuple[str, str | None, str]]:
    src = layer.get("source")
    items = src if isinstance(src, list) else [src]
    out = []
    for s in items:
        u = s if isinstance(s, str) else (s.get("url", "") if isinstance(s, dict) else "")
        if u:
            out.append(_clean_url(u))
    return out


def fetch_state(url: str) -> dict:
    """Resolve a neuroglancer state from any of the forms the viewer produces:
      - inline state:  …/#!%7B…%7D   (URL-encoded JSON embedded in the fragment)
      - gs link:       …/#!gs://bucket/path.json
      - raw json URL:  …/#!https://…/state.json  (or a plain https json URL)
      - a bare JSON string pasted directly
    """
    src = url.strip()
    if "#!" in src:
        src = src.split("#!", 1)[1]
    ref = urllib.parse.unquote(src).strip()

    if ref.startswith("{"):  # inline JSON state
        return json.loads(ref)
    if ref.startswith("gs://"):
        ref = "https://storage.googleapis.com/" + ref[len("gs://") :]
    # Only fetch over http(s): the input is user-supplied, and urlopen otherwise
    # honors file://, ftp://, … (an SSRF / local-file-read vector).
    scheme = urllib.parse.urlparse(ref).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported state URL scheme: {scheme or '(none)'!r}")
    with urllib.request.urlopen(ref, timeout=30) as r:
        return json.load(r)


def parse_state_links(text: str) -> list[tuple[str, str, float | None]]:
    """Parse a pasted/uploaded list of neuroglancer states into
    [(label, link, duration_s|None)].

    Accepts whichever form the user has on hand:
      - one state per line (the robust default — neuroglancer links are full of
        commas, so we never split a bare line on commas)
      - a neuroglancer `video_tool` keypoint script: `#` comment lines are skipped
        (a comment right before a state becomes that keyframe's label), and a line
        that is just a number is the transition DURATION (seconds) into the next
        state — exactly the format `python -m neuroglancer.tool.video_tool` consumes
      - a CSV *with a header* naming a state column (state/url/link/ngl) and an
        optional label column (label/name/title); commas in links must be quoted.
    Lines that are blank or a lone header word are skipped.
    """
    import csv
    import io
    import re

    text = (text or "").strip()
    if not text:
        return []
    lines = text.splitlines()
    header = lines[0].lower()
    state_keys = ("state", "url", "link", "ngl", "neuroglancer")
    label_keys = ("label", "name", "title")
    is_csv_header = "," in header and any(k in header for k in state_keys)

    out: list[tuple[str, str, float | None]] = []
    if is_csv_header:
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames or []
        scol = next((c for c in fields if c.strip().lower() in state_keys), fields[-1])
        lcol = next((c for c in fields if c.strip().lower() in label_keys), None)
        for i, row in enumerate(reader):
            link = (row.get(scol) or "").strip()
            if not link:
                continue
            label = (row.get(lcol) or "").strip() if lcol else ""
            out.append((label or f"state {i + 1}", link, None))
    else:
        last_comment: str | None = None
        pending_duration: float | None = None
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):  # comment -> potential label for the next state
                last_comment = s.lstrip("#").strip() or last_comment
                continue
            if re.fullmatch(r"[+-]?\d+(\.\d+)?", s):  # bare number -> transition duration
                pending_duration = float(s)
                continue
            s = s.strip('"').strip("'")
            if s.lower() in (*state_keys, "states"):
                continue
            out.append((last_comment or f"state {len(out) + 1}", s, pending_duration))
            last_comment = None
            pending_duration = None
    return out


def analyze_state(url: str) -> Manifest:
    state = fetch_state(url)
    dim = state.get("dimensions", {})
    # dimensions are {axis: [resolution_m, unit]} in x,y,z order
    voxel_nm = [dim.get(a, [8e-9, "m"])[0] * 1e9 for a in ("x", "y", "z")]

    em: EMSource | None = None
    meshes: list[MeshSource] = []
    for layer in state.get("layers", []):
        name = layer.get("name", "")
        ltype = layer.get("type")
        srcs = _sources(layer)
        if ltype == "image" and em is None:
            vol = next((u for u, _, role in srcs if role == "volume"), None)
            if vol:
                em = EMSource(name=name, zarr_url=vol, voxel_size_nm=voxel_nm)
        elif ltype == "segmentation":
            mesh_url = next((u for u, _, role in srcs if role == "mesh"), None)
            label_zarr = next((u for u, _, role in srcs if role == "volume"), None)
            skel_url = next((u for u, _, role in srcs if role == "skeleton"), None)
            if mesh_url or label_zarr or skel_url:
                seg_ids = [int(s) for s in (layer.get("segments") or []) if str(s).isdigit()]
                shader = (layer.get("skeletonRendering") or {}).get("shader", "") if skel_url else ""
                meshes.append(MeshSource(name=name, mesh_url=mesh_url or "",
                                         label_zarr=label_zarr or "", skeleton_url=skel_url or "",
                                         skeleton_shader=shader, segment_ids=seg_ids))

    # A mesh-only layer (precomputed mesh, no label volume) can still be generated
    # cleanly from labels: borrow the label volume of a layer sharing the same
    # structure name (last path component of the mesh dir == that of the labels).
    def _struct(u: str) -> str:
        return u.rstrip("/").rsplit("/", 1)[-1]

    label_by_struct = {_struct(m.label_zarr): m.label_zarr for m in meshes if m.label_zarr}
    for m in meshes:
        if m.mesh_url and not m.label_zarr:
            m.label_zarr = label_by_struct.get(_struct(m.mesh_url), "")

    server = ""
    if em:
        # https://host/...  -> https://host
        parts = em.zarr_url.split("/", 3)
        server = "/".join(parts[:3]) if len(parts) >= 3 else ""

    return Manifest(
        title=state.get("title", "untitled"),
        server=server,
        em=em,
        meshes=meshes,
        voxel_size_nm=voxel_nm,
    )
