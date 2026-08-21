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
_VOLUME_FMTS = ("zarr", "zarr2", "zarr3", "n5")  # sliceable label/EM volume formats
_pre_role_cache: dict[str, str] = {}


def _to_http(u: str) -> str:
    if u.startswith("gs://"):
        return "https://storage.googleapis.com/" + u[len("gs://"):]
    if u.startswith("s3://"):
        return "https://s3.amazonaws.com/" + u[len("s3://"):]
    return u


def _precomputed_role(url: str) -> str:
    """mesh / skeleton / volume for a precomputed source, by its `info` `@type`
    (how neuroglancer itself classifies it) — dir names like 'multires' or
    'simplified' aren't reliable. Falls back to a path heuristic if info is
    unreachable."""
    key = url.rstrip("/")
    if key in _pre_role_cache:
        return _pre_role_cache[key]
    role = None
    http = _to_http(key)
    if urllib.parse.urlparse(http).scheme in ("http", "https"):
        try:
            with urllib.request.urlopen(http + "/info", timeout=10) as r:
                t = json.load(r).get("@type", "")
            role = {"neuroglancer_skeletons": "skeleton",
                    "neuroglancer_multilod_draco": "mesh",
                    "neuroglancer_legacy_mesh": "mesh",
                    "neuroglancer_multiscale_volume": "volume"}.get(t)
        except Exception:  # noqa: BLE001  (network/format issues -> path fallback)
            role = None
    if role is None:
        low = key.lower()
        role = ("skeleton" if "skeleton" in low
                else "mesh" if ("/mesh" in low or low.endswith("multires") or "multilod" in low)
                else "volume")
    _pre_role_cache[key] = role
    return role


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
    # A bare URL (no explicit driver) is what neuroglancer infers from the data: a
    # `.zarr`/`.n5` path is a sliceable volume, anything else is precomputed (and may
    # be a mesh/skeleton source — e.g. a layer's 2nd source is its multires meshes).
    if fmt is None:
        low = u.lower()
        fmt = "n5" if ".n5" in low else "zarr" if ".zarr" in low else "precomputed"
    # zarr/n5 sources are volumes; precomputed sources are classified by their info.
    role = "volume" if fmt in _VOLUME_FMTS else _precomputed_role(u)
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
    # A real CSV header has no link in it; an inline neuroglancer state link
    # contains commas and "url", so guard against misdetecting it as a CSV header.
    looks_like_link = "://" in header or "#!" in header
    is_csv_header = (not looks_like_link and "," in header
                     and any(k in header for k in state_keys))

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
    return analyze_state_dict(fetch_state(url))


def analyze_state_dict(state: dict) -> Manifest:
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
            # only OME-Zarr (zarr/n5) images can be sliced by the tensorstore slice
            # loader; a precomputed image (e.g. flyem JPEG) isn't sliceable here, so
            # we skip it rather than crash later trying to read its `.zattrs`.
            vol = next((u for u, fmt, role in srcs
                        if role == "volume" and fmt in _VOLUME_FMTS), None)
            if vol:
                em = EMSource(name=name, zarr_url=vol, voxel_size_nm=voxel_nm)
        elif ltype == "segmentation":
            mesh_url = next((u for u, _, role in srcs if role == "mesh"), None)
            skel_url = next((u for u, _, role in srcs if role == "skeleton"), None)
            # a "volume" source is either an OME-Zarr label volume (zarr/n5 -> sliced +
            # marching cubes) or a precomputed segmentation that carries its own meshes
            # (e.g. flyem hemibrain -> meshes read directly by cloud-volume).
            vol_url, vol_fmt = next(((u, f) for u, f, role in srcs if role == "volume"),
                                    (None, None))
            label_zarr = vol_url if vol_fmt in _VOLUME_FMTS else None
            if vol_fmt in ("precomputed", "neuroglancer-precomputed") and not mesh_url:
                mesh_url = vol_url
            if mesh_url or label_zarr or skel_url:
                seg_ids = [int(s) for s in (layer.get("segments") or []) if str(s).isdigit()]
                skel_render = (layer.get("skeletonRendering") or {}) if skel_url else {}
                shader = skel_render.get("shader", "")
                shader_controls = dict(skel_render.get("shaderControls") or {})
                meshes.append(MeshSource(name=name, mesh_url=mesh_url or "",
                                         label_zarr=label_zarr or "", skeleton_url=skel_url or "",
                                         skeleton_shader=shader,
                                         skeleton_shader_controls=shader_controls,
                                         segment_ids=seg_ids))

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
