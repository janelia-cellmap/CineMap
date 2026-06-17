"""operations — the single funnel both the UI and the Claude agent call through.

Every capability that mutates a project lives here. The web routes and (later) the
agent tools are thin wrappers over these functions, so manual edits and agent
edits compose on one project.json (the single source of truth).
"""
from __future__ import annotations

import datetime
import math
import re
import uuid

from . import store
from .data.manifest import analyze_state
from .data.slice_loader import get_volume
from .models import (
    Camera,
    ClipPlane,
    Keyframe,
    MeshInstance,
    Project,
    RenderJob,
    RenderSettings,
    SlicePlane,
)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _slug(text: str, maxlen: int = 40) -> str:
    """A filesystem/URL-safe lowercase slug (used to name project dirs readably)."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:maxlen].strip("-") or "untitled"


def _stamp() -> str:
    return f"{datetime.datetime.now():%Y%m%d-%H%M%S}"


def _disambiguate(base: str, taken) -> str:
    """`base`, else `base-2`, `base-3`, … — the first name `taken(name)` rejects.
    Keeps dir names clean (datetime only); a numeric suffix appears solely when two
    are created in the same second."""
    if not taken(base):
        return base
    n = 2
    while taken(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def project_id(name: str) -> str:
    """`<name-slug>` — the project is the stable named dir; its movies are the
    datetime-stamped subdirs under it (see `render_id`)."""
    return _disambiguate(_slug(name), store.exists)


def render_id(project: Project, prefix: str = "") -> str:
    """A movie's dir under `<project>/renders/`: `<YYYYmmdd-HHMMSS>` (or
    `<prefix>-<YYYYmmdd-HHMMSS>`, e.g. `thumb-…`), so renders sort chronologically."""
    rdir = store.project_dir(project.id) / "renders"
    taken = {j.id for j in project.renders}
    base = f"{prefix}-{_stamp()}" if prefix else _stamp()
    return _disambiguate(base, lambda i: i in taken or (rdir / i).exists())


# ----------------------------- analysis / framing helpers -----------------------------
def volume_extent_nm(project: Project) -> tuple[list[float], list[float]]:
    """(center_xyz_nm, size_xyz_nm) of the EM volume from multiscale metadata."""
    em = project.manifest.em
    if not em:
        return [0, 0, 0], [10000, 10000, 10000]
    vol = get_volume(em.zarr_url)
    shp = vol.level_shape_zyx(0)  # z,y,x
    sc = vol.level_scale_nm[0]  # z,y,x nm
    size_zyx = [shp[i] * sc[i] for i in range(3)]
    size_xyz = [size_zyx[2], size_zyx[1], size_zyx[0]]
    center = [s / 2 for s in size_xyz]
    return center, size_xyz


def mesh_bbox_nm(mesh_url: str, segment_ids: list[int]):
    """(center_xyz_nm, radius_nm) of the given mesh segments, or None on failure."""
    from .data.mesh_loader import MeshLoader

    loader = MeshLoader(mesh_url)
    ids = segment_ids or list(loader.list_segments()[:1])
    lo = hi = None
    for sid in ids[:8]:  # cap cost
        try:
            v = loader.load(sid).vertices
        except Exception:
            continue
        vlo, vhi = v.min(0), v.max(0)
        lo = vlo if lo is None else [min(a, b) for a, b in zip(lo, vlo)]
        hi = vhi if hi is None else [max(a, b) for a, b in zip(hi, vhi)]
    if lo is None:
        return None
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = 0.5 * max(hi[i] - lo[i] for i in range(3))
    return center, max(radius, 1.0)


def frame_camera(center, radius_nm, fov_deg=40.0, azimuth_deg=35.0, elevation_deg=22.0) -> Camera:
    """Place a camera that frames a sphere of `radius_nm` around `center`."""
    dist = radius_nm / max(0.1, math.sin(math.radians(fov_deg) / 2)) * 1.1
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    dirv = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    pos = [center[i] + dirv[i] * dist for i in range(3)]
    return Camera(position_nm=pos, look_at_nm=list(center), fov_deg=fov_deg)


# ----------------------------- project lifecycle -----------------------------
def create_project(name: str, data_path: str) -> Project:
    """Create a project with NO keyframes — the user bakes the first one from the
    scouting view (the empty timeline says "Bake one from the scouting view").

    We only resolve the data manifest here. Camera framing and which segments/
    layers are captured happen at bake time from the live neuroglancer state, so
    there's no need (and no expensive whole-volume read) at creation."""
    manifest = analyze_state(data_path)
    project = Project(id=project_id(name), name=name, data_path=data_path, manifest=manifest)
    store.save(project)
    return project


# ----------------------------- keyframe ops -----------------------------
def add_keyframe(project: Project, keyframe: Keyframe | None = None, label: str = "") -> Keyframe:
    if keyframe is None:
        base = project.keyframes[-1] if project.keyframes else None
        if base is None:
            center, size = volume_extent_nm(project)
            keyframe = Keyframe(id=_uid("kf"), label=label or "keyframe",
                               camera=frame_camera(center, 0.5 * max(size)))
        else:
            keyframe = base.model_copy(deep=True)
            keyframe.id = _uid("kf")
            keyframe.label = label or "keyframe"
    project.keyframes.append(keyframe)
    store.save(project)
    return keyframe


def delete_keyframe(project: Project, keyframe_id: str) -> None:
    project.keyframes = [k for k in project.keyframes if k.id != keyframe_id]
    store.save(project)


def reorder_keyframes(project: Project, order: list[str]) -> None:
    by_id = {k.id: k for k in project.keyframes}
    project.keyframes = [by_id[i] for i in order if i in by_id]
    store.save(project)


def update_keyframe(project: Project, keyframe_id: str, **fields) -> Keyframe:
    kf = next(k for k in project.keyframes if k.id == keyframe_id)
    updated = kf.model_copy(update=fields)
    project.keyframes = [updated if k.id == keyframe_id else k for k in project.keyframes]
    store.save(project)
    return updated


# per-layer settings that can be edited on a keyframe and propagated across keyframes.
# segment_colors is handled specially (per-segment override keyed by segment id).
_PROPAGATABLE_FIELDS = {
    "color", "color_seed", "default_color", "object_alpha", "silhouette",
    "visible", "saturation", "render_3d",
}


def _colors_equal(a, b, tol=1.0 / 512) -> bool:
    if a is None or b is None:
        return a is b or a == b
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def _values_equal(field, a, b) -> bool:
    if field in ("color", "default_color"):
        return _colors_equal(a, b)
    if field in ("object_alpha", "silhouette", "saturation"):
        return abs(float(a) - float(b)) <= 1e-4
    return a == b


_UNSET = object()


def diff_layer_settings(old_meshes, new_meshes) -> list[dict]:
    """Per-layer setting changes between two keyframe mesh lists (matched by mesh_name),
    over the propagatable fields + per-segment colors. Each change is
    {mesh_name, field, old, new[, segment_id]} — used to offer propagation after a
    keyframe is updated from the neuroglancer view."""
    old_by = {m.mesh_name: m for m in old_meshes}
    changes: list[dict] = []
    for nm in new_meshes:
        om = old_by.get(nm.mesh_name)
        if om is None:
            continue
        for f in ("color", "color_seed", "default_color", "object_alpha",
                  "silhouette", "visible", "saturation"):
            ov, nv = getattr(om, f), getattr(nm, f)
            if not _values_equal(f, ov, nv):
                changes.append({"mesh_name": nm.mesh_name, "field": f, "old": ov, "new": nv})
        for sid in set(om.segment_colors) | set(nm.segment_colors):
            ov, nv = om.segment_colors.get(sid), nm.segment_colors.get(sid)
            if not _colors_equal(ov, nv):
                changes.append({"mesh_name": nm.mesh_name, "field": "segment_color",
                                "segment_id": int(sid), "old": ov, "new": nv})
    return changes


def propagate_layer_field(project: Project, from_keyframe_id: str, mesh_name: str,
                          field: str, value, segment_id: int | None = None,
                          direction: str = "right", match_old: bool = True,
                          match_value=_UNSET) -> dict:
    """Edit a layer (`mesh_name`) setting on one keyframe and propagate it to others.

    `field` is a MeshInstance field, or "segment_color" (then `segment_id` selects which
    segment's per-segment override to set). `direction`: "this" | "right" (this + later) |
    "left" (this + earlier) | "all". With `match_old` (the default), a target keyframe is
    changed ONLY where its current value equals the source keyframe's OLD value — so e.g.
    "make it blue wherever it was red" won't clobber a keyframe deliberately set green.
    The source keyframe is always updated. Returns {changed: [keyframe_id, ...]}.
    """
    if field != "segment_color" and field not in _PROPAGATABLE_FIELDS:
        raise ValueError(f"field not propagatable: {field}")
    kfs = project.keyframes
    idx = next((i for i, k in enumerate(kfs) if k.id == from_keyframe_id), None)
    if idx is None:
        raise ValueError("no such keyframe")

    def layer(kf):
        return next((m for m in kf.meshes if m.mesh_name == mesh_name), None)

    src = layer(kfs[idx])
    if src is None:
        raise ValueError(f"layer {mesh_name} not in keyframe {from_keyframe_id}")

    sid = str(segment_id) if segment_id is not None else None
    # value to MATCH against in target keyframes. Normally the source's current value, but
    # callers can pass an explicit match_value (e.g. after the source was already updated
    # from the neuroglancer view, so its "current" is the new value, not the old one).
    if match_value is not _UNSET:
        old = match_value
    else:
        old = src.segment_colors.get(sid) if field == "segment_color" else getattr(src, field)

    if direction == "this":
        rng = {idx}
    elif direction == "right":
        rng = set(range(idx, len(kfs)))
    elif direction == "left":
        rng = set(range(0, idx + 1))
    else:  # all
        rng = set(range(len(kfs)))

    changed: list[str] = []
    for i in rng:
        lyr = layer(kfs[i])
        if lyr is None:
            continue
        cur = lyr.segment_colors.get(sid) if field == "segment_color" else getattr(lyr, field)
        is_src = i == idx
        if not is_src and match_old and not _values_equal(
                "color" if field == "segment_color" else field, cur, old):
            continue
        if field == "segment_color":
            new_overrides = dict(lyr.segment_colors)
            new_overrides[sid] = value
            new_meshes = [m.model_copy(update={"segment_colors": new_overrides})
                          if m.mesh_name == mesh_name else m for m in kfs[i].meshes]
        else:
            new_meshes = [m.model_copy(update={field: value})
                          if m.mesh_name == mesh_name else m for m in kfs[i].meshes]
        kfs[i] = kfs[i].model_copy(update={"meshes": new_meshes})
        changed.append(kfs[i].id)
    project.keyframes = kfs
    store.save(project)
    return {"changed": changed, "old": old, "value": value}


# ----------------------------- preset shot generators -----------------------------
def _base_framing(project: Project, target, radius_nm):
    """Derive orbit target + radius from the current keyframe's camera, so presets
    preserve the framing the user/agent already set (fall back to the volume)."""
    base = project.keyframes[-1] if project.keyframes else None
    if base is not None and target is None and radius_nm is None:
        cam = base.camera
        target = list(cam.look_at_nm)
        dist = math.dist(cam.position_nm, cam.look_at_nm)
        radius_nm = max(1.0, dist * math.sin(math.radians(cam.fov_deg) / 2) / 1.1)
        return base, target, radius_nm
    center, size = volume_extent_nm(project)
    return base, target or center, radius_nm or 0.5 * max(size)


def make_orbit(project: Project, degrees: float = 360.0, n: int = 12,
               elevation_deg: float = 22.0, target=None, radius_nm=None,
               duration_per_kf_s: float = 0.6) -> list[Keyframe]:
    base, target, radius = _base_framing(project, target, radius_nm)
    # start the orbit at the current camera's azimuth so the first keyframe doesn't
    # swing away from the framing the user/agent set (fall back to 35°).
    az0 = 35.0
    if base is not None:
        dx = base.camera.position_nm[0] - target[0]
        dy = base.camera.position_nm[1] - target[1]
        if dx or dy:
            az0 = math.degrees(math.atan2(dy, dx))
    new = []
    for i in range(n):
        az = az0 + degrees * i / max(1, n - 1)
        kf = Keyframe(
            id=_uid("kf"), label=f"orbit {int(az)}°",
            camera=frame_camera(target, radius, azimuth_deg=az, elevation_deg=elevation_deg),
            slices=[s.model_copy() for s in (base.slices if base else [])],
            meshes=[m.model_copy() for m in (base.meshes if base else [])],
            duration_in_s=duration_per_kf_s,
        )
        new.append(kf)
    project.keyframes.extend(new)
    store.save(project)
    return new


def _plane_default_range(project: Project, base: Keyframe, ax_i: int,
                         mesh_name: str | None) -> tuple[float, float]:
    """Default scan range along an axis = the bounds of the mesh layer(s) involved
    (so the plane scans across the actual structure, not empty volume), padded; falls
    back to the EM volume extent when no mesh bounds resolve."""
    lo = hi = None
    for m in base.meshes:
        if (mesh_name is not None and m.mesh_name != mesh_name) or not m.segment_ids:
            continue
        src = next((s for s in project.manifest.meshes if s.name == m.mesh_name), None)
        bb = mesh_bbox_nm(src.mesh_url, m.segment_ids) if src else None
        if not bb:
            continue
        c, r = bb
        lo = c[ax_i] - r if lo is None else min(lo, c[ax_i] - r)
        hi = c[ax_i] + r if hi is None else max(hi, c[ax_i] + r)
    if lo is None:
        center, size = volume_extent_nm(project)
        lo, hi = center[ax_i] - 0.55 * size[ax_i], center[ax_i] + 0.55 * size[ax_i]
    pad = (hi - lo) * 0.1 or 1000.0
    return lo - pad, hi + pad


def _unit(v):
    import math as _m
    n = _m.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def _dominant_axis(normal) -> str:
    return ["x", "y", "z"][max(range(3), key=lambda i: abs(normal[i]))]


def plane_move(project: Project, axis: str = "z", mode: str = "slice",
               start_nm: float | None = None, stop_nm: float | None = None,
               from_xyz: list[float] | None = None, to_xyz: list[float] | None = None,
               n: int = 12, mesh_name: str | None = None, side: int = 1,
               duration_per_kf_s: float = 0.4) -> list[Keyframe]:
    """Lay down keyframes for a plane scanning from a start to a stop with the camera
    held fixed. Two ways to specify the path:
      - axis-aligned: `axis` + scalar `start_nm`/`stop_nm` (default = the involved
        layer's bounds). The plane stays perpendicular to `axis`.
      - oblique:      `from_xyz` -> `to_xyz` (full points). The plane normal is the
        travel direction, so it sweeps face-first along an arbitrary line.
    `mode`: 'slice' (EM cross-section), 'cull' (cut a layer on `side`), or 'both'."""
    base = project.keyframes[-1] if project.keyframes else None
    if base is None:
        raise ValueError("plane_move needs an existing keyframe to scan from")
    do_slice, do_cull = mode in ("slice", "both"), mode in ("cull", "both")
    em_name = project.manifest.em.name if project.manifest.em else "em"

    oblique = from_xyz is not None and to_xyz is not None
    if oblique:
        a_pt, b_pt = [float(c) for c in from_xyz], [float(c) for c in to_xyz]
        normal = _unit([b_pt[i] - a_pt[i] for i in range(3)])
        axis = _dominant_axis(normal)
        def point_at(t):  # along the A->B line
            return [a_pt[i] + (b_pt[i] - a_pt[i]) * t for i in range(3)]
    else:
        ax_i = {"x": 0, "y": 1, "z": 2}[axis]
        lo, hi = _plane_default_range(project, base, ax_i, mesh_name if do_cull else None)
        a = lo if start_nm is None else float(start_nm)
        b = hi if stop_nm is None else float(stop_nm)
        if do_cull and side < 0 and start_nm is None and stop_nm is None:
            a, b = b, a   # reverse the reveal direction when using the default range
        normal = None
        focus = list(base.camera.look_at_nm)
        def point_at(t):  # axis-aligned: move the axis coord, keep the camera focus elsewhere
            p = list(focus); p[ax_i] = a + (b - a) * t; return p

    new = []
    for i in range(n):
        t = i / max(1, n - 1)
        pt = point_at(t)
        offset = (sum(normal[j] * pt[j] for j in range(3)) if normal
                  else pt[{"x": 0, "y": 1, "z": 2}[axis]])
        meshes = []
        for msh in base.meshes:
            mc = msh.model_copy(deep=True)
            if do_cull and (mesh_name is None or msh.mesh_name == mesh_name):
                mc.clip = ClipPlane(axis=axis, position_nm=offset, normal=normal,
                                    side=side, enabled=True)
            meshes.append(mc)
        slices = ([SlicePlane(em_name=em_name, axis=axis, position_nm=offset,
                              normal=normal, visible=True)]
                  if do_slice else [s.model_copy() for s in base.slices])
        new.append(Keyframe(
            id=_uid("kf"), label=f"{mode} {axis}={int(offset)}nm{' (oblique)' if normal else ''}",
            camera=base.camera.model_copy(), slices=slices,
            meshes=meshes, duration_in_s=duration_per_kf_s,
        ))
    project.keyframes.extend(new)
    store.save(project)
    return new


def sweep_slice(project: Project, axis: str = "z", n: int = 12,
                duration_per_kf_s: float = 0.4) -> list[Keyframe]:
    return plane_move(project, axis=axis, mode="slice", n=n,
                      duration_per_kf_s=duration_per_kf_s)


def sweep_clip(project: Project, mesh_name: str | None = None, axis: str = "z",
               n: int = 12, side: int = 1, duration_per_kf_s: float = 0.4) -> list[Keyframe]:
    return plane_move(project, axis=axis, mode="cull", n=n, mesh_name=mesh_name,
                      side=side, duration_per_kf_s=duration_per_kf_s)


# ----------------------------- render -----------------------------
def create_render_job(project: Project, settings: RenderSettings | None = None,
                      kf_range=None) -> RenderJob:
    job = RenderJob(id=render_id(project), kf_range=kf_range, settings=settings or RenderSettings())
    project.renders.append(job)
    store.save(project)
    return job
