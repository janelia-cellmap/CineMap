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
    Sweep,
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


def mesh_aabb_nm(mesh_url: str, segment_ids: list[int]):
    """(lo_xyz, hi_xyz) nm axis-aligned bounding box of the segments, or None.

    Uses each segment's COARSEST-LOD bounds (a tiny fetch via seg_bbox) and the shared
    on-disk mesh cache — sizing a cutaway/scan must not re-decode full-resolution
    geometry (that made 'Add cutaway' take minutes). Coarse bounds ≈ true bounds, and
    the callers pad the range anyway."""
    from .config import PROJECTS_DIR
    from .data.mesh_loader import MeshLoader

    loader = MeshLoader(mesh_url, cache_dir=PROJECTS_DIR / ".mesh_cache")
    ids = segment_ids or list(loader.list_segments()[:1])
    lo = hi = None
    for sid in ids[:8]:  # cap cost
        bb = loader.seg_bbox(sid)            # coarsest-LOD bounds; None on failure
        if bb is None:
            continue
        blo, bhi = bb
        lo = list(blo) if lo is None else [min(a, b) for a, b in zip(lo, blo)]
        hi = list(bhi) if hi is None else [max(a, b) for a, b in zip(hi, bhi)]
    return (lo, hi) if lo is not None else None


def mesh_bbox_nm(mesh_url: str, segment_ids: list[int]):
    """(center_xyz_nm, radius_nm) of the given mesh segments, or None on failure."""
    bb = mesh_aabb_nm(mesh_url, segment_ids)
    if bb is None:
        return None
    lo, hi = bb
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = 0.5 * max(hi[i] - lo[i] for i in range(3))
    return center, max(radius, 1.0)


def frame_camera(center, radius_nm, fov_deg=40.0, azimuth_deg=35.0, elevation_deg=22.0,
                 axis: str = "z") -> Camera:
    """Place a camera that frames a sphere of `radius_nm` around `center`. `axis` is the
    orbit POLE: the camera circles in the plane perpendicular to it (azimuth) and tilts
    toward it (elevation). 'z' = circle in XY (default), 'y' = XZ, 'x' = YZ."""
    dist = radius_nm / max(0.1, math.sin(math.radians(fov_deg) / 2)) * 1.1
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    c, s = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az)
    pole = math.sin(el)
    # order the (in-plane, in-plane, pole) components onto world axes by the chosen pole
    dirv = {"z": [c, s, pole], "y": [c, pole, s], "x": [pole, c, s]}.get(axis, [c, s, pole])
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


def duplicate_keyframe(project: Project, source_id: str, after_id: str | None = None) -> Keyframe:
    src_idx = next((i for i, k in enumerate(project.keyframes) if k.id == source_id), None)
    if src_idx is None:
        raise ValueError("no such source keyframe")
    insert_after = src_idx
    if after_id:
        insert_after = next((i for i, k in enumerate(project.keyframes) if k.id == after_id), None)
        if insert_after is None:
            raise ValueError("no such target keyframe")

    src = project.keyframes[src_idx]
    dup = src.model_copy(deep=True)
    dup.id = _uid("kf")
    dup.label = (src.label or "keyframe") + " copy"
    # A pasted frame is a normal manual keyframe, even if copied from a generated orbit/scan.
    dup.group = None
    dup.group_label = ""
    project.keyframes.insert(insert_after + 1, dup)
    store.save(project)
    return dup


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
    "visible", "saturation", "render_3d", "metallic", "roughness",
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


def _rgb_to_hex(rgb) -> str:
    """[r,g,b] in 0–1 -> '#rrggbb' (neuroglancer color string)."""
    r, g, b = (max(0, min(255, round(float(c) * 255))) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _patch_ng_state_field(ng_state: dict, mesh_name: str, field, value, segment_id=None) -> None:
    """Mirror a propagated MeshInstance change into the keyframe's neuroglancer state
    layer, so the stored NG link reflects it (round-trips through gotoKf / re-capture).
    Maps CineMap fields -> NG layer JSON keys; 'color' is a CineMap-only mesh tint with
    no NG equivalent, so it's left alone."""
    if not isinstance(ng_state, dict):
        return
    layer = next((L for L in ng_state.get("layers", []) if L.get("name") == mesh_name), None)
    if layer is None:
        return
    if field == "segment_color" and segment_id is not None:
        layer.setdefault("segmentColors", {})[str(segment_id)] = _rgb_to_hex(value)
    elif field == "default_color":
        layer["segmentDefaultColor"] = _rgb_to_hex(value)
    elif field == "color_seed":
        layer["colorSeed"] = int(value)
    elif field == "visible":
        layer["visible"] = bool(value)
    elif field == "object_alpha":
        layer["objectAlpha"] = float(value)
    elif field == "silhouette":
        layer["meshSilhouetteRendering"] = float(value)
    elif field == "saturation":
        layer["saturation"] = float(value)


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
        upd = {"meshes": new_meshes}
        # mirror the change into the keyframe's stored neuroglancer state too, so the NG
        # link round-trips (gotoKf / re-capture keep the new value instead of reverting).
        if kfs[i].ng_state:
            import copy
            ng = copy.deepcopy(kfs[i].ng_state)
            _patch_ng_state_field(ng, mesh_name, field, value, segment_id)
            upd["ng_state"] = ng
        kfs[i] = kfs[i].model_copy(update=upd)
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
               duration_per_kf_s: float = 0.6,
               total_duration_s: float | None = None, axis: str = "z") -> list[Keyframe]:
    base, target, radius = _base_framing(project, target, radius_nm)
    # start the orbit at the current camera's azimuth (in the orbit plane) so the first
    # keyframe doesn't swing away from the framing the user/agent set (fall back to 35°).
    # The two in-plane world axes depend on the orbit pole `axis`.
    plane_ax = {"z": (0, 1), "y": (0, 2), "x": (1, 2)}.get(axis, (0, 1))
    az0 = 35.0
    if base is not None:
        d0 = base.camera.position_nm[plane_ax[0]] - target[plane_ax[0]]
        d1 = base.camera.position_nm[plane_ax[1]] - target[plane_ax[1]]
        if d0 or d1:
            az0 = math.degrees(math.atan2(d1, d0))
    gid = _uid("grp"); glabel = f"orbit {int(degrees)}° ×{n}"
    new = []
    for i in range(n):
        az = az0 + degrees * i / max(1, n - 1)
        kf = Keyframe(
            id=_uid("kf"), label=f"orbit {int(az)}°",
            camera=frame_camera(target, radius, azimuth_deg=az,
                                elevation_deg=elevation_deg, axis=axis),
            slices=[s.model_copy() for s in (base.slices if base else [])],
            meshes=[m.model_copy() for m in (base.meshes if base else [])],
            # duration-driven: spread total_duration_s across the orbit's keyframes so the
            # whole orbit takes that long regardless of how many sample its arc.
            duration_in_s=(duration_per_kf_s if total_duration_s is None
                           else total_duration_s / max(1, n)),
            group=gid, group_label=glabel,
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


def add_sweep(project: Project, layer: str = "", axis: str = "z", normal=None, side: int = 1,
              from_nm=None, to_nm=None, start_s=None, duration_s=None,
              from_ng=None, to_ng=None, easing: str = "linear",
              kind: str = "cutaway", em_name: str = "", mirror: bool = False,
              cap: bool = True,
              commit: bool = True) -> Sweep:
    """Add an independent plane sweep on its OWN timeline (decoupled from the camera).
    kind='cutaway' slides a mesh layer's clip plane; kind='slice' sweeps an EM cross-section.
    Slides from `from_nm` to `to_nm` over [start_s, start_s+duration_s]. Defaults: the full
    layer/volume bounds along `axis`, start 0, lasting the movie's length (or 4s).
    `from_ng`/`to_ng` accept neuroglancer coords (a triple => oblique plane along A->B; a
    single value => a depth)."""
    base = project.keyframes[-1] if project.keyframes else None
    oblique = False
    bbox_layer = layer if kind == "cutaway" else None
    if base is not None and (from_ng is not None or to_ng is not None):
        fx, tx, a, b = _ng_coords_to_nm(base, project, axis, from_ng, to_ng)
        if fx and tx:
            normal = _unit([tx[i] - fx[i] for i in range(3)]); axis = _dominant_axis(normal)
            ai = {"x": 0, "y": 1, "z": 2}[axis]
            # If the travel is along ONE axis, this is an axis-aligned slice — keep it
            # axis-aligned (fast read_slice + correct sign) rather than an oblique resample.
            if kind == "slice" and abs(normal[ai]) > 0.999:
                from_nm, to_nm = fx[ai], tx[ai]    # literal coordinate (positive), axis read
            else:
                # From/To set the cut DIRECTION; the sweep range fits the layer's extent
                # along that normal (falls back to the literal point offsets if no bounds).
                rng = _layer_offset_range(project, base, bbox_layer, normal) if kind == "cutaway" else None
                if rng:
                    from_nm, to_nm = rng
                else:
                    from_nm = sum(normal[j] * fx[j] for j in range(3))
                    to_nm = sum(normal[j] * tx[j] for j in range(3))
                oblique = True
        else:
            from_nm = a if a is not None else from_nm
            to_nm = b if b is not None else to_nm
    ax_i = {"x": 0, "y": 1, "z": 2}[axis]
    if from_nm is None or to_nm is None:
        lo, hi = _plane_default_range(project, base, ax_i, bbox_layer) if base else (0.0, 1.0)
        from_nm = lo if from_nm is None else from_nm
        to_nm = hi if to_nm is None else to_nm
    if kind == "cutaway":
        # Default direction = REVEAL: start with nothing cut and progressively cut toward
        # the end. The clip removes the dot(x,normal) > position side for side>=0 ('nothing
        # cut' is the HIGH-offset end) and < position for side<0. Order from/to so the cut
        # grows over time regardless of which corner came first.
        lo_off, hi_off = (from_nm, to_nm) if from_nm <= to_nm else (to_nm, from_nm)
        from_nm, to_nm = (hi_off, lo_off) if side >= 0 else (lo_off, hi_off)
    # (slice keeps from->to literal: the EM plane travels from the start position to the stop)
    total = sum(k.duration_in_s for k in project.keyframes) or 4.0
    sw = Sweep(id=_uid("sw"), kind=kind, layer=layer, em_name=em_name, axis=axis,
               normal=(normal if oblique else None),
               side=int(side), from_nm=float(from_nm), to_nm=float(to_nm),
               start_s=float(start_s if start_s is not None else 0.0),
               duration_s=float(duration_s if duration_s is not None else total),
               easing=easing, mirror=bool(mirror), cap=bool(cap))
    project.sweeps.append(sw)
    if commit:                 # commit=False -> build the sweep for a preview without saving
        store.save(project)
    return sw


def remove_sweep(project: Project, sweep_id: str) -> None:
    project.sweeps = [s for s in project.sweeps if s.id != sweep_id]
    store.save(project)


def update_sweep(project: Project, sweep_id: str, **fields) -> Sweep | None:
    """Patch an existing sweep's fields (start_s, duration_s, side, easing, from_nm,
    to_nm, axis, layer, cap, enabled). Unknown/None fields are ignored."""
    sw = next((s for s in project.sweeps if s.id == sweep_id), None)
    if sw is None:
        return None
    for k, v in fields.items():
        if v is not None and hasattr(sw, k):
            setattr(sw, k, v)
    store.save(project)
    return sw


def _layer_offset_range(project: Project, base: Keyframe, layer, normal):
    """[lo, hi] of the mesh layer's extent projected onto `normal` (nm), or None — so an
    OBLIQUE cutaway sweeps across the actual mesh instead of running far past it (using the
    literal From/To point offsets often spans the whole volume, so most of the sweep shows
    no change)."""
    lo = hi = None
    for m in (base.meshes if base else []):
        if (layer and m.mesh_name != layer) or not m.segment_ids:
            continue
        src = next((s for s in project.manifest.meshes if s.name == m.mesh_name), None)
        bb = mesh_aabb_nm(src.mesh_url, m.segment_ids) if src else None
        if not bb:
            continue
        blo, bhi = bb
        center = [(blo[i] + bhi[i]) / 2 for i in range(3)]
        off = sum(center[i] * normal[i] for i in range(3))
        # exact half-extent of the AABB projected onto the (possibly oblique) normal
        half = 0.5 * sum(abs(normal[i]) * (bhi[i] - blo[i]) for i in range(3)) * 1.03  # tiny pad
        lo = off - half if lo is None else min(lo, off - half)
        hi = off + half if hi is None else max(hi, off + half)
    return (lo, hi) if lo is not None else None


def _unit(v):
    import math as _m
    n = _m.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def _dominant_axis(normal) -> str:
    return ["x", "y", "z"][max(range(3), key=lambda i: abs(normal[i]))]


def _ng_coords_to_nm(base: Keyframe, project: Project, axis: str,
                     from_ng, to_ng):
    """Convert neuroglancer-unit coords (as shown in NG, in its dimension order) to nm,
    using the base keyframe's NG grid. A triple -> an (x,y,z) nm point; a single value
    -> a scalar depth (nm) along `axis`. Returns (from_xyz, to_xyz, start_nm, stop_nm)."""
    from .data.ng_camera import _voxel_nm_from_state, _xyz_perm
    st = base.ng_state or {}
    vox = _voxel_nm_from_state(st, project.manifest.voxel_size_nm)  # nm/unit, dim order
    perm = _xyz_perm(st)

    def pt(vals):
        scaled = [float(vals[i]) * vox[i] for i in range(3)]        # dim order -> nm
        return [scaled[perm[0]], scaled[perm[1]], scaled[perm[2]]]  # -> x, y, z

    if from_ng and to_ng and len(from_ng) >= 3 and len(to_ng) >= 3:
        return pt(from_ng), pt(to_ng), None, None
    sc = vox[perm[{"x": 0, "y": 1, "z": 2}[axis]]]                  # nm/unit on this axis
    a = float(from_ng[0]) * sc if from_ng else None
    b = float(to_ng[0]) * sc if to_ng else None
    return None, None, a, b


def plane_move(project: Project, axis: str = "z", mode: str = "slice",
               start_nm: float | None = None, stop_nm: float | None = None,
               from_xyz: list[float] | None = None, to_xyz: list[float] | None = None,
               from_ng: list[float] | None = None, to_ng: list[float] | None = None,
               n: int = 12, mesh_name: str | None = None, side: int = 1,
               duration_per_kf_s: float = 0.4,
               total_duration_s: float | None = None) -> list[Keyframe]:
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
    if from_ng is not None or to_ng is not None:   # neuroglancer-unit coords -> nm
        from_xyz, to_xyz, start_nm, stop_nm = _ng_coords_to_nm(base, project, axis, from_ng, to_ng)
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

    gid = _uid("grp")   # shared group so the timeline collapses the scan to one card
    glabel = f"{mode} scan {axis}{' (oblique)' if normal else ''} ×{n}"
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
        # inherit the base keyframe's slice appearance (opacity + NG contrast/shader) so a
        # scan doesn't silently reset the EM look to neuroglancer's defaults mid-shot
        base_sl = next((s for s in base.slices if s.em_name == em_name), None)
        if do_slice:
            fields = ({"opacity": base_sl.opacity, "shader": base_sl.shader,
                       "shader_controls": dict(base_sl.shader_controls)} if base_sl else {})
            slices = [SlicePlane(em_name=em_name, axis=axis, position_nm=offset,
                                 normal=normal, visible=True, **fields)]
        else:
            slices = [s.model_copy() for s in base.slices]
        # duration-driven: spread total_duration_s across the scan (first keyframe is the
        # instant lead-in to the start, the rest divide the span) so the whole scan takes
        # total_duration_s regardless of how many keyframes sample it.
        dur = (duration_per_kf_s if total_duration_s is None
               else (0.0 if i == 0 else total_duration_s / max(1, n - 1)))
        new.append(Keyframe(
            id=_uid("kf"), label=f"{mode} {axis}={int(offset)}nm{' (oblique)' if normal else ''}",
            camera=base.camera.model_copy(), slices=slices,
            meshes=meshes, duration_in_s=dur,
            group=gid, group_label=glabel,
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
