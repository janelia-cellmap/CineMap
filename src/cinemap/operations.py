"""operations — the single funnel both the UI and the Claude agent call through.

Every capability that mutates a project lives here. The web routes and (later) the
agent tools are thin wrappers over these functions, so manual edits and agent
edits compose on one project.json (the single source of truth).
"""
from __future__ import annotations

import math
import uuid

from . import store
from .data.manifest import analyze_state
from .data.slice_loader import get_volume
from .models import (
    Camera,
    Keyframe,
    MeshInstance,
    Project,
    RenderJob,
    RenderSettings,
    SlicePlane,
)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


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
    project = Project(id=_uid("proj"), name=name, data_path=data_path, manifest=manifest)
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


def sweep_slice(project: Project, axis: str = "z", n: int = 12,
                duration_per_kf_s: float = 0.4) -> list[Keyframe]:
    center, size = volume_extent_nm(project)
    ax_i = {"x": 0, "y": 1, "z": 2}[axis]
    base = project.keyframes[-1] if project.keyframes else None
    cam = base.camera if base else frame_camera(center, 0.5 * max(size))
    em_name = project.manifest.em.name if project.manifest.em else "em"
    new = []
    for i in range(n):
        pos = size[ax_i] * (0.2 + 0.6 * i / max(1, n - 1))
        kf = Keyframe(
            id=_uid("kf"), label=f"slice {axis}={int(pos)}nm",
            camera=cam.model_copy(),
            slices=[SlicePlane(em_name=em_name, axis=axis, position_nm=pos)],
            meshes=[m.model_copy() for m in (base.meshes if base else [])],
            duration_in_s=duration_per_kf_s,
        )
        new.append(kf)
    project.keyframes.extend(new)
    store.save(project)
    return new


# ----------------------------- render -----------------------------
def create_render_job(project: Project, settings: RenderSettings | None = None,
                      kf_range=None) -> RenderJob:
    job = RenderJob(id=_uid("job"), kf_range=kf_range, settings=settings or RenderSettings())
    project.renders.append(job)
    store.save(project)
    return job
