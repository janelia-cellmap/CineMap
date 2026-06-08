"""Interpolate a keyframe list into a flat list of per-frame scene states.

Camera interpolation matches neuroglancer's video_tool exactly: the look-at center
is interpolated linearly, the view ORIENTATION via spherical-linear interpolation
(slerp), and the zoom (eye->center distance) exponentially. So an orbit/rotation
(same center, changing orientation) actually arcs the camera around the target the
way neuroglancer does, instead of cutting a straight chord. Slices/meshes/annotations
are matched by identity and their opacity lerped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..models import Keyframe


def _ease(t: float, mode: str) -> float:
    if mode == "ease-in-out":
        return t * t * (3 - 2 * t)  # smoothstep
    return t


def _cam_basis(cam):
    """(center, world<-view Rotation, eye->center distance) for a Camera. The
    rotation R maps view axes to world: forward = R@[0,0,-1], up = R@[0,1,0], the
    inverse of neuroglancer's projectionOrientation convention (see ng_camera)."""
    center = np.array(cam.look_at_nm, dtype=float)
    eye = np.array(cam.position_nm, dtype=float)
    d = center - eye
    dist = float(np.linalg.norm(d)) or 1.0
    fwd = d / dist
    up = np.array(cam.up, dtype=float)
    z = -fwd
    y = up - np.dot(up, z) * z
    n = np.linalg.norm(y)
    y = y / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    return center, Rotation.from_matrix(np.column_stack([x, y, z])), dist


def _interp_camera(a, b, t):
    """Neuroglancer-style camera interpolation -> (position_nm, look_at_nm, up, fov)."""
    ca, ra, da = _cam_basis(a)
    cb, rb, db = _cam_basis(b)
    center = ca + (cb - ca) * t
    rot = Slerp([0.0, 1.0], Rotation.concatenate([ra, rb]))([t])[0]  # slerp orientation
    dist = da * (db / da) ** t                                       # exponential zoom
    fov = a.fov_deg + (b.fov_deg - a.fov_deg) * t
    fwd = rot.apply([0.0, 0.0, -1.0])
    up = rot.apply([0.0, 1.0, 0.0])
    eye = center - fwd * dist
    return eye.tolist(), center.tolist(), up.tolist(), fov


@dataclass
class FrameSlice:
    em_name: str
    axis: str
    position_nm: float
    scale_level: int | None
    opacity: float


@dataclass
class FrameMesh:
    mesh_name: str
    segment_ids: list[int]
    color: list[float]
    opacity: float
    render_3d: bool = True
    color_seed: int = 0
    default_color: list[float] | None = None
    segment_colors: dict = field(default_factory=dict)
    saturation: float = 1.0     # NG layer saturation (0 = grayscale)
    object_alpha: float = 1.0   # NG "Opacity (3d)"
    silhouette: float = 0.0     # NG "Silhouette (3d)"


@dataclass
class FrameAnnotation:
    name: str
    color: list[float]
    opacity: float
    points: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    ellipsoids: list = field(default_factory=list)
    point_radius_nm: float = 80.0
    line_radius_nm: float = 40.0


@dataclass
class FrameState:
    position_nm: list[float]
    look_at_nm: list[float]
    fov_deg: float
    up: list[float]
    slices: list[FrameSlice] = field(default_factory=list)
    meshes: list[FrameMesh] = field(default_factory=list)
    annotations: list[FrameAnnotation] = field(default_factory=list)


def _state_at(a: Keyframe, b: Keyframe, t: float) -> FrameState:
    pos, look_at, up, fov = _interp_camera(a.camera, b.camera, t)
    fs = FrameState(position_nm=pos, look_at_nm=look_at, fov_deg=fov, up=up)
    # slices matched by (em_name, axis)
    a_sl = {(s.em_name, s.axis): s for s in a.slices}
    b_sl = {(s.em_name, s.axis): s for s in b.slices}
    for key in dict.fromkeys(list(a_sl) + list(b_sl)):
        sa, sb = a_sl.get(key), b_sl.get(key)
        if sa and sb:
            fs.slices.append(FrameSlice(
                key[0], key[1],
                sa.position_nm + (sb.position_nm - sa.position_nm) * t,
                sb.scale_level,
                (sa.opacity if sa.visible else 0.0) * (1 - t) + (sb.opacity if sb.visible else 0.0) * t,
            ))
        else:  # appearing or disappearing -> fade
            s = sa or sb
            base = (s.opacity if s.visible else 0.0)
            op = base * (1 - t) if sa else base * t
            fs.slices.append(FrameSlice(key[0], key[1], s.position_nm, s.scale_level, op))
    # meshes matched by (layer name + exact segment set): a different segment set
    # is different geometry, so it cross-fades (old set fades out, new fades in)
    def mkey(m):
        return (m.mesh_name, tuple(sorted(m.segment_ids)))

    a_m = {mkey(m): m for m in a.meshes}
    b_m = {mkey(m): m for m in b.meshes}
    for key in dict.fromkeys(list(a_m) + list(b_m)):
        ma, mb = a_m.get(key), b_m.get(key)
        name, ids = key[0], list(key[1])
        src = mb or ma  # color config from the target keyframe of the transition
        cc = dict(color_seed=src.color_seed, default_color=src.default_color,
                  segment_colors=src.segment_colors, saturation=src.saturation)
        if ma and mb:
            op = (ma.opacity if ma.visible else 0.0) * (1 - t) + (mb.opacity if mb.visible else 0.0) * t
            oa = ma.object_alpha * (1 - t) + mb.object_alpha * t       # Opacity (3d) lerps
            si = ma.silhouette * (1 - t) + mb.silhouette * t           # Silhouette (3d) lerps
            fs.meshes.append(FrameMesh(name, ids, mb.color, op, mb.render_3d,
                                       object_alpha=oa, silhouette=si, **cc))
        else:
            m = ma or mb
            base = (m.opacity if m.visible else 0.0)
            op = base * (1 - t) if ma else base * t   # ma-only fades out; mb-only fades in
            fs.meshes.append(FrameMesh(name, ids, m.color, op, m.render_3d,
                                       object_alpha=m.object_alpha, silhouette=m.silhouette, **cc))
    # annotations matched by layer name; geometry is identical frame-to-frame, so
    # only opacity fades (appearing/disappearing layers fade in/out).
    a_an = {an.name: an for an in a.annotations}
    b_an = {an.name: an for an in b.annotations}
    for key in dict.fromkeys(list(a_an) + list(b_an)):
        aa, ab = a_an.get(key), b_an.get(key)
        src = ab or aa  # geometry/color from the transition's target keyframe
        if aa and ab:
            op = (aa.opacity if aa.visible else 0.0) * (1 - t) + (ab.opacity if ab.visible else 0.0) * t
        else:
            base = (src.opacity if src.visible else 0.0)
            op = base * (1 - t) if aa else base * t
        fs.annotations.append(FrameAnnotation(
            src.name, src.color, op, src.points, src.lines, src.boxes, src.ellipsoids,
            src.point_radius_nm, src.line_radius_nm))
    return fs


def build_frames(keyframes: list[Keyframe], fps: int) -> list[FrameState]:
    """Flatten keyframes to per-frame states, matching neuroglancer's video_tool
    exactly: for each transition i->i+1 emit round(duration*fps) frames at t = k/n
    for k in [0, n) (so keyframe i is shown at the START of its outgoing transition),
    then one final frame holding the last keyframe. `duration_in_s` is stored on the
    DESTINATION keyframe (the transition into it). A duration of 0 emits no transition
    frames (an instant cut)."""
    if not keyframes:
        return []
    frames: list[FrameState] = []
    for i in range(len(keyframes) - 1):
        a, b = keyframes[i], keyframes[i + 1]
        n = 0 if b.duration_in_s <= 0 else max(1, int(round(b.duration_in_s * fps)))
        for k in range(n):
            frames.append(_state_at(a, b, _ease(k / n, b.easing)))
    frames.append(_state_at(keyframes[-1], keyframes[-1], 0.0))  # final keyframe, 1 frame
    return frames
