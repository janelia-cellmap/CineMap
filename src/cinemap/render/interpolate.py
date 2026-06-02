"""Interpolate a keyframe list into a flat list of per-frame scene states.

Camera is stored as position + look_at (+ fov), so interpolation is a lerp of
both points (no quaternion bookkeeping needed for v1). Slices are matched by
(em_name, axis) and their position + opacity lerped; meshes matched by name and
their opacity lerped. Easing is applied to the [0,1] transition parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Keyframe


def _ease(t: float, mode: str) -> float:
    if mode == "ease-in-out":
        return t * t * (3 - 2 * t)  # smoothstep
    return t


def _lerp(a, b, t):
    return [ai + (bi - ai) * t for ai, bi in zip(a, b)]


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


@dataclass
class FrameState:
    position_nm: list[float]
    look_at_nm: list[float]
    fov_deg: float
    up: list[float]
    slices: list[FrameSlice] = field(default_factory=list)
    meshes: list[FrameMesh] = field(default_factory=list)


def _state_at(a: Keyframe, b: Keyframe, t: float) -> FrameState:
    fs = FrameState(
        position_nm=_lerp(a.camera.position_nm, b.camera.position_nm, t),
        look_at_nm=_lerp(a.camera.look_at_nm, b.camera.look_at_nm, t),
        fov_deg=a.camera.fov_deg + (b.camera.fov_deg - a.camera.fov_deg) * t,
        up=a.camera.up,
    )
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
        if ma and mb:
            op = (ma.opacity if ma.visible else 0.0) * (1 - t) + (mb.opacity if mb.visible else 0.0) * t
            fs.meshes.append(FrameMesh(name, ids, mb.color, op))
        else:
            m = ma or mb
            base = (m.opacity if m.visible else 0.0)
            op = base * (1 - t) if ma else base * t   # ma-only fades out; mb-only fades in
            fs.meshes.append(FrameMesh(name, ids, m.color, op))
    return fs


def build_frames(keyframes: list[Keyframe], fps: int) -> list[FrameState]:
    """Flatten keyframes to per-frame states. One frame for a lone keyframe."""
    if not keyframes:
        return []
    if len(keyframes) == 1:
        return [_state_at(keyframes[0], keyframes[0], 0.0)]
    frames: list[FrameState] = [_state_at(keyframes[0], keyframes[0], 0.0)]
    for i in range(1, len(keyframes)):
        a, b = keyframes[i - 1], keyframes[i]
        n = max(1, int(round(b.duration_in_s * fps)))
        for k in range(1, n + 1):
            t = _ease(k / n, b.easing)
            frames.append(_state_at(a, b, t))
    return frames
