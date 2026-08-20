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
        return t * t * (3 - 2 * t)              # smoothstep (slow in AND out)
    if mode == "ease-in":
        return t * t                            # slow start, full speed at end
    if mode == "ease-out":
        return t * (2 - t)                      # full speed start, slow stop
    return t                                    # linear (constant velocity)


def _mat_lerp(av, bv, t: float, default: float):
    """Interpolate a per-keyframe material knob (metallic/roughness). Returns None when
    NEITHER keyframe sets it (so the render leaves the global look/director base intact);
    otherwise treats an unset side as `default` and lerps — a clean ramp into reflective."""
    if av is None and bv is None:
        return None
    a = default if av is None else av
    b = default if bv is None else bv
    return a * (1 - t) + b * t


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
    normal: list[float] | None = None   # oblique plane normal (xyz); None = axis-aligned
    # Neuroglancer's intensity mapping for this layer, carried through interpolation so
    # the worker can bake the same contrast the viewer shows. These do NOT blend: a
    # contrast window is a lookup, and cross-fading two windows would show a frame of
    # contrast that neither keyframe asked for. They snap with the rest of the layer
    # appearance instead (see the `target_layer` switch below).
    shader: str = ""
    shader_controls: dict = field(default_factory=dict)


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
    clip: dict | None = None    # cutaway plane {axis, position_nm, side} or None
    metallic: float | None = None    # per-frame material override (None = leave look base)
    roughness: float | None = None


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
    fade_alpha: float = 0.0


def _clip_dict(c) -> dict | None:
    """A keyframe ClipPlane -> the render-side clip dict, or None when disabled."""
    if c is None or not getattr(c, "enabled", False):
        return None
    return {"axis": c.axis, "position_nm": c.position_nm, "side": c.side, "normal": c.normal}


def _lerp_clip(ca, cb, t: float) -> dict | None:
    """Interpolate a cutaway between two keyframes. When both clip the same axis/side/
    normal the plane scrolls (offset lerps); otherwise snap to the target's clip so the
    cut doesn't jump through an interpolated mismatch."""
    da, db = _clip_dict(ca), _clip_dict(cb)
    if (da and db and da["axis"] == db["axis"] and da["side"] == db["side"]
            and da["normal"] == db["normal"]):
        return {"axis": db["axis"], "side": db["side"], "normal": db["normal"],
                "position_nm": da["position_nm"] * (1 - t) + db["position_nm"] * t}
    return db if t >= 0.5 else da


def _layer_cut_at(v: float | None) -> float:
    return max(0.0, min(1.0, float(1.0 if v is None else v)))


def _target_layer_active(layer_transition: str, layer_t: float, layer_transition_at: float) -> bool:
    return layer_transition == "cut" and layer_t >= _layer_cut_at(layer_transition_at)


def _blend_value(av: float, bv: float, t: float, layer_transition: str,
                 layer_t: float, layer_transition_at: float) -> float:
    if layer_transition == "cut":
        return bv if _target_layer_active(layer_transition, layer_t, layer_transition_at) else av
    return av * (1 - t) + bv * t


def _appear_opacity(base: float, has_a: bool, t: float, layer_transition: str,
                    layer_t: float, layer_transition_at: float) -> float:
    if layer_transition == "cut":
        target = _target_layer_active(layer_transition, layer_t, layer_transition_at)
        return base if (has_a and not target) or ((not has_a) and target) else 0.0
    return base * (1 - t) if has_a else base * t


def _state_at(a: Keyframe, b: Keyframe, t: float,
              layer_transition: str = "fade", layer_t: float | None = None,
              layer_transition_at: float = 1.0) -> FrameState:
    pos, look_at, up, fov = _interp_camera(a.camera, b.camera, t)
    fs = FrameState(position_nm=pos, look_at_nm=look_at, fov_deg=fov, up=up)
    lt = t if layer_t is None else max(0.0, min(1.0, float(layer_t)))
    target_layer = _target_layer_active(layer_transition, lt, layer_transition_at)
    # slices matched by (em_name, axis)
    a_sl = {(s.em_name, s.axis): s for s in a.slices}
    b_sl = {(s.em_name, s.axis): s for s in b.slices}
    for key in dict.fromkeys(list(a_sl) + list(b_sl)):
        sa, sb = a_sl.get(key), b_sl.get(key)
        if sa and sb:
            # plane offset scrolls; normal snaps to target if it differs (same on a scan)
            same_n = sa.normal == sb.normal
            pos_nm = sb.position_nm if target_layer else (
                sa.position_nm + (sb.position_nm - sa.position_nm) * t)
            fs.slices.append(FrameSlice(
                key[0], key[1],
                pos_nm,
                sb.scale_level if target_layer else sa.scale_level,
                _blend_value(sa.opacity if sa.visible else 0.0,
                             sb.opacity if sb.visible else 0.0, t, layer_transition,
                             lt, layer_transition_at),
                normal=(sb.normal if (same_n or t >= 0.5 or target_layer) else sa.normal),
                # snap with the rest of the layer appearance rather than blending
                shader=(sb.shader if target_layer else sa.shader),
                shader_controls=dict((sb if target_layer else sa).shader_controls),
            ))
        else:  # appearing or disappearing
            s = sa or sb
            base = (s.opacity if s.visible else 0.0)
            op = _appear_opacity(base, bool(sa), t, layer_transition, lt, layer_transition_at)
            fs.slices.append(FrameSlice(key[0], key[1], s.position_nm, s.scale_level, op,
                                        normal=s.normal, shader=s.shader,
                                        shader_controls=dict(s.shader_controls)))
    # Meshes are matched by (layer name + exact segment set). A different segment set
    # is different geometry; layer_transition decides whether that change cross-fades
    # or cuts hard at the destination keyframe.
    def mkey(m):
        return (m.mesh_name, tuple(sorted(m.segment_ids)))

    a_m = {mkey(m): m for m in a.meshes}
    b_m = {mkey(m): m for m in b.meshes}
    for key in dict.fromkeys(list(a_m) + list(b_m)):
        ma, mb = a_m.get(key), b_m.get(key)
        name, ids = key[0], list(key[1])
        src = ma if (layer_transition == "cut" and not target_layer and ma) else (mb or ma)
        cc = dict(color_seed=src.color_seed, default_color=src.default_color,
                  segment_colors=src.segment_colors, saturation=src.saturation)
        if ma and mb:
            op = _blend_value(ma.opacity if ma.visible else 0.0,
                              mb.opacity if mb.visible else 0.0, t, layer_transition,
                              lt, layer_transition_at)
            oa = _blend_value(ma.object_alpha, mb.object_alpha, t, layer_transition,
                              lt, layer_transition_at)
            si = _blend_value(ma.silhouette, mb.silhouette, t, layer_transition,
                              lt, layer_transition_at)
            clip = (_clip_dict(mb.clip) if target_layer else _clip_dict(ma.clip)) \
                if layer_transition == "cut" else _lerp_clip(ma.clip, mb.clip, t)
            # material lerps too -> a layer can turn reflective over a transition. Only
            # emitted when a keyframe actually sets it, else None (leave the look base).
            mtl_t = 1.0 if target_layer else (0.0 if layer_transition == "cut" else t)
            mtl = _mat_lerp(getattr(ma, "metallic", None), getattr(mb, "metallic", None), mtl_t, 0.0)
            rgh = _mat_lerp(getattr(ma, "roughness", None), getattr(mb, "roughness", None), mtl_t, 0.5)
            fs.meshes.append(FrameMesh(name, ids, src.color, op, src.render_3d,
                                       object_alpha=oa, silhouette=si, clip=clip,
                                       metallic=mtl, roughness=rgh, **cc))
        else:
            m = ma or mb
            base = (m.opacity if m.visible else 0.0)
            op = _appear_opacity(base, bool(ma), t, layer_transition, lt, layer_transition_at)
            fs.meshes.append(FrameMesh(name, ids, m.color, op, m.render_3d,
                                       object_alpha=m.object_alpha, silhouette=m.silhouette,
                                       clip=_clip_dict(m.clip),
                                       metallic=getattr(m, "metallic", None),
                                       roughness=getattr(m, "roughness", None), **cc))
    # Annotations are matched by layer name. Their appearance follows layer_transition.
    a_an = {an.name: an for an in a.annotations}
    b_an = {an.name: an for an in b.annotations}
    for key in dict.fromkeys(list(a_an) + list(b_an)):
        aa, ab = a_an.get(key), b_an.get(key)
        src = aa if (layer_transition == "cut" and not target_layer and aa) else (ab or aa)
        if aa and ab:
            op = _blend_value(aa.opacity if aa.visible else 0.0,
                              ab.opacity if ab.visible else 0.0, t, layer_transition,
                              lt, layer_transition_at)
        else:
            base = (src.opacity if src.visible else 0.0)
            op = _appear_opacity(base, bool(aa), t, layer_transition, lt, layer_transition_at)
        fs.annotations.append(FrameAnnotation(
            src.name, src.color, op, src.points, src.lines, src.boxes, src.ellipsoids,
            src.point_radius_nm, src.line_radius_nm))
    return fs


def _with_fade(fs: FrameState, alpha: float) -> FrameState:
    fs.fade_alpha = max(0.0, min(1.0, float(alpha)))
    return fs


def _transition_state(a: Keyframe, b: Keyframe, t: float, style: str,
                      easing: str, layer_transition: str = "fade",
                      layer_transition_at: float = 1.0) -> FrameState:
    """Frame for one transition, before any global sweep overlays are evaluated."""
    if style == "cut":
        # The move has duration, but the scene itself does not interpolate: hold the
        # source pose until the next keyframe's arrival time, then the following frame
        # starts from the destination pose.
        return _state_at(a, a, 0.0)
    if style == "fade":
        # Fade through black. This avoids mixing two full 3D renders in post while still
        # giving an editorial fade between poses/states.
        if t < 0.5:
            return _with_fade(_state_at(a, a, 0.0), t * 2.0)
        return _with_fade(_state_at(b, b, 0.0), (1.0 - t) * 2.0)
    return _state_at(a, b, _ease(t, easing), layer_transition=layer_transition,
                     layer_t=t, layer_transition_at=layer_transition_at)


def state_at_time(keyframes: list[Keyframe], t: float,
                  smooth_ends: bool = False) -> FrameState | None:
    """Evaluate the timeline at global time `t` seconds, including holds and transition
    styles. Used by sweep/snapshot previews so their timing matches full renders."""
    if not keyframes:
        return None
    t = max(0.0, float(t))
    n_trans = len(keyframes) - 1
    cum = 0.0
    for i in range(n_trans):
        a, b = keyframes[i], keyframes[i + 1]
        hold = max(0.0, float(getattr(a, "hold_in_s", 0.0) or 0.0))
        if t < cum + hold:
            return _state_at(a, a, 0.0)
        cum += hold
        dur = max(0.0, float(getattr(b, "duration_in_s", 0.0) or 0.0))
        if dur > 0.0 and t < cum + dur:
            if smooth_ends:
                ease = ("ease-in-out" if n_trans == 1 else
                        "ease-in" if i == 0 else "ease-out" if i == n_trans - 1 else "linear")
            else:
                ease = b.easing
            local = (t - cum) / dur
            return _transition_state(a, b, local, getattr(b, "transition", "glide"), ease,
                                     getattr(b, "layer_transition", "fade"),
                                     getattr(b, "layer_transition_at", 1.0))
        cum += dur
    return _state_at(keyframes[-1], keyframes[-1], 0.0)


def build_frames(keyframes: list[Keyframe], fps: int,
                 smooth_ends: bool = False) -> list[FrameState]:
    """Flatten keyframes to per-frame states, matching neuroglancer's video_tool
    for normal glide transitions: for each transition i->i+1 emit round(duration*fps)
    frames at t = k/n for k in [0, n). `duration_in_s` is stored on the DESTINATION
    keyframe (the transition into it). A duration of 0 emits no transition frames.
    `transition` controls the camera/edit transition, while `layer_transition` controls
    whether layer appearance changes fade during the move or cut at a chosen time.
    `hold_in_s` on a keyframe adds dwell time before its outgoing transition.

    `smooth_ends` (the director's cinematic motion) eases into the FIRST transition
    and out of the LAST, but keeps the middle LINEAR — so the camera glides to a
    gentle start/stop without pausing at every intermediate keyframe (per-transition
    ease-in-out would decelerate to a near-stop at each one). Same frame count and
    total duration as video_tool either way."""
    if not keyframes:
        return []
    frames: list[FrameState] = []
    n_trans = len(keyframes) - 1
    for i in range(n_trans):
        a, b = keyframes[i], keyframes[i + 1]
        hold_n = int(round(max(0.0, float(getattr(a, "hold_in_s", 0.0) or 0.0)) * fps))
        for _ in range(hold_n):
            frames.append(_state_at(a, a, 0.0))
        n = 0 if b.duration_in_s <= 0 else max(1, int(round(b.duration_in_s * fps)))
        if smooth_ends:
            ease = ("ease-in-out" if n_trans == 1 else
                    "ease-in" if i == 0 else "ease-out" if i == n_trans - 1 else "linear")
        else:
            ease = b.easing
        style = getattr(b, "transition", "glide")
        layer_transition = getattr(b, "layer_transition", "fade")
        layer_transition_at = getattr(b, "layer_transition_at", 1.0)
        for k in range(n):
            frames.append(_transition_state(a, b, k / n, style, ease, layer_transition,
                                            layer_transition_at))
    frames.append(_state_at(keyframes[-1], keyframes[-1], 0.0))  # final keyframe, 1 frame
    final_hold_n = int(round(max(0.0, float(getattr(keyframes[-1], "hold_in_s", 0.0) or 0.0)) * fps))
    for _ in range(final_hold_n):
        frames.append(_state_at(keyframes[-1], keyframes[-1], 0.0))
    return frames
