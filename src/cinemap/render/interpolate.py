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
    """Neuroglancer-style camera interpolation.

    Returns (position_nm, look_at_nm, up, fov, projection, ortho_scale_nm). 2D
    cross-section keyframes use Blender's orthographic camera; when either endpoint
    is orthographic the transition carries an interpolated orthographic scale so
    2D-to-2D pans/zooms behave like Neuroglancer movie states.
    """
    ca, ra, da = _cam_basis(a)
    cb, rb, db = _cam_basis(b)
    center = ca + (cb - ca) * t
    rot = Slerp([0.0, 1.0], Rotation.concatenate([ra, rb]))([t])[0]  # slerp orientation
    dist = da * (db / da) ** t                                       # exponential zoom
    fov = a.fov_deg + (b.fov_deg - a.fov_deg) * t
    fwd = rot.apply([0.0, 0.0, -1.0])
    up = rot.apply([0.0, 1.0, 0.0])
    eye = center - fwd * dist
    aproj = getattr(a, "projection", "PERSP")
    bproj = getattr(b, "projection", "PERSP")
    projection = "ORTHO" if (aproj == "ORTHO" or bproj == "ORTHO") else "PERSP"
    ortho = None
    if projection == "ORTHO":
        av = getattr(a, "ortho_scale_nm", None)
        bv = getattr(b, "ortho_scale_nm", None)
        if av is None:
            av = 2.0 * da * np.tan(np.radians(a.fov_deg) / 2.0)
        if bv is None:
            bv = 2.0 * db * np.tan(np.radians(b.fov_deg) / 2.0)
        av, bv = max(float(av), 1.0), max(float(bv), 1.0)
        ortho = av * (bv / av) ** t
    return eye.tolist(), center.tolist(), up.tolist(), fov, projection, ortho


@dataclass
class FrameSlice:
    em_name: str
    axis: str
    position_nm: float
    scale_level: int | None
    opacity: float
    normal: list[float] | None = None   # oblique plane normal (xyz); None = axis-aligned
    contrast_limits: list[float] | None = None


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
    saturation: float = 1.0     # NG layer saturation (0 = white, 1 = full color)
    object_alpha: float = 1.0   # NG "Opacity (3d)"
    silhouette: float = 0.0     # NG "Silhouette (3d)"
    clip: dict | None = None    # cutaway plane {axis, position_nm, side} or None
    metallic: float | None = None    # per-frame material override (None = leave look base)
    roughness: float | None = None
    color_mix: float | None = None   # 0..1 progress through this mesh's OWN appear/disappear
                                      # transition (None = steady, not mid-transition). Lets the
                                      # render worker detect several same-source layers fading
                                      # into/out of each other (a recolor, not independent objects)
                                      # and merge them into one mesh with a smooth color blend
                                      # instead of stacking coincident, independently-fading copies.
    transition_role: str | None = None  # "from" (disappearing) | "to" (appearing) | None (steady)


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
    projection: str = "PERSP"
    ortho_scale_nm: float | None = None
    slices: list[FrameSlice] = field(default_factory=list)
    meshes: list[FrameMesh] = field(default_factory=list)
    annotations: list[FrameAnnotation] = field(default_factory=list)
    fade_alpha: float = 0.0
    background: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # NG 3D bg (linear)


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
    pos, look_at, up, fov, projection, ortho = _interp_camera(a.camera, b.camera, t)
    fs = FrameState(
        position_nm=pos,
        look_at_nm=look_at,
        fov_deg=fov,
        up=up,
        projection=projection,
        ortho_scale_nm=ortho,
    )
    # Background crossfades between the two keyframes (camera transition `t`), so a
    # white->black NG change blends across the move like every other appearance value.
    ba = getattr(a.lighting, "background", None) or [0.0, 0.0, 0.0]
    bb = getattr(b.lighting, "background", None) or [0.0, 0.0, 0.0]
    fs.background = [float(ba[i]) * (1 - t) + float(bb[i]) * t for i in range(3)]
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
                contrast_limits=(sb.contrast_limits if (t >= 0.5 or target_layer)
                                 else sa.contrast_limits),
            ))
        else:  # appearing or disappearing
            s = sa or sb
            base = (s.opacity if s.visible else 0.0)
            op = _appear_opacity(base, bool(sa), t, layer_transition, lt, layer_transition_at)
            fs.slices.append(FrameSlice(
                key[0], key[1], s.position_nm, s.scale_level, op,
                normal=s.normal,
                contrast_limits=s.contrast_limits,
            ))
    # Meshes are matched by layer name, then split by segment membership. Shared
    # segment ids stay continuously visible, removed ids fade out, and added ids fade
    # in. Treating the whole selected-id set as one identity made common segments blink
    # when a layer changed from "many ids" to "one id" across a transition.
    a_m = {m.mesh_name: m for m in a.meshes}
    b_m = {m.mesh_name: m for m in b.meshes}

    def append_mesh(src, ids, op, oa, si, clip, mtl=None, rgh=None,
                    color_mix=None, transition_role=None):
        ids = sorted(int(x) for x in ids)
        if not ids:
            return
        cc = dict(color_seed=src.color_seed, default_color=src.default_color,
                  segment_colors=src.segment_colors, saturation=src.saturation)
        fs.meshes.append(FrameMesh(
            src.mesh_name, ids, src.color, op, src.render_3d,
            object_alpha=oa, silhouette=si, clip=clip,
            metallic=mtl, roughness=rgh, color_mix=color_mix,
            transition_role=transition_role, **cc,
        ))

    for name in dict.fromkeys(list(a_m) + list(b_m)):
        ma, mb = a_m.get(name), b_m.get(name)
        if ma and mb and layer_transition != "cut":
            a_ids, b_ids = set(ma.segment_ids), set(mb.segment_ids)
            common = a_ids & b_ids
            removed = a_ids - b_ids
            added = b_ids - a_ids
            op = _blend_value(ma.opacity if ma.visible else 0.0,
                              mb.opacity if mb.visible else 0.0, t, layer_transition,
                              lt, layer_transition_at)
            oa = _blend_value(ma.object_alpha, mb.object_alpha, t, layer_transition,
                              lt, layer_transition_at)
            si = _blend_value(ma.silhouette, mb.silhouette, t, layer_transition,
                              lt, layer_transition_at)
            clip = _lerp_clip(ma.clip, mb.clip, t)
            mtl = _mat_lerp(getattr(ma, "metallic", None), getattr(mb, "metallic", None), t, 0.0)
            rgh = _mat_lerp(getattr(ma, "roughness", None), getattr(mb, "roughness", None), t, 0.5)
            append_mesh(mb, common, op, oa, si, clip, mtl, rgh)
            append_mesh(ma, removed, (ma.opacity if ma.visible else 0.0) * (1 - t),
                        ma.object_alpha, ma.silhouette, _clip_dict(ma.clip),
                        getattr(ma, "metallic", None), getattr(ma, "roughness", None),
                        color_mix=t, transition_role="from")
            append_mesh(mb, added, (mb.opacity if mb.visible else 0.0) * t,
                        mb.object_alpha, mb.silhouette, _clip_dict(mb.clip),
                        getattr(mb, "metallic", None), getattr(mb, "roughness", None),
                        color_mix=t, transition_role="to")
            continue

        src = ma if (layer_transition == "cut" and not target_layer and ma) else (mb or ma)
        if ma and mb:
            op = _blend_value(ma.opacity if ma.visible else 0.0,
                              mb.opacity if mb.visible else 0.0, t, layer_transition,
                              lt, layer_transition_at)
            oa = _blend_value(ma.object_alpha, mb.object_alpha, t, layer_transition,
                              lt, layer_transition_at)
            si = _blend_value(ma.silhouette, mb.silhouette, t, layer_transition,
                              lt, layer_transition_at)
            clip = _clip_dict(mb.clip) if target_layer else _clip_dict(ma.clip)
            mtl_t = 1.0 if target_layer else 0.0
            mtl = _mat_lerp(getattr(ma, "metallic", None), getattr(mb, "metallic", None), mtl_t, 0.0)
            rgh = _mat_lerp(getattr(ma, "roughness", None), getattr(mb, "roughness", None), mtl_t, 0.5)
            append_mesh(src, src.segment_ids, op, oa, si, clip, mtl, rgh)
        else:
            m = ma or mb
            base = (m.opacity if m.visible else 0.0)
            op = _appear_opacity(base, bool(ma), t, layer_transition, lt, layer_transition_at)
            append_mesh(m, m.segment_ids, op, m.object_alpha, m.silhouette, _clip_dict(m.clip),
                        getattr(m, "metallic", None), getattr(m, "roughness", None),
                        color_mix=lt, transition_role=("from" if ma else "to"))
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


def _seg_ease(a: Keyframe, b: Keyframe, i: int, n_trans: int, smooth_ends: bool) -> str:
    """Easing for the transition a->b, accounting for fly-through waypoints.

    A keyframe is a STOP (camera rests there) unless flagged `waypoint`. The arriving
    end decelerates (ease-out) only into a stop; the departing end accelerates
    (ease-in) only out of a stop — so a waypoint is entered/left at speed, giving
    continuous (non-stopping) motion through it:
      stop->stop = ease-in-out, stop->wp = ease-in, wp->stop = ease-out, wp->wp = linear.
    When neither endpoint is a waypoint, behavior is unchanged (the director's
    smooth_ends curve, else the keyframe's own easing)."""
    a_wp = bool(getattr(a, "waypoint", False))
    b_wp = bool(getattr(b, "waypoint", False))
    if a_wp or b_wp:
        if not a_wp and not b_wp:
            return "ease-in-out"
        if not a_wp:
            return "ease-in"
        if not b_wp:
            return "ease-out"
        return "linear"
    if smooth_ends:
        return ("ease-in-out" if n_trans == 1 else
                "ease-in" if i == 0 else "ease-out" if i == n_trans - 1 else "linear")
    return b.easing


def _kf_hold_s(kf: Keyframe) -> float:
    """Dwell seconds on a keyframe — zero for a fly-through waypoint."""
    if bool(getattr(kf, "waypoint", False)):
        return 0.0
    return max(0.0, float(getattr(kf, "hold_in_s", 0.0) or 0.0))


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
        hold = _kf_hold_s(a)
        if t < cum + hold:
            return _state_at(a, a, 0.0)
        cum += hold
        dur = max(0.0, float(getattr(b, "duration_in_s", 0.0) or 0.0))
        if dur > 0.0 and t < cum + dur:
            ease = _seg_ease(a, b, i, n_trans, smooth_ends)
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
        hold_n = int(round(_kf_hold_s(a) * fps))
        for _ in range(hold_n):
            frames.append(_state_at(a, a, 0.0))
        n = 0 if b.duration_in_s <= 0 else max(1, int(round(b.duration_in_s * fps)))
        ease = _seg_ease(a, b, i, n_trans, smooth_ends)
        style = getattr(b, "transition", "glide")
        layer_transition = getattr(b, "layer_transition", "fade")
        layer_transition_at = getattr(b, "layer_transition_at", 1.0)
        for k in range(n):
            frames.append(_transition_state(a, b, k / n, style, ease, layer_transition,
                                            layer_transition_at))
    frames.append(_state_at(keyframes[-1], keyframes[-1], 0.0))  # final keyframe, 1 frame
    final_hold_n = int(round(_kf_hold_s(keyframes[-1]) * fps))
    for _ in range(final_hold_n):
        frames.append(_state_at(keyframes[-1], keyframes[-1], 0.0))
    return frames
