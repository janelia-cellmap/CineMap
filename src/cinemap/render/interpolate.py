"""Interpolate a keyframe list into a flat list of per-frame scene states.

Camera interpolation matches neuroglancer's video_tool exactly: the look-at center
is interpolated linearly, the view ORIENTATION via spherical-linear interpolation
(slerp), and the zoom (eye->center distance) exponentially. So an orbit/rotation
(same center, changing orientation) actually arcs the camera around the target the
way neuroglancer does, instead of cutting a straight chord. Meshes/annotations are
matched by identity and their opacity lerped.

EM cross-sections follow neuroglancer the same way (`Keyframe.em_transition`, default
"glide"): a cross-section TRAVELS between keyframes — the plane point lerps and the
normal slerps, so a tilted plane rotates about the camera target exactly as dragging
neuroglancer's crossSectionOrientation does — rather than the old cross-fade, which put
the source and destination planes on screen simultaneously whenever their dominant axis
differed. Planes are paired across keyframes by layer + closest orientation, so a
multi-panel layout's two or three cross-sections each glide to their counterpart.
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
    # Orthographic snaps (it's a shot-level choice, not something to blend mid-transition);
    # ortho_scale_nm lerps like fov_deg, falling back to the other side when only one is set.
    orthographic = a.orthographic if t < 0.5 else b.orthographic
    osa = a.ortho_scale_nm if a.ortho_scale_nm is not None else (b.ortho_scale_nm or 0.0)
    osb = b.ortho_scale_nm if b.ortho_scale_nm is not None else (a.ortho_scale_nm or 0.0)
    ortho_scale_nm = osa + (osb - osa) * t
    fwd = rot.apply([0.0, 0.0, -1.0])
    up = rot.apply([0.0, 1.0, 0.0])
    eye = center - fwd * dist
    return eye.tolist(), center.tolist(), up.tolist(), fov, orthographic, ortho_scale_nm


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
    # Identity of this cross-section across frames, passed through to the frame spec.
    # A gliding plane changes `axis` mid-transition (once its normal rotates past 45deg),
    # so em_name+axis — the old identity — is NOT stable; None falls back to that pair.
    slot: str | None = None


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
    slice_opacity: float | None = None   # NG 2D opacity on the EM cross-section
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
    # The annotation shader and the per-annotation properties it reads. These MUST be
    # carried here: a field the frame state drops is a field the renderer never sees,
    # however carefully it was captured.
    shader: str = ""
    shader_controls: dict = field(default_factory=dict)
    point_props: dict = field(default_factory=dict)
    line_props: dict = field(default_factory=dict)
    box_props: dict = field(default_factory=dict)
    ellipsoid_props: dict = field(default_factory=dict)


@dataclass
class FrameState:
    position_nm: list[float]
    look_at_nm: list[float]
    fov_deg: float
    up: list[float]
    orthographic: bool = False
    ortho_scale_nm: float = 0.0
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


def _an_transition(an, layer_transition: str, layer_transition_at: float) -> tuple[str, float]:
    """An annotation layer may override the keyframe's layer_transition (see
    AnnotationInstance.layer_transition). An override cut defaults to switching at the
    START of the move; an inherited one keeps the keyframe's own cut point."""
    lt = getattr(an, "layer_transition", None) or layer_transition
    at = getattr(an, "layer_transition_at", None)
    if at is None:
        at = layer_transition_at if lt == layer_transition else 0.0
    return lt, _layer_cut_at(at)


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


def plane_normal(s) -> np.ndarray:
    """A slice's unit plane normal — its oblique `normal` if set, else the axis it's
    perpendicular to. Gives axis-aligned and oblique planes ONE representation, so a
    scan that tilts off-axis interpolates instead of being treated as two planes."""
    n = getattr(s, "normal", None)
    if n:
        v = np.array(n, dtype=float)
        m = float(np.linalg.norm(v))
        if m > 1e-9:
            return v / m
    e = np.zeros(3)
    e["xyz".index(s.axis)] = 1.0
    return e


def _plane_anchor(look_at, n: np.ndarray, offset: float) -> np.ndarray:
    """The point on the plane closest to that keyframe's camera target.

    Interpolating a POINT ON THE PLANE (rather than the bare scalar offset) is what makes
    a rotating cross-section pivot around what the camera is looking at, exactly like
    neuroglancer — whose cross-section always passes through `position`, the same point
    the 3D camera orbits. Baked keyframes satisfy dot(look_at, n) == offset to within
    rounding, so this is that keyframe's own center."""
    p = np.array(look_at, dtype=float)
    return p + (offset - float(np.dot(p, n))) * n


def _slerp_dir(na: np.ndarray, nb: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc slerp between two plane normals — neuroglancer slerps its
    crossSectionOrientation, so a tilt interpolates at constant angular speed instead of
    the uneven sweep a lerp of the normal would give. A plane's normal sign is arbitrary
    (n and -n describe the same plane), so flip to the near hemisphere first: otherwise a
    view captured from the other side spins the plane a pointless 180 degrees."""
    if float(np.dot(na, nb)) < 0.0:
        nb = -nb
    d = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    if d > 1.0 - 1e-9:
        return na
    om = float(np.arccos(d))
    s = float(np.sin(om))
    v = (np.sin((1.0 - t) * om) / s) * na + (np.sin(t * om) / s) * nb
    return v / (float(np.linalg.norm(v)) or 1.0)


def _glide_plane(sa, sb, look_a, look_b, t: float) -> tuple[str, float, list[float] | None]:
    """One MOVING cross-section between two keyframe planes -> (axis, offset_nm, normal).

    Reduces exactly to the old scalar-offset lerp when both planes share a normal (a
    straight depth scan), and to a rotation about the camera target when they don't."""
    na, nb = plane_normal(sa), plane_normal(sb)
    # anchors first: they're sign-invariant, unlike the (normal, offset) pair
    anchor = (_plane_anchor(look_a, na, sa.position_nm) * (1.0 - t)
              + _plane_anchor(look_b, nb, sb.position_nm) * t)
    n = _slerp_dir(na, nb, t)
    ai = int(np.argmax(np.abs(n)))
    if abs(n[ai]) >= 0.999:
        # axis-aligned: drop the normal, and give the offset in the +axis convention
        # (`position_nm` with no normal IS the axis coordinate) — a plane whose normal
        # slerped to -y would otherwise be read at +y, mirroring it across the origin.
        return "xyz"[ai], float(anchor[ai]), None
    return "xyz"[ai], float(np.dot(anchor, n)), [float(v) for v in n]


def _match_slices(a_slices, b_slices):
    """Pair the two keyframes' cross-sections for a glide -> [(sa|None, sb|None), …].

    Same layer, most-similar ORIENTATION first. A layer really can have several
    cross-sections at once (neuroglancer's 4-panel layout shows xy/xz/yz of one layer),
    so this can't collapse to one plane per layer — but each plane must pair with the
    panel it IS, not with a perpendicular one, or a scroll comes out as a 90 degree
    tumble. Unpaired planes are genuine appear/disappear."""
    cand = []
    for i, sa in enumerate(a_slices):
        na = plane_normal(sa)
        for j, sb in enumerate(b_slices):
            if sa.em_name != sb.em_name:
                continue
            # |dot|: a plane's normal sign is arbitrary, so anti-parallel == same panel
            cand.append((abs(float(np.dot(na, plane_normal(sb)))), i, j))
    cand.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    out = []
    for _, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((a_slices[i], b_slices[j]))
    out += [(s, None) for i, s in enumerate(a_slices) if i not in used_a]
    out += [(None, s) for j, s in enumerate(b_slices) if j not in used_b]
    return out


def _state_at(a: Keyframe, b: Keyframe, t: float,
              layer_transition: str = "fade", layer_t: float | None = None,
              layer_transition_at: float = 1.0,
              em_transition: str = "glide",
              em_transition_at: float | None = None) -> FrameState:
    pos, look_at, up, fov, orthographic, ortho_scale_nm = _interp_camera(a.camera, b.camera, t)
    fs = FrameState(position_nm=pos, look_at_nm=look_at, fov_deg=fov, up=up,
                    orthographic=orthographic, ortho_scale_nm=ortho_scale_nm)
    lt = t if layer_t is None else max(0.0, min(1.0, float(layer_t)))
    target_layer = _target_layer_active(layer_transition, lt, layer_transition_at)
    # EM cross-sections. Under "glide"/"cut" they're paired by layer + closest
    # ORIENTATION (see _match_slices) rather than by (layer, dominant axis) — a tilted
    # plane whose dominant axis changes between keyframes is still the same panel, and
    # keying on the axis split it into two planes that cross-faded past each other
    # instead of one plane sweeping. "fade" keeps the old (layer, axis) keying.
    em_glide = em_transition == "glide"
    em_cut = em_transition == "cut"
    if em_glide or em_cut:
        matched = _match_slices(list(a.slices), list(b.slices))
    else:
        a_sl = {(s.em_name, s.axis): s for s in a.slices}
        b_sl = {(s.em_name, s.axis): s for s in b.slices}
        matched = [(a_sl.get(k), b_sl.get(k))
                   for k in dict.fromkeys(list(a_sl) + list(b_sl))]
    # An EM cut has its own switch point: plane-to-plane it keeps the keyframe's timing,
    # but a plane that ARRIVES or LEAVES snaps at the start of the move by default — an EM
    # scan that ends before a rotation should be off the moment the camera moves, not
    # hang around as a ghost. An explicit em_transition_at always wins.
    em_at = layer_transition_at if em_transition_at is None else em_transition_at
    em_gone_at = _layer_cut_at(0.0 if em_transition_at is None else em_transition_at)
    em_snap = lt >= _layer_cut_at(em_at)
    n_slot: dict[str, int] = {}
    for sa, sb in matched:
        em_name = (sa or sb).em_name
        # one slot per (layer, panel), stable across the frames of this transition
        slot = f"{em_name}#{n_slot.get(em_name, 0)}" if (em_glide or em_cut) else None
        n_slot[em_name] = n_slot.get(em_name, 0) + 1
        if sa and sb:
            if em_cut:
                src = sb if em_snap else sa
                axis, pos_nm, nrm = src.axis, src.position_nm, src.normal
            elif em_glide:
                axis, pos_nm, nrm = _glide_plane(sa, sb, a.camera.look_at_nm,
                                                 b.camera.look_at_nm, t)
            else:  # "fade": the plane offset scrolls, the normal snaps at the midpoint
                axis = sa.axis
                pos_nm = sb.position_nm if target_layer else (
                    sa.position_nm + (sb.position_nm - sa.position_nm) * t)
                nrm = (sb.normal if (sa.normal == sb.normal or t >= 0.5 or target_layer)
                       else sa.normal)
            fs.slices.append(FrameSlice(
                em_name, axis,
                pos_nm,
                sb.scale_level if target_layer else sa.scale_level,
                _blend_value(sa.opacity if sa.visible else 0.0,
                             sb.opacity if sb.visible else 0.0, t, layer_transition,
                             lt, layer_transition_at),
                normal=nrm,
                # Snap to the destination at the halfway point, like `normal` above.
                # NOT keyed on target_layer alone: that is only ever true for a "cut"
                # transition, so under the default "fade" the destination's contrast
                # would never be used — a re-contrast on the last keyframe would never
                # render at all, since no later segment promotes it to the `a` side.
                shader=(sb.shader if (t >= 0.5 or target_layer) else sa.shader),
                shader_controls=dict(
                    (sb if (t >= 0.5 or target_layer) else sa).shader_controls),
                slot=slot,
            ))
        else:  # appearing or disappearing
            s = sa or sb
            base = (s.opacity if s.visible else 0.0)
            # em_transition owns the cross-section, including its arrival/departure: under
            # "cut" the plane is simply present or absent, never a ghost dissolving over
            # the move. ("glide" has nothing to glide from, so it falls back to the fade.)
            if em_cut:
                op = _appear_opacity(base, bool(sa), t, "cut", lt, em_gone_at)
            else:
                op = _appear_opacity(base, bool(sa), t, layer_transition, lt,
                                     layer_transition_at)
            fs.slices.append(FrameSlice(em_name, s.axis, s.position_nm, s.scale_level, op,
                                        normal=s.normal, shader=s.shader,
                                        shader_controls=dict(s.shader_controls),
                                        slot=slot))
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
            # the 2D cross-section opacity lerps like the 3D one (0.6 = renderer default
            # when a side leaves it unset), so fading labels off the slice is animatable
            so = _mat_lerp(getattr(ma, "slice_opacity", None),
                           getattr(mb, "slice_opacity", None), mtl_t, 0.6)
            fs.meshes.append(FrameMesh(name, ids, src.color, op, src.render_3d,
                                       object_alpha=oa, silhouette=si, clip=clip,
                                       metallic=mtl, roughness=rgh,
                                       slice_opacity=so, **cc))
        else:
            m = ma or mb
            base = (m.opacity if m.visible else 0.0)
            op = _appear_opacity(base, bool(ma), t, layer_transition, lt, layer_transition_at)
            fs.meshes.append(FrameMesh(name, ids, m.color, op, m.render_3d,
                                       object_alpha=m.object_alpha, silhouette=m.silhouette,
                                       clip=_clip_dict(m.clip),
                                       metallic=getattr(m, "metallic", None),
                                       roughness=getattr(m, "roughness", None),
                                       slice_opacity=getattr(m, "slice_opacity", None), **cc))
    # Annotations are matched by layer name. Their appearance follows layer_transition.
    a_an = {an.name: an for an in a.annotations}
    b_an = {an.name: an for an in b.annotations}
    for key in dict.fromkeys(list(a_an) + list(b_an)):
        aa, ab = a_an.get(key), b_an.get(key)
        an_lt, an_at = _an_transition(ab or aa, layer_transition, layer_transition_at)
        an_target = _target_layer_active(an_lt, lt, an_at)
        src = aa if (an_lt == "cut" and not an_target and aa) else (ab or aa)
        if aa and ab:
            op = _blend_value(aa.opacity if aa.visible else 0.0,
                              ab.opacity if ab.visible else 0.0, t, an_lt, lt, an_at)
        else:
            base = (src.opacity if src.visible else 0.0)
            op = _appear_opacity(base, bool(aa), t, an_lt, lt, an_at)
        fs.annotations.append(FrameAnnotation(
            src.name, src.color, op, src.points, src.lines, src.boxes, src.ellipsoids,
            src.point_radius_nm, src.line_radius_nm,
            src.shader, dict(src.shader_controls), dict(src.point_props),
            dict(src.line_props), dict(src.box_props), dict(src.ellipsoid_props)))
    return fs


def _with_fade(fs: FrameState, alpha: float) -> FrameState:
    fs.fade_alpha = max(0.0, min(1.0, float(alpha)))
    return fs


def _transition_state(a: Keyframe, b: Keyframe, t: float, style: str,
                      easing: str, layer_transition: str = "fade",
                      layer_transition_at: float = 1.0,
                      em_transition: str = "glide",
                      em_transition_at: float | None = None) -> FrameState:
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
                     layer_t=t, layer_transition_at=layer_transition_at,
                     em_transition=em_transition, em_transition_at=em_transition_at)


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
                                     getattr(b, "layer_transition_at", 1.0),
                                     getattr(b, "em_transition", "glide"),
                                     getattr(b, "em_transition_at", None))
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
        em_transition = getattr(b, "em_transition", "glide")
        em_transition_at = getattr(b, "em_transition_at", None)
        for k in range(n):
            frames.append(_transition_state(a, b, k / n, style, ease, layer_transition,
                                            layer_transition_at, em_transition,
                                            em_transition_at))
    frames.append(_state_at(keyframes[-1], keyframes[-1], 0.0))  # final keyframe, 1 frame
    final_hold_n = int(round(max(0.0, float(getattr(keyframes[-1], "hold_in_s", 0.0) or 0.0)) * fps))
    for _ in range(final_hold_n):
        frames.append(_state_at(keyframes[-1], keyframes[-1], 0.0))
    return frames
