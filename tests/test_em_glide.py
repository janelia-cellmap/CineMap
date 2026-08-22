"""The EM cross-section transition (`Keyframe.em_transition`).

Neuroglancer moves a cross-section by lerping `position` and slerping
`crossSectionOrientation` — one plane travels. CineMap used to key slices on
(layer, dominant axis) and cross-fade, so a tilted plane whose dominant axis changed
between keyframes became TWO planes on screen: one held still fading out, one fading in
at a different angle. These cover the gliding plane and its endpoints.
"""
from __future__ import annotations

import math

import numpy as np

from cinemap.models import Camera, Keyframe, SlicePlane
from cinemap.render.interpolate import _state_at, plane_normal


def _kf(kid: str, look_at, slices) -> Keyframe:
    return Keyframe(
        id=kid,
        camera=Camera(position_nm=[look_at[0], look_at[1], look_at[2] + 1000.0],
                      look_at_nm=list(look_at)),
        slices=slices,
    )


def _sl(**kw) -> SlicePlane:
    kw.setdefault("em_name", "em")
    return SlicePlane(**kw)


def _through(look_at, normal) -> float:
    """The plane offset that puts a plane with `normal` through `look_at` (what baking
    from a neuroglancer view produces: the cross-section passes through `position`)."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)          # offsets are measured along the UNIT normal
    return float(np.dot(np.asarray(look_at, float), n))


# ------------------------------------------------------------- one plane, not two
def test_tilted_planes_glide_as_one_plane():
    """A y-dominant oblique plane -> an x-dominant one stays ONE cross-section."""
    la, lb = [0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]
    na = [-0.681, 0.728, 0.071]
    nb = [-0.983, -0.009, 0.185]
    a = _kf("a", la, [_sl(axis="y", normal=na, position_nm=_through(la, na))])
    b = _kf("b", lb, [_sl(axis="x", normal=nb, position_nm=_through(lb, nb))])
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        fs = _state_at(a, b, t)
        assert len(fs.slices) == 1, f"t={t} produced {len(fs.slices)} planes"
        assert fs.slices[0].opacity == 1.0        # no cross-fade dip


def test_glide_endpoints_are_the_keyframe_planes():
    la, lb = [0.0, 0.0, 0.0], [1000.0, 500.0, -200.0]
    na, nb = [-0.681, 0.728, 0.071], [-0.983, -0.009, 0.185]
    a = _kf("a", la, [_sl(axis="y", normal=na, position_nm=_through(la, na))])
    b = _kf("b", lb, [_sl(axis="x", normal=nb, position_nm=_through(lb, nb))])
    for t, src, look in ((0.0, na, la), (1.0, nb, lb)):
        s = _state_at(a, b, t).slices[0]
        n = plane_normal(s)
        want = np.asarray(src, float)
        want = want / np.linalg.norm(want)
        sign = math.copysign(1.0, float(np.dot(n, want)))   # n and -n are one plane
        assert np.allclose(n, sign * want, atol=1e-6)
        # same plane => the keyframe's own center still lies on it
        assert abs(float(np.dot(np.asarray(look), n)) - s.position_nm) < 1e-6


def test_glide_rotates_at_constant_angular_speed():
    """Slerp, not a lerp of the normal: half the transition is half the angle."""
    la = [0.0, 0.0, 0.0]
    na, nb = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    a = _kf("a", la, [_sl(axis="x", position_nm=0.0)])
    b = _kf("b", la, [_sl(axis="y", position_nm=0.0)])
    mid = plane_normal(_state_at(a, b, 0.5).slices[0])
    ang = math.degrees(math.acos(abs(float(np.dot(mid, na)))))
    assert abs(ang - 45.0) < 0.5
    assert abs(np.dot(mid, nb) - np.dot(mid, na)) < 1e-6


def test_glide_pivots_on_the_camera_target():
    """A rotating plane stays under what the camera is looking at, like neuroglancer —
    whose cross-section always passes through `position`."""
    la, lb = [1000.0, 2000.0, 3000.0], [1200.0, 2500.0, 2500.0]
    na, nb = [0.0, 0.0, 1.0], [0.6, 0.0, 0.8]
    a = _kf("a", la, [_sl(axis="z", position_nm=_through(la, na))])
    b = _kf("b", lb, [_sl(axis="x", normal=nb, position_nm=_through(lb, nb))])
    for t in (0.0, 0.3, 0.6, 1.0):
        fs = _state_at(a, b, t)
        s = fs.slices[0]
        n = plane_normal(s)
        # the interpolated camera target is on the interpolated plane
        assert abs(float(np.dot(np.asarray(fs.look_at_nm), n)) - s.position_nm) < 1e-6


def test_axis_aligned_scan_still_lerps_the_offset():
    """The common case (straight depth scan) must be byte-for-byte the old behavior."""
    la = [0.0, 0.0, 0.0]
    a = _kf("a", la, [_sl(axis="z", position_nm=0.0)])
    b = _kf("b", la, [_sl(axis="z", position_nm=1000.0)])
    for t in (0.0, 0.25, 0.5, 1.0):
        s = _state_at(a, b, t).slices[0]
        assert s.axis == "z" and s.normal is None
        assert abs(s.position_nm - 1000.0 * t) < 1e-6


def test_negative_axis_normal_keeps_its_sign_convention():
    """A plane captured with normal [0,-1,0] stores position_nm = -y. Dropping the
    normal on an axis-aligned glide must re-express that as +y, or the plane mirrors
    across the origin."""
    la = [0.0, -5000.0, 0.0]
    a = _kf("a", la, [_sl(axis="y", normal=[0.0, -1.0, 0.0], position_nm=5000.0)])
    b = _kf("b", la, [_sl(axis="y", normal=[0.0, -1.0, 0.0], position_nm=5000.0)])
    s = _state_at(a, b, 0.5).slices[0]
    if s.normal is None:
        assert abs(s.position_nm - (-5000.0)) < 1e-6   # +y convention
    else:
        assert abs(s.position_nm - 5000.0) < 1e-6


# ------------------------------------------------------------------ multiple panels
def test_multi_panel_planes_pair_by_orientation():
    """Neuroglancer's 4-panel layout shows xy/xz/yz of ONE layer at once. Each plane
    must glide to its own counterpart, not to a perpendicular one."""
    la = [0.0, 0.0, 0.0]
    a = _kf("a", la, [_sl(axis="z", position_nm=0.0), _sl(axis="x", position_nm=0.0)])
    b = _kf("b", la, [_sl(axis="x", position_nm=500.0),           # listed in the OTHER order
                      _sl(axis="z", normal=[0.05, 0.0, 0.9987], position_nm=100.0)])
    fs = _state_at(a, b, 0.5)
    assert len(fs.slices) == 2
    assert {s.axis for s in fs.slices} == {"x", "z"}
    assert len({s.slot for s in fs.slices}) == 2      # distinct blend-export slots
    z = next(s for s in fs.slices if s.axis == "z")
    assert abs(z.position_nm - 50.0) < 1.0            # scrolled, not rotated 90 deg


def test_appearing_slice_still_fades_in():
    la = [0.0, 0.0, 0.0]
    a = _kf("a", la, [])
    b = _kf("b", la, [_sl(axis="z", position_nm=0.0)])
    assert _state_at(a, b, 0.0).slices[0].opacity == 0.0
    assert _state_at(a, b, 1.0).slices[0].opacity == 1.0


# -------------------------------------------------------------------- other modes
def test_fade_mode_keeps_the_old_cross_fade():
    la = [0.0, 0.0, 0.0]
    a = _kf("a", la, [_sl(axis="y", normal=[-0.681, 0.728, 0.071], position_nm=100.0)])
    b = _kf("b", la, [_sl(axis="x", normal=[-0.983, -0.009, 0.185], position_nm=-200.0)])
    fs = _state_at(a, b, 0.5, em_transition="fade")
    assert len(fs.slices) == 2                       # two planes, as before
    assert all(abs(s.opacity - 0.5) < 1e-6 for s in fs.slices)


def test_cut_mode_snaps_at_the_switch_point():
    la = [0.0, 0.0, 0.0]
    a = _kf("a", la, [_sl(axis="z", position_nm=0.0)])
    b = _kf("b", la, [_sl(axis="z", position_nm=900.0)])
    early = _state_at(a, b, 0.4, em_transition="cut", layer_transition_at=0.5, layer_t=0.4)
    late = _state_at(a, b, 0.6, em_transition="cut", layer_transition_at=0.5, layer_t=0.6)
    assert early.slices[0].position_nm == 0.0
    assert late.slices[0].position_nm == 900.0
