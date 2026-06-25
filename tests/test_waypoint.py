"""Fly-through waypoints: the camera passes through a waypoint at speed (no ease-stop,
no hold), while non-waypoint keyframes still ease to a stop."""
from cinemap.models import Camera, Keyframe
from cinemap.render import interpolate as I


def _kf(i, waypoint=False, hold=0.0):
    return Keyframe(
        id=f"k{i}",
        camera=Camera(position_nm=[float(i), 0, 0], look_at_nm=[0, 0, 1], up=[0, 1, 0], fov_deg=45),
        duration_in_s=1.0, hold_in_s=hold, easing="ease-in-out", waypoint=waypoint,
    )


def test_seg_ease_waypoint_table():
    stop_a, stop_b = _kf(0), _kf(1)
    wp_a, wp_b = _kf(0, waypoint=True), _kf(1, waypoint=True)
    # no waypoints -> keyframe's own easing (unchanged behavior)
    assert I._seg_ease(stop_a, stop_b, 1, 3, smooth_ends=False) == "ease-in-out"
    # no waypoints + director smooth_ends -> linear in the middle
    assert I._seg_ease(stop_a, stop_b, 1, 3, smooth_ends=True) == "linear"
    # waypoint endpoints: enter/leave at speed
    assert I._seg_ease(stop_a, wp_b, 0, 3, smooth_ends=False) == "ease-in"    # accel from stop
    assert I._seg_ease(wp_a, stop_b, 2, 3, smooth_ends=False) == "ease-out"   # decel to stop
    assert I._seg_ease(wp_a, wp_b, 1, 3, smooth_ends=False) == "linear"       # constant speed


def test_waypoint_ignores_hold():
    assert I._kf_hold_s(_kf(0, hold=3.0)) == 3.0
    assert I._kf_hold_s(_kf(0, waypoint=True, hold=3.0)) == 0.0


def test_build_frames_no_stop_or_hold_at_waypoint():
    # 3 keyframes, middle is a fly-through waypoint with a (to-be-ignored) hold.
    kfs = [_kf(0), _kf(1, waypoint=True, hold=2.0), _kf(2)]
    fps = 10
    frames = I.build_frames(kfs, fps, smooth_ends=False)
    # 2 transitions x 10 + final frame = 21; the waypoint's 2s hold must NOT add frames.
    assert len(frames) == 21
    # The middle keyframe's x-position is 1.0; with a real stop (ease-out into it) the
    # camera would dwell near x≈1. As a waypoint it passes through at speed, so only a
    # few frames sit near x≈1.
    near = sum(1 for f in frames if abs(f.position_nm[0] - 1.0) < 0.05)
    assert near <= 3, f"too many frames lingering at the waypoint: {near}"
