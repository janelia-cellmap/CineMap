"""The shader/contrast state must survive every hop to the renderer.

Capturing contrast onto a keyframe is not enough: the render path rebuilds slices through
interpolation and through sweeps, and either can silently drop fields. That is exactly
what happened to slice opacity before — captured in the model, absent at render time.
"""
from __future__ import annotations

import numpy as np
import pytest

from cinemap.models import Camera, Keyframe, SlicePlane
from cinemap.render.interpolate import state_at_time


def _ctrl(lo, hi):
    return {"normalized": {"range": [lo, hi]}}


def _kf(kid, pos, **sl):
    return Keyframe(
        id=kid, duration_in_s=1.0,
        camera=Camera(position_nm=[0, 0, 100], look_at_nm=[0, 0, 0]),
        slices=[SlicePlane(em_name="em", axis="z", position_nm=pos, **sl)],
    )


def _slice_at(kfs, t, **kw):
    fs = state_at_time(kfs, t, **kw)
    assert fs.slices, "expected a slice in the interpolated frame"
    return fs.slices[0]


# ------------------------------------------------------------------- interpolation
def test_shader_state_survives_interpolation():
    kfs = [_kf("a", 0.0, shader_controls=_ctrl(40, 180)),
           _kf("b", 100.0, shader_controls=_ctrl(40, 180))]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert _slice_at(kfs, t).shader_controls == _ctrl(40, 180), f"lost at t={t}"


def test_contrast_snaps_rather_than_blending():
    """A half-blended contrast window is one neither keyframe asked for."""
    kfs = [_kf("a", 0.0, shader_controls=_ctrl(0, 255)),
           _kf("b", 100.0, shader_controls=_ctrl(40, 180))]
    seen = {tuple(_slice_at(kfs, t).shader_controls["normalized"]["range"])
            for t in (0.0, 0.3, 0.5, 0.8, 1.0)}
    assert seen <= {(0, 255), (40, 180)}, f"blended intermediate windows: {seen}"


def test_custom_glsl_survives_interpolation():
    src = "#uicontrol invlerp normalized\nvoid main(){emitGrayscale(1.0-normalized());}"
    kfs = [_kf("a", 0.0, shader=src), _kf("b", 100.0, shader=src)]
    assert _slice_at(kfs, 0.5).shader == src


def test_appearing_slice_keeps_its_shader():
    """A slice present in only one keyframe still carries its contrast while fading."""
    a = Keyframe(id="a", duration_in_s=1.0,
                 camera=Camera(position_nm=[0, 0, 100], look_at_nm=[0, 0, 0]))
    b = _kf("b", 50.0, shader_controls=_ctrl(10, 20))
    assert _slice_at([a, b], 0.5).shader_controls == _ctrl(10, 20)


# ----------------------------------------------------------------- worker resolution
def test_worker_resolves_frame_slice_to_the_right_range():
    """_shader_for reads the FrameSlice the renderer actually holds."""
    from cinemap.render.worker import RenderWorker

    fs = _slice_at([_kf("a", 0.0, shader_controls=_ctrl(57, 150)),
                    _kf("b", 10.0, shader_controls=_ctrl(57, 150))], 0.5)
    sh = RenderWorker._shader_for(None, fs, np.uint8)
    assert sh.primary_range == (57.0, 150.0)


def test_worker_defaults_to_dtype_range_when_state_is_silent():
    from cinemap.render.worker import RenderWorker

    fs = _slice_at([_kf("a", 0.0), _kf("b", 10.0)], 0.5)
    assert RenderWorker._shader_for(None, fs, np.uint8).primary_range == (0, 255)
    assert RenderWorker._shader_for(None, fs, np.uint16).primary_range == (0, 65535)


def test_frame_slice_shader_actually_changes_pixels():
    """End of the chain: the interpolated slice's state produces windowed pixels."""
    from cinemap.data import ng_shader as ns
    from cinemap.render.worker import RenderWorker

    fs = _slice_at([_kf("a", 0.0, shader_controls=_ctrl(64, 192)),
                    _kf("b", 10.0, shader_controls=_ctrl(64, 192))], 0.5)
    out, warn = ns.shade(RenderWorker._shader_for(None, fs, np.uint8),
                         np.array([[64, 128, 192]], dtype=np.uint8))
    assert warn == ""
    assert out[0, 0, 0] == 0 and out[0, 2, 0] == 255
    assert out[0, 1, 0] == pytest.approx(128, abs=1)
