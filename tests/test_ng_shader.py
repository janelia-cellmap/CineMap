"""Parity tests for the neuroglancer shader replication.

Expected values are computed from neuroglancer's own formulas (see the citations in
`cinemap/data/ng_shader.py`), not from our implementation, so these fail if we drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from cinemap.data import ng_shader as ns


def _invlerp(x, lo, hi, clamp=True):
    v = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
    return np.clip(v, 0.0, 1.0) if clamp else v


def _u8(v):
    return (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ dtype defaults
def test_default_ranges_match_neuroglancer():
    # src/util/lerp.ts defaultDataTypeRange
    assert ns.default_range_for(np.uint8) == (0, 255)
    assert ns.default_range_for(np.uint16) == (0, 65535)
    assert ns.default_range_for(np.int8) == (-128, 127)
    assert ns.default_range_for(np.int16) == (-32768, 32767)
    assert ns.default_range_for(np.uint32) == (0, 4294967295)
    # float normalizes over the UNIT interval, not the dtype extent
    assert ns.default_range_for(np.float32) == (0.0, 1.0)


# ------------------------------------------------------------------ default shader
def test_default_shader_is_identity_for_uint8():
    """With the default range [0,255], uint8 EM must pass through unchanged."""
    data = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    sh = ns.from_layer({}, np.uint8)
    out, warn = ns.shade(sh, data)
    assert warn == ""
    assert out.shape == (1, 4, 3)
    np.testing.assert_array_equal(out[..., 0], data)
    # grayscale -> all three channels identical
    np.testing.assert_array_equal(out[..., 0], out[..., 2])


def test_uint16_default_range_is_not_a_truncating_cast():
    """Regression: the old code did astype(uint8), which wraps mod 256.

    40000 -> 40000 % 256 == 64 under the old cast, but neuroglancer shows
    40000/65535 == 0.610 -> 156.
    """
    data = np.array([[0, 40000, 65535]], dtype=np.uint16)
    out, _ = ns.shade(ns.from_layer({}, np.uint16), data)
    expected = _u8(_invlerp(data, 0, 65535))
    np.testing.assert_array_equal(out[..., 0], expected)
    assert out[0, 1, 0] == 156          # NOT 64
    assert out[0, 2, 0] == 255          # NOT 255-by-accident-of-wrapping


# ------------------------------------------------------------------ shaderControls
def test_shader_controls_range_narrows_contrast():
    data = np.array([[0, 50, 100, 150, 200, 255]], dtype=np.uint8)
    layer = {"shaderControls": {"normalized": {"range": [50, 150]}}}
    out, warn = ns.shade(ns.from_layer(layer, np.uint8), data)
    assert warn == ""
    np.testing.assert_array_equal(out[..., 0], _u8(_invlerp(data, 50, 150)))
    assert out[0, 0, 0] == 0            # below the window -> clamped black
    assert out[0, 1, 0] == 0            # exactly at min
    assert out[0, 3, 0] == 255          # exactly at max
    assert out[0, 5, 0] == 255          # above the window -> clamped white


def test_window_is_ignored():
    """`window` only zooms the UI slider; applying it to pixels would be wrong."""
    data = np.array([[0, 128, 255]], dtype=np.uint8)
    with_window = {"shaderControls": {"normalized": {"range": [0, 255],
                                                     "window": [100, 120]}}}
    out_w, _ = ns.shade(ns.from_layer(with_window, np.uint8), data)
    out_plain, _ = ns.shade(ns.from_layer({}, np.uint8), data)
    np.testing.assert_array_equal(out_w, out_plain)


def test_legacy_bare_array_shader_controls():
    data = np.array([[0, 100, 200]], dtype=np.uint8)
    out, _ = ns.shade(ns.from_layer({"shaderControls": {"normalized": [50, 150]}},
                                    np.uint8), data)
    np.testing.assert_array_equal(out[..., 0], _u8(_invlerp(data, 50, 150)))


def test_inverted_range_inverts_the_image():
    """NG allows range[0] > range[1]; the image inverts rather than breaking."""
    data = np.array([[0, 128, 255]], dtype=np.uint8)
    out, _ = ns.shade(ns.from_layer(
        {"shaderControls": {"normalized": {"range": [255, 0]}}}, np.uint8), data)
    np.testing.assert_array_equal(out[..., 0], _u8(_invlerp(data, 255, 0)))
    assert out[0, 0, 0] == 255 and out[0, 2, 0] == 0


# ------------------------------------------------------------------ directive parsing
def test_directive_declares_range_without_shader_controls():
    """A custom shader's declared default must be honored when the state omits it."""
    src = "#uicontrol invlerp normalized(range=[10, 20])\nvoid main(){emitGrayscale(normalized());}"
    sh = ns.from_layer({"shader": src}, np.uint8)
    assert sh.primary_range == (10.0, 20.0)


def test_clamp_false_is_respected():
    src = ("#uicontrol invlerp normalized(range=[0,100], clamp=false)\n"
           "void main(){emitGrayscale(normalized());}")
    sh = ns.from_layer({"shader": src}, np.uint8)
    assert sh.controls["normalized"].clamp is False
    # 200 with range [0,100] -> 2.0, clipped only at the final 0-255 encode
    out, _ = ns.shade(sh, np.array([[200]], dtype=np.uint8))
    assert out[0, 0, 0] == 255


def test_clamp_defaults_to_true():
    sh = ns.from_layer({"shader": "#uicontrol invlerp normalized\nvoid main(){emitGrayscale(normalized());}"},
                       np.uint8)
    assert sh.controls["normalized"].clamp is True


def test_commented_out_directive_is_ignored():
    src = ("// #uicontrol invlerp bogus(range=[1,2])\n"
           "#uicontrol invlerp normalized\n"
           "void main(){emitGrayscale(normalized());}")
    sh = ns.from_layer({"shader": src}, np.uint8)
    assert "bogus" not in sh.controls
    assert "normalized" in sh.controls


def test_shader_controls_override_directive_default():
    src = "#uicontrol invlerp normalized(range=[10, 20])\nvoid main(){emitGrayscale(normalized());}"
    sh = ns.from_layer({"shader": src,
                        "shaderControls": {"normalized": {"range": [30, 40]}}}, np.uint8)
    assert sh.primary_range == (30.0, 40.0)


# ------------------------------------------------------------------ emit variants
def test_emit_rgb_with_color_control():
    src = ("#uicontrol invlerp normalized\n"
           "#uicontrol vec3 tint color(default=\"#ff0000\")\n"
           "void main(){ emitRGB(tint * normalized()); }")
    data = np.array([[0, 255]], dtype=np.uint8)
    out, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8), data)
    assert warn == ""
    np.testing.assert_array_equal(out[0, 1], [255, 0, 0])   # full red at max
    np.testing.assert_array_equal(out[0, 0], [0, 0, 0])


def test_emit_rgb_channel_split():
    src = ("#uicontrol invlerp normalized\n"
           "void main(){ emitRGB(vec3(normalized(), 0.0, 0.0)); }")
    out, _ = ns.shade(ns.from_layer({"shader": src}, np.uint8),
                      np.array([[255]], dtype=np.uint8))
    np.testing.assert_array_equal(out[0, 0], [255, 0, 0])


def test_slider_control_arithmetic():
    src = ("#uicontrol invlerp normalized\n"
           "#uicontrol float gain slider(min=0, max=4, default=2)\n"
           "void main(){ emitGrayscale(normalized() * gain); }")
    out, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8),
                         np.array([[64]], dtype=np.uint8))
    assert warn == ""
    assert out[0, 0, 0] == _u8(np.array((64 / 255) * 2.0))


def test_slider_value_from_shader_controls():
    src = ("#uicontrol invlerp normalized\n"
           "#uicontrol float gain slider(min=0, max=4, default=1)\n"
           "void main(){ emitGrayscale(normalized() * gain); }")
    sh = ns.from_layer({"shader": src, "shaderControls": {"gain": 0.5}}, np.uint8)
    out, _ = ns.shade(sh, np.array([[255]], dtype=np.uint8))
    assert out[0, 0, 0] == 128


def test_emit_transparent():
    src = "void main(){ emitTransparent(); }"
    out, _ = ns.shade(ns.from_layer({"shader": src}, np.uint8),
                      np.array([[200]], dtype=np.uint8))
    np.testing.assert_array_equal(out[0, 0], [0, 0, 0])


# ------------------------------------------------------------------ colormaps
def test_colormap_jet_matches_glsl():
    src = ("#uicontrol invlerp normalized\n"
           "void main(){ emitRGB(colormapJet(normalized())); }")
    data = np.array([[0, 128, 255]], dtype=np.uint8)
    out, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8), data)
    assert warn == ""
    x = data.astype(np.float64) / 255.0
    r = np.where(x < 0.89, (x - 0.35) / 0.31, 1.0 - (x - 0.89) / 0.11 * 0.5)
    np.testing.assert_array_equal(out[..., 0], _u8(np.clip(r, 0, 1)))


# ------------------------------------------------------------------ opacity + fallback
def test_opacity_is_captured():
    assert ns.from_layer({"opacity": 0.35}, np.uint8).opacity == pytest.approx(0.35)
    assert ns.from_layer({}, np.uint8).opacity == 1.0
    # opacity 0 must survive (it is how NG hides a layer), not fall back to 1.0
    assert ns.from_layer({"opacity": 0}, np.uint8).opacity == 0.0


def test_unsupported_shader_falls_back_loudly_but_keeps_contrast():
    """An out-of-scope shader must still honor the contrast slider, and say so."""
    src = ("#uicontrol invlerp normalized\n"
           "void main(){ emitGrayscale(someUnknownFn(normalized())); }")
    data = np.array([[0, 100, 200]], dtype=np.uint8)
    sh = ns.from_layer({"shader": src,
                        "shaderControls": {"normalized": {"range": [50, 150]}}}, np.uint8)
    out, warn = ns.shade(sh, data)
    assert warn                                     # not silent
    assert "unsupported" in warn.lower()
    np.testing.assert_array_equal(out[..., 0], _u8(_invlerp(data, 50, 150)))


def test_malformed_shader_does_not_raise():
    sh = ns.from_layer({"shader": "#uicontrol \nvoid main(){"}, np.uint8)
    out, warn = ns.shade(sh, np.array([[10]], dtype=np.uint8))
    assert out.shape == (1, 1, 3)
    assert warn


def test_shading_is_layer_agnostic():
    """The same parse must work for a non-image layer dict (annotation/skeleton)."""
    layer = {"type": "annotation", "shader": "#uicontrol vec3 col color(default=\"blue\")\n"
                                             "void main(){ emitRGB(col); }"}
    out, warn = ns.shade(ns.from_layer(layer, np.uint8),
                         np.zeros((2, 2), dtype=np.uint8))
    assert warn == ""
    np.testing.assert_array_equal(out[0, 0], [0, 0, 255])


# ------------------------------------------------- regressions from code review
def test_custom_invlerp_name_falls_back_without_crashing():
    """An unsupported shader whose invlerp is NOT called `normalized`.

    The fallback renders neuroglancer's default shader, which calls normalized(); the
    kept control has to be re-keyed to match or evaluation raises straight back out and
    the slice vanishes from every frame.
    """
    src = ("#uicontrol invlerp myWindow\n"
           "void main(){ emitGrayscale(bogusFn(myWindow())); }")
    data = np.array([[0, 100, 200]], dtype=np.uint8)
    sh = ns.from_layer({"shader": src,
                        "shaderControls": {"myWindow": {"range": [50, 150]}}}, np.uint8)
    out, warn = ns.shade(sh, data)          # must not raise
    assert warn
    np.testing.assert_array_equal(out[..., 0], _u8(_invlerp(data, 50, 150)))


def test_conditional_shader_refuses_rather_than_baking_black():
    """Two emit* calls = per-pixel branching we don't model. Taking the first would
    silently produce an all-black slice with no warning."""
    src = ("#uicontrol invlerp normalized\n"
           "void main(){ if (normalized() < 0.1) { emitTransparent(); }\n"
           "             else { emitGrayscale(normalized()); } }")
    data = np.array([[0, 128, 255]], dtype=np.uint8)
    out, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8), data)
    assert warn and "emit" in warn
    np.testing.assert_array_equal(out[..., 0], data)   # fell back to plain invlerp


def test_smoothstep_applies_edges_to_vectors():
    src = ("#uicontrol invlerp normalized\n"
           "void main(){ emitRGB(smoothstep(vec3(0.0), vec3(1.0), vec3(normalized()))); }")
    out, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8),
                         np.array([[0, 255]], dtype=np.uint8))
    assert warn == ""
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 255
