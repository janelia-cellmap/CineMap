"""Parity tests for the neuroglancer shader replication.

Expected values are computed from neuroglancer's own formulas (see the citations in
`cinemap/data/ng_shader.py`), not from our implementation, so these fail if we drift.
"""
from __future__ import annotations

import json

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


def test_bare_array_shader_controls_are_ignored_like_neuroglancer():
    """A bare `[lo, hi]` instead of `{"range": [lo, hi]}` must NOT change the image.

    Neuroglancer's parseImageInvlerpParameters calls verifyObject first, which throws on
    an array, and restoreState swallows the error — so the control keeps its default.
    Confirmed against a live viewer (spikes/ng_parity/shader_parity.py): neuroglancer
    renders the identity ramp for this input, not a [50,150] window.
    """
    data = np.array([[0, 100, 200]], dtype=np.uint8)
    out, _ = ns.shade(ns.from_layer({"shaderControls": {"normalized": [50, 150]}},
                                    np.uint8), data)
    plain, _ = ns.shade(ns.from_layer({}, np.uint8), data)
    np.testing.assert_array_equal(out, plain)


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


# ------------------------------------- real hemibrain state (public demo script)
HEMIBRAIN_SYNAPSE_SHADER = json.loads(r'''"#uicontrol bool showPsds checkbox(default=true)\n#uicontrol vec3 preColor color(default=\"red\")\n#uicontrol vec3 postColor color(default=\"blue\")\n#uicontrol float preConfidence slider(min=0, max=1, default=0)\n#uicontrol float postConfidence slider(min=0, max=1, default=0)\n\nvoid main() {\n  setColor(defaultColor());\n}\n"''')


def test_real_annotation_layer_controls_parse():
    """An ANNOTATION layer from the public hemibrain video script.

    Proves the parsing is layer-agnostic: this is not an image layer and uses
    checkbox/color/slider controls rather than invlerp.
    """
    layer = {"type": "annotation", "shader": HEMIBRAIN_SYNAPSE_SHADER,
             "shaderControls": {"showPsds": False}}
    sh = ns.from_layer(layer, np.uint8)
    assert sh.controls["preColor"].value == (1.0, 0.0, 0.0)      # "red"
    assert sh.controls["postColor"].value == (0.0, 0.0, 1.0)     # "blue"
    assert sh.controls["preConfidence"].value == 0.0
    assert sh.controls["showPsds"].value is False                # state overrode the default


def test_escaped_quote_color_still_resolves():
    """A double-encoded state keeps the backslashes; the color must not fall back."""
    src = '#uicontrol vec3 c color(default=\\"red\\")\nvoid main(){ emitRGB(c); }'
    assert ns.from_layer({"shader": src}, np.uint8).controls["c"].value == (1.0, 0.0, 0.0)


# --------------------------------------------------------------- effective opacity
# Neuroglancer's image `opacity` does NOT scale displayed pixels for the bottom-most
# image layer: it disables GL blending entirely for renderLayerNum 0 with the default
# blend mode (sliceview/volume/image_renderlayer.ts:152), and the panel composites the
# background with `if (sampledColor.a == 0.0)` rather than an alpha blend
# (sliceview/frontend.ts:753). Measured against a live viewer: opacity 0.25/0.5/0.75/1.0
# all render an identical full-strength ramp; only 0.0 shows the background.

def _state(*layers):
    return {"layers": list(layers)}


def test_bottom_image_layer_ignores_opacity():
    """The case that matters: one EM layer at neuroglancer's DEFAULT opacity of 0.5.

    Taking the raw value would render every EM slice at half strength in Blender against
    a viewer showing it whole.
    """
    em = {"type": "image", "name": "em"}                  # no explicit opacity -> 0.5
    assert ns.effective_image_opacity(_state(em), em) == 1.0
    dimmed = {"type": "image", "name": "em", "opacity": 0.25}
    assert ns.effective_image_opacity(_state(dimmed), dimmed) == 1.0


def test_zero_opacity_hides_the_layer():
    """alpha == 0 is the one value the panel's if/else treats as 'show the background'."""
    em = {"type": "image", "name": "em", "opacity": 0}
    assert ns.effective_image_opacity(_state(em), em) == 0.0


def test_image_layer_above_the_first_does_blend():
    """renderLayerNum > 0 enables blending, so opacity is real for stacked image layers."""
    em = {"type": "image", "name": "em"}
    overlay = {"type": "image", "name": "overlay", "opacity": 0.3}
    st = _state(em, overlay)
    assert ns.effective_image_opacity(st, em) == 1.0
    assert ns.effective_image_opacity(st, overlay) == 0.3


def test_additive_blend_keeps_opacity_even_at_the_bottom():
    """`BLEND_MODES.ADDITIVE` takes the gl.enable(BLEND) branch regardless of layer num."""
    em = {"type": "image", "name": "em", "opacity": 0.4, "blend": "additive"}
    assert ns.effective_image_opacity(_state(em), em) == 0.4


def test_hidden_layers_do_not_claim_the_bottom_slot():
    """A hidden or archived layer is not drawn, so the next one is renderLayerNum 0."""
    hidden = {"type": "image", "name": "hidden", "visible": False}
    em = {"type": "image", "name": "em", "opacity": 0.5}
    assert ns.effective_image_opacity(_state(hidden, em), em) == 1.0
    archived = {"type": "image", "name": "old", "archived": True}
    assert ns.effective_image_opacity(_state(archived, em), em) == 1.0


def test_segmentation_layers_are_not_image_render_layers():
    """Only image layers occupy the image renderLayerNum sequence."""
    seg = {"type": "segmentation", "name": "seg"}
    em = {"type": "image", "name": "em", "opacity": 0.5}
    assert ns.effective_image_opacity(_state(seg, em), em) == 1.0


# ------------------------------------------------- full-GLSL shaders (functions, locals)
# Real neuroglancer shaders are not single expressions: they define helper functions,
# declare `const` coefficient vectors, and compute locals before emitting. The cellmap
# skeleton shaders do all three, which is why matching shader TEMPLATES lost their color.

TURBO = """
#uicontrol float minRadiusNm slider(min=1.0, max=300.0, step=1.0, default=30.0)
#uicontrol float maxRadiusNm slider(min=100.0, max=3000.0, step=10.0, default=1300.0)

vec3 turbo(float x) {
  x = clamp(x, 0.0, 1.0);
  const vec4 kRed = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kGreen = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kBlue = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kRed2 = vec2(-152.94239396, 59.28637943);
  const vec2 kGreen2 = vec2(4.27729857, 2.82956604);
  const vec2 kBlue2 = vec2(-89.90310912, 27.34824973);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return clamp(vec3(dot(v4, kRed) + dot(v2, kRed2),
                    dot(v4, kGreen) + dot(v2, kGreen2),
                    dot(v4, kBlue) + dot(v2, kBlue2)), 0.0, 1.0);
}

void main() {
  float radiusNm = max(prop_radius(), 0.001);
  float lo = log(max(minRadiusNm, 0.001));
  float hi = log(max(maxRadiusNm, minRadiusNm + 0.001));
  float t = clamp((log(radiusNm) - lo) / (hi - lo), 0.0, 1.0);
  emitRGB(turbo(t));
}
"""


def test_shader_property_names_finds_prop_accessors():
    assert ns.shader_property_names(TURBO) == {"radius"}
    assert ns.shader_property_names("void main(){emitGrayscale(0.5);}") == set()


def test_turbo_skeleton_shader_evaluates():
    """The exact shader shape that used to fall back to one flat color."""
    radii = np.array([10.0, 30.0, 100.0, 300.0, 1000.0, 1300.0, 5000.0])
    rgb, warn = ns.shade_properties(TURBO, {"radius": radii})
    assert warn == "", warn
    assert rgb.shape == (7, 3)
    # turbo runs dark blue -> cyan -> yellow -> red as t goes 0..1
    assert rgb[2][2] > rgb[2][0]        # 100nm: blue-ish channel dominates red
    assert rgb[4][0] > rgb[4][2]        # 1000nm: red channel dominates blue
    # clamped at both ends of [minRadiusNm, maxRadiusNm]
    np.testing.assert_array_equal(rgb[0], rgb[1])       # 10nm and 30nm both clamp to t=0
    np.testing.assert_array_equal(rgb[5], rgb[6])       # 1300nm and 5000nm both clamp to t=1
    assert len(np.unique(rgb, axis=0)) >= 5             # genuinely varying, not flat


def test_slider_controls_reach_a_property_shader():
    """Narrowing the radius window must move the colors, or the sliders are decorative."""
    radii = np.array([50.0, 100.0, 200.0])
    a, _ = ns.shade_properties(TURBO, {"radius": radii})
    b, warn = ns.shade_properties(
        TURBO, {"radius": radii}, {"minRadiusNm": 40.0, "maxRadiusNm": 220.0})
    assert warn == ""
    assert not np.array_equal(a, b)


def test_user_function_and_locals_in_an_image_shader():
    """The same machinery has to work for image layers, not just skeletons."""
    src = """
#uicontrol invlerp normalized
float boost(float v) { float k = v * v; return k; }
void main() { emitGrayscale(boost(normalized())); }
"""
    data = np.array([[0, 128, 255]], dtype=np.uint8)
    out, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8), data)
    assert warn == ""
    want = _u8(_invlerp(data, 0, 255) ** 2)
    np.testing.assert_array_equal(out[..., 0], want)


def test_vec4_arithmetic_is_not_truncated_to_vec3():
    """A vec4 dot product must use all four components."""
    src = """
#uicontrol invlerp normalized
void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  vec4 k = vec4(0.0, 0.0, 0.0, 0.25);
  emitGrayscale(dot(v, k));
}
"""
    out, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8),
                         np.array([[0]], dtype=np.uint8))
    assert warn == ""
    assert out[0, 0, 0] == 255          # dot == 4*0.25 == 1.0; a vec3 truncation gives 0


def test_mismatched_vector_widths_are_refused_not_padded():
    """vec2 * vec3 is nonsense; padding with the last component would invent a value."""
    with pytest.raises(ns.ShaderUnsupported):
        ns._Expr("a * b", {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]}).parse()


def test_control_flow_falls_back_loudly():
    src = """
#uicontrol invlerp normalized
void main() { float v = normalized(); if (v > 0.5) { v = 1.0; } emitGrayscale(v); }
"""
    _, warn = ns.shade(ns.from_layer({"shader": src}, np.uint8),
                       np.array([[0, 255]], dtype=np.uint8))
    assert warn        # falls back rather than baking a guessed branch


def test_property_shader_reports_why_it_could_not_run():
    rgb, warn = ns.shade_properties("void main() { emitRGB(vec3(nope())); }",
                                    {"radius": np.array([1.0])})
    assert warn and rgb.shape == (1, 3)
