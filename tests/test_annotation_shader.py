"""Annotation layers must honour neuroglancer's annotation shader, not just annotationColor.

Neuroglancer's default annotation shader is `void main() { setColor(defaultColor()); }`
(annotation/annotation_layer_state.ts:127), which is the flat `annotationColor` we always
used. A custom shader instead colours each annotation from its own properties, and the
setters are kind-specific -- `setPointMarkerColor` does not touch ellipsoids.
"""
from __future__ import annotations

import numpy as np
import pytest

from cinemap.data import annotations as ann
from cinemap.data import ng_shader as ns

SCORE_SHADER = """
void main() {
  float s = prop_score();
  setColor(vec4(s, 1.0 - s, 0.0, 1.0));
}
"""

LAYER = {
    "type": "annotation",
    "name": "synapses",
    "annotationColor": "#ffff4d",
    "shader": SCORE_SHADER,
    "annotationProperties": [
        {"id": "score", "type": "float32", "default": 0.0},
        {"id": "tag", "type": "rgb", "default": "#000000"},
    ],
    "annotations": [
        {"type": "point", "point": [0, 0, 0], "props": [0.0, "#ff0000"]},
        {"type": "point", "point": [1, 0, 0], "props": [1.0, "#00ff00"]},
        {"type": "ellipsoid", "center": [2, 0, 0], "radii": [1, 1, 1], "props": [0.5, "#0000ff"]},
        {"type": "line", "pointA": [0, 0, 0], "pointB": [1, 1, 1]},   # no props -> defaults
    ],
}


# --------------------------------------------------------------------- capture
def test_parse_inline_collects_properties_per_kind():
    prims = ann.parse_inline(LAYER, (1.0, 1.0, 1.0))
    props = prims["props"]
    assert props["points"]["score"] == [0.0, 1.0]
    assert props["ellipsoids"]["score"] == [0.5]
    # an annotation with no `props` array falls back to each spec's declared default
    assert props["lines"]["score"] == [0.0]
    # rgb properties arrive as JSON colour strings and become normalized components
    assert props["points"]["tag"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_missing_property_spec_means_no_props():
    plain = {"type": "annotation", "annotations": [{"type": "point", "point": [0, 0, 0]}]}
    assert ann.parse_inline(plain, (1.0, 1.0, 1.0))["props"]["points"] == {}


# --------------------------------------------------------------------- shading
def test_default_shader_is_the_flat_annotation_color():
    out, warn = ns.shade_annotations("", {}, [1.0, 0.95, 0.30], count=2)
    assert warn == ""
    for kind in ("point", "line", "box", "ellipsoid"):
        np.testing.assert_array_equal(out[kind], np.tile([255, 242, 77, 255], (2, 1)))


def test_property_shader_colors_each_annotation():
    out, warn = ns.shade_annotations(
        SCORE_SHADER, {"score": np.array([0.0, 0.5, 1.0])}, [1.0, 1.0, 1.0], count=3)
    assert warn == ""
    np.testing.assert_array_equal(out["point"],
                                  np.array([[0, 255, 0, 255], [128, 128, 0, 255],
                                            [255, 0, 0, 255]]))
    # setColor fans out to every primitive kind (annotation/type_handler.ts:539)
    for kind in ("line", "box", "ellipsoid"):
        np.testing.assert_array_equal(out[kind], out["point"])


def test_kind_specific_setter_does_not_leak():
    """setPointMarkerColor must not repaint ellipsoids."""
    src = """
void main() {
  setColor(vec3(0.0, 0.0, 1.0));
  setPointMarkerColor(vec3(1.0, 0.0, 0.0));
}
"""
    out, warn = ns.shade_annotations(src, {}, [1.0, 1.0, 1.0], count=1)
    assert warn == ""
    np.testing.assert_array_equal(out["point"][0], [255, 0, 0, 255])
    np.testing.assert_array_equal(out["ellipsoid"][0], [0, 0, 255, 255])


def test_last_setter_wins():
    src = "void main() { setColor(vec3(1,0,0)); setColor(vec3(0,1,0)); }"
    out, warn = ns.shade_annotations(src, {}, [0, 0, 0], count=1)
    assert warn == ""
    np.testing.assert_array_equal(out["point"][0], [0, 255, 0, 255])


def test_shader_controls_reach_the_annotation_shader():
    src = """
#uicontrol vec3 tint color(default="#ff0000")
void main() { setColor(tint); }
"""
    a, _ = ns.shade_annotations(src, {}, [0, 0, 0], count=1)
    b, warn = ns.shade_annotations(src, {}, [0, 0, 0], {"tint": "#0000ff"}, count=1)
    assert warn == ""
    np.testing.assert_array_equal(a["point"][0], [255, 0, 0, 255])
    np.testing.assert_array_equal(b["point"][0], [0, 0, 255, 255])


def test_size_setters_do_not_break_the_shader():
    """setPointMarkerSize etc. are recorded but not colour setters; they must be ignored,
    not treated as unsupported GLSL."""
    src = "void main() { setPointMarkerSize(12.0); setLineWidth(3.0); setColor(vec3(1,1,0)); }"
    out, warn = ns.shade_annotations(src, {}, [0, 0, 0], count=1)
    assert warn == ""
    np.testing.assert_array_equal(out["point"][0], [255, 255, 0, 255])


def test_unsupported_shader_falls_back_to_annotation_color():
    out, warn = ns.shade_annotations("void main() { setColor(vec3(nope())); }",
                                     {}, [1.0, 0.0, 0.0], count=2)
    assert warn
    np.testing.assert_array_equal(out["point"], np.tile([255, 0, 0, 255], (2, 1)))


# --------------------------------------------------------------------- end to end
def test_mesh_vertices_carry_the_shaded_colors():
    """The colours must survive into the actual geometry, not just the shader call."""
    prims = ann.parse_inline(LAYER, (1.0, 1.0, 1.0))
    styles = {}
    for kind, key in (("point", "points"), ("line", "lines"), ("ellipsoid", "ellipsoids")):
        n = len(prims[key])
        shaded, warn = ns.shade_annotations(SCORE_SHADER, prims["props"][key],
                                            [1.0, 0.95, 0.30], count=n)
        assert warn == ""
        styles[kind] = shaded[kind]

    mesh = ann.annotations_to_mesh(prims, [1.0, 0.95, 0.30], styles=styles)
    assert mesh is not None
    colors = np.unique(np.asarray(mesh.visual.vertex_colors)[:, :3], axis=0)
    # score 0 -> green, score 1 -> red, score 0.5 -> olive; none of them the flat yellow
    assert any((c == [0, 255, 0]).all() for c in colors)
    assert any((c == [255, 0, 0]).all() for c in colors)
    assert not any((c == [255, 242, 77]).all() for c in colors)


def test_flat_color_still_used_without_a_shader():
    prims = ann.parse_inline({"type": "annotation",
                              "annotations": [{"type": "point", "point": [0, 0, 0]}]},
                             (1.0, 1.0, 1.0))
    mesh = ann.annotations_to_mesh(prims, [1.0, 0.95, 0.30])
    colors = np.unique(np.asarray(mesh.visual.vertex_colors)[:, :3], axis=0)
    assert len(colors) == 1
    np.testing.assert_array_equal(colors[0], [255, 242, 77])


def test_box_colors_expand_to_all_twelve_edges():
    """A box is swept as 12 separate tubes; every one must take that box's colour."""
    prims = {"points": [], "lines": [], "ellipsoids": [],
             "boxes": [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]}
    styles = {"box": np.array([[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)}
    mesh = ann.annotations_to_mesh(prims, [1.0, 1.0, 1.0], styles=styles)
    colors = np.unique(np.asarray(mesh.visual.vertex_colors)[:, :3], axis=0)
    assert len(colors) == 2
    assert any((c == [255, 0, 0]).all() for c in colors)
    assert any((c == [0, 0, 255]).all() for c in colors)


# ------------------------------------------------------------------- plumbing
def test_frame_annotation_carries_the_shader_through_interpolation():
    """A field the frame state drops is a field the renderer never sees -- the exact way
    the slice shader was lost before."""
    from cinemap.models import AnnotationInstance, Camera, Keyframe
    from cinemap.render.interpolate import state_at_time

    inst = AnnotationInstance(
        name="synapses", color=[1.0, 0.95, 0.30],
        points=[[0, 0, 0], [1, 0, 0]],
        shader=SCORE_SHADER, shader_controls={"tint": "#00ff00"},
        point_props={"score": [0.0, 1.0]})
    cam = Camera(position_nm=[0, 0, 100], look_at_nm=[0, 0, 0], fov_deg=45.0)
    kf = Keyframe(id="kf1", label="a", camera=cam, annotations=[inst])

    fs = state_at_time([kf], 0.0)
    fa = fs.annotations[0]
    assert fa.shader == SCORE_SHADER
    assert fa.shader_controls == {"tint": "#00ff00"}
    assert fa.point_props == {"score": [0.0, 1.0]}
