"""Line annotations rendered as 3D arrows.

A neuroglancer line annotation is two clicks (pointA = tail, pointB = tip), which makes it
the authoring path for a callout that points at something. These cover the geometry and
the layer selection that decides tube-vs-arrow.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cinemap.data.annotations import _arrow, annotations_to_mesh
from cinemap.models import (AnnotationInstance, Camera, Keyframe, Project, RenderJob,
                            RenderSettings)
from cinemap.render.worker import RenderWorker

RED = np.array([255, 0, 0, 255], dtype=np.uint8)


def _prims(lines=None, **kw):
    out = {"points": [], "lines": lines or [], "boxes": [], "ellipsoids": []}
    out.update(kw)
    return out


# ------------------------------------------------------------------ geometry
def test_arrow_tip_lands_exactly_on_point_b():
    for tip in ([2000, 0, 0], [0, 0, -1500], [-700, 700, 300]):
        m = _arrow([0, 0, 0], tip, RED, 40.0)
        # the cone apex is a real vertex at pointB, not merely near it
        assert np.min(np.linalg.norm(m.vertices - np.asarray(tip, float), axis=1)) < 1e-9


def test_arrow_is_a_closed_solid():
    m = _arrow([100, 200, 300], [900, -400, 1200], RED, 40.0)
    assert m.is_watertight          # shaft is capped, head is closed
    assert m.volume > 0


def test_arrow_is_tinted_with_the_annotation_color():
    m = _arrow([0, 0, 0], [1000, 0, 0], RED, 40.0)
    assert np.array_equal(np.unique(m.visual.vertex_colors, axis=0), RED[None, :])


def test_arrow_scales_with_its_own_length():
    """A long arrow must not come out as a hairline with a fat head."""
    short = _arrow([0, 0, 0], [500, 0, 0], RED, 40.0)
    long = _arrow([0, 0, 0], [50000, 0, 0], RED, 40.0)
    # cross-section radius ~ half the y-extent
    r_short = float(short.bounds[1][1] - short.bounds[0][1]) / 2
    r_long = float(long.bounds[1][1] - long.bounds[0][1]) / 2
    assert r_long > 10 * r_short


def test_thicker_requested_radius_is_honored():
    thin = _arrow([0, 0, 0], [1000, 0, 0], RED, 40.0)
    thick = _arrow([0, 0, 0], [1000, 0, 0], RED, 200.0)
    assert (thick.bounds[1][1] - thick.bounds[0][1]) > (thin.bounds[1][1] - thin.bounds[0][1])


def _shaft_and_head(m):
    tail = m.vertices[m.vertices[:, 0] < 1e-6]          # the flat cap at the tail
    return float(tail[:, 1].max() - tail[:, 1].min()), float(m.bounds[1][1] - m.bounds[0][1])


def test_arrow_scale_fattens_shaft_and_head():
    thin = _arrow([0, 0, 0], [4000, 0, 0], RED, 40.0, scale=1.0)
    bold = _arrow([0, 0, 0], [4000, 0, 0], RED, 40.0, scale=3.0)
    s1, h1 = _shaft_and_head(thin)
    s3, h3 = _shaft_and_head(bold)
    assert s3 == pytest.approx(3 * s1, rel=1e-6)
    assert h3 > h1
    # length is set by the two clicked points, NOT by scale
    assert bold.bounds[1][0] == pytest.approx(4000.0)


def test_a_scaled_up_short_arrow_still_looks_like_an_arrow():
    """Without a clamp the shaft catches up with the (length-capped) head and the whole
    thing reads as one lumpy cone."""
    for scale in (1.0, 2.0, 3.0, 6.0, 20.0):
        m = _arrow([0, 0, 0], [500, 0, 0], RED, 40.0, scale=scale)
        shaft, head = _shaft_and_head(m)
        assert head > 1.5 * shaft, f"scale={scale}: shaft {shaft} vs head {head}"
        assert m.is_watertight


def test_arrow_scale_reaches_through_the_public_path():
    prims = _prims(lines=[[[0, 0, 0], [4000, 0, 0]]])
    a = annotations_to_mesh(prims, [1, 0, 0], 80, 40, arrows=True, arrow_scale=1.0)
    b = annotations_to_mesh(prims, [1, 0, 0], 80, 40, arrows=True, arrow_scale=3.0)
    assert _shaft_and_head(b)[0] > _shaft_and_head(a)[0]


def test_degenerate_arrow_is_skipped_not_crashed():
    assert _arrow([5, 5, 5], [5, 5, 5], RED, 40.0) is None


# -------------------------------------------------------- through the public path
def test_arrows_flag_changes_the_line_geometry():
    prims = _prims(lines=[[[0, 0, 0], [1000, 0, 0]]])
    tube = annotations_to_mesh(prims, [1, 0, 0], 80, 40, arrows=False)
    arrow = annotations_to_mesh(prims, [1, 0, 0], 80, 40, arrows=True)
    assert tube is not None and arrow is not None
    assert len(arrow.vertices) != len(tube.vertices)
    assert arrow.is_watertight            # the plain tube is an open cylinder
    # the arrow is wider than the 40 nm tube (head radius), same span along the line
    assert (arrow.bounds[1][1] - arrow.bounds[0][1]) > (tube.bounds[1][1] - tube.bounds[0][1])


def test_boxes_stay_tubes_in_an_arrow_layer():
    """Only LINES become arrows; a bounding box is still a wireframe."""
    prims = _prims(boxes=[[[0, 0, 0], [500, 500, 500]]])
    a = annotations_to_mesh(prims, [1, 0, 0], 80, 40, arrows=True)
    b = annotations_to_mesh(prims, [1, 0, 0], 80, 40, arrows=False)
    assert len(a.vertices) == len(b.vertices)


def test_per_annotation_shader_colors_reach_the_arrows():
    prims = _prims(lines=[[[0, 0, 0], [1000, 0, 0]], [[0, 0, 0], [0, 1000, 0]]])
    styles = {"line": np.array([[10, 20, 30, 255], [200, 210, 220, 255]], dtype=np.uint8)}
    m = annotations_to_mesh(prims, [1, 0, 0], 80, 40, styles=styles, arrows=True)
    got = {tuple(c) for c in np.unique(m.visual.vertex_colors, axis=0)}
    assert got == {(10, 20, 30, 255), (200, 210, 220, 255)}


# ------------------------------------------------------------- layer selection
def _worker(arrow_layers=None, scale=1.0):
    p = Project(id="p", name="t", arrow_layers=arrow_layers or [],
                keyframes=[Keyframe(id="k", camera=Camera(position_nm=[0, 0, 1],
                                                          look_at_nm=[0, 0, 0]))])
    p.arrow_scale = scale        # bypass validation so the clamp itself is under test
    return RenderWorker(p, RenderJob(id="j", settings=RenderSettings(draft=True)))


def _an(name):
    return AnnotationInstance(name=name, lines=[[[0, 0, 0], [1000, 0, 0]]])


def test_name_convention_is_the_default():
    w = _worker()
    assert w._ann_arrows(_an("arrows"))
    assert w._ann_arrows(_an("pore arrow 2"))
    assert not w._ann_arrows(_an("landmarks"))


def test_explicit_list_overrides_the_convention():
    w = _worker(["landmarks"])
    assert w._ann_arrows(_an("landmarks"))
    assert not w._ann_arrows(_an("arrows"))   # listed layers are the whole answer


def test_arrow_flag_keys_the_asset_cache():
    """Same geometry, different rendering -> different cached npz, or an arrow layer
    would silently reuse the tube asset built before the layer was marked."""
    plain, marked = _worker(), _worker(["landmarks"])
    an = _an("landmarks")
    assert plain._ann_uid(an) != marked._ann_uid(an)


def test_arrow_scale_keys_the_asset_cache():
    """Re-scaling must rebuild the mesh, not reuse the thin one already on disk."""
    thin, bold = _worker(["arrows"]), _worker(["arrows"], scale=3.0)
    an = _an("arrows")
    assert thin._arrow_scale() == 1.0 and bold._arrow_scale() == 3.0
    assert thin._ann_uid(an) != bold._ann_uid(an)


def test_arrow_scale_is_clamped_to_something_buildable():
    assert _worker(scale=0.01)._arrow_scale() == 0.05      # floored, not invisible
    assert _worker(scale=1e9)._arrow_scale() == 20.0       # capped
    assert _worker(scale=0.0)._arrow_scale() == 1.0        # 0 / unset => the default
    assert _worker(scale=None)._arrow_scale() == 1.0
    assert _worker(scale="oops")._arrow_scale() == 1.0


# ------------------------------------------------- constant on-screen arrow size
def _frame(look_at=(0, 0, 0), dist=10000.0, fov=45.0):
    from cinemap.render.interpolate import FrameState
    la = list(look_at)
    return FrameState(position_nm=[la[0], la[1], la[2] - dist], look_at_nm=la,
                      fov_deg=fov, up=[0, 1, 0])


def _arrow_ann(length=30000.0):
    return AnnotationInstance(name="arrows",
                              lines=[[[0.0, -length, 0.0], [0.0, 0.0, 0.0]]])


def test_screen_lock_is_on_by_default():
    """One physical size can't serve a wide shot and a 3 um close-up, so new projects
    lock arrows to the view; 0 opts back out to literal nm geometry."""
    from cinemap.models import Project
    assert Project.model_fields["arrow_screen_frac"].default == 0.25
    assert _worker(["arrows"])._arrow_screen_lock(_arrow_ann(), _frame()) != {}
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.0
    assert w._arrow_screen_lock(_arrow_ann(), _frame()) == {}


def test_screen_locked_arrow_holds_its_share_of_the_frame():
    """Same apparent size at every zoom: scale x length == frac x frame height."""
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    an = _arrow_ann(30000.0)
    for dist in (2000.0, 20000.0, 400000.0):
        fr = _frame(dist=dist)
        frame_h = 2 * dist * math.tan(math.radians(fr.fov_deg) / 2)
        lock = w._arrow_screen_lock(an, fr)
        assert lock["scale"] * 30000.0 == pytest.approx(0.5 * frame_h, rel=1e-9)


def test_screen_lock_pivots_on_the_tip():
    """The point must stay on the structure while the arrow resizes."""
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    tip = [1000.0, 2000.0, 3000.0]
    an = AnnotationInstance(name="arrows", lines=[[[1000.0, -28000.0, 3000.0], tip]])
    lock = w._arrow_screen_lock(an, _frame(look_at=tip))
    assert lock["pivot_bu"] == [t / w.nm_per_bu for t in tip]


def test_screen_lock_only_applies_to_arrow_layers():
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    plain = AnnotationInstance(name="landmarks", lines=[[[0, -100, 0], [0, 0, 0]]])
    assert w._arrow_screen_lock(plain, _frame()) == {}
    assert w._arrow_screen_lock(_arrow_ann(), _frame()) != {}


def test_screen_lock_ignores_a_degenerate_arrow():
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    zero = AnnotationInstance(name="arrows", lines=[[[5, 5, 5], [5, 5, 5]]])
    assert w._arrow_screen_lock(zero, _frame()) == {}
    assert w._arrow_screen_lock(AnnotationInstance(name="arrows"), _frame()) == {}


def test_screen_lock_measures_the_frame_at_the_arrow_not_at_the_target():
    """Apparent size goes as 1/depth. An arrow floating halfway between the camera and
    the thing it points past must be scaled DOWN, or it swallows the frame."""
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    fr = _frame(dist=10000.0)                    # camera at z = -10000, target at origin
    at_target = AnnotationInstance(name="arrows", lines=[[[0.0, -30000.0, 0.0],
                                                          [0.0, 0.0, 0.0]]])
    near_lens = AnnotationInstance(name="arrows", lines=[[[0.0, -30000.0, -5000.0],
                                                          [0.0, 0.0, -5000.0]]])
    assert w._arrow_screen_lock(near_lens, fr)["scale"] == pytest.approx(
        0.5 * w._arrow_screen_lock(at_target, fr)["scale"])


def test_two_arrows_at_different_depths_read_the_same_size_on_screen():
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.3
    fr = _frame(dist=40000.0)
    for z in (-30000.0, 0.0, 30000.0):
        an = AnnotationInstance(name="arrows", lines=[[[0.0, -8000.0, z], [0.0, 0.0, z]]])
        depth = 40000.0 + z                       # distance from the camera along +z
        frame_h = 2 * depth * math.tan(math.radians(fr.fov_deg) / 2)
        assert w._arrow_screen_lock(an, fr)["scale"] * 8000.0 == pytest.approx(
            0.3 * frame_h, rel=1e-9)


def test_a_tip_behind_the_camera_does_not_explode():
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    behind = AnnotationInstance(name="arrows", lines=[[[0, -100, -30000], [0, 0, -30000]]])
    lock = w._arrow_screen_lock(behind, _frame(dist=10000.0))
    assert 0.0 < lock["scale"] < 1e6


def test_a_layer_can_ask_for_its_own_on_screen_size():
    """The wide shot that reveals all four callouts wants them smaller than the close-up
    that shows one."""
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    small = AnnotationInstance(name="arrows", screen_frac=0.1,
                               lines=[[[0.0, -30000.0, 0.0], [0.0, 0.0, 0.0]]])
    assert w._arrow_screen_lock(small, _frame())["scale"] == pytest.approx(
        0.2 * w._arrow_screen_lock(_arrow_ann(), _frame())["scale"])


def test_a_layer_can_opt_out_of_the_lock_on_its_own():
    w = _worker(["arrows"])
    w.project.arrow_screen_frac = 0.5
    literal = AnnotationInstance(name="arrows", screen_frac=0.0,
                                 lines=[[[0.0, -30000.0, 0.0], [0.0, 0.0, 0.0]]])
    assert w._arrow_screen_lock(literal, _frame()) == {}
