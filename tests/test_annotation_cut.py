"""Per-layer annotation transitions, and the EM plane's own arrival/departure.

A callout arrow is a graphic, not scenery. Dissolving one arrow into the next reads as
a mistake, so an annotation layer can cut while everything around it cross-fades. The
same argument applies to the EM cross-section leaving for good: em_transition="cut"
means the plane is present or absent, never a ghost.
"""
from __future__ import annotations

import pytest

from cinemap.models import AnnotationInstance, Camera, Keyframe, SlicePlane
from cinemap.render.interpolate import state_at_time


def _kf(id_, anns=(), slices=(), dur=2.0, **kw):
    return Keyframe(id=id_, camera=Camera(position_nm=[0, 0, 1000], look_at_nm=[0, 0, 0]),
                    annotations=list(anns), slices=list(slices), duration_in_s=dur, **kw)


def _an(name, visible=True, **kw):
    return AnnotationInstance(name=name, lines=[[[0, 0, 0], [1000, 0, 0]]],
                              visible=visible, **kw)


def _op(kfs, t, name):
    fs = state_at_time(kfs, t)
    return next((a.opacity for a in fs.annotations if a.name == name), None)


def test_annotation_without_an_override_still_fades():
    kfs = [_kf("a", [_an("x")]), _kf("b", [_an("x", visible=False)])]
    assert _op(kfs, 1.0, "x") == pytest.approx(0.5)


def test_an_override_cut_lands_on_the_destination_keyframe():
    """A callout swap belongs to the shot it introduces: the change happens when you get
    there, not four seconds early while the camera is still travelling."""
    on, off = _an("x", layer_transition="cut"), _an("x", visible=False,
                                                    layer_transition="cut")
    kfs = [_kf("a", [on]), _kf("b", [off])]
    assert _op(kfs, 0.0, "x") == 1.0
    assert _op(kfs, 1.9, "x") == 1.0        # held, at full strength, through the move
    assert _op(kfs, 2.0, "x") == 0.0        # ... and off on arrival


def test_an_override_cut_point_is_configurable():
    kfs = [_kf("a", [_an("x", layer_transition="cut", layer_transition_at=0.75)]),
           _kf("b", [_an("x", visible=False, layer_transition="cut",
                         layer_transition_at=0.75)])]
    assert _op(kfs, 1.0, "x") == 1.0        # 50% through the 2 s move
    assert _op(kfs, 1.6, "x") == 0.0        # 80%


def test_a_layer_appearing_mid_movie_pops_on_when_it_arrives():
    kfs = [_kf("a"), _kf("b", [_an("x", layer_transition="cut")])]
    assert _op(kfs, 1.9, "x") == 0.0
    assert _op(kfs, 2.0, "x") == 1.0


def test_a_layer_leaving_is_gone_as_the_camera_moves_off():
    """The mirror image: what a shot introduced does not ride along into the next move."""
    kfs = [_kf("a", [_an("x", layer_transition="cut")], hold_in_s=1.0), _kf("b")]
    assert _op(kfs, 0.5, "x") == 1.0
    assert _op(kfs, 1.0, "x") == 0.0


def test_the_override_is_per_layer_not_per_keyframe():
    """One arrow cuts while the scenery annotation beside it keeps fading."""
    kfs = [_kf("a", [_an("arrow", layer_transition="cut"), _an("mesh outline")]),
           _kf("b", [_an("arrow", visible=False, layer_transition="cut"),
                     _an("mesh outline", visible=False)])]
    assert _op(kfs, 1.0, "arrow") == 1.0                      # cut: no half-arrow
    assert _op(kfs, 2.0, "arrow") == 0.0
    assert _op(kfs, 1.0, "mesh outline") == pytest.approx(0.5)  # fade: halfway out


def test_cut_annotations_do_not_drag_the_meshes_or_slices_with_them():
    sl = SlicePlane(em_name="em", axis="z", position_nm=0.0)
    kfs = [_kf("a", [_an("arrow", layer_transition="cut")], [sl]),
           _kf("b", [_an("arrow", visible=False, layer_transition="cut")],
               [SlicePlane(em_name="em", axis="z", position_nm=1000.0)])]
    fs = state_at_time(kfs, 1.0)
    assert fs.slices[0].position_nm == pytest.approx(500.0)   # still gliding


# --------------------------------------------------------------- EM departure
def _slice_op(kfs, t):
    fs = state_at_time(kfs, t)
    return fs.slices[0].opacity if fs.slices else None


def test_a_departing_em_plane_fades_by_default():
    kfs = [_kf("a", slices=[SlicePlane(em_name="em", axis="z", position_nm=0.0)]),
           _kf("b")]
    assert _slice_op(kfs, 1.0) == pytest.approx(0.5)


def test_em_cut_turns_the_plane_off_instead_of_dissolving_it():
    """Scan, then rotate away: the plane is gone the moment the camera moves, with no
    setup beyond choosing "cut" — a departing plane needs no explicit cut point."""
    kfs = [_kf("a", hold_in_s=1.0,
               slices=[SlicePlane(em_name="em", axis="z", position_nm=0.0)]),
           _kf("b", em_transition="cut")]
    assert _slice_op(kfs, 0.5) == 1.0        # the shot itself keeps its cross-section
    assert _slice_op(kfs, 1.0) == 0.0        # gone as the camera leaves, no dissolve
    assert _slice_op(kfs, 2.9) == 0.0


def test_em_cut_can_hold_the_plane_to_the_end_of_the_move():
    """em_transition_at overrides the default, and does NOT ride on layer_transition_at:
    killing the cross-section for a rotation is a different edit from fading meshes."""
    kfs = [_kf("a", slices=[SlicePlane(em_name="em", axis="z", position_nm=0.0)]),
           _kf("b", em_transition="cut", em_transition_at=1.0, layer_transition_at=0.0)]
    assert _slice_op(kfs, 1.9) == 1.0


def test_em_cut_pops_an_arriving_plane_on_at_its_keyframe():
    """Turning the EM on at the start of the move put a cross-section on screen four
    seconds before the shot that wanted it."""
    kfs = [_kf("a"), _kf("b", em_transition="cut",
                         slices=[SlicePlane(em_name="em", axis="z", position_nm=0.0)])]
    assert _slice_op(kfs, 1.9) == 0.0
    assert _slice_op(kfs, 2.0) == 1.0        # on, full strength, no ramp


def test_em_cut_between_two_planes_still_switches_at_the_keyframe():
    """A plane-to-plane cut is not an appearance: it keeps the keyframe's own timing, so
    the destination plane arrives with the destination shot."""
    kfs = [_kf("a", slices=[SlicePlane(em_name="em", axis="z", position_nm=0.0)]),
           _kf("b", em_transition="cut",
               slices=[SlicePlane(em_name="em", axis="z", position_nm=900.0)])]
    assert state_at_time(kfs, 1.0).slices[0].position_nm == 0.0     # source held
    assert state_at_time(kfs, 2.0).slices[0].position_nm == 900.0


def test_a_freshly_scouted_arrow_layer_cuts_without_being_configured():
    """Re-scouting a shot must not silently reintroduce cross-fading arrows."""
    from cinemap.models import Manifest, Project
    from cinemap.scouting import _annotations_from_view
    st = {"dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]},
          "layers": [{"type": "annotation", "name": "pore1 arrow",
                      "annotations": [{"type": "line", "pointA": [0, 0, 0],
                                       "pointB": [10, 0, 0]}]},
                     {"type": "annotation", "name": "landmarks",
                      "annotations": [{"type": "line", "pointA": [0, 0, 0],
                                       "pointB": [10, 0, 0]}]}]}
    p = Project(id="p", name="t", manifest=Manifest(voxel_size_nm=[8, 8, 8]))
    got = {a.name: a.layer_transition for a in _annotations_from_view(p, st)}
    assert got == {"pore1 arrow": "cut", "landmarks": None}


def test_baking_with_cross_sections_hidden_in_3d_bakes_no_em_plane():
    """"Show cross sections in 3-d" off means the user is looking at meshes alone; baking
    the plane anyway put an EM sheet in the render that was nowhere on their screen."""
    from cinemap.models import EMSource, Manifest, Project
    from cinemap.scouting import _scene_from_view
    st = {"dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]},
          "position": [100, 100, 100], "layout": "3d", "showSlices": False,
          "projectionOrientation": [0, 0, 0, 1],
          "layers": [{"type": "image", "name": "em", "visible": True}]}
    p = Project(id="p", name="t",
                manifest=Manifest(voxel_size_nm=[8, 8, 8], em=EMSource(name="em", zarr_url="mem://em")))
    assert _scene_from_view(p, dict(st))[1] == []
    st["showSlices"] = True
    assert len(_scene_from_view(p, st)[1]) == 1        # the default is unchanged
    del st["showSlices"]
    assert len(_scene_from_view(p, dict(st))[1]) == 1  # ... and so is a state without it
