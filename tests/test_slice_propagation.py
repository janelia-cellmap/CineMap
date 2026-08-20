"""Slice-layer (image/EM) appearance diffing and propagation.

Contrast set in neuroglancer used to be captured onto one keyframe and then stop there,
because only mesh layers were diffed and propagatable. These cover the slice path.
"""
from __future__ import annotations

import pytest

from cinemap import operations as ops
from cinemap.models import Camera, Keyframe, SlicePlane


def _kf(kid: str, **slice_kw) -> Keyframe:
    return Keyframe(
        id=kid,
        camera=Camera(position_nm=[0, 0, 100], look_at_nm=[0, 0, 0]),
        slices=[SlicePlane(em_name="em", axis="z", **slice_kw)],
    )


def _project(tmp_path, kfs):
    """A saved Project; propagation persists through ops.store, so give it a real dir."""
    from cinemap import store as store_mod
    from cinemap.models import Project

    store_mod.PROJECTS_DIR = tmp_path
    ops.store.PROJECTS_DIR = tmp_path
    p = Project(id="p1", name="t", keyframes=kfs)
    ops.store.save(p)
    return p


def _ctrl(lo, hi):
    return {"normalized": {"range": [lo, hi]}}


# ------------------------------------------------------------------------- diffing
def test_contrast_change_is_detected():
    old = _kf("k1", shader_controls=_ctrl(0, 255)).slices
    new = _kf("k1", shader_controls=_ctrl(40, 180)).slices
    changes = ops.diff_slice_settings(old, new)
    assert len(changes) == 1
    c = changes[0]
    assert c["mesh_name"] == "em"
    assert c["field"] == "shader_controls"
    assert c["new"] == _ctrl(40, 180)
    assert c["layer_kind"] == "slice"


def test_opacity_change_is_detected():
    changes = ops.diff_slice_settings(_kf("k", opacity=1.0).slices,
                                      _kf("k", opacity=0.4).slices)
    assert [c["field"] for c in changes] == ["opacity"]


def test_no_change_reports_nothing():
    a = _kf("k", opacity=0.5, shader_controls=_ctrl(10, 20)).slices
    b = _kf("k", opacity=0.5, shader_controls=_ctrl(10, 20)).slices
    assert ops.diff_slice_settings(a, b) == []


def test_layer_absent_from_old_is_skipped():
    assert ops.diff_slice_settings([], _kf("k", opacity=0.3).slices) == []


# --------------------------------------------------------------------- propagation
def test_propagate_right_updates_this_and_later(tmp_path):
    kfs = [_kf(f"k{i}", shader_controls=_ctrl(0, 255)) for i in range(4)]
    p = _project(tmp_path, kfs)
    res = ops.propagate_slice_field(p, "k1", "em", "shader_controls", _ctrl(40, 180),
                                    direction="right")
    assert res["changed"] == ["k1", "k2", "k3"]
    got = [k.slices[0].shader_controls for k in p.keyframes]
    assert got == [_ctrl(0, 255), _ctrl(40, 180), _ctrl(40, 180), _ctrl(40, 180)]


def test_match_old_protects_deliberately_different_keyframes(tmp_path):
    kfs = [_kf("k0", shader_controls=_ctrl(0, 255)),
           _kf("k1", shader_controls=_ctrl(0, 255)),
           _kf("k2", shader_controls=_ctrl(90, 200)),   # deliberately different
           _kf("k3", shader_controls=_ctrl(0, 255))]
    p = _project(tmp_path, kfs)
    res = ops.propagate_slice_field(p, "k1", "em", "shader_controls", _ctrl(40, 180),
                                    direction="right", match_old=True)
    assert "k2" not in res["changed"]
    assert p.keyframes[2].slices[0].shader_controls == _ctrl(90, 200)
    assert p.keyframes[3].slices[0].shader_controls == _ctrl(40, 180)


def test_match_old_false_overwrites_everything(tmp_path):
    kfs = [_kf("k0", shader_controls=_ctrl(0, 255)),
           _kf("k1", shader_controls=_ctrl(90, 200))]
    p = _project(tmp_path, kfs)
    ops.propagate_slice_field(p, "k0", "em", "shader_controls", _ctrl(1, 2),
                              direction="all", match_old=False)
    assert all(k.slices[0].shader_controls == _ctrl(1, 2) for k in p.keyframes)


def test_propagate_this_only(tmp_path):
    p = _project(tmp_path, [_kf("k0", opacity=1.0), _kf("k1", opacity=1.0)])
    res = ops.propagate_slice_field(p, "k0", "em", "opacity", 0.25, direction="this")
    assert res["changed"] == ["k0"]
    assert p.keyframes[1].slices[0].opacity == 1.0


def test_propagate_left(tmp_path):
    p = _project(tmp_path, [_kf(f"k{i}", opacity=1.0) for i in range(3)])
    res = ops.propagate_slice_field(p, "k1", "em", "opacity", 0.5, direction="left")
    assert res["changed"] == ["k0", "k1"]
    assert p.keyframes[2].slices[0].opacity == 1.0


def test_source_keyframe_always_changes_even_if_it_would_not_match(tmp_path):
    """The source is the edit the user just made; match_old must not veto it."""
    p = _project(tmp_path, [_kf("k0", opacity=0.9), _kf("k1", opacity=0.1)])
    res = ops.propagate_slice_field(p, "k0", "em", "opacity", 0.3, direction="right")
    assert "k0" in res["changed"]
    assert p.keyframes[0].slices[0].opacity == pytest.approx(0.3)


def test_rejects_non_propagatable_field(tmp_path):
    p = _project(tmp_path, [_kf("k0")])
    with pytest.raises(ValueError, match="not propagatable"):
        ops.propagate_slice_field(p, "k0", "em", "axis", "x")


def test_unknown_layer_raises(tmp_path):
    p = _project(tmp_path, [_kf("k0")])
    with pytest.raises(ValueError, match="not in keyframe"):
        ops.propagate_slice_field(p, "k0", "nope", "opacity", 0.5)


def test_keyframes_without_the_layer_are_skipped(tmp_path):
    bare = Keyframe(id="k1", camera=Camera(position_nm=[0, 0, 1], look_at_nm=[0, 0, 0]))
    p = _project(tmp_path, [_kf("k0", opacity=1.0), bare, _kf("k2", opacity=1.0)])
    res = ops.propagate_slice_field(p, "k0", "em", "opacity", 0.2, direction="right")
    assert res["changed"] == ["k0", "k2"]
