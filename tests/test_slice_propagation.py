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


# ------------------------------------------- 2D (cross-section) layer opacity
def test_slice_opacity_zero_keeps_the_mesh_but_clears_the_cross_section():
    """Neuroglancer's 2D layer opacity is independent of the 3D mesh: a layer set to 0
    should stop painting labels on the EM slice while its geometry still renders."""
    from cinemap.models import (Camera as C, Manifest, MeshInstance, MeshSource, Project,
                                RenderJob, RenderSettings)
    from cinemap.render.interpolate import state_at_time
    from cinemap.render.worker import RenderWorker

    def project(slice_opacity):
        m = MeshInstance(mesh_name="anatomy", segment_ids=[1, 2],
                         slice_opacity=slice_opacity)
        kf = Keyframe(id="k", camera=C(position_nm=[0, 0, 100], look_at_nm=[0, 0, 0]),
                      meshes=[m])
        return Project(id="p", name="t", keyframes=[kf],
                       manifest=Manifest(meshes=[MeshSource(name="anatomy",
                                                            label_zarr="s3://labels/")]))

    for so, expect_overlay, expect_alpha in ((None, True, 0.6), (0.25, True, 0.25),
                                             (0.0, False, None)):
        p = project(so)
        w = RenderWorker(p, RenderJob(id="j", settings=RenderSettings(draft=True)))
        fr = state_at_time(p.keyframes, 0.0)
        overlays = w._frame_seg_overlays(fr)
        assert bool(overlays) is expect_overlay, so
        if expect_overlay:
            assert overlays[0][3] == pytest.approx(expect_alpha)
        # the 3D mesh renders regardless of what the cross-section does
        assert w._mesh_render_alpha(fr.meshes[0]) == pytest.approx(1.0)


def test_slice_overlay_alpha_keys_the_slice_image_cache():
    """The alpha is baked into the PNG, so changing it must not reuse the old image."""
    from cinemap.data.colors import LayerColors
    from cinemap.models import Manifest, Project, RenderJob, RenderSettings
    from cinemap.render.interpolate import FrameSlice
    from cinemap.render.worker import RenderWorker

    p = Project(id="p", name="t", manifest=Manifest())
    w = RenderWorker(p, RenderJob(id="j", settings=RenderSettings(draft=True)))
    sl = FrameSlice("em", "z", 0.0, None, 1.0)
    lc = LayerColors()
    key = lambda a: w._slice_cache_key(sl, ((0, 0, 0), 1000.0), [("s3://l/", [1], lc, a)])
    assert key(0.6) != key(0.25)
