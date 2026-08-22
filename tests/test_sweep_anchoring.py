"""Sweeps stay lined up with the shot they were built for.

A sweep is timed in absolute seconds, but it exists to happen at a MOMENT ("scan while
we're on the pore"). Editing keyframes moves that moment, so sweeps are pinned to a
keyframe and their start is re-derived. EM slices / mesh layers need no such thing —
they're stored on the keyframe itself and travel with it.
"""
from __future__ import annotations

import pytest

from cinemap import operations as ops
from cinemap.models import Camera, Keyframe, Project


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path):
    from cinemap import store as store_mod
    store_mod.PROJECTS_DIR = tmp_path
    ops.store.PROJECTS_DIR = tmp_path


def _kf(kid, x=0.0, dur=2.0, hold=0.0):
    return Keyframe(id=kid, camera=Camera(position_nm=[x, 0, 100], look_at_nm=[x, 0, 0]),
                    duration_in_s=dur, hold_in_s=hold)


def _project(n=4, **kw):
    kfs = [_kf(f"k{i}", x=100.0 * i, **kw) for i in range(n)]
    p = Project(id="p", name="t", keyframes=kfs)
    ops.store.save(p)
    return p


def _sweep(p, start_s, **kw):
    return ops.add_sweep(p, kind="slice", em_name="em", axis="z",
                         from_nm=0.0, to_nm=100.0, start_s=start_s, duration_s=1.0, **kw)


# --------------------------------------------------------------- arrival times
def test_keyframe_times_matches_the_renderer_walk():
    p = _project(3, dur=2.0)
    p.keyframes[0].hold_in_s = 1.0
    # kf0 at 0, hold 1s, then 2s move -> kf1 at 3, then 2s -> kf2 at 5
    assert ops.keyframe_times(p.keyframes) == [0.0, 3.0, 5.0]


def test_keyframe_times_agrees_with_build_frames():
    from cinemap.render.interpolate import build_frames
    p = _project(4, dur=2.0)
    p.keyframes[1].hold_in_s = 1.5
    fps = 20
    total = len(build_frames(p.keyframes, fps)) - 1     # the trailing single frame
    assert abs(ops.keyframe_times(p.keyframes)[-1] * fps - total) < 1.5


# ------------------------------------------------------------------- anchoring
def test_sweep_is_pinned_to_the_keyframe_it_starts_on():
    p = _project(4)                       # arrivals: 0, 2, 4, 6
    sw = _sweep(p, start_s=4.3)
    assert sw.anchor_kf == "k2"
    assert sw.anchor_offset_s == pytest.approx(0.3)


def test_inserting_a_keyframe_before_a_sweep_pushes_it_later():
    p = _project(4)
    sw = _sweep(p, start_s=4.3)           # 0.3s after landing on k2
    ops.duplicate_keyframe(p, "k0", after_id="k0")   # inserts a shot with its own 2s move
    ops.update_keyframe(p, "k1", duration_in_s=5.0)  # k1 now arrives 3s later still
    assert p.sweeps[0].start_s == pytest.approx(9.3)
    assert p.sweeps[0].anchor_kf == "k2"             # still the same shot


def test_deleting_a_keyframe_pulls_later_sweeps_back():
    p = _project(4)
    sw = _sweep(p, start_s=4.3)
    ops.delete_keyframe(p, "k1")          # removes a 2s move ahead of the sweep
    assert p.sweeps[0].start_s == pytest.approx(2.3)
    assert p.sweeps[0].anchor_kf == "k2"


def test_deleting_the_anchor_keyframe_rebases_onto_its_neighbour():
    p = _project(4)
    _sweep(p, start_s=4.3)
    ops.delete_keyframe(p, "k2")
    sw = p.sweeps[0]
    assert sw.anchor_kf == "k1"
    # k1 arrives at 2.0; the sweep kept its 2.3s distance from k1 -> 4.3, then the deleted
    # move no longer exists, so it sits where the following content now is
    assert sw.start_s == pytest.approx(4.3)
    assert sw.anchor_offset_s == pytest.approx(2.3)


def test_retiming_a_move_carries_later_sweeps():
    p = _project(4)
    _sweep(p, start_s=6.0)                # on k3
    ops.update_keyframe(p, "k2", duration_in_s=10.0)
    assert p.sweeps[0].start_s == pytest.approx(14.0)


def test_reordering_keyframes_moves_the_sweep_with_its_shot():
    p = _project(4)
    _sweep(p, start_s=6.0)                # pinned to k3, the last shot
    ops.reorder_keyframes(p, ["k0", "k3", "k1", "k2"])   # k3 becomes the second shot
    assert p.sweeps[0].anchor_kf == "k3"
    assert p.sweeps[0].start_s == pytest.approx(2.0)


def test_a_sweep_before_the_first_arrival_anchors_to_keyframe_zero():
    p = _project(4)
    sw = _sweep(p, start_s=0.0)
    assert sw.anchor_kf == "k0" and sw.anchor_offset_s == pytest.approx(0.0)
    ops.update_keyframe(p, "k1", duration_in_s=9.0)      # later moves don't drag it
    assert p.sweeps[0].start_s == pytest.approx(0.0)


def test_legacy_unanchored_sweep_is_adopted_at_its_current_time():
    """A sweep saved before anchoring existed keeps its timing, then starts tracking."""
    p = _project(4)
    sw = _sweep(p, start_s=4.3)
    sw.anchor_kf = None                   # simulate an old project.json
    sw.anchor_offset_s = 0.0
    ops.store.save(p)
    ops.delete_keyframe(p, "k1")
    assert p.sweeps[0].anchor_kf == "k2"
    assert p.sweeps[0].start_s == pytest.approx(2.3)


def test_dragging_a_sweep_repins_it():
    p = _project(4)
    sw = _sweep(p, start_s=0.5)           # on k0
    ops.update_sweep(p, sw.id, start_s=6.2)
    assert p.sweeps[0].anchor_kf == "k3"
    assert p.sweeps[0].anchor_offset_s == pytest.approx(0.2)


# ------------------------------------------------------ paste / duplicate timing
def test_duplicate_lands_after_the_source():
    p = _project(3)
    dup = ops.duplicate_keyframe(p, "k1", after_id="k1")
    assert [k.id for k in p.keyframes] == ["k0", "k1", dup.id, "k2"]


def test_duplicate_gets_the_default_move_not_the_source_s():
    """The move into a keyframe belongs to the pair it sat between — the copy isn't in
    that pair, so it starts from the default and you retime it."""
    from cinemap.models import Keyframe
    default = Keyframe.model_fields["duration_in_s"].default
    p = _project(4, dur=8.0)
    dup = ops.duplicate_keyframe(p, "k3", after_id="k1")
    assert dup.duration_in_s == default != 8.0
    assert p.keyframes[2].id == dup.id
    assert p.keyframes[3].duration_in_s == 8.0        # the frame it displaced is untouched


def test_duplicate_lands_after_the_card_you_hit_even_with_odd_timings():
    p = _project(5, dur=8.0)
    p.keyframes[2].duration_in_s = 0.0                # a same-instant pair
    for target in ("k0", "k2", "k4"):
        dup = ops.duplicate_keyframe(p, "k3", after_id=target)
        ids = [k.id for k in p.keyframes]
        assert ids[ids.index(target) + 1] == dup.id


def test_copy_labels_stay_distinguishable():
    p = _project(2)
    p.keyframes[1].label = "pore"
    a = ops.duplicate_keyframe(p, "k1", after_id="k1")
    b = ops.duplicate_keyframe(p, a.id, after_id=a.id)
    c = ops.duplicate_keyframe(p, b.id, after_id=b.id)
    assert [a.label, b.label, c.label] == ["pore copy", "pore copy 2", "pore copy 3"]


def test_duplicate_brings_its_pinned_sweep_along():
    p = _project(4)
    sw = _sweep(p, start_s=4.5)                # pinned to k2, +0.5s
    dup = ops.duplicate_keyframe(p, "k2", after_id="k2")
    assert len(p.sweeps) == 2
    twin = next(s for s in p.sweeps if s.id != sw.id)
    assert twin.anchor_kf == dup.id
    assert twin.anchor_offset_s == pytest.approx(0.5)   # same beat within the shot
    assert (twin.kind, twin.em_name, twin.from_nm, twin.to_nm) == \
           (sw.kind, sw.em_name, sw.from_nm, sw.to_nm)
    # the copy sits 2s later (its own move), so its sweep does too
    assert twin.start_s == pytest.approx(sw.start_s + 2.0)


def test_duplicate_can_opt_out_of_copying_sweeps():
    p = _project(4)
    _sweep(p, start_s=4.5)
    ops.duplicate_keyframe(p, "k2", after_id="k2", with_sweeps=False)
    assert len(p.sweeps) == 1


def test_sweeps_of_other_keyframes_are_not_copied():
    p = _project(4)
    _sweep(p, start_s=2.0)                     # on k1
    ops.duplicate_keyframe(p, "k2", after_id="k2")
    assert len(p.sweeps) == 1


# ---------------------------------------------- a render must not revert your edits
def test_patch_render_job_keeps_edits_made_while_rendering():
    """The render process holds a project snapshot from when it started; saving that
    whole snapshot at the end used to revert everything edited during the render."""
    from cinemap import store
    from cinemap.models import RenderJob

    p = _project(3)
    job = RenderJob(id="r1")
    p.renders = [job]
    store.save(p)

    stale = store.load("p").model_copy(deep=True)      # what the subprocess loaded
    ops.duplicate_keyframe(p, "k1", after_id="k1")     # ...meanwhile, the user pastes
    n_after_edit = len(store.load("p").keyframes)

    done = next(j for j in stale.renders if j.id == "r1")
    done.status = "done"
    done.output_path = "/tmp/out.mp4"
    store.patch_render_job("p", done)

    fresh = store.load("p")
    assert len(fresh.keyframes) == n_after_edit         # the paste survived
    assert fresh.renders[0].status == "done"            # and the job record landed
    assert fresh.renders[0].output_path == "/tmp/out.mp4"
