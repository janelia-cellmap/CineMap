import numpy as np

from cinemap.models import Camera, Keyframe, MeshInstance, SlicePlane
from cinemap.data.ng_camera import cross_section_plane, ng_to_cross_section_camera
from cinemap.render.interpolate import build_frames
from cinemap.render.worker import RenderWorker, _apply_contrast_window
from cinemap.scouting import _image_contrast_from_layer


def test_extracts_neuroglancer_shader_window():
    layer = {
        "type": "image",
        "name": "fibsem",
        "shaderControls": {
            "normalized": {
                "range": [0, 1],
                "window": [0.2, 0.8],
            }
        },
    }

    assert _image_contrast_from_layer(layer) == [0.2, 0.8]


def test_extracts_neuroglancer_shader_range_when_window_absent():
    layer = {
        "type": "image",
        "name": "fibsem",
        "shaderControls": {"normalized": {"range": [93, 140]}},
    }

    assert _image_contrast_from_layer(layer) == [93.0, 140.0]


def test_contrast_window_maps_uint8_values_to_display_range():
    image = np.array([[25, 75, 125]], dtype=np.uint8)

    out = _apply_contrast_window(image, [25, 125])

    assert np.allclose(out, [[0, 127.5, 255]])


def test_normalized_contrast_window_maps_to_uint8_source_range():
    image = np.array([[0, 128, 255]], dtype=np.uint8)

    out = _apply_contrast_window(image, [0.25, 0.75])

    assert out[0, 0] < 0
    assert 127 < out[0, 1] < 129
    assert out[0, 2] > 255


def test_normalized_contrast_window_uses_dtype_not_crop_max():
    image = np.array([[0, 1, 2]], dtype=np.uint8)

    out = _apply_contrast_window(image, [0.25, 0.75])

    assert out[0, 0] < 0
    assert out[0, 2] < 0


def test_interpolated_frame_preserves_slice_contrast_limits():
    cam = Camera(position_nm=[0, 0, 10], look_at_nm=[0, 0, 0], fov_deg=45)
    a = Keyframe(
        id="a",
        camera=cam,
        slices=[SlicePlane(em_name="fibsem", axis="z", position_nm=10, contrast_limits=[90, 140])],
    )
    b = Keyframe(
        id="b",
        camera=cam,
        duration_in_s=1.0,
        easing="linear",
        slices=[SlicePlane(em_name="fibsem", axis="z", position_nm=20, contrast_limits=[90, 140])],
    )

    frames = build_frames([a, b], fps=2)

    assert frames[0].slices[0].contrast_limits == [90, 140]
    assert frames[1].slices[0].contrast_limits == [90, 140]


def test_changing_segment_sets_keep_common_segments_continuous():
    cam = Camera(position_nm=[0, 0, 10], look_at_nm=[0, 0, 0], fov_deg=45)
    a = Keyframe(
        id="a",
        camera=cam,
        meshes=[MeshInstance(mesh_name="neurons", segment_ids=[1, 2, 3])],
    )
    b = Keyframe(
        id="b",
        camera=cam,
        duration_in_s=1.0,
        easing="linear",
        meshes=[MeshInstance(mesh_name="neurons", segment_ids=[3])],
    )

    mid = build_frames([a, b], fps=2)[1]
    by_ids = {tuple(m.segment_ids): m.opacity * m.object_alpha for m in mid.meshes}

    assert by_ids[(3,)] == 1.0
    assert by_ids[(1, 2)] == 0.5


def test_cross_section_state_maps_to_orthographic_camera():
    state = {
        "dimensions": {"z": [8e-9, "m"], "y": [8e-9, "m"], "x": [8e-9, "m"]},
        "position": [10, 20, 30],
        "crossSectionScale": 2.0,
        "crossSectionOrientation": [0.0, 0.0, 0.0, 1.0],
        "layout": "4panel-alt",
    }

    cam = ng_to_cross_section_camera(state, [8.0, 8.0, 8.0], viewport_height_px=100)

    assert cam.projection == "ORTHO"
    assert cam.ortho_scale_nm == 1600.0
    assert cam.look_at_nm == [240.0, 160.0, 80.0]


def test_named_slice_layout_selects_matching_panel_axis():
    base = {
        "dimensions": {"z": [8e-9, "m"], "y": [8e-9, "m"], "x": [8e-9, "m"]},
        "position": [10, 20, 30],
        "crossSectionScale": 2.0,
        "crossSectionOrientation": [0.0, 0.0, 0.0, 1.0],
    }

    assert cross_section_plane({**base, "layout": "xy"}, [8.0, 8.0, 8.0])[:2] == ("z", 80.0)
    assert cross_section_plane({**base, "layout": "xz"}, [8.0, 8.0, 8.0])[:2] == ("y", 160.0)
    assert cross_section_plane({**base, "layout": "yz"}, [8.0, 8.0, 8.0])[:2] == ("x", 240.0)


def test_cross_section_camera_uses_captured_viewport_height():
    state = {
        "dimensions": {"z": [8e-9, "m"], "y": [8e-9, "m"], "x": [8e-9, "m"]},
        "position": [10, 20, 30],
        "crossSectionScale": 2.0,
        "layout": "xy",
        "_cinemap_viewport": {"width_px": 800, "height_px": 222},
    }

    cam = ng_to_cross_section_camera(state, [8.0, 8.0, 8.0])

    assert cam.ortho_scale_nm == 3552.0


def test_worker_upgrades_saved_2d_ng_state_to_orthographic_keyframe():
    worker = object.__new__(RenderWorker)
    worker.manifest = type(
        "Manifest",
        (),
        {
            "voxel_size_nm": [8.0, 8.0, 8.0],
            "em": type("EM", (), {"name": "fibsem"})(),
        },
    )()
    state = {
        "dimensions": {"z": [8e-9, "m"], "y": [8e-9, "m"], "x": [8e-9, "m"]},
        "position": [10, 20, 30],
        "crossSectionScale": 2.0,
        "crossSectionOrientation": [0.0, 0.0, 0.0, 1.0],
        "layout": "4panel-alt",
        "layers": [
            {
                "type": "image",
                "name": "fibsem",
                "shaderControls": {"normalized": {"range": [90, 140]}},
            }
        ],
    }
    kf = Keyframe(
        id="old",
        camera=Camera(position_nm=[0, 0, 10], look_at_nm=[0, 0, 0], fov_deg=45),
        ng_state=state,
    )
    worker.project = type("Project", (), {"keyframes": [kf]})()

    upgraded = worker._render_keyframes()[0]

    assert upgraded.camera.projection == "ORTHO"
    assert upgraded.slices[0].contrast_limits == [90.0, 140.0]
