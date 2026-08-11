from pathlib import Path

import pytest

from cinemap.models import Camera, Keyframe, MeshInstance, Project
from cinemap.scouting import _state_from_keyframe, current_layer_colors


def _local_neuroglancer_path(*parts):
    path = Path("/tmp/cinemap-neuroglancer-src", *parts)
    if not path.exists():
        pytest.skip(f"local Neuroglancer checkout unavailable: {path}")
    return path


def test_state_from_keyframe_restores_saturation_and_clears_stale_colors():
    project = Project(id="p", name="p")
    keyframe = Keyframe(
        id="kf",
        camera=Camera(
            position_nm=[0.0, 0.0, 100.0],
            look_at_nm=[0.0, 0.0, 0.0],
            fov_deg=45.0,
            up=[0.0, 1.0, 0.0],
        ),
        meshes=[
            MeshInstance(
                mesh_name="seg",
                segment_ids=[1, 2],
                color_seed=7,
                saturation=0.25,
                object_alpha=0.5,
                silhouette=2.0,
            )
        ],
    )
    base = {
        "layers": [
            {
                "type": "segmentation",
                "name": "seg",
                "segments": ["9"],
                "segmentDefaultColor": "#ff0000",
                "segmentColors": {"9": "#00ff00"},
            }
        ]
    }

    state = _state_from_keyframe(project, keyframe, base)
    layer = state["layers"][0]

    assert layer["segments"] == ["1", "2"]
    assert layer["colorSeed"] == 7
    assert layer["saturation"] == 0.25
    assert layer["objectAlpha"] == 0.5
    assert layer["meshSilhouetteRendering"] == 2.0
    assert "segmentDefaultColor" not in layer
    assert "segmentColors" not in layer


def test_current_layer_colors_follow_linked_segmentation_color_group():
    project = Project(id="p", name="p")
    state = {
        "layers": [
            {
                "type": "segmentation",
                "name": "root",
                "segmentDefaultColor": "#336699",
                "saturation": 0.5,
            },
            {
                "type": "segmentation",
                "name": "child",
                "linkedSegmentationGroup": "root",
                "segmentDefaultColor": "#ff0000",
                "saturation": 1.0,
            },
        ]
    }

    colors = current_layer_colors(project, state)

    assert colors["child"].default == colors["root"].default
    assert colors["child"].saturation == colors["root"].saturation


def test_current_layer_colors_explicit_false_keeps_linked_group_colors_local():
    project = Project(id="p", name="p")
    state = {
        "layers": [
            {
                "type": "segmentation",
                "name": "root",
                "segmentDefaultColor": "#336699",
            },
            {
                "type": "segmentation",
                "name": "child",
                "linkedSegmentationGroup": "root",
                "linkedSegmentationColorGroup": False,
                "segmentDefaultColor": "#ff0000",
            },
        ]
    }

    colors = current_layer_colors(project, state)

    assert colors["child"].default == [1.0, 0.0, 0.0]


def test_linked_color_group_default_matches_neuroglancer_source():
    source = _local_neuroglancer_path("src", "layer", "segmentation", "index.ts").read_text()
    keys = _local_neuroglancer_path("src", "layer", "segmentation", "json_keys.ts").read_text()

    assert 'LINKED_SEGMENTATION_COLOR_GROUP_JSON_KEY =\n  "linkedSegmentationColorGroup";' in keys
    assert "const linkedSegmentationColorGroupName = verifyOptionalObjectProperty" in source
    assert "json_keys.LINKED_SEGMENTATION_COLOR_GROUP_JSON_KEY" in source
    assert "linkedSegmentationGroupName," in source
    assert "(x) => (x === false ? undefined : verifyString(x))" in source
