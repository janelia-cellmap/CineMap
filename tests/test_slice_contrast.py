import numpy as np

from cinemap.render.worker import _apply_contrast_window
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
