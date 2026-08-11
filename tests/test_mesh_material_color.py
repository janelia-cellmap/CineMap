from types import SimpleNamespace

from cinemap.data.colors import LayerColors
from cinemap.render.worker import RenderWorker


def test_mesh_material_color_prefers_ng_default_for_multi_id_layer():
    lc = LayerColors(default=[0.7372549019607844, 0.7019607843137254, 0.7019607843137254])
    mesh = SimpleNamespace(segment_ids=[1, 2, 3], color=[0.91, 0.45, 0.23])

    assert RenderWorker._mesh_material_color(mesh, lc) == lc.default


def test_mesh_material_color_uses_fallback_for_multi_id_layer_without_default():
    lc = LayerColors(seed=0)
    mesh = SimpleNamespace(segment_ids=[1, 2, 3], color=[0.91, 0.45, 0.23])

    assert RenderWorker._mesh_material_color(mesh, lc) == mesh.color


def test_mesh_material_color_uses_segment_color_for_single_id_without_default():
    lc = LayerColors(seed=0)
    mesh = SimpleNamespace(segment_ids=[1], color=[0.91, 0.45, 0.23])

    assert RenderWorker._mesh_material_color(mesh, lc) == list(lc.rgb(1))


def test_layer_colors_saturation_matches_neuroglancer_white_mix():
    lc = LayerColors(default=[0.2, 0.4, 0.8], saturation=0.25)

    assert lc.rgb(1) == (0.8, 0.85, 0.95)
