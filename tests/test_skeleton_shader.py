import numpy as np

from cinemap.render.worker import RenderWorker
from cinemap.data.skeleton import ExpressionShaderColorizer, parse_shader_colormap


MOSQUITO_SHADER = """
#uicontrol float minRadiusNm slider(min=1.0, max=300.0, step=1.0, default=30.0)
#uicontrol float maxRadiusNm slider(min=100.0, max=3000.0, step=10.0, default=1300.0)

vec3 turbo(float x) { return vec3(x, x, x); }

void main() {
  float radiusNm = max(prop_radius(), 0.001);
  float lo = log(max(minRadiusNm, 0.001));
  float hi = log(max(maxRadiusNm, minRadiusNm + 0.001));
  float t = clamp((log(radiusNm) - lo) / (hi - lo), 0.0, 1.0);
  emitRGB(turbo(t));
}
"""


def test_parse_turbo_radius_skeleton_shader():
    cm = parse_shader_colormap(MOSQUITO_SHADER)

    assert isinstance(cm, ExpressionShaderColorizer)
    assert cm.controls["minRadiusNm"] == 30.0
    assert cm.controls["maxRadiusNm"] == 1300.0


def test_turbo_radius_shader_uses_skeleton_radius_attribute():
    cm = parse_shader_colormap(MOSQUITO_SHADER)
    skel = type("Skel", (), {"radius": np.array([30.0, 1300.0])})()

    rgba = cm.edge_rgba(skel, np.array([[0, 1]], dtype=np.int64))

    assert rgba.shape == (1, 4)
    assert rgba[0, 3] == 255
    assert rgba[0, :3].max() > 0


def test_expression_shader_supports_non_radius_prop_formula():
    shader = """
    #uicontrol float norm slider(min=1.0, max=20.0, step=1.0, default=10.0)
    void main() {
      float x = clamp(prop_score() / norm, 0.0, 1.0);
      emitRGB(vec3(x, 0.25, 1.0 - x));
    }
    """
    cm = parse_shader_colormap(shader)
    skel = type("Skel", (), {"score": np.array([0.0, 10.0, 20.0])})()

    rgba = cm.edge_rgba(skel, np.array([[0, 1], [1, 2]], dtype=np.int64))

    assert rgba.shape == (2, 4)
    assert rgba[0, 0] < rgba[1, 0]
    assert rgba[0, 2] > rgba[1, 2]
    assert np.all(rgba[:, 3] == 255)


def test_skeleton_shader_participates_in_mesh_cache_key():
    worker = object.__new__(RenderWorker)
    worker._draft = True
    worker._prefer_labels = False
    worker._mesh_budget = 1
    worker._label_smooth_iters = 0
    worker._label_decimate_fraction = 0.0
    worker._label_blockwise = "auto"
    worker.manifest = type(
        "Manifest",
        (),
        {
            "meshes": [
                type(
                    "MeshSource",
                    (),
                    {
                        "name": "pruned",
                        "skeleton_url": "https://example.test/skeletons",
                        "mesh_url": "",
                        "label_zarr": "",
                        "skeleton_shader": MOSQUITO_SHADER,
                    },
                )()
            ]
        },
    )()

    uid_a = worker._mesh_uid("pruned", [16])
    worker.manifest.meshes[0].skeleton_shader = MOSQUITO_SHADER + "\n// changed"
    uid_b = worker._mesh_uid("pruned", [16])

    assert uid_a != uid_b
    assert worker._mesh_uses_shader_vertex_colors("pruned") is True
