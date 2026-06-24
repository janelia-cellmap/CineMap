import ast
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from cinemap.render import director
from cinemap.render.interpolate import FrameSlice, FrameState
from cinemap.render.ng_oit import OITFragment, composite_oit_over, compute_oit_weight
from cinemap.render.worker import (
    RenderWorker,
    _director_settings_for_render,
    _export_mesh_npz,
    _ng_quantize_decode_normals,
)


def _local_neuvid_path(*parts):
    path = Path("/groups/scicompsoft/home/ackermand/Programming/neuVid", *parts)
    if not path.exists():
        pytest.skip(f"local neuVid file unavailable: {path}")
    return path


def _local_neuroglancer_path(*parts):
    path = Path("/tmp/cinemap-neuroglancer-src", *parts)
    if not path.exists():
        pytest.skip(f"local Neuroglancer checkout unavailable: {path}")
    return path


def _reset_blender(blender_script):
    blender_script._clear()
    blender_script._orig_mesh.clear()
    blender_script._clip_state.clear()
    blender_script._mat_base.clear()
    blender_script._slice_objs = []
    blender_script._fade_overlay = None


def _tiny_mesh_scene(tmp_path, material):
    np = pytest.importorskip("numpy")
    mesh_path = tmp_path / "tiny_mesh.npz"
    np.savez(
        mesh_path,
        v=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        f=np.array([[0, 1, 2]], dtype=np.int32),
    )
    return {
        "world": {"nm_per_bu": 1.0},
        "meshes": [{"id": "tiny", "obj_path": str(mesh_path), "color": [0.25, 0.5, 0.75]}],
        "direction": {"material": material},
    }


def _linked_from(nt, socket):
    links = [link for link in nt.links if link.to_socket == socket]
    assert len(links) == 1
    return links[0].from_node


def _specular_value_input_node(nt, node):
    return next(
        link.from_node
        for link in nt.links
        if link.to_node == node and link.from_node.type == "VALUE" and "spec" in link.from_node.name.lower()
    )


def _local_neuvid_lamp_specs():
    path = _local_neuvid_path("neuVid", "render.py")
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "lampSpecs" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("neuVid render.py did not define lampSpecs")


def _local_neuvid_render_source_blender5_compat():
    """Local neuVid render.py with no-op guards for Blender 5's removed Cycles fields."""
    out_lines = []
    for line in _local_neuvid_path("neuVid", "render.py").read_text().splitlines():
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if stripped == "mat.cycles.use_transparent_shadow = True":
            out_lines.extend([
                f"{indent}try:",
                f"{indent}    mat.cycles.use_transparent_shadow = True",
                f"{indent}except AttributeError:",
                f"{indent}    pass",
            ])
        elif stripped == "lampData.cycles.cast_shadow = True":
            out_lines.extend([
                f"{indent}try:",
                f"{indent}    lampData.cycles.cast_shadow = True",
                f"{indent}except AttributeError:",
                f"{indent}    pass",
            ])
        elif stripped == "lampData.cycles.use_multiple_importance_sampling = True":
            out_lines.extend([
                f"{indent}try:",
                f"{indent}    lampData.cycles.use_multiple_importance_sampling = True",
                f"{indent}except AttributeError:",
                f"{indent}    pass",
            ])
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _write_minimal_neuvid_blend(tmp_path):
    pytest.importorskip("bpy")
    mathutils = pytest.importorskip("mathutils")
    from cinemap.render import blender_script

    neuvid_root = _local_neuvid_path()
    sys.path.insert(0, str(neuvid_root / "neuVid"))
    try:
        from utilsMaterials import newBasicMaterial
    finally:
        try:
            sys.path.remove(str(neuvid_root / "neuVid"))
        except ValueError:
            pass

    _reset_blender(blender_script)
    world = blender_script.bpy.data.worlds.new("World")
    blender_script.bpy.context.scene.world = world
    mesh = blender_script.bpy.data.meshes.new("Neuron.demoMesh")
    mesh.from_pydata(
        [(-2, 0, -2), (2, 0, -2), (2, 0, 2), (-2, 0, 2)],
        [],
        [(0, 2, 1), (0, 3, 2)],
    )
    mesh.update()
    obj = blender_script.bpy.data.objects.new("Neuron.demo", mesh)
    blender_script.bpy.context.collection.objects.link(obj)
    obj.data.materials.append(newBasicMaterial("Material.Neuron.demo", (0.25, 0.5, 0.75, 1.0)))

    bound = blender_script.bpy.data.objects.new(
        "Bound.neurons", blender_script.bpy.data.meshes.new("Bound.neuronsMesh")
    )
    blender_script.bpy.context.collection.objects.link(bound)
    bound.location = (0, 0, 0)
    bound["Min"] = (-2, 0, -2)
    bound["Max"] = (2, 0, 2)
    bound["Radius"] = 2.8284271247461903

    cam_data = blender_script.bpy.data.cameras.new("Camera")
    cam = blender_script.bpy.data.objects.new("Camera", cam_data)
    blender_script.bpy.context.collection.objects.link(cam)
    blender_script.bpy.context.scene.camera = cam
    cam.location = (0, -0.12, 0)
    direction = mathutils.Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens_unit = "FOV"
    cam.data.angle = 0.7
    cam.data.clip_start = 0.001
    cam.data.clip_end = 100.0
    blender_script.bpy.context.scene.frame_start = 1
    blender_script.bpy.context.scene.frame_end = 1
    blender_script.bpy.context.scene.render.fps = 30

    blend_path = tmp_path / "neuvid_minimal.blend"
    blender_script.bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    return blend_path


def _run_local_neuvid_reference_render(blend_path, tmp_path):
    script = _local_neuvid_path("neuVid", "render.py")
    json_path = tmp_path / "neuvid_minimal.json"
    json_path.write_text("{}")
    out_dir = tmp_path / "neuvid_ref"
    out_dir.mkdir()

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(script),
            "--",
            "-ib",
            str(blend_path),
            "-ij",
            str(json_path),
            "-o",
            str(out_dir),
            "-s",
            "1",
            "-e",
            "1",
            "-rx",
            "64",
            "-ry",
            "64",
            "-sa",
            "32",
            "-nc",
            "-ndn",
        ]
        g = {"__name__": "__main__", "__file__": str(script)}
        exec(compile(_local_neuvid_render_source_blender5_compat(), str(script), "exec"), g)
    finally:
        sys.argv = old_argv
    return out_dir / "0001.png"


def test_mesh_npz_export_writes_neuroglancer_style_vertex_normals(tmp_path):
    np = pytest.importorskip("numpy")

    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int32),
    )
    out = tmp_path / "mesh.npz"

    _export_mesh_npz(mesh, out)

    arr = np.load(out)
    s = 2 ** -0.5
    raw_expected = np.array(
        [
            [0.0, s, s],
            [0.0, s, s],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert "n" in arr.files
    assert arr["n"] == pytest.approx(_ng_quantize_decode_normals(raw_expected), abs=1e-6)
    assert arr["n"] != pytest.approx(raw_expected, abs=1e-6)


def test_mesh_and_warm_cache_versions_include_ng_normal_generation():
    source = Path("src/cinemap/render/worker.py").read_text()

    assert "|geom8" in source
    assert "blender-material-v4" in source


def test_mesh_render_alpha_clamps_like_neuroglancer_object_alpha():
    worker = object.__new__(RenderWorker)

    assert worker._mesh_render_alpha(
        SimpleNamespace(visible=True, opacity=0.8, object_alpha=2.0, render_3d=True)
    ) == pytest.approx(0.8)
    assert worker._mesh_render_alpha(
        SimpleNamespace(visible=True, opacity=2.0, object_alpha=0.75, render_3d=True)
    ) == pytest.approx(0.75)
    assert worker._mesh_render_alpha(
        SimpleNamespace(visible=True, opacity=1.0, object_alpha=-0.5, render_3d=True)
    ) == pytest.approx(0.0)


def test_neuroglancer_oit_reference_matches_single_layer_alpha():
    out = composite_oit_over([OITFragment(rgb=(0.5, 0.0, 0.0), alpha=0.5, depth=0.5)])

    assert out == pytest.approx((0.5, 0.0, 0.0))


def test_neuroglancer_oit_reference_is_order_independent_for_equal_depth():
    red = OITFragment(rgb=(0.5, 0.0, 0.0), alpha=0.5, depth=0.5)
    green = OITFragment(rgb=(0.0, 0.5, 0.0), alpha=0.5, depth=0.5)

    assert composite_oit_over([red, green]) == pytest.approx((0.375, 0.375, 0.0))
    assert composite_oit_over([green, red]) == pytest.approx((0.375, 0.375, 0.0))


def test_neuroglancer_oit_reference_weights_by_depth():
    near = OITFragment(rgb=(0.5, 0.0, 0.0), alpha=0.5, depth=0.1)
    far = OITFragment(rgb=(0.0, 0.5, 0.0), alpha=0.5, depth=0.9)
    out = composite_oit_over([near, far])

    assert compute_oit_weight(0.5, 0.1) > compute_oit_weight(0.5, 0.9)
    assert out[0] > out[1]
    assert out[0] + out[1] == pytest.approx(0.75)


def test_active_slice_sweep_replaces_keyframe_slice_for_same_slot():
    worker = object.__new__(RenderWorker)
    worker.project = SimpleNamespace(
        sweeps=[
            SimpleNamespace(
                enabled=True,
                kind="slice",
                em_name="raw",
                axis="z",
                normal=[0.0, 0.0, 1.0],
                from_nm=10.0,
                to_nm=20.0,
                start_s=0.0,
                duration_s=1.0,
                easing="linear",
                opacity=1.0,
                overlay_layers=[],
            )
        ]
    )
    worker.manifest = SimpleNamespace(em=SimpleNamespace(name="raw"))
    worker.job = SimpleNamespace(settings=SimpleNamespace(still=False))
    worker._vol_for = lambda _name: (None, False)

    frame = SimpleNamespace(
        slices=[FrameSlice("raw", "z", 100.0, None, 1.0)],
        meshes=[],
    )

    reads = worker._frame_slice_reads(frame, 0.5)

    assert len(reads) == 1
    assert reads[0][0].position_nm == 15.0
    assert reads[0][2] is False


def test_neuroglancer_preset_uses_shadowless_ng_shader():
    settings = director.make_settings({"preset": "neuroglancer"})

    assert settings.material.ng_shader is True
    assert settings.material.cast_shadows is False
    assert settings.material.specular == 0.0
    assert settings.material.blend_method == "BLEND"
    assert settings.material.shadow_method == "NONE"
    assert settings.material.show_transparent_back is True
    assert settings.material.color_space == "display"
    assert settings.lighting.kind == "none"
    assert settings.view_transform == "Raw"
    assert settings.dof.enabled is False
    assert settings.bloom.enabled is False


def test_neuvid_preset_matches_neuvid_material_and_lights():
    settings = director.make_settings({"preset": "neuvid"})

    assert settings.material.roughness == 0.25
    assert settings.material.specular == 0.5
    assert settings.material.specular_tint == 0.75
    assert settings.material.alpha_scaled_specular is True
    assert settings.material.color_space == "linear"
    assert settings.material.emission_strength == 0.0
    assert settings.material.transparent_max_bounces == 32
    assert settings.material.transparent_shadows is True
    assert settings.lighting.kind == "neuvid"
    assert settings.lighting.ambient == 0.0
    assert settings.lighting.key_color == (1.0, 1.0, 1.0)
    assert settings.lighting.fill_color == (1.0, 1.0, 1.0)
    assert settings.lighting.rim_color == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("look", "auto_direct", "ng_shader", "light_kind"),
    [
        ({}, True, False, "neuvid"),
        ({}, False, True, "none"),
        ({"preset": "ng"}, False, True, "none"),
        ({"preset": "neuvid"}, False, False, "neuvid"),
    ],
)
def test_render_look_resolves_independently_from_auto_direct(
    look,
    auto_direct,
    ng_shader,
    light_kind,
):
    settings = _director_settings_for_render(SimpleNamespace(look=look), auto_direct)

    assert settings.material.ng_shader is ng_shader
    assert settings.lighting.kind == light_kind


def test_scene_spec_includes_selected_look_when_auto_direct_off(tmp_path):
    class Settings(SimpleNamespace):
        def model_dump(self):
            return dict(self.__dict__)

    worker = object.__new__(RenderWorker)
    worker.project = SimpleNamespace(
        look={"preset": "ng"},
        lighting=SimpleNamespace(background=[0.0, 0.0, 0.0], key_energy=0.0),
        keyframes=[],
        sweeps=[],
    )
    worker.manifest = SimpleNamespace(meshes=[], em=None)
    worker.job = SimpleNamespace(
        settings=Settings(
            width=64,
            height=64,
            fps=30,
            still=False,
            export_blend=True,
            engine="CYCLES",
        )
    )
    worker._auto_direct = False
    worker._lod_mode = "single"
    worker._show_bbox = False
    worker._flip_handed = False
    worker._cb = None
    worker.cancel = SimpleNamespace(is_set=lambda: False)
    worker.nm_per_bu = 1.0
    worker.frames_dir = tmp_path
    worker.assets_dir = tmp_path / "assets"
    worker.blend_path = tmp_path / "scene.blend"
    frame = FrameState(
        position_nm=[0.0, 0.0, 1000.0],
        look_at_nm=[0.0, 0.0, 0.0],
        fov_deg=45.0,
        up=[0.0, 1.0, 0.0],
    )

    spec = worker._build_scene_spec([frame])

    assert spec["direction"]["material"]["ng_shader"] is True
    assert spec["direction"]["lighting"]["kind"] == "none"
    assert spec["direction"]["view"]["transform"] == "Raw"


def test_neuroglancer_shader_light_uses_frame_camera_direction():
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    frame = {"camera": {"position_bu": [1.0, 2.0, 3.0], "look_at_bu": [1.0, 2.0, 8.0]}}

    light = blender_script._ng_light_direction(frame)

    assert tuple(round(float(v), 6) for v in light) == (0.0, 0.0, 1.0)


def test_neuroglancer_preset_matches_local_shader_source():
    single_mesh = _local_neuroglancer_path("src", "single_mesh", "frontend.ts").read_text()
    perspective = _local_neuroglancer_path("src", "perspective_view", "panel.ts").read_text()
    segmentation = _local_neuroglancer_path("src", "segmentation_display_state", "frontend.ts").read_text()
    mesh_frontend = _local_neuroglancer_path("src", "mesh", "frontend.ts").read_text()
    mesh_backend = _local_neuroglancer_path("src", "mesh", "backend.ts").read_text()
    settings = director.make_settings({"preset": "neuroglancer"})

    assert "vLightingFactor = abs(dot(normal, uLightDirection.xyz)) + uLightDirection.w;" in single_mesh
    assert "emit(vec4(color.rgb * vLightingFactor, color.a), vPickID);" in single_mesh
    assert "color.xyz *= color.a;" in single_mesh
    assert "float computeOITWeight(float alpha, float depth)" in perspective
    assert "vec4 accum = color * weight;" in perspective
    assert "colorBuffers: makeTextureBuffers(" in perspective
    assert "this.gl.RGBA32F" in perspective
    assert "gl.depthMask(false);" in perspective
    assert "WebGL2RenderingContext.ONE_MINUS_SRC_ALPHA" in perspective
    assert "this.transparencyCopyHelper.draw(" in perspective
    assert "vec3.scale(lightVec, lightDirection, directionalLighting);" in single_mesh
    assert "lightVec[3] = ambientLighting;" in single_mesh
    assert "const ambient = 0.2;" in perspective
    assert "const directional = 1 - ambient;" in perspective
    assert "color[0] *= alpha;" in segmentation
    assert "color[1] *= alpha;" in segmentation
    assert "color[2] *= alpha;" in segmentation
    assert "const alpha = Math.min(1, displayState.objectAlpha.value);" in segmentation
    assert "color[i] = color[i] * saturation + (1 - saturation);" in segmentation
    assert "float absCosAngle = abs(dot(normal, uLightDirection.xyz));" in mesh_frontend
    assert "vColor *= pow(1.0 - absCosAngle, uSilhouettePower);" in mesh_frontend
    assert "const glsl_decodeNormalOctahedronSnorm8" in mesh_frontend
    assert "decodeNormalOctahedronSnorm8(aVertexNormal)" in mesh_frontend
    assert "encodeNormals32fx3ToOctahedron8x2" in mesh_backend
    assert "vec3.cross(faceNormal, v1v0, v2v1);" in mesh_backend
    assert "vertexNormals[offset + j] += faceNormal[j];" in mesh_backend
    assert "vec3.normalize(vec, vec);" in mesh_backend

    assert settings.material.ng_shader is True
    assert settings.material.cast_shadows is False
    assert settings.material.blend_method == "BLEND"
    assert settings.material.shadow_method == "NONE"
    assert settings.material.show_transparent_back is True
    assert settings.material.color_space == "display"
    assert settings.lighting.kind == "none"
    assert settings.view_transform == "Raw"


def test_neuvid_render_setup_raises_transparent_max_bounces():
    bpy = pytest.importorskip("bpy")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    material = asdict(director.make_settings({"preset": "neuvid"}).material)
    blender_script._setup_render(
        {
            "render": {"engine": "CYCLES", "width": 16, "height": 16, "samples": 1},
            "world": {"background": [0.0, 0.0, 0.0]},
            "direction": {
                "material": material,
                "lighting": asdict(director.make_settings({"preset": "neuvid"}).lighting),
                "view": {"transform": "", "look": ""},
            },
        }
    )

    assert bpy.context.scene.cycles.transparent_max_bounces == 32


def test_neuvid_light_rig_matches_cycles_area_light_setup(tmp_path):
    pytest.importorskip("bpy")
    mathutils = pytest.importorskip("mathutils")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    settings = director.make_settings({"preset": "neuvid"})
    blender_script._import_meshes(_tiny_mesh_scene(tmp_path, asdict(settings.material)))
    center, radius = blender_script._scene_center_radius()
    rig = asdict(settings.lighting)

    blender_script._add_neuvid_lights({"direction": {"lighting": rig}}, rig)

    distance = radius * 2.5
    expected_energy = 2_400_000.0 * (distance / 425.1282) ** 2
    light_rotation = mathutils.Matrix.Rotation(math.radians(180.0), 4, "Y").to_3x3()
    source_dirs = [
        (-0.892, 0.3, 0.9),
        (0.588, 0.46, 0.248),
        (0.216, -0.392, -0.216),
    ]

    for i, source_dir in enumerate(source_dirs):
        obj = blender_script.bpy.data.objects[f"Lamp.{i}"]
        light = obj.data
        expected_dir = light_rotation @ mathutils.Vector(source_dir)
        expected_dir.normalize()
        expected_location = center + expected_dir * distance

        assert light.type == "AREA"
        assert tuple(light.color) == (1.0, 1.0, 1.0)
        assert light.size == pytest.approx(radius)
        assert light.energy == pytest.approx(expected_energy)
        assert tuple(obj.location) == pytest.approx(tuple(expected_location))
        assert (obj.location - center).length == pytest.approx(distance)


def test_neuvid_light_rotation_matches_local_render_source(tmp_path):
    pytest.importorskip("bpy")
    mathutils = pytest.importorskip("mathutils")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    settings = director.make_settings({"preset": "neuvid"})
    blender_script._import_meshes(_tiny_mesh_scene(tmp_path, asdict(settings.material)))
    center, radius = blender_script._scene_center_radius()
    rig = asdict(settings.lighting)
    rig["neuvid_light_rotation"] = [10.0, 20.0, 30.0]

    blender_script._add_neuvid_lights({"direction": {"lighting": rig}}, rig)

    source_dir = mathutils.Vector((-0.892, 0.3, 0.9))
    local_rotation = mathutils.Euler((10.0, 20.0, math.radians(30.0)), "XYZ").to_matrix()
    parent_rotation = mathutils.Matrix.Rotation(math.radians(180.0), 4, "Y").to_3x3()
    expected_dir = parent_rotation @ (local_rotation @ source_dir)
    expected_dir.normalize()
    expected_location = center + expected_dir * (radius * 2.5)

    assert tuple(blender_script.bpy.data.objects["Lamp.0"].location) == pytest.approx(
        tuple(expected_location)
    )


def test_add_light_clears_stale_lights_for_neuroglancer_and_neuvid(tmp_path):
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    stale_data = blender_script.bpy.data.lights.new("stale", type="SUN")
    stale = blender_script.bpy.data.objects.new("stale", stale_data)
    blender_script.bpy.context.scene.collection.objects.link(stale)

    blender_script._add_light({"direction": {"lighting": asdict(director.make_settings({"preset": "ng"}).lighting)}})

    assert not [obj for obj in blender_script.bpy.context.scene.objects if obj.type == "LIGHT"]

    settings = director.make_settings({"preset": "neuvid"})
    blender_script._import_meshes(_tiny_mesh_scene(tmp_path, asdict(settings.material)))
    blender_script._add_light({"direction": {"lighting": asdict(settings.lighting)}})
    blender_script._add_light({"direction": {"lighting": asdict(settings.lighting)}})

    assert sorted(obj.name for obj in blender_script.bpy.context.scene.objects if obj.type == "LIGHT") == [
        "Lamp.0",
        "Lamp.1",
        "Lamp.2",
    ]


def test_export_blend_preserves_preset_light_counts(tmp_path):
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    def spec_for(preset, path):
        settings = director.make_settings({"preset": preset})
        return {
            "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
            "lighting": {"key_energy": 0.0},
            "render": {"engine": "CYCLES", "width": 16, "height": 16, "fps": 30, "samples": 1},
            "meshes": _tiny_mesh_scene(tmp_path, asdict(settings.material))["meshes"],
            "frames": [
                {
                    "camera": {
                        "position_bu": [0.0, 0.0, 4.0],
                        "look_at_bu": [0.0, 0.0, 0.0],
                        "fov_rad": 0.7,
                        "up": [0.0, 1.0, 0.0],
                    },
                    "slices": [],
                    "mesh_overrides": {
                        "tiny": {"opacity": 1.0, "visible": True, "silhouette": 0.0, "color": [0.25, 0.5, 0.75]}
                    },
                    "fade_alpha": 0.0,
                    "index": 0,
                }
            ],
            "output_dir": str(tmp_path),
            "fps": 30,
            "export_blend": str(path),
            "direction": director.plan([], settings),
            "warm_blend": None,
        }

    blender_script.export_blend(spec_for("ng", tmp_path / "ng.blend"))
    assert (tmp_path / "ng.blend").exists()
    assert not [obj for obj in blender_script.bpy.context.scene.objects if obj.type == "LIGHT"]

    blender_script.export_blend(spec_for("neuvid", tmp_path / "neuvid.blend"))
    assert (tmp_path / "neuvid.blend").exists()
    lights = [obj for obj in blender_script.bpy.context.scene.objects if obj.type == "LIGHT"]
    assert sorted(obj.name for obj in lights) == ["Lamp.0", "Lamp.1", "Lamp.2"]
    assert {obj.data.type for obj in lights} == {"AREA"}


def test_export_blend_bakes_per_frame_material_state(tmp_path):
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    settings = director.make_settings({"preset": "neuvid"})
    mesh = _tiny_mesh_scene(tmp_path, asdict(settings.material))["meshes"][0]
    spec = {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {"engine": "CYCLES", "width": 16, "height": 16, "fps": 30, "samples": 1},
        "meshes": [mesh],
        "frames": [
            {
                "camera": {
                    "position_bu": [0.0, 0.0, 4.0],
                    "look_at_bu": [0.0, 0.0, 0.0],
                    "fov_rad": 0.7,
                    "up": [0.0, 1.0, 0.0],
                },
                "slices": [],
                "mesh_overrides": {
                    "tiny": {
                        "opacity": 1.0,
                        "visible": True,
                        "silhouette": 0.0,
                        "color": [0.25, 0.5, 0.75],
                        "emphasis": 0.4,
                        "metallic": 0.7,
                        "roughness": 0.9,
                    }
                },
                "fade_alpha": 0.0,
                "index": 0,
            },
            {
                "camera": {
                    "position_bu": [0.0, 0.0, 4.0],
                    "look_at_bu": [0.0, 0.0, 0.0],
                    "fov_rad": 0.7,
                    "up": [0.0, 1.0, 0.0],
                },
                "slices": [],
                "mesh_overrides": {
                    "tiny": {
                        "opacity": 1.0,
                        "visible": True,
                        "silhouette": 0.0,
                        "color": [0.25, 0.5, 0.75],
                    }
                },
                "fade_alpha": 0.0,
                "index": 1,
            },
        ],
        "output_dir": str(tmp_path),
        "fps": 30,
        "export_blend": str(tmp_path / "material_state.blend"),
        "direction": director.plan([], settings),
        "warm_blend": None,
    }

    blender_script.export_blend(spec)

    scene = blender_script.bpy.context.scene
    mat = blender_script.bpy.data.objects["tiny"].active_material
    nodes = mat.node_tree.nodes
    scene.frame_set(1)
    assert nodes["cm_emit"].outputs[0].default_value == pytest.approx(0.4)
    assert nodes["cm_metal"].outputs[0].default_value == pytest.approx(0.7)
    assert nodes["cm_rough"].outputs[0].default_value == pytest.approx(0.9)
    scene.frame_set(2)
    assert nodes["cm_emit"].outputs[0].default_value == pytest.approx(0.0)
    assert nodes["cm_metal"].outputs[0].default_value == pytest.approx(0.0)
    assert nodes["cm_rough"].outputs[0].default_value == pytest.approx(0.25)


def test_export_blend_bakes_camera_relative_light_rotation(tmp_path):
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    settings = director.make_settings({"preset": "rake"})
    mesh = _tiny_mesh_scene(tmp_path, asdict(settings.material))["meshes"][0]
    frames = []
    for index, camera in enumerate(
        [
            {"position_bu": [0.0, 0.0, 4.0], "look_at_bu": [0.0, 0.0, 0.0], "fov_rad": 0.7, "up": [0.0, 1.0, 0.0]},
            {"position_bu": [4.0, 0.0, 0.0], "look_at_bu": [0.0, 0.0, 0.0], "fov_rad": 0.7, "up": [0.0, 1.0, 0.0]},
        ]
    ):
        frames.append(
            {
                "camera": camera,
                "slices": [],
                "mesh_overrides": {
                    "tiny": {"opacity": 1.0, "visible": True, "silhouette": 0.0, "color": [0.25, 0.5, 0.75]}
                },
                "fade_alpha": 0.0,
                "index": index,
            }
        )
    spec = {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {"engine": "CYCLES", "width": 16, "height": 16, "fps": 30, "samples": 1},
        "meshes": [mesh],
        "frames": frames,
        "output_dir": str(tmp_path),
        "fps": 30,
        "export_blend": str(tmp_path / "camera_relative_lights.blend"),
        "direction": director.plan([], settings),
        "warm_blend": None,
    }

    blender_script.export_blend(spec)

    scene = blender_script.bpy.context.scene
    key = blender_script.bpy.data.objects["Key"]
    scene.frame_set(1)
    rot1 = tuple(float(x) for x in key.rotation_euler)
    scene.frame_set(2)
    rot2 = tuple(float(x) for x in key.rotation_euler)
    assert rot1 != pytest.approx(rot2)
    assert key.animation_data is not None


def test_neuvid_light_rig_constants_match_local_render_source():
    specs = _local_neuvid_lamp_specs()
    render_source = _local_neuvid_path("neuVid", "render.py").read_text()
    light_loop = render_source[
        render_source.index("lampSpecs = ["): render_source.index("# Put the lights", render_source.index("lampSpecs = ["))
    ]

    assert [spec["direction"] for spec in specs] == [
        (-0.892, 0.3, 0.9),
        (0.588, 0.46, 0.248),
        (0.216, -0.392, -0.216),
    ]
    assert [spec["color"][:3] for spec in specs] == [
        (0.8, 0.8, 0.8),
        (0.498, 0.5, 0.6),
        (0.798, 0.838, 1.0),
    ]
    assert "lampDistance = neuronsBoundRadius * 2.5" in light_loop
    assert "lampData.energy = 2400000" in light_loop
    assert "powerScale = (lampDistance / 425.1282)**2" in light_loop
    assert "lightRotationX = math.radians(jsonLightRotationX)" in light_loop
    assert "lightRotationY = math.radians(jsonLightRotationY)" in light_loop
    assert "lightRotationZ = math.radians(jsonLightRotationZ)" in light_loop
    assert "lightRotationE = mathutils.Euler((jsonLightRotationX, jsonLightRotationY, lightRotationZ))" in light_loop
    assert 'lampRotator.rotation_euler = mathutils.Euler((0, math.radians(180), 0), "XYZ")' in render_source

    cycles_setup = light_loop[
        light_loop.index("elif args.useCycles:"): light_loop.index("else:", light_loop.index("elif args.useCycles:"))
    ]
    assert "lampData.color" not in cycles_setup
    settings = director.make_settings({"preset": "neuvid"})
    assert settings.lighting.key_color == (1.0, 1.0, 1.0)
    assert settings.lighting.fill_color == (1.0, 1.0, 1.0)
    assert settings.lighting.rim_color == (1.0, 1.0, 1.0)


def test_neuvid_preset_render_matches_local_neuvid_reference(tmp_path):
    pytest.importorskip("bpy")
    np = pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    from cinemap.render import blender_script

    blend_path = _write_minimal_neuvid_blend(tmp_path)
    ref_path = _run_local_neuvid_reference_render(blend_path, tmp_path)
    ref = np.asarray(image_mod.open(ref_path).convert("RGB"))

    _reset_blender(blender_script)
    mesh_path = tmp_path / "cinemap_minimal.npz"
    np.savez(
        mesh_path,
        v=np.array(
            [[-0.02, 0.0, -0.02], [0.02, 0.0, -0.02], [0.02, 0.0, 0.02], [-0.02, 0.0, 0.02]],
            dtype=np.float32,
        ),
        f=np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32),
    )
    out_dir = tmp_path / "cinemap_out"
    out_dir.mkdir()
    settings = director.make_settings({"preset": "neuvid"})
    spec = {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {"engine": "CYCLES", "width": 64, "height": 64, "fps": 30, "samples": 32, "noise_threshold": 0.001},
        "meshes": [{"id": "demo", "obj_path": str(mesh_path), "color": [0.25, 0.5, 0.75]}],
        "frames": [
            {
                "camera": {
                    "position_bu": [0.0, -0.12, 0.0],
                    "look_at_bu": [0.0, 0.0, 0.0],
                    "fov_rad": 0.7,
                    "up": [0.0, 0.0, 1.0],
                },
                "slices": [],
                "mesh_overrides": {
                    "demo": {"opacity": 1.0, "visible": True, "silhouette": 0.0, "color": [0.25, 0.5, 0.75]}
                },
                "fade_alpha": 0.0,
                "index": 0,
            }
        ],
        "output_dir": str(out_dir),
        "fps": 30,
        "export_blend": None,
        "direction": director.plan([], settings),
        "warm_blend": None,
    }
    scene_path = tmp_path / "cinemap_scene.json"
    scene_path.write_text(json.dumps(spec))

    blender_script.main(str(scene_path))

    cine = np.asarray(image_mod.open(out_dir / "frame_00000.png").convert("RGB"))
    ref_mask = ref.max(axis=2) > 2
    cine_mask = cine.max(axis=2) > 2
    assert int(abs(int(ref_mask.sum()) - int(cine_mask.sum()))) <= 40
    assert cine[32, 32] == pytest.approx(ref[32, 32], abs=4)
    assert cine[cine_mask].mean(axis=0) == pytest.approx(ref[ref_mask].mean(axis=0), abs=6)


def test_neuroglancer_material_graph_uses_constant_light_vector(tmp_path):
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    material = asdict(director.make_settings({"preset": "ng"}).material)
    meshes = blender_script._import_meshes(_tiny_mesh_scene(tmp_path, material))
    mat = meshes["tiny"][1]
    nt = mat.node_tree

    assert mat.blend_method == "BLEND"
    if hasattr(mat, "shadow_method"):
        assert mat.shadow_method == "NONE"
    if hasattr(mat, "show_transparent_back"):
        assert mat.show_transparent_back is True
    if hasattr(mat, "surface_render_method"):
        assert mat.surface_render_method == "BLENDED"
    assert tuple(nt.nodes["cm_color"].outputs[0].default_value) == (0.25, 0.5, 0.75, 1.0)
    assert nt.nodes.get("cm_ng_light_x") is not None
    assert nt.nodes.get("cm_ng_light_y") is not None
    assert nt.nodes.get("cm_ng_light_z") is not None
    dot = next(n for n in nt.nodes if n.type == "VECT_MATH" and n.operation == "DOT_PRODUCT")
    linked_to_dot_light = [link for link in nt.links if link.to_node == dot and link.to_socket == dot.inputs[1]]

    assert len(linked_to_dot_light) == 1
    assert linked_to_dot_light[0].from_node.type == "COMBXYZ"
    assert not any(link.from_socket.name == "Incoming" and link.to_node == dot for link in nt.links)


def test_blender_import_applies_cached_vertex_normals(tmp_path):
    pytest.importorskip("bpy")
    np = pytest.importorskip("numpy")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    mesh_path = tmp_path / "normal_mesh.npz"
    np.savez(
        mesh_path,
        v=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        f=np.array([[0, 1, 2]], dtype=np.int32),
        n=np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
    )
    material = asdict(director.make_settings({"preset": "ng"}).material)
    scene = {
        "world": {"nm_per_bu": 1.0},
        "meshes": [{"id": "normal_mesh", "obj_path": str(mesh_path), "color": [1.0, 0.0, 0.0]}],
        "direction": {"material": material},
    }

    meshes = blender_script._import_meshes(scene)

    obj = meshes["normal_mesh"][0]
    assert obj.data.has_custom_normals is True
    assert tuple(obj.data.corner_normals[0].vector) == pytest.approx((0.0, 0.0, 1.0))


def test_neuroglancer_render_alpha_is_display_space_half_red(tmp_path):
    pytest.importorskip("bpy")
    np = pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    mesh_path = tmp_path / "square.npz"
    np.savez(
        mesh_path,
        v=np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float32),
        f=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    settings = director.make_settings({"preset": "ng"})
    spec = {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {"engine": "CYCLES", "width": 48, "height": 48, "fps": 30, "samples": 8, "noise_threshold": 0.02},
        "meshes": [{"id": "square", "obj_path": str(mesh_path), "color": [1.0, 0.0, 0.0]}],
        "frames": [
            {
                "camera": {
                    "position_bu": [0.0, 0.0, 4.0],
                    "look_at_bu": [0.0, 0.0, 0.0],
                    "fov_rad": 0.7,
                    "up": [0.0, 1.0, 0.0],
                },
                "slices": [],
                "mesh_overrides": {
                    "square": {"opacity": 0.5, "visible": True, "silhouette": 0.0, "color": [1.0, 0.0, 0.0]}
                },
                "fade_alpha": 0.0,
                "index": 0,
            }
        ],
        "output_dir": str(out_dir),
        "fps": 30,
        "export_blend": None,
        "direction": director.plan([], settings),
        "warm_blend": None,
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(spec))

    blender_script.main(str(scene_path))

    img = image_mod.open(out_dir / "frame_00000.png").convert("RGB")
    center = img.getpixel((24, 24))
    assert 120 <= center[0] <= 135
    assert center[1] <= 4
    assert center[2] <= 4


def test_neuroglancer_render_vertex_colors_stay_display_rgb(tmp_path):
    pytest.importorskip("bpy")
    np = pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    mesh_path = tmp_path / "vertex_color_square.npz"
    rgba = np.array([[64, 128, 191, 255]] * 4, dtype=np.uint8)
    np.savez(
        mesh_path,
        v=np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float32),
        f=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        c=rgba,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    settings = director.make_settings({"preset": "ng"})
    spec = {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {"engine": "CYCLES", "width": 48, "height": 48, "fps": 30, "samples": 8, "noise_threshold": 0.02},
        "meshes": [{"id": "square", "obj_path": str(mesh_path), "color": [1.0, 1.0, 1.0]}],
        "frames": [
            {
                "camera": {
                    "position_bu": [0.0, 0.0, 4.0],
                    "look_at_bu": [0.0, 0.0, 0.0],
                    "fov_rad": 0.7,
                    "up": [0.0, 1.0, 0.0],
                },
                "slices": [],
                "mesh_overrides": {
                    "square": {"opacity": 1.0, "visible": True, "silhouette": 0.0, "color": [1.0, 1.0, 1.0]}
                },
                "fade_alpha": 0.0,
                "index": 0,
            }
        ],
        "output_dir": str(out_dir),
        "fps": 30,
        "export_blend": None,
        "direction": director.plan([], settings),
        "warm_blend": None,
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(spec))

    blender_script.main(str(scene_path))

    img = image_mod.open(out_dir / "frame_00000.png").convert("RGB")
    center = img.getpixel((24, 24))
    assert center == pytest.approx((64, 128, 191), abs=5)


def test_neuroglancer_render_lighting_factor_matches_shader_formula(tmp_path):
    pytest.importorskip("bpy")
    np = pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    u = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0, -math.sqrt(3.0)], dtype=np.float32)
    mesh_path = tmp_path / "tilted_square.npz"
    np.savez(
        mesh_path,
        v=np.array([-u - v, u - v, u + v, -u + v], dtype=np.float32),
        f=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    settings = director.make_settings({"preset": "ng"})
    spec = {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {"engine": "CYCLES", "width": 64, "height": 64, "fps": 30, "samples": 8, "noise_threshold": 0.02},
        "meshes": [{"id": "tilted_square", "obj_path": str(mesh_path), "color": [0.0, 1.0, 0.0]}],
        "frames": [
            {
                "camera": {
                    "position_bu": [0.0, 0.0, 6.0],
                    "look_at_bu": [0.0, 0.0, 0.0],
                    "fov_rad": 0.5,
                    "up": [0.0, 1.0, 0.0],
                },
                "slices": [],
                "mesh_overrides": {
                    "tilted_square": {"opacity": 1.0, "visible": True, "silhouette": 0.0, "color": [0.0, 1.0, 0.0]}
                },
                "fade_alpha": 0.0,
                "index": 0,
            }
        ],
        "output_dir": str(out_dir),
        "fps": 30,
        "export_blend": None,
        "direction": director.plan([], settings),
        "warm_blend": None,
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(spec))

    blender_script.main(str(scene_path))

    img = image_mod.open(out_dir / "frame_00000.png").convert("RGB")
    center = img.getpixel((32, 32))
    expected = round(255 * (abs(0.5) * 0.8 + 0.2))
    assert expected - 8 <= center[1] <= expected + 8
    assert center[0] <= 4
    assert center[2] <= 4


def test_neuroglancer_render_silhouette_uses_scaled_abs_cos_angle(tmp_path):
    pytest.importorskip("bpy")
    np = pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    u = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0, -math.sqrt(3.0)], dtype=np.float32)
    mesh_path = tmp_path / "tilted_silhouette_square.npz"
    np.savez(
        mesh_path,
        v=np.array([-u - v, u - v, u + v, -u + v], dtype=np.float32),
        f=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    settings = director.make_settings({"preset": "ng"})
    spec = {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {"engine": "CYCLES", "width": 64, "height": 64, "fps": 30, "samples": 8, "noise_threshold": 0.02},
        "meshes": [{"id": "tilted_square", "obj_path": str(mesh_path), "color": [0.0, 1.0, 0.0]}],
        "frames": [
            {
                "camera": {
                    "position_bu": [0.0, 0.0, 6.0],
                    "look_at_bu": [0.0, 0.0, 0.0],
                    "fov_rad": 0.5,
                    "up": [0.0, 1.0, 0.0],
                },
                "slices": [],
                "mesh_overrides": {
                    "tilted_square": {"opacity": 1.0, "visible": True, "silhouette": 2.0, "color": [0.0, 1.0, 0.0]}
                },
                "fade_alpha": 0.0,
                "index": 0,
            }
        ],
        "output_dir": str(out_dir),
        "fps": 30,
        "export_blend": None,
        "direction": director.plan([], settings),
        "warm_blend": None,
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(spec))

    blender_script.main(str(scene_path))

    img = image_mod.open(out_dir / "frame_00000.png").convert("RGB")
    center = img.getpixel((32, 32))
    lighting = abs(0.5) * 0.8 + 0.2
    silhouette = (1.0 - abs(0.5) * 0.8) ** 2.0
    expected = round(255 * lighting * silhouette)
    assert expected - 8 <= center[1] <= expected + 8
    assert center[0] <= 4
    assert center[2] <= 4


def test_neuvid_material_graph_scales_specular_by_alpha_and_tints_to_color(tmp_path):
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    _reset_blender(blender_script)
    material = asdict(director.make_settings({"preset": "neuvid"}).material)
    meshes = blender_script._import_meshes(_tiny_mesh_scene(tmp_path, material))
    mat = meshes["tiny"][1]
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    assert tuple(nt.nodes["cm_color"].outputs[0].default_value) == (0.25, 0.5, 0.75, 1.0)
    if hasattr(mat, "cycles") and hasattr(mat.cycles, "use_transparent_shadow"):
        assert mat.cycles.use_transparent_shadow is True
    spec_socket = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    spec_link = next(link for link in nt.links if link.to_socket == spec_socket)
    spec_mult = spec_link.from_node

    assert spec_mult.type == "MATH"
    assert spec_mult.operation == "MULTIPLY"
    assert {link.from_node.name for link in nt.links if link.to_node == spec_mult} == {
        "cm_alpha",
        "cm_specular",
    }

    tint_socket = bsdf.inputs.get("Specular Tint")
    assert tint_socket is not None
    if hasattr(tint_socket.default_value, "__len__"):
        tint_node = _linked_from(nt, tint_socket)
        assert tint_node.type == "MIX_RGB"
        assert tint_node.inputs["Fac"].default_value == 0.75
        assert tuple(tint_node.inputs["Color1"].default_value) == (1.0, 1.0, 1.0, 1.0)
    else:
        assert tint_socket.default_value == 0.75


def test_neuvid_material_graph_matches_local_neuvid_new_basic_material(tmp_path):
    pytest.importorskip("bpy")
    from cinemap.render import blender_script

    neuvid_root = Path("/groups/scicompsoft/home/ackermand/Programming/neuVid")
    if not neuvid_root.exists():
        pytest.skip("local neuVid checkout unavailable")
    sys.path.insert(0, str(neuvid_root))
    try:
        from neuVid import utilsMaterials
    finally:
        try:
            sys.path.remove(str(neuvid_root))
        except ValueError:
            pass

    _reset_blender(blender_script)
    color = (0.25, 0.5, 0.75, 1.0)
    neuvid_mat = utilsMaterials.newBasicMaterial("neuvid_ref", color)
    material = asdict(director.make_settings({"preset": "neuvid"}).material)
    cinemap_mat = blender_script._import_meshes(_tiny_mesh_scene(tmp_path, material))["tiny"][1]
    neuvid_bsdf = neuvid_mat.node_tree.nodes["Principled BSDF"]
    cinemap_bsdf = cinemap_mat.node_tree.nodes["Principled BSDF"]

    assert cinemap_mat.blend_method == neuvid_mat.blend_method
    if hasattr(neuvid_mat, "surface_render_method"):
        assert cinemap_mat.surface_render_method == neuvid_mat.surface_render_method

    assert cinemap_bsdf.inputs["Roughness"].default_value == pytest.approx(
        neuvid_bsdf.inputs["Roughness"].default_value
    )
    assert tuple(cinemap_mat.node_tree.nodes["cm_color"].outputs[0].default_value) == pytest.approx(
        tuple(neuvid_mat.node_tree.nodes["diffuse_color"].outputs[0].default_value)
    )

    spec_key = "Specular IOR Level" if "Specular IOR Level" in cinemap_bsdf.inputs else "Specular"
    neuvid_spec = _linked_from(neuvid_mat.node_tree, neuvid_bsdf.inputs[spec_key])
    cinemap_spec = _linked_from(cinemap_mat.node_tree, cinemap_bsdf.inputs[spec_key])
    assert neuvid_spec.type == cinemap_spec.type == "MATH"
    assert neuvid_spec.operation == cinemap_spec.operation == "MULTIPLY"
    assert _specular_value_input_node(neuvid_mat.node_tree, neuvid_spec).outputs[0].default_value == pytest.approx(
        _specular_value_input_node(cinemap_mat.node_tree, cinemap_spec).outputs[0].default_value
    )

    neuvid_tint = _linked_from(neuvid_mat.node_tree, neuvid_bsdf.inputs["Specular Tint"])
    cinemap_tint = _linked_from(cinemap_mat.node_tree, cinemap_bsdf.inputs["Specular Tint"])
    assert neuvid_tint.type == cinemap_tint.type == "MIX_RGB"
    assert cinemap_tint.inputs["Fac"].default_value == pytest.approx(
        neuvid_tint.inputs["Fac"].default_value
    )
    assert tuple(cinemap_tint.inputs["Color1"].default_value) == pytest.approx(
        tuple(neuvid_tint.inputs["Color1"].default_value)
    )
