"""End-to-end color parity through a real Blender render.

The unit tests in test_ng_shader.py prove we compute the same pixel values neuroglancer
would. That is only half the problem: those values then travel through Blender's color
management, which is where they used to be corrupted (loaded as "Non-Color", i.e. treated
as linear, then pushed through the AgX filmic transform — a mid-gray 128 came out around
188). These tests render an actual EM-slice quad with bpy and assert the pixels survive.

Skipped automatically when bpy is unavailable. Slow (a few seconds per render), so they
are marked and can be deselected with `-m "not slow"`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

bpy = pytest.importorskip("bpy", reason="bpy (Blender) not installed")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "src" / "cinemap" / "render" / "blender_script.py"

# Values spanning the range, including the mid-gray that made the old bug obvious.
PROBE_VALUES = [0, 32, 64, 128, 192, 255]


def _write_probe_png(path: Path) -> None:
    """A 1-pixel-tall strip, one PROBE_VALUES entry per column."""
    from PIL import Image

    row = np.array([PROBE_VALUES], dtype=np.uint8)
    Image.fromarray(np.repeat(row[:, :, None], 3, axis=2)).save(path)


def _scene_spec(png: Path, out_dir: Path, *, engine: str = "CYCLES") -> dict:
    """A minimal scene: one EM-slice quad filling an orthographic camera's view.

    The quad spans x in [-3,3], y in [-1,1] at z=0; the camera looks straight down -Z
    with ortho_scale 6, so the strip maps cleanly across the frame.
    """
    return {
        "world": {"nm_per_bu": 1.0, "background": [0.0, 0.0, 0.0]},
        "lighting": {"key_energy": 0.0},
        "render": {
            "width": len(PROBE_VALUES) * 20, "height": 20,
            "samples": 1, "noise_threshold": 0.1, "engine": engine,
            "fps": 1, "export_blend": False, "draft": True, "still": True,
        },
        "meshes": [],
        "frames": [{
            "index": 0,
            "camera": {"position_bu": [0.0, 0.0, 5.0], "type": "ORTHO",
                       "ortho_scale": 6.0, "look_at_bu": [0.0, 0.0, 0.0],
                       "quaternion": [0.0, 0.0, 0.0, 1.0]},
            "slices": [{
                "image_path": str(png),
                "origin_bu": [-3.0, -1.0, 0.0],
                "u_bu": [6.0, 0.0, 0.0],
                "v_bu": [0.0, 2.0, 0.0],
                "opacity": 1.0,
                "occlude": True,
                "slot": "probe:z",
            }],
            "mesh_overrides": {},
        }],
        "output_dir": str(out_dir),
        "fps": 1,
        "export_blend": None,
    }


def _render(tmp_path: Path, spec: dict) -> np.ndarray:
    from PIL import Image

    spec_path = tmp_path / "scene.json"
    spec_path.write_text(json.dumps(spec))
    out = subprocess.run([sys.executable, str(SCRIPT), str(spec_path)],
                         capture_output=True, text=True, timeout=900)
    frame = Path(spec["output_dir"]) / "frame_00000.png"
    if not frame.exists():
        raise AssertionError(
            f"render produced no frame (exit {out.returncode})\n"
            f"--- stdout ---\n{out.stdout[-3000:]}\n--- stderr ---\n{out.stderr[-3000:]}")
    return np.asarray(Image.open(frame).convert("RGB"))


def _sampled_columns(img: np.ndarray) -> list[int]:
    """Center-sample one pixel per probe column, avoiding quad edges."""
    h, w = img.shape[:2]
    n = len(PROBE_VALUES)
    return [int(img[h // 2, int((i + 0.5) * w / n), 0]) for i in range(n)]


@pytest.fixture(scope="module")
def probe_png(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("probe") / "probe.png"
    _write_probe_png(p)
    return p


def test_em_slice_survives_color_management(tmp_path, probe_png):
    """A baked slice must render back to the value neuroglancer displayed.

    This is the regression that made EM look washed out: with the texture tagged
    Non-Color and an AgX view transform, 128 rendered near 188.
    """
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    img = _render(tmp_path, _scene_spec(probe_png, out_dir))
    got = _sampled_columns(img)
    for want, have in zip(PROBE_VALUES, got):
        assert abs(have - want) <= 2, (
            f"EM slice value {want} rendered as {have} (all: {got}). "
            "Check the slice texture colorspace and the view transform.")


def test_midgray_is_not_lifted(tmp_path, probe_png):
    """Explicit guard on the exact symptom, in case tolerances above are ever loosened."""
    out_dir = tmp_path / "frames_mid"
    out_dir.mkdir()
    img = _render(tmp_path, _scene_spec(probe_png, out_dir))
    mid = _sampled_columns(img)[PROBE_VALUES.index(128)]
    assert mid < 140, f"mid-gray rendered at {mid}: the old Non-Color/AgX path is back"
    assert mid > 116, f"mid-gray rendered at {mid}: over-darkened (double sRGB decode?)"


def test_contrast_window_reaches_the_render(tmp_path):
    """Narrowing the invlerp range must visibly change rendered pixels.

    Guards the whole chain: shaderControls -> ng_shader -> baked PNG -> Blender.
    """
    from PIL import Image

    from cinemap.data import ng_shader as ns

    data = np.array([PROBE_VALUES], dtype=np.uint8)
    shaded, warn = ns.shade(
        ns.from_layer({"shaderControls": {"normalized": {"range": [64, 192]}}}, np.uint8),
        data)
    assert warn == ""

    png = tmp_path / "windowed.png"
    Image.fromarray(shaded).save(png)
    out_dir = tmp_path / "frames_win"
    out_dir.mkdir()
    got = _sampled_columns(_render(tmp_path, _scene_spec(png, out_dir)))

    expected = shaded[0, :, 0].tolist()
    for want, have in zip(expected, got):
        assert abs(have - want) <= 2, f"windowed EM: wanted {expected}, got {got}"
    # and the window really did something: 64 -> black, 192 -> white
    assert got[PROBE_VALUES.index(64)] <= 2
    assert got[PROBE_VALUES.index(192)] >= 253
