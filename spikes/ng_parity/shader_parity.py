#!/usr/bin/env python
"""Check `cinemap.data.ng_shader` against the real neuroglancer WebGL rasterizer.

The unit tests prove our evaluator matches neuroglancer's *formulas as written in its
source*. This closes the remaining gap: it renders a known ramp through an actual
neuroglancer viewer in a real browser and diffs the pixels against what our numpy
evaluator predicts. A divergence we never thought to encode as a formula still gets
caught here.

How it avoids the usual screenshot pitfalls (all found the hard way):
  - Data is served as a `LocalVolume`, so there is no network and no cache.
  - `LocalVolume` data is indexed in the SAME order as its `CoordinateSpace` names, and
    the viewer position is too. We use x,y,z with the test pattern varying along x, so
    one screen row is a complete transfer-function sample.
  - `crossSectionScale = 1.0` with 1 nm voxels is exactly 1 voxel per screen pixel, so
    pixel i of the row is voxel i. Neuroglancer samples with GL_NEAREST, so there is no
    resampling to explain away.
  - The cross-section background is set to pure red. Neuroglancer leaves the background
    showing wherever a chunk has not loaded, and its screenshot can return before the
    layer appears; a red pixel therefore means "not rendered", not "black data". Every
    case asserts the row contains no red.

Requires a browser. Neuroglancer rasterizes client-side, so its screenshot API drives
selenium. On this machine Firefox's true-headless compositor cannot map a framebuffer
(`RenderCompositorSWGL failed mapping default framebuffer`), so we run a real Firefox
under a virtual X server instead:

    conda activate cinemap        # puts geckodriver 0.37.1 ahead of /snap/bin's 0.37.0
    xvfb-run -a -s "-screen 0 1280x1024x24" python spikes/ng_parity/shader_parity.py

Without the conda env on PATH, selenium picks up the snap geckodriver, which is confined
and cannot see /groups, and fails with "binary is not a Firefox executable".
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cinemap.data import ng_shader  # noqa: E402

N = 256
BG = (255, 0, 0)          # "nothing was drawn here"


def firefox_binary() -> str:
    """The real Firefox ELF, not conda's wrapper script (geckodriver rejects the script)."""
    wrapper = shutil.which("firefox")
    if not wrapper:
        sys.exit("no firefox on PATH; mamba install -n cinemap -c conda-forge firefox geckodriver")
    app = Path(wrapper).parent / "FirefoxApp" / "firefox"
    return str(app if app.exists() else wrapper)


def ramp(dtype) -> np.ndarray:
    """A [x, y, z] volume whose value varies only along x, covering the dtype's range."""
    info = np.iinfo(dtype)
    xs = np.linspace(info.min, info.max, N).astype(dtype)
    return np.ascontiguousarray(np.tile(xs[:, None, None], (1, N, 1)))


# --------------------------------------------------------------------------- cases
COLOR_SHADER = """
#uicontrol invlerp normalized
#uicontrol vec3 tint color(default="#ff8000")
void main() { emitRGB(tint * normalized()); }
"""

JET_SHADER = """
#uicontrol invlerp normalized
void main() { emitRGB(colormapJet(normalized())); }
"""

CLAMP_OFF_SHADER = """
#uicontrol invlerp normalized(range=[50,200], clamp=false)
void main() { emitGrayscale(normalized()); }
"""

RGBA_SHADER = """
#uicontrol invlerp normalized
void main() { emitRGBA(vec4(normalized(), 1.0 - normalized(), 0.25, 1.0)); }
"""

CASES = [
    ("uint8 default shader",        np.uint8,  {}),
    ("uint8 range [50,200]",        np.uint8,  {"shaderControls": {"normalized": {"range": [50, 200]}}}),
    ("uint8 inverted range",        np.uint8,  {"shaderControls": {"normalized": {"range": [200, 50]}}}),
    ("uint8 bare array -> ignored", np.uint8,  {"shaderControls": {"normalized": [30, 220]}}),
    ("uint8 window must be ignored", np.uint8, {"shaderControls": {"normalized": {"range": [50, 200],
                                                                                  "window": [0, 255]}}}),
    ("uint8 clamp=false directive", np.uint8,  {"shader": CLAMP_OFF_SHADER}),
    ("uint8 color control",         np.uint8,  {"shader": COLOR_SHADER}),
    ("uint8 color override",        np.uint8,  {"shader": COLOR_SHADER,
                                                "shaderControls": {"tint": "#00c0ff"}}),
    ("uint8 colormapJet",           np.uint8,  {"shader": JET_SHADER}),
    ("uint8 emitRGBA",              np.uint8,  {"shader": RGBA_SHADER}),
    ("uint16 default shader",       np.uint16, {}),
    ("uint16 range [1000,40000]",   np.uint16, {"shaderControls": {"normalized": {"range": [1000, 40000]}}}),
    ("uint8 opacity 0.5 == full",   np.uint8,  {"opacity": 0.5}),
    ("uint8 opacity 0.25 == full",  np.uint8,  {"opacity": 0.25}),
    ("uint8 opacity 0 -> hidden",   np.uint8,  {"opacity": 0.0}),
]


def expected_row(layer: dict, data_row: np.ndarray) -> tuple[np.ndarray, str]:
    """What CineMap bakes into the slice PNG for this layer, as a (N, 3) uint8 row."""
    shader = ng_shader.from_layer(layer, data_row.dtype)
    rgb, warn = ng_shader.shade(shader, data_row[None, :])
    return rgb[0], warn


def composite(rgb: np.ndarray, layer: dict) -> np.ndarray:
    """Apply what neuroglancer's `opacity` ACTUALLY does, per ng_shader.

    Not an alpha blend. The bottom-most image layer is drawn with GL blending disabled,
    so its opacity never modulates RGB; the panel then shows the background only where
    alpha is exactly 0. This function deliberately calls the production rule so a change
    to it is checked against the live viewer rather than against a second copy of the
    assumption.
    """
    eff = ng_shader.effective_image_opacity({"layers": [{**layer, "type": "image",
                                                         "name": "em"}]},
                                            {**layer, "type": "image", "name": "em"})
    if eff >= 1.0:
        return rgb
    bg = np.array(BG, dtype=np.float64)
    return np.clip(rgb * eff + bg * (1.0 - eff) + 0.5, 0, 255).astype(np.uint8)


def contact_sheet(rows: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    """One strip per case: neuroglancer on top, our prediction below, 16 px each."""
    from PIL import Image, ImageDraw

    band, gap = 16, 10
    h = len(rows) * (2 * band + gap) + gap
    img = Image.new("RGB", (N + 220, h), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    for i, (name, ng, ours) in enumerate(rows):
        y = gap + i * (2 * band + gap)
        img.paste(Image.fromarray(np.tile(ng[None], (band, 1, 1))), (210, y))
        img.paste(Image.fromarray(np.tile(ours[None], (band, 1, 1))), (210, y + band))
        draw.text((6, y + 4), f"{name}", fill=(230, 230, 230))
        draw.text((6, y + band + 4), f"  ng / cinemap", fill=(140, 140, 140))
    img.save(path)


def run_dtype(neuroglancer, dtype, cases, sheet) -> int:
    """Run every case for one dtype in its own viewer + browser session.

    One session per dtype, not one for everything: repointing a layer at a DIFFERENT
    LocalVolume mid-session makes `viewer.screenshot` block forever waiting on chunk
    statistics that never settle. Switching shader/opacity on a FIXED source is fine, so
    grouping by dtype keeps this to one browser launch per dtype.
    """
    import neuroglancer.webdriver

    viewer = neuroglancer.Viewer()
    dims = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=[1, 1, 1])
    volume = neuroglancer.LocalVolume(data=ramp(dtype), dimensions=dims)

    with viewer.txn() as s:
        s.dimensions = dims
        s.layout = "xy"
        s.position = [N / 2, N / 2, 0.5]
        s.cross_section_scale = 1.0                 # 1 voxel == 1 screen pixel
        s.cross_section_background_color = "#%02x%02x%02x" % BG
        s.show_axis_lines = False
        s.show_scale_bar = False
        s.show_default_annotations = False
        s.layers["em"] = neuroglancer.ImageLayer(source=volume)
    source_url = viewer.state.layers["em"].source[0].url

    driver = neuroglancer.webdriver.Webdriver(
        viewer, headless=False, browser="firefox",
        browser_binary_path=firefox_binary(), window_size=(600, 600), print_logs=False)

    failures = 0
    try:
        for name, layer in cases:
            # Layers are built from raw JSON, exactly as a captured state stores them. The
            # python wrapper's setters reject the bare-array `shaderControls` form, and how
            # neuroglancer treats that form is one of the things we need to observe.
            layer_json = {"type": "image", "source": source_url, **layer}
            with viewer.txn() as s:
                s.layers["em"] = neuroglancer.viewer_state.ImageLayer(json_data=layer_json)

            got = np.asarray(viewer.screenshot(size=(N, N)).screenshot.image_pixels)[N // 2, :, :3]
            if layer.get("opacity") == 0.0:
                # The background SHOWING is the expected result here, not a load failure.
                hidden = bool((got == np.array(BG)).all())
                print(f"{'PASS' if hidden else 'FAIL'}  {name:<32} "
                      f"background shown: {hidden}")
                failures += not hidden
                continue
            if (got == np.array(BG)).all(axis=-1).any():
                print(f"SKIP  {name}: background showed through (chunk not loaded)")
                failures += 1
                continue

            want, warn = expected_row(layer, ramp(dtype)[:, 0, 0])
            want = composite(want, layer)
            diff = np.abs(got.astype(int) - want.astype(int))
            ok = diff.max() <= 1                    # 1/255 for float->byte rounding
            failures += not ok
            note = f"  [fallback: {warn}]" if warn else ""
            print(f"{'PASS' if ok else 'FAIL'}  {name:<32} max|d|={diff.max():3}  "
                  f"mean|d|={diff.mean():5.2f}  bias={(got.astype(int) - want).mean():+6.2f}{note}",
                  flush=True)
            sheet.append((name, got, want))
            if not ok:
                bad = int(np.argmax(diff.max(axis=-1)))
                print(f"        worst at x={bad}: ng={got[bad].tolist()} ours={want[bad].tolist()}")
    finally:
        driver.driver.quit()
    return failures


def main() -> int:
    import neuroglancer

    neuroglancer.set_server_bind_address("127.0.0.1")
    by_dtype: dict[type, list] = {}
    for name, dtype, layer in CASES:
        by_dtype.setdefault(dtype, []).append((name, layer))

    failures = 0
    sheet: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        for dtype, cases in by_dtype.items():
            failures += run_dtype(neuroglancer, dtype, cases, sheet)
    finally:
        # Written even on a hang/abort, so the strips for completed cases survive.
        if sheet:
            out = (Path(sys.argv[1]) if len(sys.argv) > 1
                   else REPO / "spikes" / "ng_parity" / "out" / "parity.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            contact_sheet(sheet, out)
            print(f"contact sheet: {out}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} cases match neuroglancer")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
