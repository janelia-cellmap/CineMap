#!/usr/bin/env python
"""Compare a CineMap render against a real neuroglancer screenshot of the same state.

The unit tests prove our shader math matches neuroglancer's formulas and that Blender's
color management round-trips the values (tests/test_ng_shader.py,
tests/test_blender_parity.py). This closes the loop against the actual viewer, so a
divergence we did not think to encode as a formula still gets caught.

Usage:
    python spikes/ng_parity/compare.py STATE [-o OUT] [--size W H]

STATE is a neuroglancer state: a share link, a URL, or a path to a JSON file.

Requires a browser: neuroglancer renders in WebGL, so its screenshot API drives a
headless chromedriver. `selenium` is already a project dependency, but the chromium and
chromedriver BINARIES may not be:

    mamba install -n cinemap -c conda-forge chromium chromedriver

Without them this script exits with that message rather than a stack trace.

What it reports, per test case, is the mean and max absolute per-channel difference
(0-255) plus the mean signed bias. Bias matters more than magnitude: a uniform positive
bias means a gamma/transfer-function problem (the class of bug that made EM look washed
out), whereas scattered high max with ~0 mean is usually resampling at edges.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


# --------------------------------------------------------------------------- state
def load_state(spec: str) -> dict:
    """A state dict from a file path, a raw URL, or a neuroglancer share link."""
    p = Path(spec)
    if p.exists():
        return json.loads(p.read_text())
    if "#!" in spec:                       # share link: fragment holds the state
        frag = spec.split("#!", 1)[1]
        frag = urllib.parse.unquote(frag)
        if frag.startswith("http"):        # #!<url-to-json>
            with urllib.request.urlopen(frag, timeout=30) as r:
                return json.load(r)
        return json.loads(frag)
    with urllib.request.urlopen(spec, timeout=30) as r:
        return json.load(r)


def _image_layers(state: dict) -> list[dict]:
    return [l for l in state.get("layers", []) if l.get("type") == "image"]


def variants(state: dict) -> list[tuple[str, dict]]:
    """Test cases: the state as given, plus contrast/opacity perturbations.

    Rendering only the state as-is would pass even if shaderControls were ignored
    entirely (the default window is usually the full dtype range). Perturbing contrast is
    what actually proves the value is read and applied.
    """
    out: list[tuple[str, dict]] = [("as-given", state)]
    imgs = _image_layers(state)
    if not imgs:
        return out
    name = imgs[0].get("name", "image")

    def with_layer(desc: str, **fields):
        s = json.loads(json.dumps(state))
        for layer in s.get("layers", []):
            if layer.get("name") == name:
                layer.update(fields)
        out.append((desc, s))

    with_layer("contrast 40-180",
               shaderControls={"normalized": {"range": [40, 180]}})
    with_layer("contrast 80-120 (narrow)",
               shaderControls={"normalized": {"range": [80, 120]}})
    with_layer("opacity 0.5", opacity=0.5)
    with_layer("inverted range",
               shaderControls={"normalized": {"range": [255, 0]}})
    return out


# ---------------------------------------------------------------------- capture
BROWSER_HELP = (
    "neuroglancer renders in WebGL, so every one of its capture paths — viewer.screenshot,\n"
    "neuroglancer.tool.screenshot and neuroglancer.tool.video_tool alike — drives a real\n"
    "browser through selenium. There is no pure-Python rasterizer to fall back on.\n"
    "Install the binaries (selenium itself is already a project dependency):\n"
    "    mamba install -n cinemap -c conda-forge chromium chromedriver")


def require_browser() -> None:
    """Fail fast with an actionable message if no chromedriver/chromium is present.

    Checked up front because otherwise Viewer() binds a server and the webdriver call
    blocks for a long time before failing, which looks like a hang.
    """
    import shutil

    driver = shutil.which("chromedriver")
    browser = next((b for b in ("chromium", "chromium-browser", "google-chrome",
                                "chrome", "firefox") if shutil.which(b)), None)
    if driver and browser:
        return
    missing = ", ".join(x for x, ok in (("chromedriver", driver), ("a browser", browser))
                        if not ok)
    sys.exit(f"cannot capture neuroglancer screenshots: {missing} not on PATH.\n"
             + BROWSER_HELP)


def ng_screenshot(state: dict, size: tuple[int, int]):
    """Screenshot `state` from a real neuroglancer, as an HxWx3 uint8 array."""
    try:
        import neuroglancer
        import neuroglancer.webdriver
    except ImportError as e:
        sys.exit(f"neuroglancer python package unavailable: {e}")

    viewer = neuroglancer.Viewer()
    viewer.set_state(state)
    try:
        driver = neuroglancer.webdriver.Webdriver(viewer, headless=True,
                                                  window_size=size)
    except Exception as e:  # noqa: BLE001 — almost always a missing browser binary
        sys.exit(
            "could not start headless chromedriver "
            f"({type(e).__name__}: {e}).\n"
            "neuroglancer renders in WebGL, so a browser is required. Install one:\n"
            "    mamba install -n cinemap -c conda-forge chromium chromedriver")
    with driver:
        with viewer.config_state.txn() as s:
            s.show_ui_controls = False
            s.show_panel_borders = False
        shot = viewer.screenshot(size=size).screenshot
    return np.asarray(shot.image_pixels)[:, :, :3]


def cinemap_render(state: dict, size: tuple[int, int], out_dir: Path):
    """Render the same state through CineMap, as an HxWx3 uint8 array."""
    from PIL import Image

    from cinemap import operations as ops, scouting, store
    from cinemap.models import RenderJob, RenderSettings
    from cinemap.render.worker import RenderWorker

    project = ops.new_project("ng-parity")
    scouting.bake_keyframe_from_state(project, state, label="parity")
    store.save(project)

    job = RenderJob(id="parity", settings=RenderSettings(
        width=size[0], height=size[1], still=True, draft=False, engine="CYCLES",
        auto_direct=False))
    worker = RenderWorker(project, job)
    worker.run()
    frames = sorted((out_dir if out_dir.exists() else worker.frames_dir).glob("frame_*.png"))
    if not frames:
        frames = sorted(worker.frames_dir.glob("frame_*.png"))
    if not frames:
        raise SystemExit("CineMap produced no frame")
    return np.asarray(Image.open(frames[0]).convert("RGB"))


# ---------------------------------------------------------------------- compare
def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference stats between two images, resized to a common shape if needed."""
    if a.shape != b.shape:
        from PIL import Image
        b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]),
                                                 Image.NEAREST))
    d = a.astype(np.int32) - b.astype(np.int32)
    return {
        "mean_abs": float(np.abs(d).mean()),
        "max_abs": int(np.abs(d).max()),
        "bias": float(d.mean()),          # signed: systematic gamma/transfer errors
        "p99_abs": float(np.percentile(np.abs(d), 99)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", help="share link, URL, or path to a neuroglancer state JSON")
    ap.add_argument("-o", "--out", default="spikes/ng_parity/out",
                    help="directory for ng/cinemap/diff PNGs")
    ap.add_argument("--size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    ap.add_argument("--tolerance", type=float, default=4.0,
                    help="max acceptable mean absolute difference (0-255)")
    args = ap.parse_args()

    from PIL import Image

    require_browser()
    state = load_state(args.state)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = (args.size[0], args.size[1])

    failures = 0
    for desc, st in variants(state):
        slug = desc.replace(" ", "_").replace("/", "-")
        print(f"\n=== {desc} ===", flush=True)
        ng = ng_screenshot(st, size)
        cm = cinemap_render(st, size, out / slug)
        stats = compare(ng, cm)
        Image.fromarray(ng).save(out / f"{slug}.ng.png")
        Image.fromarray(cm).save(out / f"{slug}.cinemap.png")
        diff = np.abs(ng.astype(np.int32) - cm.astype(np.int32)).astype(np.uint8)
        Image.fromarray(diff).save(out / f"{slug}.diff.png")

        ok = stats["mean_abs"] <= args.tolerance
        failures += not ok
        print(f"  mean|d| {stats['mean_abs']:6.2f}   p99 {stats['p99_abs']:6.2f}   "
              f"max {stats['max_abs']:3d}   bias {stats['bias']:+6.2f}   "
              f"{'PASS' if ok else 'FAIL'}")
        if abs(stats["bias"]) > 2.0:
            print("  ^ systematic bias — suspect a color-management / transfer-function "
                  "mismatch, not resampling")

    print(f"\nwrote images to {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
