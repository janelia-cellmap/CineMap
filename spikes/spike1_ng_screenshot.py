"""Spike #1 — Neuroglancer headless screenshot at a given state.

Proves the mvideo capture primitive: load a scene, drive a headless browser,
capture a PNG, mutate the camera (projection orientation quaternion), capture a
SECOND PNG. Two different frames from two states == a keyframe-driven video.

Uses a synthetic LocalVolume (image + segmentation) so the spike is fast,
deterministic, and offline — it isolates the *capture mechanism* from data I/O.

NOTE (production finding, see plan.md): capturing the real large remote EM volume
needs GPU-accelerated headless Chrome (EGL on the RTX 5090). Under xvfb,
`--use-gl=egl` hangs at browser launch, and software GL (`--disable-gpu`) is too
slow to ever finish loading a full EM volume. So headless-GPU capture tuning is a
tracked production task; the keyframe->frame mechanism itself is proven here.

Run:  xvfb-run -a conda run -n mv_env python spikes/spike1_ng_screenshot.py
Out:  spikes/out/ng_frame_000.png , ng_frame_001.png
"""
import math
import os

import neuroglancer
import numpy as np
from neuroglancer import webdriver

OUT = os.path.join(os.path.dirname(__file__), "out")
SIZE = [800, 600]


def synthetic_scene(viewer) -> None:
    """A tiny image volume + a labelled segmentation, as a stand-in for EM+seg."""
    n = 96
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    cx = cy = cz = n / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)

    # "EM" image: noisy shells
    image = (np.sin(r / 3.0) * 110 + 128 + np.random.default_rng(0).normal(0, 8, r.shape))
    image = np.clip(image, 0, 255).astype(np.uint8)

    # segmentation: 3 nested blobs => 3 instance labels
    seg = np.zeros_like(image, dtype=np.uint32)
    seg[r < 38] = 1
    seg[r < 26] = 2
    seg[(xx - cx) ** 2 + (yy - cy + 18) ** 2 + (zz - cz) ** 2 < 12**2] = 3

    space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[8, 8, 8]
    )
    with viewer.txn() as s:
        s.dimensions = space
        s.layers["em"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(image, dimensions=space)
        )
        s.layers["seg"] = neuroglancer.SegmentationLayer(
            source=neuroglancer.LocalVolume(seg, dimensions=space),
            segments=[1, 2, 3],
        )
        s.layout = "3d"
        s.position = [n / 2, n / 2, n / 2]


def save_png(reply, path: str) -> None:
    img = reply.screenshot.image
    with open(path, "wb") as f:
        f.write(img)
    print(f"[spike1] wrote {os.path.basename(path)} ({len(img)} bytes)", flush=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    neuroglancer.set_server_bind_address("127.0.0.1")
    viewer = neuroglancer.Viewer()
    synthetic_scene(viewer)
    print(f"[spike1] viewer url: {viewer.get_viewer_url()}", flush=True)

    # Key findings (the whole reason this spike exists):
    #  - do NOT use docker=True: it adds --disable-gpu which kills WebGL entirely,
    #    so neuroglancer never renders and screenshot() hangs forever.
    #  - default flags give working WebGL2 via SwiftShader (ANGLE/Vulkan, software).
    #  - print_logs=False: the BiDi console log-listener also blocks under xvfb.
    print("[spike1] launching headless Chrome (SwiftShader WebGL2)…", flush=True)
    wd = webdriver.Webdriver(
        viewer,
        headless=True,
        window_size=SIZE,
        print_logs=False,
        extra_command_line_args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    try:
        save_png(viewer.screenshot(size=SIZE), os.path.join(OUT, "ng_frame_000.png"))

        half = math.radians(35) / 2  # rotate projection 35° about Y — the keyframe move
        with viewer.txn() as s:
            s.projection_orientation = [0.0, math.sin(half), 0.0, math.cos(half)]
        save_png(viewer.screenshot(size=SIZE), os.path.join(OUT, "ng_frame_001.png"))

        f0 = os.path.join(OUT, "ng_frame_000.png")
        f1 = os.path.join(OUT, "ng_frame_001.png")
        ok = all(os.path.getsize(p) > 0 for p in (f0, f1))
        print(f"[spike1] {'PASS' if ok else 'FAIL'} — two frames at two camera states", flush=True)
    finally:
        wd.driver.quit()


if __name__ == "__main__":
    main()
