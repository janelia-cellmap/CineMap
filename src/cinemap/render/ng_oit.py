"""Reference math for Neuroglancer's weighted blended transparency pass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OITFragment:
    """One transparent fragment as Neuroglancer's OIT emitter receives it.

    ``rgb`` must already be premultiplied by ``alpha`` and any mesh lighting or
    silhouette factor, matching Neuroglancer's mesh shader output.
    ``depth`` is WebGL ``gl_FragCoord.z`` in [0, 1].
    """

    rgb: tuple[float, float, float]
    alpha: float
    depth: float


def compute_oit_weight(alpha: float, depth: float) -> float:
    """Port Neuroglancer's ``computeOITWeight(alpha, depth)`` GLSL."""
    a = min(1.0, float(alpha)) * 8.0 + 0.01
    b = -float(depth) * 0.95 + 1.0
    return a * a * a * b * b * b


def composite_oit_over(
    fragments: Iterable[OITFragment],
    background: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Composite transparent fragments using Neuroglancer's OIT copy pass.

    This mirrors the perspective panel path:
    accumulate premultiplied color and weighted alpha, track revealage as the
    product of ``(1 - alpha)``, then copy over the existing color buffer with
    blend factors ``ONE_MINUS_SRC_ALPHA, SRC_ALPHA``.
    """
    accum_rgb = [0.0, 0.0, 0.0]
    accum_alpha = 0.0
    revealage = 1.0
    for frag in fragments:
        alpha = max(0.0, float(frag.alpha))
        weight = compute_oit_weight(alpha, frag.depth)
        accum_rgb[0] += float(frag.rgb[0]) * weight
        accum_rgb[1] += float(frag.rgb[1]) * weight
        accum_rgb[2] += float(frag.rgb[2]) * weight
        accum_alpha += alpha * weight
        revealage *= 1.0 - alpha
    bg = [float(background[i]) for i in range(3)]
    if accum_alpha <= 0.0:
        return (bg[0], bg[1], bg[2])
    color = [c / accum_alpha for c in accum_rgb]
    opacity = 1.0 - revealage
    return (
        color[0] * opacity + bg[0] * revealage,
        color[1] * opacity + bg[1] * revealage,
        color[2] * opacity + bg[2] * revealage,
    )
