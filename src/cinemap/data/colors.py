"""Neuroglancer-matching segment colors.

Reproduces what neuroglancer shows for a segmentation layer:
  - `segmentColors`        per-segment fixed colors (overrides)
  - `segmentDefaultColor`  one fixed color for all segments
  - `colorSeed`            hash-based coloring (neuroglancer's own algorithm)

This is captured per keyframe, so the coloring can differ frame to frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import neuroglancer.segment_colors as _ngsc


def hex_to_rgb(h: str) -> list[float]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


@dataclass
class LayerColors:
    seed: int = 0
    default: list[float] | None = None          # segmentDefaultColor (rgb 0-1) or None
    overrides: dict[int, list[float]] = field(default_factory=dict)  # segmentColors

    def rgb(self, seg_id: int) -> tuple[float, float, float]:
        sid = int(seg_id)
        if sid in self.overrides:
            r, g, b = self.overrides[sid]
            return (r, g, b)
        if self.default is not None:
            r, g, b = self.default
            return (r, g, b)
        # neuroglancer's exact hash coloring for (colorSeed, segment id)
        return tuple(hex_to_rgb(_ngsc.hex_string_from_segment_id(self.seed, sid)))

    def cache_key(self):
        return (self.seed, tuple(self.default) if self.default else None,
                tuple(sorted((k, tuple(v)) for k, v in self.overrides.items())))


def from_layer_dict(layer: dict) -> LayerColors:
    """Build LayerColors from a neuroglancer layer JSON dict."""
    seed = int(layer.get("colorSeed", 0) or 0)
    dc = layer.get("segmentDefaultColor")
    default = hex_to_rgb(dc) if isinstance(dc, str) and dc else None
    overrides: dict[int, list[float]] = {}
    for k, v in (layer.get("segmentColors") or {}).items():
        try:
            overrides[int(k)] = hex_to_rgb(v) if isinstance(v, str) else list(v)
        except Exception:  # noqa: BLE001
            pass
    return LayerColors(seed=seed, default=default, overrides=overrides)
