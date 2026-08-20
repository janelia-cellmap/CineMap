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
    """'#rrggbb' -> [r,g,b] in 0-1, still sRGB-encoded (what neuroglancer displays)."""
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def srgb_to_linear(c: float) -> float:
    """One sRGB channel (0-1) -> linear.

    Neuroglancer's colors are display-referred sRGB, but Blender's color inputs are
    linear and its Standard view transform re-encodes to sRGB on output. Feeding an
    sRGB value straight in therefore renders it too bright and slightly off-hue. This
    is the same conversion `scouting._bg_from_state` already applies to the background
    so it round-trips exactly; mesh colors need it for the same reason.
    """
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


@dataclass
class LayerColors:
    seed: int = 0
    default: list[float] | None = None          # segmentDefaultColor (rgb 0-1) or None
    overrides: dict[int, list[float]] = field(default_factory=dict)  # segmentColors
    saturation: float = 1.0                     # layer `saturation` (0 = grayscale)

    def _desaturate(self, r, g, b) -> tuple[float, float, float]:
        if self.saturation >= 0.999:
            return (r, g, b)
        # neuroglancer mixes the color toward its luminance by (1 - saturation)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        s = self.saturation
        return (lum + (r - lum) * s, lum + (g - lum) * s, lum + (b - lum) * s)

    def rgb(self, seg_id: int) -> tuple[float, float, float]:
        sid = int(seg_id)
        if sid in self.overrides:
            r, g, b = self.overrides[sid]
        elif self.default is not None:
            r, g, b = self.default
        else:  # neuroglancer's exact hash coloring for (colorSeed, segment id)
            r, g, b = hex_to_rgb(_ngsc.hex_string_from_segment_id(self.seed, sid))
        return self._desaturate(r, g, b)

    def cache_key(self):
        return (self.seed, tuple(self.default) if self.default else None,
                tuple(sorted((k, tuple(v)) for k, v in self.overrides.items())),
                round(self.saturation, 3))


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
    sat = layer.get("saturation", 1.0)
    saturation = float(sat) if sat is not None else 1.0
    return LayerColors(seed=seed, default=default, overrides=overrides, saturation=saturation)


def render3d_from_layer(layer: dict) -> dict:
    """The neuroglancer 3D mesh render-tab settings from a layer JSON dict:
    'Opacity (3d)' (objectAlpha) and 'Silhouette (3d)' (meshSilhouetteRendering).
    Note: objectAlpha=0 is how neuroglancer HIDES a layer, so we must NOT fold 0
    into the default with `or` (0 is falsy) — only None falls back to the default."""
    oa = layer.get("objectAlpha", 1.0)
    sil = layer.get("meshSilhouetteRendering", 0.0)
    return {
        "object_alpha": float(oa) if oa is not None else 1.0,
        "silhouette": float(sil) if sil is not None else 0.0,
    }
