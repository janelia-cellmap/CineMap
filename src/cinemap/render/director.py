"""Director — a non-destructive presentation pass over the scientific keyframes.

CineMap stays neuroglancer-first: geometry, positions, colors, visibility and
camera states all come from neuroglancer and are NEVER changed here. The director
only *adds* presentation directives that the Blender scene spec carries as optional
fields, so the look is "professionally directed" without any manual input:

  - a camera-relative key / fill / rim light rig (soft rim, consistent as the
    camera orbits)
  - publication-quality material parameters (specular / sheen) over the NG colors
  - subtle depth-of-field on the framed subject
  - an inferred "hero" object per keyframe (focus/emphasis metadata; used more by
    later phases — appear/highlight emphasis, focus pulls, reveals)

With auto-direct off, `plan()` is never called and the render is the plain,
neuroglancer-faithful scene (every directive field is optional in the spec).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MaterialProfile:
    """Principled-BSDF tuning applied over each layer's neuroglancer base color."""
    roughness: float = 0.38
    specular: float = 0.5        # Blender "Specular IOR Level"
    sheen: float = 0.08          # subtle soft-edge sheen for a publication look
    coat: float = 0.0
    emission_strength: float = 0.05  # faint self-illum floor so nothing is pure black


@dataclass
class LightRig:
    """Three-point rig, oriented relative to the camera each frame."""
    key_energy: float = 4.5      # SUN irradiance (W/m^2)
    fill_ratio: float = 0.4      # fill = key * this
    rim_ratio: float = 0.85      # rim/back = key * this
    camera_relative: bool = True
    ambient: float = 0.25        # world background multiplier (soft global fill)


@dataclass
class DepthOfField:
    enabled: bool = True
    fstop: float = 4.0           # subtle; higher = less background blur


@dataclass
class DirectorSettings:
    material: MaterialProfile = field(default_factory=MaterialProfile)
    lighting: LightRig = field(default_factory=LightRig)
    dof: DepthOfField = field(default_factory=DepthOfField)


def _visible_layer_segments(kf) -> dict[str, frozenset]:
    """{layer name: visible segment set} for this keyframe's 3D mesh layers."""
    out: dict[str, frozenset] = {}
    for m in kf.meshes:
        if (getattr(m, "render_3d", True) and getattr(m, "visible", True)
                and m.segment_ids):
            out[m.mesh_name] = frozenset(m.segment_ids)
    return out


def infer_heroes(keyframes) -> list[dict]:
    """Per keyframe, the most salient ("hero") layer and why. Salience priority,
    matching how a viewer's eye is drawn:
      introduced (new this keyframe) > highlighted (color/opacity changed) >
      focal (fewest segments — the specific object vs the bulk context) > first 3D.
    Pure presentation metadata (focus/emphasis); it never alters the data."""
    heroes: list[dict] = []
    prev_segs: dict[str, frozenset] = {}
    prev_style: dict[str, tuple] = {}
    for kf in keyframes:
        segs = _visible_layer_segments(kf)
        style = {m.mesh_name: (tuple(m.color or []),
                               round(float(getattr(m, "object_alpha", 1.0)), 3))
                 for m in kf.meshes}
        hero, reason = None, ""
        appeared = [n for n in segs if n not in prev_segs]
        changed = [n for n in segs if n in prev_style and style.get(n) != prev_style[n]]
        if appeared:
            hero, reason = min(appeared, key=lambda n: len(segs[n])), "introduced"
        elif changed:
            hero, reason = changed[0], "highlighted"
        elif segs:
            hero, reason = min(segs, key=lambda n: len(segs[n])), "focal"
        heroes.append({"hero": hero, "reason": reason})
        prev_segs, prev_style = segs, style
    return heroes


def plan(keyframes, settings: DirectorSettings | None = None) -> dict:
    """The presentation directives the worker injects into the scene spec."""
    s = settings or DirectorSettings()
    return {
        "material": asdict(s.material),
        "lighting": asdict(s.lighting),
        "dof": asdict(s.dof),
        "heroes": infer_heroes(keyframes),
    }
