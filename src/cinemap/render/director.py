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
    """Principled-BSDF tuning. Default is the neuVid-style beauty look: a glossy-ish
    Principled (roughness ~0.25 + specular -> highlights) lit by the 3-point rig, which
    gives real form-contrast (neuVid is Janelia's neuron-video tool; this mirrors its
    material). Smooth-shaded. Paired with AgX so highlights roll off (no oversaturation)."""
    roughness: float = 0.25      # glossy-ish (neuVid) -> specular highlights = pop/contrast
    specular: float = 0.5        # specular highlights catch the 3-point lights
    metallic: float = 0.0        # 0 = dielectric (default); 1 = metal (shiny, tinted reflection)
    transmission: float = 0.0    # 0 = opaque; >0 = glassy/translucent
    sheen: float = 0.0           # soft velvety edge sheen (waxy/clay look)
    coat: float = 0.0
    emission_strength: float = 0.02  # near-zero: shading comes from the lights, not self-glow
    edge_glow: float = 0.0       # off — the rim glow washed out the faceting
    ao: float = 0.0              # neuVid uses no AO; the 3-point lighting carries the form
    ao_distance_nm: float = 2000  # AO reach (only used if ao > 0)
    cavity: float = 0.0          # curvature (pointiness) shading: convex ridges brighten,
                                 # concave creases darken -> crisp surface contrast that
                                 # follows the geometry (sharper than AO; MeshLab-like).
                                 # 0 = off; ~0.5 = pronounced. Cheap (Cycles computes it).
    # Fresnel edge-darken border: OFF. On thin tubular meshes nearly all surface is at a
    # grazing angle, so it dims broadly instead of drawing clean borders (and Freestyle is
    # infeasible here). Real object outlines would need a compositor object-ID edge pass.
    edge_darken: float = 0.0
    edge_power: float = 2.5      # rim tightness (only used when edge_darken > 0)
    backface_cull: bool = False  # make back faces transparent: thin tubes/cell-bodies stop
                                 # doubling up (front+back) at low opacity -> glassier, more
                                 # see-through transparent state (closer to neuroglancer)
    flat_shading: bool = False   # smooth (per-vertex) normals like neuVid/NG -> rounded
                                 # form-shading on the tubes (not faceted)
    ng_shader: bool = False      # faithful neuroglancer mesh shader (emission-only headlight,
                                 # abs(N·view)*0.8+0.2) instead of lit Principled. The "ng" look.


@dataclass
class LightRig:
    """Three-point rig, oriented relative to the camera each frame."""
    # neuVid-style 3-point: a gray KEY (raking) + camera-front FILL + cool RIM. Colors
    # follow neuVid (key neutral-gray, fill slightly cool, rim cool/blue) -> a subtle
    # warm/cool studio dimension. With AgX the glossy highlights roll off (no clipping).
    key_energy: float = 6.0      # SUN irradiance (W/m^2)
    fill_ratio: float = 0.6      # camera-front fill (lifts the shadow side)
    rim_ratio: float = 0.5       # rim separates silhouettes from the dark background
    camera_relative: bool = True
    ambient: float = 0.18        # modest ambient
    key_color: tuple = (0.8, 0.8, 0.8)     # neutral gray key (neuVid)
    fill_color: tuple = (0.5, 0.5, 0.6)    # slightly cool fill (neuVid)
    rim_color: tuple = (0.8, 0.84, 1.0)    # cool/blue rim (neuVid)
    ambient_color: tuple = (1.0, 1.0, 1.0)
    # Optional 2nd back/edge light on the OPPOSITE side from the rim, in a contrasting
    # color -> cinematic two-tone edge separation (off by default; set kick_ratio > 0).
    kick_ratio: float = 0.0
    kick_color: tuple = (1.0, 1.0, 1.0)


@dataclass
class DepthOfField:
    enabled: bool = True
    fstop: float = 4.0           # subtle; higher = less background blur


@dataclass
class Emphasis:
    """Transient appear/highlight cue. DISABLED by default (glow/spotlight = 0): a
    per-object brightness flash is gaudy when many objects appear, and it lands a beat
    late (at the keyframe, after the fade-in). The reveal is carried by the fade-in
    itself; structures read as 'lit' via the constant bloom + edge-glow instead."""
    seconds: float = 0.7
    glow: float = 0.0
    spotlight: float = 0.0       # context layers dip to (1 - this) at the peak


@dataclass
class Bloom:
    """Soft glow on bright/emissive areas (compositor) — bright structures bloom
    against the dark background, the 'publication glow'. Constant, so it works the
    same with one object or thousands (unlike a per-object flash)."""
    enabled: bool = True
    threshold: float = 0.6       # brightness above which it blooms
    size: int = 7                # blur radius (larger = softer/wider glow)
    mix: float = -0.55           # -1 image only … +1 glare only; small = subtle add


@dataclass
class DirectorSettings:
    material: MaterialProfile = field(default_factory=MaterialProfile)
    lighting: LightRig = field(default_factory=LightRig)
    dof: DepthOfField = field(default_factory=DepthOfField)
    emphasis: Emphasis = field(default_factory=Emphasis)
    bloom: Bloom = field(default_factory=Bloom)
    smooth_camera: bool = False  # cinematic ease of the FIRST/LAST transition. Off by
                                 # default: neuroglancer's video_tool is pure linear, so
                                 # linear keeps our timing/motion exactly NG-faithful.
    # AgX rolls the bright raking highlights off instead of clipping to neon. Plain AgX
    # (no "Punchy") keeps the lighter, more even non-dramatic look closer to neuroglancer.
    view_transform: str = "AgX"
    view_look: str = ""


def make_settings(look: dict | None = None) -> "DirectorSettings":
    """Build DirectorSettings from a project `look` dict: an optional `preset`
    (neuvid | ng | rake | drama) sets a starting look, then explicit per-knob overrides
    (glow, edge_glow, roughness, specular, ng_shader, view_transform/look) apply on top.
    Lets the UI experiment with textures/glow/looks without code changes. Empty -> the
    default (neuVid-style) look."""
    s = DirectorSettings()
    look = look or {}
    preset = (look.get("preset") or "").lower()
    m, lr = s.material, s.lighting
    if preset == "ng":                       # faithful neuroglancer flat headlight shader
        m.ng_shader = True; m.flat_shading = False; m.ao = 0.0
        s.view_transform = "Standard"; s.view_look = ""
    elif preset == "rake":                    # raking key + camera fill, matte
        m.ng_shader = False; m.roughness = 0.55; m.specular = 0.15; m.ao = 0.6
        m.ao_distance_nm = 2000; m.flat_shading = True
        lr.key_energy = 7.5; lr.fill_ratio = 0.5; lr.rim_ratio = 0.5; lr.ambient = 0.18
        lr.key_color = lr.fill_color = lr.rim_color = lr.ambient_color = (1.0, 1.0, 1.0)
        s.view_transform = "AgX"; s.view_look = ""
    elif preset == "drama":                   # deep raking shadows + warm/cool + punchy
        m.ng_shader = False; m.roughness = 0.55; m.specular = 0.05; m.ao = 0.75
        m.ao_distance_nm = 1800; m.flat_shading = True
        lr.key_energy = 8.5; lr.fill_ratio = 0.3; lr.rim_ratio = 0.8; lr.ambient = 0.10
        lr.key_color = (1.0, 0.88, 0.72); lr.fill_color = (0.72, 0.82, 1.0)
        lr.rim_color = (0.78, 0.85, 1.0); lr.ambient_color = (0.85, 0.9, 1.0)
        s.view_transform = "AgX"; s.view_look = "AgX - Punchy"
    # else: neuvid (the defaults) — no change
    # explicit per-knob overrides on top of the preset
    if look.get("roughness") is not None: m.roughness = float(look["roughness"])
    if look.get("specular") is not None: m.specular = float(look["specular"])
    if look.get("metallic") is not None: m.metallic = float(look["metallic"])
    if look.get("transmission") is not None: m.transmission = float(look["transmission"])
    if look.get("sheen") is not None: m.sheen = float(look["sheen"])
    if look.get("coat") is not None: m.coat = float(look["coat"])
    if look.get("glow") is not None:      # one "glow" knob -> emission + edge glow
        m.emission_strength = float(look["glow"]); m.edge_glow = float(look["glow"])
    if look.get("edge_glow") is not None: m.edge_glow = float(look["edge_glow"])
    if look.get("emission_strength") is not None: m.emission_strength = float(look["emission_strength"])
    if look.get("ng_shader") is not None: m.ng_shader = bool(look["ng_shader"])
    if look.get("view_transform"): s.view_transform = look["view_transform"]
    if look.get("view_look") is not None: s.view_look = look["view_look"]
    return s


def _frame_starts(keyframes, fps: int) -> tuple[list[int], int]:
    """Frame index where each keyframe is shown (start of its outgoing transition),
    mirroring interpolate.build_frames, plus the total frame count (incl. the final
    held frame). Lets the director map per-keyframe events onto frame ranges."""
    starts: list[int] = []
    f = 0
    for i, kf in enumerate(keyframes):
        starts.append(f)
        if i < len(keyframes) - 1:
            d = keyframes[i + 1].duration_in_s
            f += 0 if d <= 0 else max(1, int(round(d * fps)))
    return starts, f + 1


def emphasis_track(keyframes, fps: int, settings: DirectorSettings | None = None):
    """Per-frame emphasis as a list of (hero_layer, glow_add, spotlight_factor). A
    glow pulse + context dip fires when a layer is introduced or highlighted, peaking
    as it appears and decaying over `emphasis.seconds`. The opening keyframe is
    skipped (nothing is 'revealed' there)."""
    s = settings or DirectorSettings()
    starts, total = _frame_starts(keyframes, fps)
    heroes = infer_heroes(keyframes)
    pulse = max(1, int(round(s.emphasis.seconds * fps)))
    track = [(None, 0.0, 1.0)] * total
    for i in range(1, len(heroes)):           # skip the opening keyframe
        h = heroes[i]
        if h["reason"] not in ("introduced", "highlighted") or not h["hero"]:
            continue
        for df in range(pulse):
            fi = starts[i] + df
            if fi >= total:
                break
            x = df / pulse                    # smooth bump: quick rise, soft falloff
            env = (x / 0.2) if x < 0.2 else (1.0 - (x - 0.2) / 0.8) ** 2
            glow = s.emphasis.glow * env
            spot = 1.0 - s.emphasis.spotlight * env
            cur = track[fi]
            if glow >= cur[1]:                # strongest overlapping event wins
                track[fi] = (h["hero"], glow, min(spot, cur[2]))
    return track


_VISIBLE_ALPHA = 0.1   # a layer counts as on-screen only above this effective opacity


def _layer_state(kf) -> dict[str, tuple]:
    """{layer name: (segment_count, effective_alpha, color)} for 3D mesh layers."""
    out: dict[str, tuple] = {}
    for m in kf.meshes:
        if getattr(m, "render_3d", True) and m.segment_ids:
            a = float(getattr(m, "object_alpha", 1.0)) if getattr(m, "visible", True) else 0.0
            out[m.mesh_name] = (len(m.segment_ids), round(a, 3), tuple(m.color or []))
    return out


def infer_heroes(keyframes) -> list[dict]:
    """Per keyframe, the most salient ("hero") layer and why — emphasis should fire
    only when a structure becomes MORE prominent, never as it fades out:
      introduced (newly visible — new layer, or one that just rose above invisible) >
      highlighted (recolored, or opacity clearly increased) >
      focal (fewest visible segments; no emphasis — just metadata).
    A layer that's (near-)invisible, or merely fading away, is never a hero. Pure
    presentation metadata; it never alters the data."""
    heroes: list[dict] = []
    prev: dict[str, tuple] = {}
    for kf in keyframes:
        st = _layer_state(kf)
        introduced, highlighted = [], []
        for name, (n, a, col) in st.items():
            if a < _VISIBLE_ALPHA:                         # not visibly shown
                continue
            if name not in prev or prev[name][1] < _VISIBLE_ALPHA:
                introduced.append((name, n))               # new, or just became visible
            else:
                _, pa, pcol = prev[name]
                if col != pcol or a > pa + 0.05:           # recolored / more opaque
                    highlighted.append((name, n))
        if introduced:
            hero, reason = min(introduced, key=lambda x: x[1])[0], "introduced"
        elif highlighted:
            hero, reason = min(highlighted, key=lambda x: x[1])[0], "highlighted"
        else:
            vis = [(name, s[0]) for name, s in st.items() if s[1] >= _VISIBLE_ALPHA]
            hero, reason = (min(vis, key=lambda x: x[1])[0] if vis else None), "focal"
        heroes.append({"hero": hero, "reason": reason})
        prev = st
    return heroes


def plan(keyframes, settings: DirectorSettings | None = None) -> dict:
    """The presentation directives the worker injects into the scene spec."""
    s = settings or DirectorSettings()
    return {
        "material": asdict(s.material),
        "lighting": asdict(s.lighting),
        "dof": asdict(s.dof),
        "bloom": asdict(s.bloom),
        "view": {"transform": s.view_transform, "look": s.view_look},
        "heroes": infer_heroes(keyframes),
    }
