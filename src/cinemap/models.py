"""Pydantic data model — the project, keyframes, and render jobs.

This mirrors the data model in plan.md. A Keyframe is a Blender-renderable scene
state (camera + EM slice planes + meshes); a video is interpolation between
consecutive keyframes. project.json is the single source of truth that both the
UI and the Claude agent mutate (through operations.py).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Axis = Literal["x", "y", "z"]


# ----------------------------- data sources -----------------------------
class MeshSource(BaseModel):
    """A precomputed (neuroglancer) mesh layer: name -> mesh dir + selected ids."""

    name: str
    mesh_url: str = ""  # multilod-draco mesh dir + segment_properties (for ids/bbox)
    label_zarr: str = ""  # OME-Zarr label volume — preferred geometry via marching cubes
    skeleton_url: str = ""  # precomputed neuroglancer_skeletons dir (rendered as tubes)
    skeleton_shader: str = ""  # NG skeletonRendering.shader -> matched colormap on tubes
    segment_ids: list[int] = Field(default_factory=list)


class EMSource(BaseModel):
    """An OME-Zarr multiscale EM volume to slice from."""

    name: str
    zarr_url: str  # directory holding .zattrs (multiscales) + sN/ levels
    voxel_size_nm: list[float] = Field(default_factory=lambda: [8.0, 8.0, 8.0])


class Manifest(BaseModel):
    """Result of analyzing a dataset (from a neuroglancer state or a path)."""

    title: str = "untitled"
    server: str = ""
    em: Optional[EMSource] = None
    meshes: list[MeshSource] = Field(default_factory=list)
    voxel_size_nm: list[float] = Field(default_factory=lambda: [8.0, 8.0, 8.0])
    bbox_nm: Optional[list[list[float]]] = None  # [[x0,y0,z0],[x1,y1,z1]]


# ----------------------------- scene / keyframe -----------------------------
class Camera(BaseModel):
    # All in nm world coordinates; orientation is a quaternion [x,y,z,w].
    position_nm: list[float]
    look_at_nm: list[float]
    fov_deg: float = 40.0
    up: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])


class SlicePlane(BaseModel):
    em_name: str = "em"
    axis: Axis = "z"
    position_nm: float = 0.0           # offset of the plane = dot(point, normal)
    # None => axis-aligned plane (perpendicular to `axis`). A unit xyz vector =>
    # an oblique plane with that normal (the EM is resampled on the tilted plane).
    normal: Optional[list[float]] = None
    scale_level: Optional[int] = None  # None => auto-pick from on-screen extent
    opacity: float = 1.0
    visible: bool = True


class ClipPlane(BaseModel):
    """A cutaway plane for a single mesh layer: geometry on the hidden side of the
    plane is made transparent in the 3D render, revealing what's inside/behind.
    Independent of EM slices, and applied per layer (cut the cell, keep the mitos)."""
    axis: Axis = "z"
    position_nm: float = 0.0          # offset of the plane = dot(point, normal)
    normal: Optional[list[float]] = None  # unit xyz; None => axis-aligned (e_axis)
    side: int = 1          # +1 hides the +normal side of the plane; -1 hides the other
    enabled: bool = False


class MeshInstance(BaseModel):
    mesh_name: str
    segment_ids: list[int] = Field(default_factory=list)
    color: list[float] = Field(default_factory=lambda: [0.91, 0.45, 0.23])
    opacity: float = 1.0
    visible: bool = True
    # render as a 3D mesh? A layer with a precomputed-mesh source does; a
    # label-only segmentation layer is shown only on the EM slice (like neuroglancer).
    render_3d: bool = True
    # neuroglancer color config (captured per keyframe -> coloring can change frame
    # to frame): hash seed, one fixed color for all, and per-segment overrides.
    color_seed: int = 0
    default_color: Optional[list[float]] = None
    segment_colors: dict[str, list[float]] = Field(default_factory=dict)
    saturation: float = 1.0     # NG layer saturation (0 = grayscale meshes)
    # neuroglancer 3D mesh render state (per keyframe -> can change frame to frame)
    object_alpha: float = 1.0   # NG "Opacity (3d)"  (objectAlpha)
    silhouette: float = 0.0     # NG "Silhouette (3d)" (meshSilhouetteRendering)
    clip: Optional[ClipPlane] = None   # cutaway plane (render-only; per layer)


class AnnotationInstance(BaseModel):
    """A neuroglancer annotation layer captured for a keyframe. Geometry is stored
    normalized in nm (x/y/z) so the render path is source-agnostic (inline now;
    precomputed later)."""

    name: str
    color: list[float] = Field(default_factory=lambda: [1.0, 0.95, 0.30])
    visible: bool = True
    opacity: float = 1.0
    points: list[list[float]] = Field(default_factory=list)         # [[x,y,z], ...]
    lines: list[list[list[float]]] = Field(default_factory=list)    # [[[x,y,z],[x,y,z]], ...]
    boxes: list[list[list[float]]] = Field(default_factory=list)    # [[[lo],[hi]], ...]
    ellipsoids: list[dict] = Field(default_factory=list)            # [{center, radii}, ...]
    point_radius_nm: float = 80.0
    line_radius_nm: float = 40.0


class Lighting(BaseModel):
    key_energy: float = 3000.0
    background: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])  # black, like NG


class Keyframe(BaseModel):
    id: str
    label: str = ""
    camera: Camera
    slices: list[SlicePlane] = Field(default_factory=list)
    meshes: list[MeshInstance] = Field(default_factory=list)
    annotations: list[AnnotationInstance] = Field(default_factory=list)
    lighting: Lighting = Field(default_factory=Lighting)
    duration_in_s: float = 2.0  # transition duration INTO this keyframe (the move)
    hold_in_s: float = 0.0      # rest/dwell ON this keyframe's pose before moving on
    easing: Literal["linear", "ease-in-out", "ease-in", "ease-out"] = "ease-in-out"
    ng_state: Optional[dict] = None  # originating scouting state (round-trip)
    thumbnail_path: Optional[str] = None
    # keyframes produced together by a generated move (plane scan, orbit, …) share a
    # group id + label so the timeline can collapse them into one stacked card.
    group: Optional[str] = None
    group_label: str = ""


class RenderSettings(BaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 30
    samples: int = 64
    engine: Literal["CYCLES", "BLENDER_EEVEE_NEXT"] = "CYCLES"
    # when set, the job produces a self-contained .blend (camera, mesh and slice
    # animation baked to F-curves; textures packed in) instead of rendering frames.
    export_blend: bool = False
    # draft = fast preview/thumbnail quality: coarse EM slice level + low-voxel
    # meshes (see RenderWorker). Off = full resolution for the final video.
    draft: bool = False
    # still = render exactly ONE frame (a per-keyframe thumbnail), ignoring holds and
    # sweep timeline-extension so it stays a single PNG (not an mp4).
    still: bool = False
    # By default a layer's precomputed meshes are downloaded (fast, LOD-adaptive,
    # matches neuroglancer). Set this to instead regenerate watertight meshes from
    # the OME-Zarr label volume via marching cubes when one is available.
    mesh_from_labels: bool = False
    # Mesh detail multiplier on the per-layer vertex budget (1.0 = default 5M full /
    # 1.2M draft). Higher = crisper meshes but more VRAM; the worker hard-caps the
    # budget and auto-retries at lower detail if the GPU runs out of memory.
    mesh_detail: float = 1.0
    # Auto-direction: a non-destructive presentation pass (camera-relative key/fill/
    # rim lighting, publication materials, subtle depth-of-field on the framed
    # subject). On by default; off renders the plain neuroglancer-faithful scene.
    auto_direct: bool = True
    # Mesh LOD strategy:
    #   "single" — one LOD for the whole shot (built at the closest frame's scale).
    #   "frame"  — per-frame LOD: coarser when a layer is far/small on screen, finer
    #              on close-ups (like neuroglancer); collapses to one build on orbits.
    #   "chunk"  — per-chunk spatial LOD (different LODs within one mesh); not yet
    #              implemented, currently falls back to "frame".
    lod_mode: Literal["single", "frame", "chunk"] = "frame"


class RenderJob(BaseModel):
    id: str
    kf_range: Optional[list[int]] = None  # [start, end] keyframe indices; None => all
    settings: RenderSettings = Field(default_factory=RenderSettings)
    status: Literal["pending", "running", "done", "error", "cancelled"] = "pending"
    progress: float = 0.0
    message: str = ""
    output_path: Optional[str] = None


class Sweep(BaseModel):
    """An animated effect on its OWN timeline, independent of the camera keyframes.
    A cutaway sweep slides a layer's clip plane from `from_nm` to `to_nm` over a span of
    the movie's global time [start_s, start_s+duration_s]; the camera meanwhile does
    whatever its keyframes say. This decouples 'what the plane does' from 'where the
    camera is' — a sweep is one self-contained thing laid on top of the camera track."""
    id: str
    # cutaway = slice a mesh layer's clip plane; slice = sweep an EM cross-section plane.
    kind: Literal["cutaway", "slice"] = "cutaway"
    layer: str = ""                        # mesh layer name the clip plane cuts (cutaway)
    em_name: str = ""                      # EM layer name the slice shows (slice)
    axis: Axis = "z"
    normal: Optional[list[float]] = None   # oblique plane; None => axis-aligned
    side: int = 1
    from_nm: float = 0.0
    to_nm: float = 0.0
    start_s: float = 0.0                   # global-timeline start (seconds)
    duration_s: float = 2.0
    easing: Literal["linear", "ease-in-out", "ease-in", "ease-out"] = "linear"
    opacity: float = 1.0                   # EM slice overlay strength (slice kind)
    mirror: bool = False                   # ping-pong: sweep from->to then back to->from
    enabled: bool = True


class Project(BaseModel):
    id: str
    name: str
    data_path: str = ""  # neuroglancer state URL or dataset path
    manifest: Manifest = Field(default_factory=Manifest)
    lighting: Lighting = Field(default_factory=Lighting)
    keyframes: list[Keyframe] = Field(default_factory=list)
    # independent animated effects (cutaway sweeps) overlaid on the camera timeline
    sweeps: list[Sweep] = Field(default_factory=list)
    renders: list[RenderJob] = Field(default_factory=list)
    look: dict[str, Any] = Field(default_factory=dict)  # render look overrides (preset +
    # glow/roughness/specular/ng_shader/view) — the look-experiment panel writes this.
