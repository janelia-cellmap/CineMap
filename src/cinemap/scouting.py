"""Neuroglancer scouting viewer — interactive exploration, NOT in the render path.

Holds one live neuroglancer python Viewer whose state mirrors to the embedded
browser iframe. `bake_keyframe` converts the current scouting view into a Blender
keyframe (camera target from the NG position; framing from the project defaults).
"""
from __future__ import annotations

import os

import neuroglancer

from . import operations as ops
from .data.manifest import fetch_state
from .data.ng_shader import effective_image_opacity
from .models import AnnotationInstance, Camera, Keyframe, MeshInstance, Project, SlicePlane

_viewer: neuroglancer.Viewer | None = None


def get_viewer() -> neuroglancer.Viewer:
    global _viewer
    if _viewer is None:
        # Bind on all interfaces so the embedded viewer is reachable when the app
        # is opened from another machine (the URL host is rewritten per-request to
        # whatever host the browser used — see server._ng_url_for).
        neuroglancer.set_server_bind_address(
            os.environ.get("CINEMAP_NG_BIND", "0.0.0.0"),
            int(os.environ.get("CINEMAP_NG_PORT", "0") or "0"),
        )
        _viewer = neuroglancer.Viewer()
    return _viewer


def viewer_url() -> str:
    return get_viewer().get_viewer_url()


def load_dataset(data_path: str) -> None:
    """Point the scouting viewer at the dataset (a neuroglancer state)."""
    state = fetch_state(data_path)
    get_viewer().set_state(state)


def current_state() -> dict:
    return get_viewer().state.to_json()


def current_layer_visibility(project: Project) -> dict[str, bool]:
    """{layer_name: is_visible} for every layer in the scouting viewer."""
    v = get_viewer()
    return {m.name: getattr(m, "visible", True) is not False for m in v.state.layers}


def _visible_segments_from_state(st: dict) -> dict[str, list[int]]:
    """{layer_name: [visible segment ids]} parsed straight from a neuroglancer state
    dict — no live-viewer round-trip, so no async race (the viewer may not have
    applied a just-set state yet). '!'-prefixed segments are selected-but-hidden and
    excluded, matching what the 3D view shows."""
    out: dict[str, list[int]] = {}
    for layer in st.get("layers", []):
        if layer.get("type") != "segmentation" or layer.get("visible", True) is False:
            continue
        ids = [int(s) for s in (str(x) for x in (layer.get("segments") or []))
               if not s.startswith("!") and s.lstrip("-").isdigit()]
        if ids:
            out[layer.get("name", "")] = sorted(ids)
    return out


def current_visible_segments(project: Project) -> dict[str, list[int]]:
    """{layer_name: [visible segment ids]} from the scouting viewer right now.

    Uses neuroglancer's `visible_segments`, so segments that are selected-but-
    hidden (the "!"-prefixed ones in the side panel) are excluded — exactly the
    set of ids currently shown in the 3D view.
    """
    v = get_viewer()
    by_name = {m.name for m in project.manifest.meshes}
    out: dict[str, list[int]] = {}
    for managed in v.state.layers:
        layer = managed.layer
        if managed.name in by_name and isinstance(layer, neuroglancer.SegmentationLayer):
            if getattr(managed, "visible", True) is False:
                continue
            vis = sorted(int(s) for s in layer.visible_segments)
            if vis:
                out[managed.name] = vis
    return out


def current_layer_colors(project: Project, st: dict | None = None) -> dict:
    """{layer_name: LayerColors} captured from the current neuroglancer state.
    Pass an already-serialized `st` to avoid a redundant full-state to_json()."""
    from .data import colors as _colors

    if st is None:
        st = get_viewer().state.to_json()
    return {l.get("name"): _colors.from_layer_dict(l)
            for l in st.get("layers", []) if l.get("type") == "segmentation"}


def slice_from_layer(em_name: str, layer: dict | None, st: dict | None = None,
                     **kw) -> SlicePlane:
    """Build a SlicePlane carrying the layer's neuroglancer appearance.

    EVERY SlicePlane should be built through here. Opacity and the shader/contrast state
    used to be dropped on the floor because each construction site set only geometry, so
    the render always used opacity 1.0 and neuroglancer's default contrast no matter what
    the viewer showed. Centralizing it means a new call site can't silently regress that.

    `opacity` is the EFFECTIVE opacity, not the raw layer value -- see
    `ng_shader.effective_image_opacity`. Neuroglancer defaults image opacity to 0.5 but
    does not blend the bottom-most image layer at all, so the raw value would render
    every EM slice at half strength against a viewer that shows it whole.
    """
    layer = layer or {}
    return SlicePlane(
        em_name=em_name,
        opacity=effective_image_opacity(st, layer),
        shader=layer.get("shader") or "",
        shader_controls=dict(layer.get("shaderControls") or {}),
        **kw,
    )


def _layer_by_name(st: dict, name: str) -> dict | None:
    return next((l for l in st.get("layers", []) if l.get("name") == name), None)


def _meshes_from_visible(project: Project, prev: list[MeshInstance] | None = None,
                         st: dict | None = None) -> list[MeshInstance]:
    """Build mesh instances from the current NG visible segments. A layer renders
    a 3D mesh only if it has a precomputed-mesh source (like neuroglancer);
    label-only segmentation layers are shown on the EM slice only. Also captures
    the layer's neuroglancer coloring (seed / fixed colors).

    `st` (a pre-serialized viewer state) lets the caller serialize once and reuse
    it here for colors + per-layer 3D state instead of re-serializing per call —
    each to_json() scales with the selected-segment count.
    """
    from .data import colors as _colors

    if st is None:
        st = get_viewer().state.to_json()
    # a layer renders 3D geometry if it has a precomputed mesh OR a skeleton source;
    # label-only layers stay on the EM slice (render_3d=False).
    has_mesh = {m.name: bool(m.mesh_url or m.skeleton_url) for m in project.manifest.meshes}
    lcolors = current_layer_colors(project, st)
    # per-layer 3D render state (Opacity/Silhouette) from the serialized state
    layers = {l.get("name"): l for l in st.get("layers", [])}
    vis = _visible_segments_from_state(st)
    # Resolve linkedSegmentationGroup: a visible segmentation layer with no segments
    # of its own but linked to another layer shows that layer's segments (e.g.
    # mito-objects-grouped, keyed by neuron id, linked to the neuron layer). Fetch
    # the linked layer's segments for it so the linked meshes render too.
    by_name = {m.name for m in project.manifest.meshes}
    for name, ldict in layers.items():
        link = ldict.get("linkedSegmentationGroup")
        if (link and name in by_name and ldict.get("visible", True) is not False
                and not vis.get(name) and vis.get(link)):
            vis[name] = list(vis[link])
    # CineMap-only material (metallic/roughness) isn't in the neuroglancer state, so a
    # freshly-baked keyframe would lose it. Inherit it per-layer from the previous keyframe
    # so "set reflective, then add a keyframe" carries the look forward (incl. after import).
    prev_by_name = {m.mesh_name: m for m in (prev or [])}
    meshes: list[MeshInstance] = []
    for name, ids in vis.items():
        lc = lcolors.get(name)
        fields = {"segment_ids": ids, "render_3d": has_mesh.get(name, True)}
        if lc is not None:
            fields.update(color_seed=lc.seed, default_color=lc.default,
                          segment_colors={str(k): v for k, v in lc.overrides.items()},
                          saturation=lc.saturation)
        if name in layers:
            fields.update(_colors.render3d_from_layer(layers[name]))  # Opacity/Silhouette (3d)
        pm = prev_by_name.get(name)
        if pm is not None:
            if getattr(pm, "metallic", None) is not None:
                fields["metallic"] = pm.metallic
            if getattr(pm, "roughness", None) is not None:
                fields["roughness"] = pm.roughness
        meshes.append(MeshInstance(mesh_name=name, **fields))
    return meshes


def _scene_from_view(project: Project, st: dict | None = None):
    """Capture a scene as (camera, slices, meshes, annotations, ng_state) from a
    neuroglancer state dict. `st` defaults to the live scouting viewer; importing
    passes the saved state directly so capture never races the viewer's async load."""
    from .data.ng_camera import ng_to_camera

    if st is None:
        st = get_viewer().state.to_json()  # serialize the live viewer ONCE; reuse below
    cam = ng_to_camera(st, project.manifest.voxel_size_nm)
    em_name = project.manifest.em.name if project.manifest.em else "em"
    layer_vis = {l.get("name"): l.get("visible", True) is not False
                 for l in st.get("layers", [])}
    # a "3d" layout shows no cross-section in neuroglancer, so bake no EM slice
    show_slice = layer_vis.get(em_name, True) and st.get("layout") != "3d"
    slices = ([slice_from_layer(em_name, _layer_by_name(st, em_name), st,
                                axis="z", position_nm=cam.look_at_nm[2])]
              if show_slice else [])
    prev = project.keyframes[-1].meshes if project.keyframes else None
    meshes = _meshes_from_visible(project, prev=prev, st=st)   # inherit material from last kf
    annotations = _annotations_from_view(project, st)
    return cam, slices, meshes, annotations, st


def _state_declares_render_layers(project: Project, st: dict) -> bool:
    """Whether this NG state explicitly says something about renderable layers.

    If a state has known mesh/segmentation or EM layers but they are hidden or empty, that
    is an intentional blank/slice-only state. Do not treat it as a failed capture and copy
    previous meshes forward.
    """
    mesh_names = {m.name for m in project.manifest.meshes}
    em_name = project.manifest.em.name if project.manifest.em else None
    for layer in st.get("layers", []):
        name = layer.get("name")
        typ = layer.get("type")
        if typ == "segmentation" and name in mesh_names:
            return True
        if typ == "image" and name == em_name:
            return True
    return False


def _hex_to_rgb(h: str) -> list[float]:
    h = (h or "").lstrip("#")
    if len(h) != 6:
        return [1.0, 0.95, 0.30]
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _annotations_from_view(project: Project, st: dict) -> list[AnnotationInstance]:
    """Capture visible annotation layers (inline points/lines/boxes/ellipsoids) from
    the serialized state. Layers backed only by a precomputed source (no inline
    `annotations`) are skipped for now."""
    from .data import annotations as _ann
    from .data.ng_camera import _voxel_nm_from_state, _xyz_perm

    vox = _voxel_nm_from_state(st, project.manifest.voxel_size_nm)  # NG grid, dim order
    perm = _xyz_perm(st)
    out: list[AnnotationInstance] = []
    for layer in st.get("layers", []):
        if layer.get("type") != "annotation":
            continue
        prims = _ann.parse_inline(layer, vox, perm)
        if not _ann.has_geometry(prims):
            continue
        out.append(AnnotationInstance(
            name=layer.get("name", "annotations"),
            color=_hex_to_rgb(layer.get("annotationColor", "#ffff4d")),
            visible=layer.get("visible", True) is not False,
            points=prims["points"], lines=prims["lines"],
            boxes=prims["boxes"], ellipsoids=prims["ellipsoids"]))
    return out


def bake_keyframe(project: Project, label: str = "scouted", st: dict | None = None) -> Keyframe:
    """Build a NEW keyframe from a neuroglancer state (the live view by default)."""
    cam, slices, meshes, annotations, st = _scene_from_view(project, st)
    _merge_manifest(project, st)   # learn layers new to this view (e.g. an EM image just added)
    if (not meshes and not slices and project.keyframes
            and not _state_declares_render_layers(project, st)):
        meshes = [m.model_copy() for m in project.keyframes[-1].meshes]
    kf = Keyframe(id=ops._uid("kf"), label=label, camera=cam, slices=slices,
                  meshes=meshes, annotations=annotations, ng_state=st)
    return ops.add_keyframe(project, kf)


def bake_keyframe_from_state(project: Project, state: dict, label: str = "imported") -> Keyframe:
    """Bake a keyframe from an arbitrary neuroglancer state (not the live view).

    Drives the scouting viewer to `state`, then bakes exactly as if the user had
    loaded that view and clicked Bake — so an imported keyframe is identical to a
    hand-baked one (visible layers/segments, colors, 3D style, camera). Used by the
    "import states" feature to turn a list of saved views into a keyframe timeline.
    """
    get_viewer().set_state(state)            # update the iframe so the user sees it…
    return bake_keyframe(project, label=label, st=state)  # …but capture from the dict


def _merge_manifest(project: Project, state: dict) -> None:
    """Union one imported state's mesh/EM sources into the project manifest, so a
    layer that appears only in a later state (e.g. a skeleton layer not present in
    state 1) is known to the renderer. Matches by layer name; first one wins."""
    from .data.manifest import analyze_state_dict

    m = analyze_state_dict(state)
    have = {x.name for x in project.manifest.meshes}
    for ms in m.meshes:
        if ms.name not in have:
            project.manifest.meshes.append(ms)
            have.add(ms.name)
    if project.manifest.em is None and m.em is not None:
        project.manifest.em = m.em


def _bg_from_state(state: dict) -> list[float] | None:
    """The neuroglancer 3D (perspective) view background = `projectionBackgroundColor`
    (a CSS hex). Default is black. Returned LINEAR so it round-trips through Blender's
    Standard (sRGB) view transform back to exactly the color NG shows."""
    from .data.colors import hex_to_rgb

    bg = state.get("projectionBackgroundColor")
    if not isinstance(bg, str) or not bg:
        return [0.0, 0.0, 0.0]
    try:
        srgb = hex_to_rgb(bg)
    except Exception:  # noqa: BLE001
        return [0.0, 0.0, 0.0]
    # sRGB -> linear (Blender background colors are linear)
    return [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in srgb]


def import_states(project: Project, links: list[tuple]) -> tuple[list[Keyframe], list[str]]:
    """Bake one keyframe per (label, state-link[, duration]). A per-entry duration
    (from a neuroglancer video_tool script) sets the transition INTO that keyframe.
    Returns (keyframes, errors); a link that fails to fetch/parse is reported and
    skipped, not fatal."""
    from .data.manifest import fetch_state

    created: list[Keyframe] = []
    errors: list[str] = []
    for entry in links:
        label, link = entry[0], entry[1]
        duration = entry[2] if len(entry) > 2 else None
        try:
            state = fetch_state(link)
            if not created:                  # first state sets the NG view background
                project.lighting.background = _bg_from_state(state)
            _merge_manifest(project, state)  # register layers new to this state
            kf = bake_keyframe_from_state(project, state, label=label)
            # match neuroglancer's video_tool: linear interpolation between states,
            # the script's number is the transition duration into this keyframe, and
            # an OMITTED duration defaults to 1.0s (NG's load_script default) — not
            # CineMap's normal 2.0s — so the total runtime matches NG exactly.
            kf.easing = "linear"
            kf.duration_in_s = float(duration) if duration is not None else 1.0
            created.append(kf)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: {e}")
    ops.store.save(project)  # persist the easing/duration overrides once for the batch
    return created, errors


def update_keyframe_from_view(project: Project, keyframe_id: str) -> Keyframe | None:
    """Overwrite an existing keyframe with the current Neuroglancer state (camera +
    layers + segments), keeping its timing (duration/easing) and label."""
    kf = next((k for k in project.keyframes if k.id == keyframe_id), None)
    if kf is None:
        return None
    cam, slices, meshes, annotations, st = _scene_from_view(project)
    _merge_manifest(project, st)   # learn layers new to this view (e.g. an EM image just added)
    updated = kf.model_copy(update={"camera": cam, "slices": slices, "meshes": meshes,
                                    "annotations": annotations, "ng_state": st})
    project.keyframes = [updated if k.id == keyframe_id else k for k in project.keyframes]
    ops.store.save(project)
    return updated


def sync_manifest_from_view(project: Project) -> Project:
    """Union the live neuroglancer view's layers into the project manifest WITHOUT
    baking a keyframe — so a volume just added in the viewer (e.g. an EM image, or a
    segmentation with a label volume) becomes sliceable right away. Returns the project."""
    st = get_viewer().state.to_json()
    _merge_manifest(project, st)
    ops.store.save(project)
    return project


def sync_segments(project: Project, keyframe_id: str) -> Keyframe | None:
    """Push the current neuroglancer layer state onto an existing keyframe
    (camera unchanged): which segmentation layers + segments are shown, and
    whether the EM slice is shown — propagated into the frame and thus Blender."""
    kf = next((k for k in project.keyframes if k.id == keyframe_id), None)
    if kf is None:
        return None
    st = get_viewer().state.to_json()  # serialize once; reuse for meshes
    vis = current_layer_visibility(project)
    meshes = _meshes_from_visible(project, kf.meshes, st=st)

    # slice on/off follows the EM image layer; keep its axis/position
    em_name = project.manifest.em.name if project.manifest.em else "em"
    if kf.slices:
        # refresh appearance from the live layer too, so a contrast/opacity change in
        # neuroglancer lands on the existing slice instead of only new ones
        slices = []
        for s in kf.slices:
            live = _layer_by_name(st, s.em_name)
            # Key presence, not truthiness: if the layer is in the live state, its
            # appearance is authoritative. `or` would make clearing a custom shader or
            # resetting the contrast in neuroglancer un-syncable, since the empty value
            # would silently fall back to the stale captured one.
            in_state = s.em_name in {l.get("name") for l in st.get("layers", [])}
            live = live or {}
            slices.append(s.model_copy(update={
                "visible": vis.get(s.em_name, True),
                "opacity": (effective_image_opacity(st, live) if in_state
                            else s.opacity),
                "shader": (live.get("shader") or "") if in_state else s.shader,
                "shader_controls": (dict(live.get("shaderControls") or {}) if in_state
                                    else dict(s.shader_controls)),
            }))
    elif vis.get(em_name, True):  # EM turned on but keyframe had no slice -> add one
        slices = [slice_from_layer(em_name, _layer_by_name(st, em_name), st,
                                   axis="z", position_nm=kf.camera.look_at_nm[2])]
    else:
        slices = []

    updated = kf.model_copy(update={"meshes": meshes, "slices": slices})
    project.keyframes = [updated if k.id == keyframe_id else k for k in project.keyframes]
    ops.store.save(project)
    return updated


def _rgb_to_hex(rgb) -> str:
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _state_from_keyframe(project: Project, kf: Keyframe, base: dict) -> dict:
    """Reconstruct a full neuroglancer state for a keyframe that has no captured
    ng_state (orbit/sweep/duplicate/establish frames). Starts from the live state
    (so all layers exist), then applies the keyframe's own view: per-segmentation-
    layer visible segments + visibility + colors, EM-image-layer visibility, and
    finally the camera/zoom. This makes clicking ANY keyframe restore everything,
    not just the camera."""
    from .data.ng_camera import camera_to_ng

    state = camera_to_ng(kf.camera, project.manifest.voxel_size_nm, base)
    meshes = {m.mesh_name: m for m in kf.meshes}
    shown_em = {s.em_name for s in kf.slices if getattr(s, "visible", True)}
    em_name = project.manifest.em.name if project.manifest.em else None
    for layer in state.get("layers", []):
        name = layer.get("name")
        m = meshes.get(name)
        if m is not None:                                   # segmentation layer
            layer["segments"] = [str(s) for s in (m.segment_ids or [])]
            layer["visible"] = bool(m.visible)
            layer["colorSeed"] = int(getattr(m, "color_seed", 0) or 0)
            if getattr(m, "default_color", None):
                layer["segmentDefaultColor"] = _rgb_to_hex(m.default_color)
            if getattr(m, "segment_colors", None):
                layer["segmentColors"] = {str(k): _rgb_to_hex(v)
                                          for k, v in m.segment_colors.items()}
            layer["objectAlpha"] = float(getattr(m, "object_alpha", 1.0))      # Opacity (3d)
            layer["meshSilhouetteRendering"] = float(getattr(m, "silhouette", 0.0))  # Silhouette (3d)
        elif em_name and name == em_name:                   # EM image layer
            layer["visible"] = name in shown_em
    return state


def goto_keyframe(project: Project, keyframe_id: str) -> bool:
    """Navigate the scouting viewer to a keyframe's view (the iframe updates live)."""
    kf = next((k for k in project.keyframes if k.id == keyframe_id), None)
    if kf is None:
        return False
    v = get_viewer()
    if kf.ng_state:  # baked from neuroglancer -> exact round-trip
        v.set_state(kf.ng_state)
    else:  # programmatic keyframe -> rebuild full state from the keyframe's data
        v.set_state(_state_from_keyframe(project, kf, v.state.to_json()))
    return True
