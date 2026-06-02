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
from .models import Camera, Keyframe, MeshInstance, Project, SlicePlane

_viewer: neuroglancer.Viewer | None = None


def get_viewer() -> neuroglancer.Viewer:
    global _viewer
    if _viewer is None:
        neuroglancer.set_server_bind_address(
            os.environ.get("CINEMAP_NG_BIND", "127.0.0.1")
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


def _meshes_from_visible(project: Project, prev: list[MeshInstance] | None = None) -> list[MeshInstance]:
    """Build mesh instances from the current NG visible segments, preserving the
    color/opacity of any matching previous instance."""
    prev_by = {m.mesh_name: m for m in (prev or [])}
    meshes: list[MeshInstance] = []
    for name, ids in current_visible_segments(project).items():
        base = prev_by.get(name)
        if base is not None:
            meshes.append(base.model_copy(update={"segment_ids": ids}))
        else:
            meshes.append(MeshInstance(mesh_name=name, segment_ids=ids))
    return meshes


def _scene_from_view(project: Project):
    """Capture the current scouting view as (camera, slices, meshes, ng_state):
    the 3D-view camera, the EM slice (if the image layer is shown), and meshes for
    the visible segmentation layers/segments."""
    from .data.ng_camera import ng_to_camera

    st = get_viewer().state.to_json()
    cam = ng_to_camera(st, project.manifest.voxel_size_nm)
    em_name = project.manifest.em.name if project.manifest.em else "em"
    vis = current_layer_visibility(project)
    slices = ([SlicePlane(em_name=em_name, axis="z", position_nm=cam.look_at_nm[2])]
              if vis.get(em_name, True) else [])
    meshes = _meshes_from_visible(project)
    return cam, slices, meshes, st


def bake_keyframe(project: Project, label: str = "scouted") -> Keyframe:
    """Build a NEW keyframe from the current scouting view."""
    cam, slices, meshes, st = _scene_from_view(project)
    if not meshes and not slices and project.keyframes:  # nothing on -> keep previous meshes
        meshes = [m.model_copy() for m in project.keyframes[-1].meshes]
    kf = Keyframe(id=ops._uid("kf"), label=label, camera=cam, slices=slices,
                  meshes=meshes, ng_state=st)
    return ops.add_keyframe(project, kf)


def update_keyframe_from_view(project: Project, keyframe_id: str) -> Keyframe | None:
    """Overwrite an existing keyframe with the current Neuroglancer state (camera +
    layers + segments), keeping its timing (duration/easing) and label."""
    kf = next((k for k in project.keyframes if k.id == keyframe_id), None)
    if kf is None:
        return None
    cam, slices, meshes, st = _scene_from_view(project)
    updated = kf.model_copy(update={"camera": cam, "slices": slices,
                                    "meshes": meshes, "ng_state": st})
    project.keyframes = [updated if k.id == keyframe_id else k for k in project.keyframes]
    ops.store.save(project)
    return updated


def sync_segments(project: Project, keyframe_id: str) -> Keyframe | None:
    """Push the current neuroglancer layer state onto an existing keyframe
    (camera unchanged): which segmentation layers + segments are shown, and
    whether the EM slice is shown — propagated into the frame and thus Blender."""
    kf = next((k for k in project.keyframes if k.id == keyframe_id), None)
    if kf is None:
        return None
    vis = current_layer_visibility(project)
    meshes = _meshes_from_visible(project, kf.meshes)

    # slice on/off follows the EM image layer; keep its axis/position
    em_name = project.manifest.em.name if project.manifest.em else "em"
    if kf.slices:
        slices = [s.model_copy(update={"visible": vis.get(s.em_name, True)}) for s in kf.slices]
    elif vis.get(em_name, True):  # EM turned on but keyframe had no slice -> add one
        slices = [SlicePlane(em_name=em_name, axis="z", position_nm=kf.camera.look_at_nm[2])]
    else:
        slices = []

    updated = kf.model_copy(update={"meshes": meshes, "slices": slices})
    project.keyframes = [updated if k.id == keyframe_id else k for k in project.keyframes]
    ops.store.save(project)
    return updated


def goto_keyframe(project: Project, keyframe_id: str) -> bool:
    """Navigate the scouting viewer to a keyframe's view (the iframe updates live)."""
    from .data.ng_camera import camera_to_ng

    kf = next((k for k in project.keyframes if k.id == keyframe_id), None)
    if kf is None:
        return False
    v = get_viewer()
    if kf.ng_state:  # baked from neuroglancer -> exact round-trip
        v.set_state(kf.ng_state)
    else:  # programmatic keyframe -> synthesize a state from the camera
        base = v.state.to_json()
        v.set_state(camera_to_ng(kf.camera, project.manifest.voxel_size_nm, base))
    return True
