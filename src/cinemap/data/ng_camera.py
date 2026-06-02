"""Convert between a neuroglancer 3D-view and a Blender camera.

Neuroglancer's 3D panel (bottom-right) is defined by:
  - `position`            — the look-at point, in voxels of the data dimensions
  - `projectionOrientation` — quaternion [x,y,z,w] orienting the 3D camera
  - `projectionScale`     — zoom, in canonical voxels across the viewport

We map that to a Blender camera (look_at + position + fov), and back. Round-trips
for baked keyframes use the stored ng_state directly; these conversions are for
(a) giving a baked keyframe a matching render camera and (b) sending a
programmatic keyframe (orbit/sweep) back to neuroglancer.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..models import Camera


def _vox(voxel_nm):
    return np.array(voxel_nm, dtype=float)


def ng_to_camera(state: dict, voxel_nm, fov_deg: float = 40.0) -> Camera:
    pos_vox = np.array(state.get("position") or [0, 0, 0], dtype=float)
    look_at = pos_vox * _vox(voxel_nm)
    q = state.get("projectionOrientation") or [0.0, 0.0, 0.0, 1.0]
    scale = float(state.get("projectionScale", 10000.0))

    rot = Rotation.from_quat(q)  # neuroglancer & scipy both use [x,y,z,w]
    # neuroglancer view space: camera looks along -Z, up is +Y. World directions:
    fwd = rot.apply([0.0, 0.0, -1.0], inverse=True)
    up = rot.apply([0.0, 1.0, 0.0], inverse=True)

    visible_nm = scale * float(np.mean(_vox(voxel_nm)))
    dist = visible_nm / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cam_pos = look_at - fwd * dist
    return Camera(position_nm=cam_pos.tolist(), look_at_nm=look_at.tolist(),
                  fov_deg=fov_deg, up=up.tolist())


def camera_to_ng(camera: Camera, voxel_nm, base_state: dict | None = None) -> dict:
    state = dict(base_state or {})
    look_at = np.array(camera.look_at_nm, dtype=float)
    state["position"] = (look_at / _vox(voxel_nm)).tolist()

    fwd = look_at - np.array(camera.position_nm, dtype=float)
    dist = float(np.linalg.norm(fwd)) or 1.0
    fwd = fwd / dist
    up = np.array(camera.up, dtype=float)

    # build the view frame (inverse of ng_to_camera): view -Z = fwd, view +Y = up
    z = -fwd
    y = up - np.dot(up, z) * z
    ny = np.linalg.norm(y)
    y = y / ny if ny > 1e-9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    world_from_view = np.column_stack([x, y, z])
    q = Rotation.from_matrix(world_from_view.T).as_quat()  # view-from-world
    state["projectionOrientation"] = [float(v) for v in q]

    visible_nm = 2.0 * dist * math.tan(math.radians(camera.fov_deg) / 2.0)
    state["projectionScale"] = visible_nm / float(np.mean(_vox(voxel_nm)))
    return state
