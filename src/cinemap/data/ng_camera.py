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
    # neuroglancer maps view directions to world by `rot` directly (NOT its inverse),
    # and its 3D view is Y-DOWN: screen up is -Y in view space. Camera looks along -Z.
    # (Verified by matching rendered frames to neuroglancer's video_tool output.)
    fwd = rot.apply([0.0, 0.0, -1.0])
    up = rot.apply([0.0, -1.0, 0.0])

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

    # Inverse of ng_to_camera (which maps view->world by `rot` directly, Y-down):
    # rot maps view +Z->-fwd, view +Y->-up, so its columns are [up×fwd, -up, -fwd].
    up = up - np.dot(up, fwd) * fwd
    nu = np.linalg.norm(up)
    up = up / nu if nu > 1e-9 else np.array([0.0, -1.0, 0.0])
    c2 = -fwd
    c1 = -up
    c0 = np.cross(c1, c2)
    rot = np.column_stack([c0, c1, c2])  # view->world rotation
    q = Rotation.from_matrix(rot).as_quat()
    state["projectionOrientation"] = [float(v) for v in q]

    visible_nm = 2.0 * dist * math.tan(math.radians(camera.fov_deg) / 2.0)
    state["projectionScale"] = visible_nm / float(np.mean(_vox(voxel_nm)))
    return state
