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

# neuroglancer's perspective view, taken from its source (perspective_panel):
#   fovy = Math.PI/4 = 45deg (VERTICAL field of view), and camera distance =
#   (projectionScale/2)/tan(fovy/2) in voxels, so the visible extent at the focus is
#   exactly projectionScale*voxel. We render at this FOV/distance with a VERTICAL
#   sensor fit, reproducing neuroglancer's framing exactly — and, like neuroglancer,
#   the vertical framing is independent of the window aspect (a wider frame just
#   shows more on the sides). No empirical calibration.
NG_FOV_DEG = 45.0


def _vox(voxel_nm):
    return np.array(voxel_nm, dtype=float)


def _xyz_perm(state: dict) -> list[int]:
    """Indices into the NG dimension-ordered arrays (position / voxel / world vectors)
    that reorder them to (x, y, z). Neuroglancer lists `dimensions` in an arbitrary
    order — often z,y,x — and `position`/`projectionOrientation` follow that order, but
    our world frame (and the precomputed meshes) are x,y,z. For an x,y,z state this is
    the identity, so it's backward-compatible."""
    dims = list((state.get("dimensions") or {}).keys())
    if len(dims) < 3:
        return [0, 1, 2]

    def idx(ax, default):
        for i, d in enumerate(dims):
            if d == ax or d[:1].lower() == ax:
                return i
        return default
    return [idx("x", 0), idx("y", 1), idx("z", 2)]


_UNIT_TO_NM = {"m": 1e9, "cm": 1e7, "mm": 1e6, "um": 1e3, "µm": 1e3, "nm": 1.0, "pm": 1e-3}


def _voxel_nm_from_state(state: dict, fallback):
    """Voxel size (nm) per NG dimension, in the state's dimension order. The NG
    `position` is in these units, so this is the authoritative scale — the manifest's
    voxel size can differ from the units a given neuroglancer view is displayed in
    (e.g. a 16 nm EM shown on a 1 nm grid), which would scale the camera wrong."""
    dims = state.get("dimensions") or {}
    out = []
    for spec in dims.values():
        try:
            out.append(float(spec[0]) * _UNIT_TO_NM.get(spec[1], 1e9))
        except (TypeError, IndexError, ValueError):
            return fallback
    return out if len(out) >= 3 else fallback


def handedness_flipped(state: dict) -> bool:
    """True when the NG->xyz axis permutation is a reflection (odd), e.g. a z,y,x view.
    Reordering the camera to xyz then flips image chirality vs neuroglancer, so the
    render must reintroduce the reflection (a mirrored camera matrix) to match NG."""
    p = list(_xyz_perm(state))
    swaps = 0
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]; p[i], p[j] = p[j], p[i]; swaps += 1
    return swaps % 2 == 1


def ng_to_camera(state: dict, voxel_nm, fov_deg: float = NG_FOV_DEG) -> Camera:
    perm = _xyz_perm(state)
    voxel_nm = _voxel_nm_from_state(state, voxel_nm)
    pos_vox = np.array(state.get("position") or [0, 0, 0], dtype=float)
    # position & voxel are in NG dimension order; multiply elementwise, then reorder to xyz
    look_at = (pos_vox * _vox(voxel_nm))[perm]
    q = state.get("projectionOrientation") or [0.0, 0.0, 0.0, 1.0]
    scale = float(state.get("projectionScale", 10000.0))

    rot = Rotation.from_quat(q)  # neuroglancer & scipy both use [x,y,z,w]
    # neuroglancer maps view directions to world by `rot` directly (NOT its inverse).
    # Its 3D view is Y-DOWN (screen up = -Y) and the camera looks along +Z in view
    # space (so depth ordering matches: closer objects sit in front). Verified by
    # matching rendered frames — including depth — to neuroglancer's video_tool output.
    # the orientation maps view->world in NG's dimension order; reorder the world
    # vectors to xyz so the camera matches the (x,y,z) mesh world.
    fwd = rot.apply([0.0, 0.0, 1.0])[perm]
    up = rot.apply([0.0, -1.0, 0.0])[perm]

    # NG: visible extent at the focus = projectionScale*voxel; dist back-computed from
    # the vertical FOV. (Exactly NG's (projectionScale/2)/tan(fovy/2) * voxel.)
    visible_nm = scale * float(np.mean(_vox(voxel_nm)))
    dist = visible_nm / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cam_pos = look_at - fwd * dist
    return Camera(position_nm=cam_pos.tolist(), look_at_nm=look_at.tolist(),
                  fov_deg=fov_deg, up=up.tolist())


def camera_to_ng(camera: Camera, voxel_nm, base_state: dict | None = None) -> dict:
    state = dict(base_state or {})
    perm = _xyz_perm(state)
    inv = list(np.argsort(perm))            # reorder an (x,y,z) vector back to NG dim order
    look_at = np.array(camera.look_at_nm, dtype=float)   # xyz
    vox = _vox(_voxel_nm_from_state(state, voxel_nm))    # NG dimension order
    state["position"] = (look_at[inv] / vox).tolist()    # xyz -> dim order, then to voxels

    fwd = look_at - np.array(camera.position_nm, dtype=float)
    dist = float(np.linalg.norm(fwd)) or 1.0
    fwd = fwd / dist
    up = np.array(camera.up, dtype=float)

    # Inverse of ng_to_camera (which maps view->world by `rot` directly, Y-down,
    # looking +Z): rot maps view +Z->fwd, view +Y->-up, columns are [(-up)×fwd, -up, fwd].
    up = up - np.dot(up, fwd) * fwd
    nu = np.linalg.norm(up)
    up = up / nu if nu > 1e-9 else np.array([0.0, -1.0, 0.0])
    fwd, up = fwd[inv], up[inv]              # xyz -> NG dim order for the orientation
    c2 = fwd
    c1 = -up
    c0 = np.cross(c1, c2)
    rot = np.column_stack([c0, c1, c2])  # view->world rotation
    q = Rotation.from_matrix(rot).as_quat()
    state["projectionOrientation"] = [float(v) for v in q]

    visible_nm = 2.0 * dist * math.tan(math.radians(camera.fov_deg) / 2.0)
    state["projectionScale"] = visible_nm / float(np.mean(_vox(voxel_nm)))
    return state
