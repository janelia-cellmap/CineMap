"""Read EM cross-section slices from an OME-Zarr multiscale volume over https.

A slice is a cheap 2D read: we pick an appropriate multiscale level for the
requested on-screen extent, read one plane perpendicular to an axis, and return
it as an image plus the world-space placement Blender needs.

Validated path (spike): tensorstore zarr driver + http kvstore, zstd, '/' sep.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import tensorstore as ts

# OME-Zarr axes are z,y,x; cinemap world coords are x,y,z (nm).
_AXIS_TO_ZYX = {"z": 0, "y": 1, "x": 2}


@dataclass
class SliceResult:
    image: np.ndarray  # 2D uint8
    axis: str
    position_nm: float
    # world-space rectangle the plane covers, as (origin_nm, u_nm, v_nm) where the
    # plane spans origin + s*u + t*v for s,t in [0,1]. In x,y,z nm.
    origin_nm: tuple[float, float, float]
    u_nm: tuple[float, float, float]
    v_nm: tuple[float, float, float]
    scale_level: int


class EMVolume:
    """Multiscale OME-Zarr EM volume; opens levels lazily and reads planes."""

    def __init__(self, zarr_url: str):
        self.url = zarr_url.rstrip("/")
        self.multiscales = self._read_attrs()
        self.datasets = self.multiscales["datasets"]  # [{path, coordinateTransformations}]
        # nm scale + translation (z,y,x) per level. The OME-Zarr translation aligns
        # each downsampled level to the s0 grid; world = voxel*scale + translation.
        # Ignoring it offsets coarse levels (and makes meshes drift from the EM).
        self.level_scale_nm: list[list[float]] = []
        self.level_translation_nm: list[list[float]] = []
        for ds in self.datasets:
            ct = ds["coordinateTransformations"]
            scale = next(t["scale"] for t in ct if t["type"] == "scale")
            trans = next((t["translation"] for t in ct if t["type"] == "translation"), [0, 0, 0])
            self.level_scale_nm.append([float(s) for s in scale])
            self.level_translation_nm.append([float(t) for t in trans])

    def _read_attrs(self) -> dict:
        with urllib.request.urlopen(f"{self.url}/.zattrs", timeout=30) as r:
            attrs = json.load(r)
        return attrs["multiscales"][0]

    @lru_cache(maxsize=16)
    def _open_level(self, level: int):
        path = self.datasets[level]["path"]
        base = f"{self.url}/{path}/"
        # Some cellmap arrays add a non-standard "checksum" field to the zstd
        # compressor that tensorstore's strict parser rejects. Fetch the .zarray,
        # drop it, and open with assume_metadata so tensorstore skips re-parsing.
        with urllib.request.urlopen(f"{base}.zarray", timeout=30) as r:
            meta = json.load(r)
        comp = meta.get("compressor")
        if isinstance(comp, dict):
            comp.pop("checksum", None)
        return ts.open({
            "driver": "zarr",
            "kvstore": {"driver": "http", "base_url": base},
            "metadata": meta,
            "open": True,
            "assume_metadata": True,
        }).result()

    def level_shape_zyx(self, level: int) -> tuple[int, int, int]:
        return tuple(int(x) for x in self._open_level(level).shape)

    def pick_level_for_box(self, bbox_xyz_nm, target_voxels: int = 8_000_000) -> int:
        """Finest level whose voxel count inside the bbox is <= target_voxels."""
        ext = [bbox_xyz_nm[1][i] - bbox_xyz_nm[0][i] for i in range(3)]  # x,y,z nm
        for lvl, sc in enumerate(self.level_scale_nm):  # sc is z,y,x
            nvox = (ext[0] / sc[2]) * (ext[1] / sc[1]) * (ext[2] / sc[0])
            if nvox <= target_voxels:
                return lvl
        return len(self.level_scale_nm) - 1

    def read_box(self, bbox_xyz_nm, level: int, pad: int = 2):
        """Read a 3D subvolume covering bbox at `level`.

        Returns (array_zyx, origin_voxel_zyx, scale_zyx_nm, translation_zyx_nm),
        where world_zyx = (voxel_index * scale) + translation.
        """
        arr = self._open_level(level)
        sc = self.level_scale_nm[level]          # z,y,x nm/voxel
        tr = self.level_translation_nm[level]    # z,y,x nm
        (x0n, y0n, z0n), (x1n, y1n, z1n) = bbox_xyz_nm
        z0 = max(0, int((z0n - tr[0]) / sc[0]) - pad); z1 = min(arr.shape[0], int((z1n - tr[0]) / sc[0]) + pad)
        y0 = max(0, int((y0n - tr[1]) / sc[1]) - pad); y1 = min(arr.shape[1], int((y1n - tr[1]) / sc[1]) + pad)
        x0 = max(0, int((x0n - tr[2]) / sc[2]) - pad); x1 = min(arr.shape[2], int((x1n - tr[2]) / sc[2]) + pad)
        z1 = max(z1, z0 + 1); y1 = max(y1, y0 + 1); x1 = max(x1, x0 + 1)
        sub = np.asarray(arr[z0:z1, y0:y1, x0:x1].read().result())
        return sub, (z0, y0, x0), tuple(sc), tuple(tr)

    def pick_level(self, extent_nm: float, target_px: int = 1600) -> int:
        """Coarsest level that still gives >= target_px across `extent_nm`."""
        best = 0
        for lvl, scale in enumerate(self.level_scale_nm):
            in_plane_nm = max(scale[1], scale[2])  # y,x nm/voxel
            px = extent_nm / in_plane_nm
            if px >= target_px:
                best = lvl
            else:
                break
        return best

    # in-plane world axes for each slice axis: (u_axis, v_axis) in x/y/z terms
    _INPLANE = {"z": ("x", "y"), "y": ("x", "z"), "x": ("y", "z")}
    _XYZ = {"x": 0, "y": 1, "z": 2}

    def read_slice(
        self, axis: str, position_nm: float, level: int | None = None, target_px: int = 1600,
        region: tuple[tuple[float, float, float], float] | None = None,
    ) -> SliceResult:
        """Read one cross-section. If `region`=((cx,cy,cz)_nm, half_nm) is given,
        read only that square crop around the camera target at a level chosen for
        the crop extent (sharp when zoomed); otherwise read the whole plane."""
        zyx = _AXIS_TO_ZYX[axis]
        ua, va = self._INPLANE[axis]  # world axis names for the two in-plane dirs

        if region is not None:
            extent_nm = 2 * region[1]
        else:
            s0, shp0 = self.level_scale_nm[0], self.level_shape_zyx(0)
            extent_nm = max(
                shp0[_AXIS_TO_ZYX[ua]] * s0[_AXIS_TO_ZYX[ua]],
                shp0[_AXIS_TO_ZYX[va]] * s0[_AXIS_TO_ZYX[va]],
            )
        if level is None:
            level = self.pick_level(extent_nm, target_px)

        arr = self._open_level(level)
        scale = self.level_scale_nm[level]        # z,y,x nm/voxel
        trans = self.level_translation_nm[level]  # z,y,x nm; world = voxel*scale + trans
        idx = max(0, min(int(round((position_nm - trans[zyx]) / scale[zyx])), arr.shape[zyx] - 1))

        # in-plane world ranges
        if region is not None:
            c, h = region[0], region[1]
            urange = (c[self._XYZ[ua]] - h, c[self._XYZ[ua]] + h)
            vrange = (c[self._XYZ[va]] - h, c[self._XYZ[va]] + h)
        else:
            urange = (trans[_AXIS_TO_ZYX[ua]], trans[_AXIS_TO_ZYX[ua]] + arr.shape[_AXIS_TO_ZYX[ua]] * scale[_AXIS_TO_ZYX[ua]])
            vrange = (trans[_AXIS_TO_ZYX[va]], trans[_AXIS_TO_ZYX[va]] + arr.shape[_AXIS_TO_ZYX[va]] * scale[_AXIS_TO_ZYX[va]])

        u_zyx, v_zyx = _AXIS_TO_ZYX[ua], _AXIS_TO_ZYX[va]
        u0 = max(0, int((urange[0] - trans[u_zyx]) / scale[u_zyx])); u1 = min(arr.shape[u_zyx], int((urange[1] - trans[u_zyx]) / scale[u_zyx]))
        v0 = max(0, int((vrange[0] - trans[v_zyx]) / scale[v_zyx])); v1 = min(arr.shape[v_zyx], int((vrange[1] - trans[v_zyx]) / scale[v_zyx]))
        u1 = max(u1, u0 + 1); v1 = max(v1, v0 + 1)

        sel = [slice(None)] * 3
        sel[zyx] = idx
        sel[u_zyx] = slice(u0, u1)
        sel[v_zyx] = slice(v0, v1)
        sub = np.asarray(arr[tuple(sel)].read().result()).astype(np.uint8)
        # orient so rows=v, cols=u
        sub = np.moveaxis(sub, (0, 1), (0, 1)) if v_zyx < u_zyx else sub.T

        # actual world rectangle covered (snap to voxel bounds; world = voxel*scale + trans)
        uo, ue = u0 * scale[u_zyx] + trans[u_zyx], u1 * scale[u_zyx] + trans[u_zyx]
        vo, ve = v0 * scale[v_zyx] + trans[v_zyx], v1 * scale[v_zyx] + trans[v_zyx]
        origin = [0.0, 0.0, 0.0]; origin[self._XYZ[axis]] = position_nm
        origin[self._XYZ[ua]] = uo; origin[self._XYZ[va]] = vo
        u = [0.0, 0.0, 0.0]; u[self._XYZ[ua]] = ue - uo
        v = [0.0, 0.0, 0.0]; v[self._XYZ[va]] = ve - vo

        return SliceResult(
            image=sub, axis=axis, position_nm=position_nm,
            origin_nm=tuple(origin), u_nm=tuple(u), v_nm=tuple(v), scale_level=level,
        )
