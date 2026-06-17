"""Read EM cross-section slices from an OME-Zarr multiscale volume over https.

A slice is a cheap 2D read: we pick an appropriate multiscale level for the
requested on-screen extent, read one plane perpendicular to an axis, and return
it as an image plus the world-space placement Blender needs.

Validated path (spike): tensorstore zarr driver + http kvstore, zstd, '/' sep.
"""
from __future__ import annotations

import json
import urllib.error
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


@lru_cache(maxsize=32)
def _cached_volume(zarr_url: str) -> "EMVolume":
    return EMVolume(zarr_url)


def get_volume(zarr_url: str) -> "EMVolume":
    """Cached EMVolume per URL. Constructing one does a blocking `.zattrs` HTTP
    fetch (and each level opens with a `.zarray` fetch), so reusing the instance
    across keyframe/preset/mesh operations avoids redundant network round-trips
    within a session. Volume metadata is immutable, so sharing is safe. The URL is
    normalized first so trailing-slash variants hit the same cache entry."""
    return _cached_volume(zarr_url.rstrip("/"))


get_volume.cache_clear = _cached_volume.cache_clear  # expose for tests/refresh


class EMVolume:
    """Multiscale OME-Zarr EM volume; opens levels lazily and reads planes."""

    def __init__(self, zarr_url: str):
        self.url = zarr_url.rstrip("/")
        self.zarr_v3 = False  # set by _read_attrs (v3 datasets expose zarr.json)
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

    @staticmethod
    def _get_json(url: str) -> dict:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)

    def _read_attrs(self) -> dict:
        """Multiscales metadata, supporting both OME-Zarr layouts:
          - v2 / OME-0.4:  `.zattrs` with a top-level `multiscales`
          - v3 / OME-0.5:  `zarr.json` with `attributes.ome.multiscales`
        Sets `self.zarr_v3` so `_open_level` picks the matching tensorstore driver."""
        try:
            attrs = self._get_json(f"{self.url}/.zattrs")
            self.zarr_v3 = False
            return attrs["multiscales"][0]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        # No .zattrs -> assume Zarr v3 group metadata.
        grp = self._get_json(f"{self.url}/zarr.json")
        self.zarr_v3 = True
        attrs = grp.get("attributes", {})
        ome = attrs.get("ome", attrs)  # OME-Zarr 0.5 nests multiscales under "ome"
        return ome["multiscales"][0]

    @lru_cache(maxsize=16)
    def _open_level(self, level: int):
        path = self.datasets[level]["path"]
        base = f"{self.url}/{path}/"
        if self.zarr_v3:
            # tensorstore's zarr3 driver reads the level's own `zarr.json` (codecs,
            # sharding, etc.) directly — no manual metadata massaging needed.
            return ts.open({
                "driver": "zarr3",
                "kvstore": {"driver": "http", "base_url": base},
                "open": True,
            }).result()
        # Zarr v2: some cellmap arrays add a non-standard "checksum" field to the
        # zstd compressor that tensorstore's strict parser rejects. Fetch the
        # .zarray, drop it, and open with assume_metadata to skip re-parsing.
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

    @staticmethod
    def plane_basis(normal_xyz):
        """Orthonormal in-plane basis (U, V) and unit normal for an oblique plane."""
        n = np.asarray(normal_xyz, float)
        n = n / (np.linalg.norm(n) or 1.0)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(ref, n); u /= (np.linalg.norm(u) or 1.0)
        v = np.cross(n, u)
        return u, v, n

    def read_oblique_slice(self, normal_xyz, point_xyz, half_nm: float,
                           target_px: int = 640, target_voxels: int = 24_000_000) -> SliceResult:
        """Resample a tilted plane (unit `normal` through `point`) over a 2*half_nm
        square patch. Reads the bounding-box subvolume at a level bounded by
        target_voxels, then nearest-samples the plane grid (fast; fine for EM)."""
        u, v, n = self.plane_basis(normal_xyz)
        p = np.asarray(point_xyz, float)
        px = max(8, int(target_px))
        s = np.linspace(-half_nm, half_nm, px)
        su, sv = np.meshgrid(s, s)                                  # (px,px)
        world = p[None, None, :] + su[..., None] * u + sv[..., None] * v   # (px,px,3) xyz nm
        flat = world.reshape(-1, 3)
        wmin, wmax = flat.min(0), flat.max(0)
        bbox = ((wmin[0], wmin[1], wmin[2]), (wmax[0], wmax[1], wmax[2]))
        level = self.pick_level_for_box(bbox, target_voxels)
        sub, (z0, y0, x0), sc, tr = self.read_box(bbox, level)      # sub is z,y,x
        sub = np.asarray(sub).astype(np.uint8)
        fx = (world[..., 0] - tr[2]) / sc[2] - x0
        fy = (world[..., 1] - tr[1]) / sc[1] - y0
        fz = (world[..., 2] - tr[0]) / sc[0] - z0
        oob = ((fz < 0) | (fz > sub.shape[0] - 1) | (fy < 0) | (fy > sub.shape[1] - 1)
               | (fx < 0) | (fx > sub.shape[2] - 1))
        iz = np.clip(np.round(fz).astype(int), 0, sub.shape[0] - 1)
        iy = np.clip(np.round(fy).astype(int), 0, sub.shape[1] - 1)
        ix = np.clip(np.round(fx).astype(int), 0, sub.shape[2] - 1)
        img = sub[iz, iy, ix]
        img[oob] = 0
        origin = p - half_nm * u - half_nm * v   # corner at su=-half, sv=-half (rows=v, cols=u)
        return SliceResult(
            image=img, axis="oblique", position_nm=float(np.dot(p, n)),
            origin_nm=tuple(origin), u_nm=tuple(2 * half_nm * u), v_nm=tuple(2 * half_nm * v),
            scale_level=level,
        )

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
