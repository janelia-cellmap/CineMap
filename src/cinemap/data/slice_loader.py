"""Read EM cross-section slices from an OME-Zarr multiscale volume over https.

A slice is a cheap 2D read: we pick an appropriate multiscale level for the
requested on-screen extent, read one plane perpendicular to an axis, and return
it as an image plus the world-space placement Blender needs.

Validated path (spike): tensorstore zarr driver + http kvstore, zstd, '/' sep.
"""
from __future__ import annotations

import os
import urllib.error
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import tensorstore as ts

from .local_paths import read_json, tensorstore_kvstore

# OME-Zarr axes are z,y,x; cinemap world coords are x,y,z (nm).
_AXIS_TO_ZYX = {"z": 0, "y": 1, "x": 2}

# Shared tensorstore context across ALL opened levels/volumes. The default context
# gives each open its own cache pool with total_bytes_limit=0 (no caching), so every
# slice re-fetched + re-decompressed its chunks over HTTP from scratch — and a render's
# slices sweep through heavily OVERLAPPING regions, so the same chunks were paid for
# dozens of times (cold slices were ~50s each). One shared pool with a real byte limit
# lets overlapping reads hit cache, and the bumped concurrency fans out the cold chunk
# fetches instead of serializing them. Sharing is safe: volume data is immutable.
#
# Sizing matters more than it looks. A cross-section is one voxel deep, but these volumes
# are chunked 64^3, so reading a single plane forces tensorstore to fetch and decode the
# whole 64-plane band of every chunk it touches — about 1 GB decompressed for a 16 MB
# plane. That is fine ONLY if the band survives long enough for the other planes in it to
# be read (a warm plane measures ~0.05 s against ~3.8 s cold). With several render threads
# each working a different band, a small pool evicts bands before they pay off, so the
# amplification is paid over and over. Hence a pool sized in bands, not megabytes.
# Override with CINEMAP_TS_CACHE_BYTES on memory-constrained machines.
_TS_CACHE_BYTES = int(os.environ.get("CINEMAP_TS_CACHE_BYTES") or 12_000_000_000)
_TS_CONTEXT = ts.Context({
    "cache_pool": {"total_bytes_limit": _TS_CACHE_BYTES},
    "data_copy_concurrency": {"limit": 16},
})


def _to_display_dtype(sub: np.ndarray) -> np.ndarray:
    """Keep image data in its NATIVE dtype for the shader to window.

    This used to be `sub.astype(np.uint8)`, which silently wrapped modulo 256 on any
    non-8-bit EM (a uint16 value of 40000 came out as 64) and threw away neuroglancer's
    contrast window even on 8-bit data. Intensity mapping is neuroglancer's job and is
    now done by `data/ng_shader.py`, which needs the raw values plus the dtype to pick
    the right default invlerp range.

    Shape is deliberately left alone. This also runs on the 3D subvolume the oblique
    reader samples from, and squeezing a 1-voxel-thick box down to 2D there breaks the
    `sub.shape[2]` indexing that follows, dropping the slice.
    """
    return sub


@dataclass
class SliceResult:
    image: np.ndarray  # 2D, in the source's NATIVE dtype (see _to_display_dtype)
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
        return read_json(url, timeout=30)

    def _read_attrs(self) -> dict:
        """Multiscales metadata, supporting both OME-Zarr layouts:
          - v2 / OME-0.4:  `.zattrs` with a top-level `multiscales`
          - v3 / OME-0.5:  `zarr.json` with `attributes.ome.multiscales`
        Sets `self.zarr_v3` so `_open_level` picks the matching tensorstore driver."""
        try:
            attrs = self._get_json(f"{self.url}/.zattrs")
            self.zarr_v3 = False
            return attrs["multiscales"][0]
        except (urllib.error.HTTPError, FileNotFoundError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code != 404:
                raise
        # No .zattrs -> assume Zarr v3 group metadata.
        grp = self._get_json(f"{self.url}/zarr.json")
        self.zarr_v3 = True
        attrs = grp.get("attributes", {})
        ome = attrs.get("ome", attrs)  # OME-Zarr 0.5 nests multiscales under "ome"
        return ome["multiscales"][0]

    @property
    def dtype(self):
        """The volume's numpy dtype — what neuroglancer resolves a default invlerp
        range against (uint8 -> 0..255, uint16 -> 0..65535). Cheap: the coarsest level
        is already open or is a metadata-only read."""
        return self._open_level(len(self.datasets) - 1).dtype.numpy_dtype

    @lru_cache(maxsize=16)
    def _open_level(self, level: int):
        path = self.datasets[level]["path"]
        base = f"{self.url}/{path}/"
        if self.zarr_v3:
            # tensorstore's zarr3 driver reads the level's own `zarr.json` (codecs,
            # sharding, etc.) directly — no manual metadata massaging needed.
            return ts.open({
                "driver": "zarr3",
                "kvstore": tensorstore_kvstore(base),
                "open": True,
            }, context=_TS_CONTEXT).result()
        # Zarr v2: some cellmap arrays add a non-standard "checksum" field to the
        # zstd compressor that tensorstore's strict parser rejects. Fetch the
        # .zarray, drop it, and open with assume_metadata to skip re-parsing.
        meta = read_json(f"{base}.zarray", timeout=30)
        comp = meta.get("compressor")
        if isinstance(comp, dict):
            comp.pop("checksum", None)
        return ts.open({
            "driver": "zarr",
            "kvstore": tensorstore_kvstore(base),
            "metadata": meta,
            "open": True,
            "assume_metadata": True,
        }, context=_TS_CONTEXT).result()

    def level_shape_zyx(self, level: int) -> tuple[int, int, int]:
        return tuple(int(x) for x in self._open_level(level).shape)

    def extent_nm(self) -> tuple[list[float], list[float]]:
        """Full data extent as (lo_xyz, hi_xyz) in nm world coords — the source's
        bounding box, like neuroglancer draws. Derived from level 0: world =
        voxel*scale + translation, with shape/scale/translation in z,y,x order."""
        shp = self.level_shape_zyx(0)              # z,y,x voxels
        sc, tr = self.level_scale_nm[0], self.level_translation_nm[0]   # z,y,x nm
        lo_zyx = [float(tr[i]) for i in range(3)]
        hi_zyx = [float(tr[i]) + shp[i] * float(sc[i]) for i in range(3)]
        return [lo_zyx[2], lo_zyx[1], lo_zyx[0]], [hi_zyx[2], hi_zyx[1], hi_zyx[0]]

    def pick_level_for_box(self, bbox_xyz_nm, target_voxels: int = 8_000_000,
                           nm_per_px: float | None = None) -> int:
        """Level for reading a 3D bbox.

        If `nm_per_px` is given, prefers the same physical-zoom reasoning as
        `pick_level_for_nm_per_px` — the coarsest level that still resolves at
        least one voxel per screen pixel — so tilted/oblique planes track
        camera zoom instead of only the box's raw voxel count (which used to
        make oblique reads go coarser than an equivalent axis-aligned view
        whenever the bbox was large, independent of how zoomed-in the camera
        actually was). `target_voxels` remains a memory safety net: the
        result never reads more than that many voxels, even if the zoom-ideal
        level would.
        """
        ext = [bbox_xyz_nm[1][i] - bbox_xyz_nm[0][i] for i in range(3)]  # x,y,z nm

        def nvox_at(lvl: int) -> float:
            sc = self.level_scale_nm[lvl]  # z,y,x nm/voxel
            return (ext[0] / sc[2]) * (ext[1] / sc[1]) * (ext[2] / sc[0])

        budget_lvl = len(self.level_scale_nm) - 1
        for lvl in range(len(self.level_scale_nm)):
            if nvox_at(lvl) <= target_voxels:
                budget_lvl = lvl
                break

        if nm_per_px is None or nm_per_px <= 0:
            return budget_lvl

        zoom_lvl = 0
        for lvl, sc in enumerate(self.level_scale_nm):
            in_plane_nm = max(sc)
            if in_plane_nm <= nm_per_px:
                zoom_lvl = lvl
            else:
                break
        return max(zoom_lvl, budget_lvl)  # never finer than the memory budget allows

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
                           target_px: int = 640, target_voxels: int = 800_000_000,
                           raw: bool = False) -> SliceResult:
        """Resample a tilted plane (unit `normal` through `point`) over a 2*half_nm
        square patch, matching neuroglancer's approach on two axes:

        - Level is chosen from the plane's physical on-screen scale (nm/px), same
          as the axis-aligned path, so a tilted plane gets the same resolution an
          equivalent axis-aligned view would.
        - Only the chunks the plane's sample grid actually touches are fetched —
          not the full axis-aligned bounding cuboid, which for a steeply tilted
          plane can be orders of magnitude bigger than the plane itself (a thin
          tilted rectangle's AABB is roughly cube-shaped, mostly empty space far
          from the plane).

        Samples are trilinearly interpolated (`raw=False`, continuous EM/image
        data) or nearest-neighbor (`raw=True`, segmentation ids — blending ids
        would be meaningless).
        """
        u, v, n = self.plane_basis(normal_xyz)
        p = np.asarray(point_xyz, float)
        px = max(8, int(target_px))
        s = np.linspace(-half_nm, half_nm, px)
        su, sv = np.meshgrid(s, s)                                  # (px,px)
        world = p[None, None, :] + su[..., None] * u + sv[..., None] * v   # (px,px,3) xyz nm
        nm_per_px = (2.0 * half_nm) / max(1, px - 1)

        flat = world.reshape(-1, 3)
        wmin, wmax = flat.min(0), flat.max(0)
        bbox = ((wmin[0], wmin[1], wmin[2]), (wmax[0], wmax[1], wmax[2]))
        level = self.pick_level_for_box(bbox, target_voxels, nm_per_px=nm_per_px)

        arr = self._open_level(level)
        sc = self.level_scale_nm[level]          # z,y,x nm/voxel
        tr = self.level_translation_nm[level]    # z,y,x nm
        shape = arr.shape                        # z,y,x voxels

        fx = (world[..., 0] - tr[2]) / sc[2]
        fy = (world[..., 1] - tr[1]) / sc[1]
        fz = (world[..., 2] - tr[0]) / sc[0]
        oob = ((fz < 0) | (fz > shape[0] - 1) | (fy < 0) | (fy > shape[1] - 1)
               | (fx < 0) | (fx > shape[2] - 1))

        img = self._gather_plane(arr, shape, fz, fy, fx, raw=raw)
        img[oob] = 0

        origin = p - half_nm * u - half_nm * v   # corner at su=-half, sv=-half (rows=v, cols=u)
        return SliceResult(
            image=img, axis="oblique", position_nm=float(np.dot(p, n)),
            origin_nm=tuple(origin), u_nm=tuple(2 * half_nm * u), v_nm=tuple(2 * half_nm * v),
            scale_level=level,
        )

    @staticmethod
    def _gather_plane(arr, shape, fz, fy, fx, raw: bool) -> np.ndarray:
        """Sample `arr` at fractional voxel coords (fz,fy,fx), fetching only the
        chunks actually touched instead of one dense box covering their full
        extent. `raw=True` -> nearest-neighbor (segmentation ids); otherwise
        trilinear. Coordinates may be out of [0, shape-1]; callers are expected
        to mask those out afterward (this clips them in for the gather).

        Samples are grouped by chunk with one sort per lookup, so each chunk reads
        exactly its own samples. (Masking the whole sample grid once per chunk, as
        before, cost chunks x samples: a full-resolution tilted plane touches
        thousands of chunks with 2.5M samples, which took minutes per frame.)"""
        cz, cy, cx = arr.chunk_layout.read_chunk.shape
        dtype = arr.dtype.numpy_dtype
        out_shape = fz.shape
        ny = -(-shape[1] // cy)
        nx = -(-shape[2] // cx)

        def chunk_keys(za, ya, xa):
            return ((za // cz) * ny + (ya // cy)) * nx + (xa // cx)

        def fetch_chunks(keys):
            pending = {}
            for k in keys:
                zc, rem = divmod(int(k), ny * nx)
                yc, xc = divmod(rem, nx)
                z0, z1 = zc * cz, min(shape[0], (zc + 1) * cz)
                y0, y1 = yc * cy, min(shape[1], (yc + 1) * cy)
                x0, x1 = xc * cx, min(shape[2], (xc + 1) * cx)
                pending[int(k)] = (arr[z0:z1, y0:y1, x0:x1].read(), (z0, y0, x0))
            return {k: (np.asarray(fut.result()), origin) for k, (fut, origin) in pending.items()}

        def lookup(chunks, za, ya, xa):
            zf, yf, xf = za.ravel(), ya.ravel(), xa.ravel()
            key = chunk_keys(zf, yf, xf)
            order = np.argsort(key, kind="stable")
            sk = key[order]
            cut = np.flatnonzero(np.diff(sk)) + 1
            out = np.empty(key.size, dtype=dtype)
            for s0, s1 in zip(np.r_[0, cut], np.r_[cut, key.size]):
                idx = order[s0:s1]
                data, (z0, y0, x0) = chunks[int(sk[s0])]
                out[idx] = data[zf[idx] - z0, yf[idx] - y0, xf[idx] - x0]
            return out.reshape(za.shape)

        if raw:
            iz = np.clip(np.round(fz), 0, shape[0] - 1).astype(np.int64)
            iy = np.clip(np.round(fy), 0, shape[1] - 1).astype(np.int64)
            ix = np.clip(np.round(fx), 0, shape[2] - 1).astype(np.int64)
            chunks = fetch_chunks(np.unique(chunk_keys(iz, iy, ix)))
            return lookup(chunks, iz, iy, ix)

        iz0 = np.clip(np.floor(fz), 0, shape[0] - 1).astype(np.int64)
        iy0 = np.clip(np.floor(fy), 0, shape[1] - 1).astype(np.int64)
        ix0 = np.clip(np.floor(fx), 0, shape[2] - 1).astype(np.int64)
        iz1 = np.clip(iz0 + 1, 0, shape[0] - 1)
        iy1 = np.clip(iy0 + 1, 0, shape[1] - 1)
        ix1 = np.clip(ix0 + 1, 0, shape[2] - 1)
        wz = np.clip(fz - iz0, 0.0, 1.0)
        wy = np.clip(fy - iy0, 0.0, 1.0)
        wx = np.clip(fx - ix0, 0.0, 1.0)

        corners = [
            (za, ya, xa, wa * wb * wc)
            for za, wa in ((iz0, 1 - wz), (iz1, wz))
            for ya, wb in ((iy0, 1 - wy), (iy1, wy))
            for xa, wc in ((ix0, 1 - wx), (ix1, wx))
        ]
        chunks = fetch_chunks(np.unique(np.concatenate(
            [chunk_keys(za, ya, xa).ravel() for za, ya, xa, _ in corners])))

        img_f = np.zeros(out_shape, dtype=np.float64)
        for za, ya, xa, wgt in corners:
            img_f += wgt * lookup(chunks, za, ya, xa).astype(np.float64)

        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            img_f = np.clip(np.round(img_f), info.min, info.max)
        return img_f.astype(dtype)

    def pick_level_for_nm_per_px(self, axis: str, nm_per_px: float) -> int:
        """Neuroglancer-style multiscale choice from physical screen scale.

        Pick the coarsest level whose in-plane voxel spacing is no larger than one
        rendered pixel. This keeps the loaded EM resolution tied to the view's physical
        scale instead of to an arbitrary output image size.
        """
        if nm_per_px <= 0:
            return 0
        ua, va = self._INPLANE[axis]
        best = 0
        for lvl, scale in enumerate(self.level_scale_nm):
            in_plane_nm = max(scale[_AXIS_TO_ZYX[ua]], scale[_AXIS_TO_ZYX[va]])
            if in_plane_nm <= nm_per_px:
                best = lvl
            else:
                break
        return best

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
        target_nm_per_px: float | None = None,
        raw: bool = False,
    ) -> SliceResult:
        """Read one cross-section. If `region`=((cx,cy,cz)_nm, half_nm) is given,
        read only that square crop around the camera target. When `target_nm_per_px`
        is supplied, choose the multiscale level by physical screen scale; otherwise
        fall back to the older target-pixel-count heuristic.

        `raw=True` keeps the array's native dtype — required for LABEL volumes whose
        segment ids exceed 255 (the default uint8 cast, fine for 8-bit EM, would
        truncate ids mod 256 and collide segments)."""
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
        if level is None and target_nm_per_px is not None:
            level = self.pick_level_for_nm_per_px(axis, target_nm_per_px)
        elif level is None:
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
        sub = np.asarray(arr[tuple(sel)].read().result())
        if not raw:
            sub = _to_display_dtype(sub)
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
