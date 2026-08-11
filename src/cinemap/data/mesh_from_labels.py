"""Generate clean meshes for segments directly from OME-Zarr label volumes.

This bypasses the precomputed multilod-draco meshes (which cloud-volume decodes
with per-chunk gaps, producing a 'stippled' look). The preferred path uses zmesh
to generate separate meshes per selected label ID, preserving instance colors and
boundaries while choosing the finest label scale that fits the render budget.
"""
from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import trimesh
from skimage import measure
import zmesh

from .slice_loader import get_volume


_ZMESH_VERTEX_PER_VOXEL_ESTIMATE = 0.15
_ZMESH_VERTEX_PER_SURFACE_FACE_ESTIMATE = 1.0
_ROI_SCAN_VOXELS = 24_000_000
_ROI_REFINE_MAX_VOXELS = 256_000_000
_PLAN_EXACT_COUNT_MAX_VOXELS = 96_000_000
_PLAN_MAX_READ_VOXELS = 512_000_000
_PLAN_NEXT_LEVEL_VERTEX_FACTOR = 4.0
_PLAN_NEXT_LEVEL_TOLERANCE = 1.10
_ROI_REFINE_PAD_FRACTION = 0.50
_ROI_FINAL_PAD_FRACTION = 0.15
_ROI_EDGE_MARGIN_VOXELS = 2
_LOSSLESS_VERTEX_DEDUP_LIMIT = 8_000_000
_MIN_DECIMATE_FACES = 100_000
# Smallest per-segment face target for an explicit decimate_fraction. Low enough that
# the keep-fraction is honored even on many-small-segment layers (the old 256 floor
# capped tiny segments and pushed the aggregate well above the requested fraction),
# but above a degenerate sliver so each segment stays a valid closed-ish surface.
_DECIMATE_MIN_FACES = 16
_PYFQMR_MISSING = False


def _layer_offset(layer: str) -> float:
    """Stable hue offset per layer name, so different layers (nuc vs mito) sit in
    different color families even when they share segment ids."""
    if not layer:
        return 0.0
    h = 0
    for c in layer:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    # golden-ratio spread so similar names (nuc/mito) land far apart on the hue wheel
    return (h * 0.6180339887498949) % 1.0


def seg_color(seg_id: int, layer: str = "") -> tuple[float, float, float]:
    """Stable, well-spread color per segment id (golden-ratio hue), shifted by a
    per-layer offset so segments of different layers are distinguishable."""
    h = (_layer_offset(layer) + int(seg_id) * 0.6180339887498949) % 1.0
    return colorsys.hsv_to_rgb(h, 0.62, 0.95)


@dataclass(frozen=True)
class _ReadPlan:
    level: int
    stride: int
    bbox_xyz_nm: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    surface_faces_estimate: int | None = None
    surface_budget: int | None = None


def _shape_voxels(shape) -> int:
    return int(shape[0]) * int(shape[1]) * int(shape[2])


def _bbox_voxels_for_level(vol, bbox_xyz_nm, level: int, pad: int = 2) -> int:
    if bbox_xyz_nm is None:
        return _shape_voxels(vol.level_shape_zyx(level))
    sc = vol.level_scale_nm[level]
    tr = vol.level_translation_nm[level]
    shape = vol.level_shape_zyx(level)
    (x0n, y0n, z0n), (x1n, y1n, z1n) = bbox_xyz_nm
    z0 = max(0, int((z0n - tr[0]) / sc[0]) - pad)
    z1 = min(shape[0], int((z1n - tr[0]) / sc[0]) + pad)
    y0 = max(0, int((y0n - tr[1]) / sc[1]) - pad)
    y1 = min(shape[1], int((y1n - tr[1]) / sc[1]) + pad)
    x0 = max(0, int((x0n - tr[2]) / sc[2]) - pad)
    x1 = min(shape[2], int((x1n - tr[2]) / sc[2]) + pad)
    return max(1, z1 - z0) * max(1, y1 - y0) * max(1, x1 - x0)


def _volume_extent_xyz_nm(vol):
    return vol.extent_nm()


def _pad_bbox_xyz_nm(vol, bbox_xyz_nm, fraction: float):
    (lo, hi) = bbox_xyz_nm
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    span = np.maximum(hi - lo, 1.0)
    pad = span * float(fraction)
    ext_lo, ext_hi = _volume_extent_xyz_nm(vol)
    out_lo = np.maximum(lo - pad, np.asarray(ext_lo, dtype=float))
    out_hi = np.minimum(hi + pad, np.asarray(ext_hi, dtype=float))
    return (tuple(float(v) for v in out_lo), tuple(float(v) for v in out_hi))


def _bbox_from_indices_xyz_nm(vol, level: int, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int):
    sc = vol.level_scale_nm[level]
    tr = vol.level_translation_nm[level]
    lo = (x0 * sc[2] + tr[2], y0 * sc[1] + tr[1], z0 * sc[0] + tr[0])
    hi = (x1 * sc[2] + tr[2], y1 * sc[1] + tr[1], z1 * sc[0] + tr[0])
    return (lo, hi)


def _selected_bbox_in_array(arr, ids):
    zz, yy, xx = np.where(np.isin(arr, list(ids)))
    if len(zz) == 0:
        return None
    return (
        int(zz.min()), int(zz.max()) + 1,
        int(yy.min()), int(yy.max()) + 1,
        int(xx.min()), int(xx.max()) + 1,
    )


def _touches_read_edge(local_bbox, shape) -> bool:
    z0, z1, y0, y1, x0, x1 = local_bbox
    margin = _ROI_EDGE_MARGIN_VOXELS
    return (
        z0 <= margin or y0 <= margin or x0 <= margin
        or z1 >= int(shape[0]) - margin
        or y1 >= int(shape[1]) - margin
        or x1 >= int(shape[2]) - margin
    )


def _selected_voxel_count_for_level(vol, bbox_xyz_nm, level: int, ids, cache: dict[int, int]) -> int:
    if level in cache:
        return cache[level]
    arr, *_ = vol.read_box(bbox_xyz_nm, level, pad=2)
    count = int(np.isin(arr, list(ids)).sum())
    cache[level] = count
    return count


def _selected_surface_faces(arr: np.ndarray, ids) -> int:
    """Count selected-label voxel faces that become mesh boundary candidates.

    For multiple selected labels, a face between two different selected IDs counts
    once for each ID, because zmesh emits separate instance surfaces.
    """
    id_values = list(ids)
    total = 0
    for axis in range(3):
        first = np.take(arr, 0, axis=axis)
        last = np.take(arr, -1, axis=axis)
        total += int(np.isin(first, id_values).sum())
        total += int(np.isin(last, id_values).sum())

        left = np.take(arr, range(arr.shape[axis] - 1), axis=axis)
        right = np.take(arr, range(1, arr.shape[axis]), axis=axis)
        changed = left != right
        total += int(np.count_nonzero(np.isin(left, id_values) & changed))
        total += int(np.count_nonzero(np.isin(right, id_values) & changed))
    return total


def _selected_surface_faces_for_level(vol, bbox_xyz_nm, level: int, ids, cache: dict[int, int]) -> int:
    if level in cache:
        return cache[level]
    arr, *_ = vol.read_box(bbox_xyz_nm, level, pad=2)
    count = _selected_surface_faces(arr, ids)
    cache[level] = count
    return count


def _surface_area_scale_nm(vol, level: int) -> float:
    z, y, x = (float(v) for v in vol.level_scale_nm[level])
    return max(1.0, (x * y + x * z + y * z) / 3.0)


def _estimate_surface_faces_for_level(vol, bbox_xyz_nm, level: int, ids, cache: dict[int, int]) -> int:
    """Estimate selected-label surface faces without treating empty ROI as mesh.

    zmesh output grows roughly with boundary surface area, not filled volume.
    Count exact boundary faces where the ROI read is modest; for finer levels,
    scale from the nearest coarser exact count by voxel face area. This avoids
    the 8x-per-octave overestimate that selected-volume counts produce.
    """
    voxels = _bbox_voxels_for_level(vol, bbox_xyz_nm, level)
    if voxels <= _PLAN_EXACT_COUNT_MAX_VOXELS:
        return _selected_surface_faces_for_level(vol, bbox_xyz_nm, level, ids, cache)

    for coarser in range(level + 1, len(vol.level_scale_nm)):
        coarse_voxels = _bbox_voxels_for_level(vol, bbox_xyz_nm, coarser)
        if coarse_voxels > _PLAN_EXACT_COUNT_MAX_VOXELS:
            continue
        coarse_count = _selected_surface_faces_for_level(vol, bbox_xyz_nm, coarser, ids, cache)
        fine_area = _surface_area_scale_nm(vol, level)
        coarse_area = _surface_area_scale_nm(vol, coarser)
        return max(1, int(np.ceil(coarse_count * (coarse_area / fine_area))))

    return voxels


def _choose_plan(
    vol,
    bbox_xyz_nm,
    *,
    target_voxels: int,
    target_vertices: int | None,
    seg_ids=None,
) -> _ReadPlan:
    """Finest level/stride whose estimated mesh size fits the budget.

    When `target_vertices` is supplied, do not cap the final label read by a
    separate voxel-count budget. The quality control is the expected mesh size;
    memory failures are handled by retrying one pyramid level coarser. Surface
    estimates are an advisory starting point; actual zmesh vertex counts decide
    whether to refine to the next finer level.
    """
    chosen_level = len(vol.level_scale_nm) - 1
    chosen_voxels = _bbox_voxels_for_level(vol, bbox_xyz_nm, chosen_level)
    chosen_surface_faces = None
    surface_budget = None

    if target_vertices and seg_ids:
        ids = set(int(s) for s in seg_ids)
        surface_budget = max(1, int(target_vertices / _ZMESH_VERTEX_PER_SURFACE_FACE_ESTIMATE))
        count_cache: dict[int, int] = {}
        for level in range(len(vol.level_scale_nm)):
            voxels = _bbox_voxels_for_level(vol, bbox_xyz_nm, level)
            surface_faces = _estimate_surface_faces_for_level(vol, bbox_xyz_nm, level, ids, count_cache)
            if surface_faces <= surface_budget:
                chosen_level = level
                chosen_voxels = voxels
                chosen_surface_faces = surface_faces
                break
        read_budget = chosen_voxels
    else:
        read_budget = max(1, int(target_voxels))
        for level in range(len(vol.level_scale_nm)):
            voxels = _bbox_voxels_for_level(vol, bbox_xyz_nm, level)
            if voxels <= read_budget:
                chosen_level = level
                chosen_voxels = voxels
                break

    stride = max(1, int(np.ceil((chosen_voxels / read_budget) ** (1 / 3))))
    return _ReadPlan(
        level=chosen_level,
        stride=stride,
        bbox_xyz_nm=bbox_xyz_nm,
        surface_faces_estimate=chosen_surface_faces,
        surface_budget=surface_budget,
    )


def _candidate_plans(
    vol,
    bbox_xyz_nm,
    *,
    target_voxels: int,
    target_vertices: int | None,
    seg_ids=None,
) -> list[_ReadPlan]:
    """Return candidate plans from a conservative level toward finer levels.

    The surface estimate is only a guardrail to avoid obviously unreasonable
    fine reads. The actual zmesh vertex count is checked after each candidate is
    generated, and the refinement loop uses measured vertices at the current
    scale to decide whether the next finer scale is worth trying.
    """
    if not (target_vertices and seg_ids):
        return [
            _choose_plan(
                vol,
                bbox_xyz_nm,
                target_voxels=target_voxels,
                target_vertices=target_vertices,
                seg_ids=seg_ids,
            )
        ]

    ids = set(int(s) for s in seg_ids)
    surface_budget = max(1, int(target_vertices / _ZMESH_VERTEX_PER_SURFACE_FACE_ESTIMATE))
    count_cache: dict[int, int] = {}
    levels: list[int] = []
    for level in range(len(vol.level_scale_nm)):
        surface_faces = _estimate_surface_faces_for_level(vol, bbox_xyz_nm, level, ids, count_cache)
        if surface_faces <= 0:
            continue
        if surface_faces <= surface_budget:
            levels.append(level)
    if levels:
        finest_advisory = min(levels)
        coarsest_level = len(vol.level_scale_nm) - 1
        start = min(finest_advisory + 1, coarsest_level)
        plans = []
        for level in range(start, -1, -1):
            surface_faces = _estimate_surface_faces_for_level(vol, bbox_xyz_nm, level, ids, count_cache)
            plans.append(
                _ReadPlan(
                    level=level,
                    stride=1,
                    bbox_xyz_nm=bbox_xyz_nm,
                    surface_faces_estimate=surface_faces,
                    surface_budget=surface_budget,
                )
            )
        return plans
    return [
        _choose_plan(
            vol,
            bbox_xyz_nm,
            target_voxels=target_voxels,
            target_vertices=target_vertices,
            seg_ids=seg_ids,
        )
    ]


def _coarsen_plan(vol, plan: _ReadPlan) -> _ReadPlan | None:
    level = int(plan.level) + 1
    if level < len(vol.level_scale_nm):
        return _ReadPlan(level=level, stride=1, bbox_xyz_nm=plan.bbox_xyz_nm)
    if int(plan.stride) < 16:
        return _ReadPlan(level=plan.level, stride=int(plan.stride) * 2, bbox_xyz_nm=plan.bbox_xyz_nm)
    return None


def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return any(token in text for token in ("memory", "bad_alloc", "std::bad_alloc", "oom", "cannot allocate"))


def _label_source_name(label_zarr_url: str) -> str:
    path = urlparse(label_zarr_url).path.rstrip("/")
    return path.rsplit("/", 1)[-1] or "labels"


def _read_plan_array(vol, plan: _ReadPlan, pad: int = 2):
    """Read the plan region and return (arr, origin_voxel_zyx, scale_zyx_nm, translation)."""
    level = plan.level
    stride = plan.stride
    if plan.bbox_xyz_nm is not None:
        arr, origin, sc, tr = vol.read_box(plan.bbox_xyz_nm, level, pad=pad)
        if stride > 1:
            arr = arr[::stride, ::stride, ::stride]
            sc = tuple(s * stride for s in sc)
        return arr, origin, tuple(sc), tuple(tr)

    arr = np.asarray(vol._open_level(level)[::stride, ::stride, ::stride].read().result())
    sc = tuple(s * stride for s in vol.level_scale_nm[level])
    tr = tuple(vol.level_translation_nm[level])
    return arr, (0, 0, 0), sc, tr


def _bbox_for_ids(label_zarr_url: str, seg_ids, target_voxels: int = _ROI_SCAN_VOXELS):
    """Find a conservative xyz-nm bounding box from labels when no mesh bbox exists."""
    ids = set(int(s) for s in seg_ids)
    if not ids:
        return None
    vol = get_volume(label_zarr_url)
    level = len(vol.level_scale_nm) - 1
    for lvl in range(len(vol.level_scale_nm)):
        if _shape_voxels(vol.level_shape_zyx(lvl)) <= target_voxels:
            level = lvl
            break

    initial_bbox = None
    # Start with the coarsest cheap full-volume scan, but if a thin label
    # disappears completely at that level, step finer while the full scan remains
    # within a bounded planning read.
    for lvl in range(level, -1, -1):
        if _shape_voxels(vol.level_shape_zyx(lvl)) > _ROI_REFINE_MAX_VOXELS:
            break
        arr = np.asarray(vol._open_level(lvl)[:, :, :].read().result())
        bbox_idx = _selected_bbox_in_array(arr, ids)
        if bbox_idx is None:
            continue
        initial_bbox = _bbox_from_indices_xyz_nm(vol, lvl, *bbox_idx)
        level = lvl
        break
    if initial_bbox is None:
        return None

    bbox = initial_bbox
    # Refine the bbox at progressively finer levels. Each refinement reads a
    # generously padded version of the previous bbox; if the selected label still
    # touches the read boundary, expand and retry instead of accepting a clipped
    # ROI. Stop when the next finer read would be too large for planning.
    for lvl in range(level - 1, -1, -1):
        padded = _pad_bbox_xyz_nm(vol, bbox, _ROI_REFINE_PAD_FRACTION)
        if _bbox_voxels_for_level(vol, padded, lvl, pad=0) > _ROI_REFINE_MAX_VOXELS:
            break
        accepted = False
        for _ in range(4):
            sub, (oz, oy, ox), _sc, _tr = vol.read_box(padded, lvl, pad=0)
            local_bbox = _selected_bbox_in_array(sub, ids)
            if local_bbox is None:
                break
            z0, z1, y0, y1, x0, x1 = local_bbox
            bbox = _bbox_from_indices_xyz_nm(
                vol, lvl,
                oz + z0, oz + z1,
                oy + y0, oy + y1,
                ox + x0, ox + x1,
            )
            if not _touches_read_edge(local_bbox, sub.shape):
                accepted = True
                break
            padded = _pad_bbox_xyz_nm(vol, bbox, _ROI_REFINE_PAD_FRACTION)
            if _bbox_voxels_for_level(vol, padded, lvl, pad=0) > _ROI_REFINE_MAX_VOXELS:
                break
        if not accepted:
            break

    return _pad_bbox_xyz_nm(vol, bbox, _ROI_FINAL_PAD_FRACTION)


def _mesh_to_trimesh(mesh, seg_id: int, origin_zyx_nm, colorize=None) -> trimesh.Trimesh | None:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    verts_zyx = np.asarray(mesh.vertices, dtype=np.float64) + np.asarray(origin_zyx_nm)
    verts_xyz = verts_zyx[:, ::-1]
    faces = np.asarray(mesh.faces, dtype=np.int64)[:, ::-1]
    vcolors = None
    if colorize is not None:
        r, g, b = colorize(int(seg_id))
        vcolors = np.tile(np.array([r, g, b, 1.0]) * 255, (len(verts_xyz), 1)).astype(np.uint8)
    return trimesh.Trimesh(vertices=verts_xyz, faces=faces, vertex_colors=vcolors, process=False)


def _cleanup_mesh_lossless(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove strictly redundant topology without moving the represented surface."""
    if len(mesh.faces) == 0:
        return mesh
    try:
        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        elif hasattr(mesh, "remove_degenerate_faces"):
            mesh.remove_degenerate_faces()
    except Exception:  # noqa: BLE001
        pass
    try:
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        elif hasattr(mesh, "remove_duplicate_faces"):
            mesh.remove_duplicate_faces()
    except Exception:  # noqa: BLE001
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:  # noqa: BLE001
        pass
    if 0 < len(mesh.vertices) <= _LOSSLESS_VERTEX_DEDUP_LIMIT:
        try:
            verts, first_idx, inverse = np.unique(
                np.asarray(mesh.vertices), axis=0, return_index=True, return_inverse=True
            )
            if len(verts) < len(mesh.vertices):
                faces = inverse[np.asarray(mesh.faces)]
                colors = None
                try:
                    vc = np.asarray(mesh.visual.vertex_colors)
                    if len(vc) == len(mesh.vertices):
                        colors = vc[first_idx]
                except Exception:  # noqa: BLE001
                    colors = None
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)
                mesh.remove_unreferenced_vertices()
        except Exception:  # noqa: BLE001
            pass
    return mesh


def _simplify_with_trimesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh | None:
    """Quadric decimation via trimesh's native backend (fast_simplification).

    The fallback when pyfqmr isn't installed. Re-tiles the part's single color onto
    the decimated vertices (label-mesh parts are one color each), since the backend
    drops vertex attributes."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max(4, int(target_faces)))
    except Exception:  # noqa: BLE001  (no backend installed, or degenerate input)
        return None
    if simplified is None or len(simplified.faces) == 0:
        return None
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if len(vc):
            simplified.visual.vertex_colors = np.tile(
                vc[0], (len(simplified.vertices), 1)
            ).astype(np.uint8)
    except Exception:  # noqa: BLE001
        pass
    return simplified


def _simplify_with_pyfqmr(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh | None:
    global _PYFQMR_MISSING
    if _PYFQMR_MISSING:
        return _simplify_with_trimesh(mesh, target_faces)
    try:
        import pyfqmr  # type: ignore
    except Exception:  # noqa: BLE001
        _PYFQMR_MISSING = True
        return _simplify_with_trimesh(mesh, target_faces)
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        simp = pyfqmr.Simplify()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        if hasattr(simp, "setMesh"):
            simp.setMesh(verts, faces)
        else:
            simp.set_mesh(verts, faces)
        kwargs = {
            "target_count": max(4, int(target_faces)),
            "preserve_border": True,
            "verbose": False,
        }
        try:
            simp.simplify_mesh(**kwargs)
        except TypeError:
            kwargs.pop("verbose", None)
            simp.simplify_mesh(**kwargs)
        result = simp.getMesh() if hasattr(simp, "getMesh") else simp.get_mesh()
        out_verts, out_faces = result[:2]
        colors = None
        try:
            vc = np.asarray(mesh.visual.vertex_colors)
            if len(vc):
                rgba = vc[0]
                colors = np.tile(rgba, (len(out_verts), 1)).astype(np.uint8)
        except Exception:  # noqa: BLE001
            colors = None
        return trimesh.Trimesh(vertices=out_verts, faces=out_faces, vertex_colors=colors, process=False)
    except Exception as e:  # noqa: BLE001
        print(f"[labels] pyfqmr simplification skipped: {e}")
        return _simplify_with_trimesh(mesh, target_faces)


def _postprocess_label_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_faces: int | None = None,
    smooth_iters: int = 0,
    min_decimate_faces: int = _MIN_DECIMATE_FACES,
) -> trimesh.Trimesh:
    mesh = _cleanup_mesh_lossless(mesh)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
        mesh = _cleanup_mesh_lossless(mesh)
    if target_faces is not None and len(mesh.faces) > max(int(min_decimate_faces), int(target_faces)):
        simplified = _simplify_with_pyfqmr(mesh, int(target_faces))
        if simplified is not None:
            mesh = _cleanup_mesh_lossless(simplified)
    return mesh


_BLOCK_MESH_WORKERS_ENV = "CINEMAP_BLOCK_MESH_WORKERS"


def _resolve_max_workers(max_workers: int | None) -> int:
    """Block-meshing thread count: explicit value, else env override, else ~CPUs.

    Reads are I/O-bound (tensorstore over HTTP) and zmesh releases the GIL during
    marching cubes, so threads parallelize both the fetch and the mesh well."""
    if max_workers and int(max_workers) > 0:
        return int(max_workers)
    env = os.environ.get(_BLOCK_MESH_WORKERS_ENV)
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(8, (os.cpu_count() or 4)))


def _block_side(block_voxels: int) -> int:
    """Cube side (voxels) for a target per-block voxel budget."""
    return max(8, int(round(max(1, int(block_voxels)) ** (1.0 / 3.0))))


# Above this single-read array size, mesh blockwise instead so a sparse-but-huge
# bbox (e.g. thousands of scattered organelles at a fine scale) doesn't materialize
# the whole box at once and OOM. Capped here and also at a fraction of free RAM.
_BLOCKWISE_READ_BYTES_CAP = 4 * 1024 ** 3
_BLOCKWISE_READ_RAM_FRACTION = 0.25


def _meminfo_available_bytes() -> int | None:
    """Machine-wide free memory (MemAvailable) — what's physically free right now."""
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        return None
    return None


def _cgroup_mem_limit_bytes() -> tuple[int | None, int | None]:
    """``(limit, used)`` bytes for THIS process's memory cgroup, or ``(None, None)``.

    On a shared cluster node, a ``bsub -n 16`` job on a 48-core box is capped by the
    cgroup LSF places it in — NOT the node's free RAM — so this is the limit that
    actually matters. Handles cgroup v2 (``memory.max`` / ``memory.current``) and v1
    (``memory.limit_in_bytes`` / ``memory.usage_in_bytes``), resolving the process's
    own cgroup path from ``/proc/self/cgroup`` and falling back to the mount root."""
    # cgroup v2 unified hierarchy: lines look like "0::/some/path"
    try:
        rel = ""
        for ln in open("/proc/self/cgroup"):
            f = ln.strip().split(":")
            if len(f) == 3 and f[1] == "":
                rel = f[2].lstrip("/")
                break
        for cdir in (os.path.join("/sys/fs/cgroup", rel), "/sys/fs/cgroup"):
            mx = os.path.join(cdir, "memory.max")
            if os.path.exists(mx):
                raw = open(mx).read().strip()
                if raw and raw != "max":
                    cur_f = os.path.join(cdir, "memory.current")
                    used = int(open(cur_f).read().strip()) if os.path.exists(cur_f) else 0
                    return int(raw), used
                break
    except Exception:  # noqa: BLE001
        pass
    # cgroup v1: "<id>:memory:/path"
    try:
        rel = ""
        for ln in open("/proc/self/cgroup"):
            f = ln.strip().split(":")
            if len(f) >= 3 and "memory" in f[1].split(","):
                rel = f[2].lstrip("/")
                break
        for cdir in (os.path.join("/sys/fs/cgroup/memory", rel), "/sys/fs/cgroup/memory"):
            lf = os.path.join(cdir, "memory.limit_in_bytes")
            if os.path.exists(lf):
                limit = int(open(lf).read().strip())
                if limit < (1 << 62):  # v1 "unlimited" is a huge sentinel value
                    uf = os.path.join(cdir, "memory.usage_in_bytes")
                    used = int(open(uf).read().strip()) if os.path.exists(uf) else 0
                    return limit, used
                break
    except Exception:  # noqa: BLE001
        pass
    return None, None


def _rlimit_as_bytes() -> int | None:
    """Address-space rlimit (RLIMIT_AS), if a finite one is set (some schedulers do)."""
    try:
        import resource
        soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft not in (resource.RLIM_INFINITY, -1):
            return int(soft)
    except Exception:  # noqa: BLE001
        pass
    return None


def _available_ram_bytes() -> int | None:
    """Memory THIS process can actually use — the MIN of every cap we can detect:
    machine free memory, the job's cgroup headroom (LSF/containers enforce here), and
    the address-space rlimit. The min is the point: a memory-limited cluster job must
    size its reads off its own allocation, not the node's free RAM, or it gets
    OOM-killed at the cgroup limit long before the node runs out."""
    candidates = []
    mi = _meminfo_available_bytes()
    if mi:
        candidates.append(mi)
    limit, used = _cgroup_mem_limit_bytes()
    if limit:
        candidates.append(max(0, limit - (used or 0)))
    rl = _rlimit_as_bytes()
    if rl:
        candidates.append(rl)
    candidates = [c for c in candidates if c and c > 0]
    return min(candidates) if candidates else None


def _blockwise_read_threshold_bytes() -> int:
    avail = _available_ram_bytes()
    if avail:
        return int(min(_BLOCKWISE_READ_BYTES_CAP, _BLOCKWISE_READ_RAM_FRACTION * avail))
    return _BLOCKWISE_READ_BYTES_CAP


def _level_itemsize(vol, level: int) -> int:
    dt = vol._open_level(level).dtype
    return getattr(getattr(dt, "numpy_dtype", None), "itemsize", None) or np.dtype(str(dt)).itemsize


def _planned_read_bytes(vol, plan) -> int:
    """Bytes a single (non-blockwise) read of this plan would materialize.

    Pure arithmetic from the bbox extent and the level's voxel size (the shape/scale
    metadata is already cached) — no voxel data is read, so this is free to check
    before committing to the read."""
    box_voxels = _bbox_voxels_for_level(vol, plan.bbox_xyz_nm, plan.level, pad=3)
    if plan.stride and int(plan.stride) > 1:
        # A strided plan still reads the FULL box before subsampling, so peak memory
        # tracks the unstrided box.
        pass
    return int(box_voxels) * _level_itemsize(vol, plan.level)


def _resolve_blockwise(vol, plan, blockwise) -> bool:
    """Decide blockwise vs single read for one plan.

    ``True``/``False`` force the choice; ``"auto"`` (default) reads blockwise only
    when the single read would exceed the memory threshold. Strided plans can't use
    blockwise (striding breaks boundary welds), so they always read single."""
    if blockwise is True or blockwise is False:
        return bool(blockwise)
    if plan.stride and int(plan.stride) > 1:
        return False
    try:
        return _planned_read_bytes(vol, plan) > _blockwise_read_threshold_bytes()
    except Exception:  # noqa: BLE001 — if we can't size it, prefer the memory-safe path
        return True


def _iter_block_ranges(box, side):
    """Tile a voxel box ((z0,z1),(y0,y1),(x0,x1)) into cubic block core ranges."""
    (z0, z1), (y0, y1), (x0, x1) = box
    for bz0 in range(z0, z1, side):
        bz1 = min(bz0 + side, z1)
        for by0 in range(y0, y1, side):
            by1 = min(by0 + side, y1)
            for bx0 in range(x0, x1, side):
                bx1 = min(bx0 + side, x1)
                yield (bz0, bz1), (by0, by1), (bx0, bx1)


def _block_read_axis(c0, c1, box_lo, box_hi, dim, halo):
    """Per-axis (read_start, read_stop, pad_lo, pad_hi) for one block's meshing read.

    Marching cubes needs a voxel of context beyond the geometry it meshes, so every
    block is read with a halo:
      * interior block boundaries get a +1-voxel overlap so the straddling cube is
        meshed once (by the lower block) and both blocks emit identical vertices on
        the shared plane — a later weld stitches them;
      * at the ROI-box exterior we pull up to ``halo`` REAL neighbor voxels from the
        volume when it extends past the box (true padding for meshing), and only seal
        with a background voxel where the read reaches the actual volume edge.
    Returns indices into the full level array plus the background pad to apply."""
    if c0 <= box_lo:                       # block on the ROI-box low edge
        read0 = max(0, c0 - halo)          # pad with real data toward the volume edge
        pad_lo = 1 if read0 == 0 else 0    # only seal at the true volume boundary
    else:
        read0 = c0                         # interior: neighbor owns the straddling cube
        pad_lo = 0
    if c1 >= box_hi:                        # block on the ROI-box high edge
        read1 = min(dim, c1 + halo)
        pad_hi = 1 if read1 == dim else 0
    else:
        read1 = min(dim, c1 + 1)            # +1 overlap into the next block
        pad_hi = 0
    return read0, read1, pad_lo, pad_hi


def _mesh_one_block(arr_handle, box, block, seg_ids, sc, tr, colorize, *, halo=1):
    """Marching-cubes one block; return ``{seg_id: [trimesh, ...]}`` in world nm.

    Reads a real-voxel halo around the block (see :func:`_block_read_axis`) so each
    block's marching cubes has the neighbor context it needs, then runs zmesh with
    ``close=False`` so interior boundary planes are left open for the cross-block
    weld. Background padding is added only on faces that reach the true volume edge,
    matching the single-read path's ``np.pad(labels, 1)`` seal."""
    (Z0, Z1), (Y0, Y1), (X0, X1) = box
    (bz0, bz1), (by0, by1), (bx0, bx1) = block
    shape = arr_handle.shape
    rz0, rz1, pz0, pz1 = _block_read_axis(bz0, bz1, Z0, Z1, shape[0], halo)
    ry0, ry1, py0, py1 = _block_read_axis(by0, by1, Y0, Y1, shape[1], halo)
    rx0, rx1, px0, px1 = _block_read_axis(bx0, bx1, X0, X1, shape[2], halo)
    sub = np.asarray(arr_handle[rz0:rz1, ry0:ry1, rx0:rx1].read().result())
    keep = np.isin(sub, seg_ids)
    if not keep.any():
        return {}
    labels = np.where(keep, sub, 0).astype(np.uint32, copy=False)
    labels = np.pad(labels, ((pz0, pz1), (py0, py1), (px0, px1)))
    mesher = zmesh.Mesher(tuple(float(s) for s in sc))
    mesher.mesh(labels, close=False)
    origin_zyx_nm = (
        (rz0 - pz0) * sc[0] + tr[0],
        (ry0 - py0) * sc[1] + tr[1],
        (rx0 - px0) * sc[2] + tr[2],
    )
    out: dict[int, list] = {}
    available = set(int(s) for s in mesher.ids())
    for seg_id in seg_ids:
        if seg_id not in available:
            continue
        try:
            raw = mesher.get(seg_id, reduction_factor=0, max_error=None, voxel_centered=False)
        except TypeError:
            raw = mesher.get(seg_id)
        part = _mesh_to_trimesh(raw, seg_id, origin_zyx_nm, colorize=colorize)
        if part is not None:
            out.setdefault(seg_id, []).append(part)
    return out


def _blockwise_raw_parts(vol, plan, seg_ids, *, colorize, block_voxels, max_workers, block_halo=1):
    """Block-by-block meshing of a plan's ROI; one welded trimesh per present seg.

    Bounds peak memory to ~one block (vs. the single whole-ROI read), so a finer
    pyramid level fits the same RAM — read more vertices, then optionally decimate.
    Each block is meshed with a ``block_halo``-voxel real-data pad (see
    :func:`_block_read_axis`). Returns ``None`` for strided plans (striding would
    break boundary welds), so the caller falls back to the single-read path."""
    if plan.stride and int(plan.stride) > 1:
        return None
    level = plan.level
    arr_handle = vol._open_level(level)
    sc = tuple(float(s) for s in vol.level_scale_nm[level])
    tr = tuple(float(t) for t in vol.level_translation_nm[level])
    if plan.bbox_xyz_nm is not None:
        (z0, z1), (y0, y1), (x0, x1) = vol._box_voxel_bounds(plan.bbox_xyz_nm, level, pad=3)
    else:
        shp = vol.level_shape_zyx(level)
        z0, z1, y0, y1, x0, x1 = 0, shp[0], 0, shp[1], 0, shp[2]
    box = ((z0, z1), (y0, y1), (x0, x1))
    blocks = list(_iter_block_ranges(box, _block_side(block_voxels)))
    workers = _resolve_max_workers(max_workers)

    def _run(block):
        return _mesh_one_block(
            arr_handle, box, block, seg_ids, sc, tr, colorize, halo=max(1, int(block_halo))
        )

    if workers > 1 and len(blocks) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(blocks))) as ex:
            results = list(ex.map(_run, blocks))
    else:
        results = [_run(b) for b in blocks]

    per_seg: dict[int, list] = {}
    for res in results:
        for seg_id, parts in res.items():
            per_seg.setdefault(seg_id, []).extend(parts)

    raw_parts = []
    for seg_id in seg_ids:
        parts = per_seg.get(seg_id)
        if not parts:
            continue
        if len(parts) == 1:
            raw_parts.append(parts[0])
            continue
        welded = trimesh.util.concatenate(parts)
        try:
            welded.merge_vertices()  # weld coincident boundary vertices across blocks
        except Exception:  # noqa: BLE001
            pass
        raw_parts.append(welded)
    return raw_parts, sc


def _raw_parts_for_plan(
    vol,
    plan,
    seg_ids,
    *,
    colorize,
    blockwise,
    block_voxels,
    block_halo,
    max_workers,
    label_name,
    target_vertices,
):
    """Return ``(raw_parts, scale_zyx)`` for one plan: a welded trimesh per present
    seg id in world (xyz nm) coords, before postprocessing. An empty list signals
    the caller to try the next (coarser/finer) plan. Picks the blockwise reader when
    enabled (and the plan isn't strided); otherwise reads the whole ROI at once."""
    if blockwise:
        blk = _blockwise_raw_parts(
            vol, plan, seg_ids, colorize=colorize,
            block_voxels=block_voxels, max_workers=max_workers, block_halo=block_halo,
        )
        if blk is not None:
            raw_parts, sc = blk
            print(
                "[labels] zmesh probe (blockwise) "
                f"{label_name} ids={len(seg_ids)} level=s{plan.level} "
                f"scale_zyx_nm=({sc[0]:.3g},{sc[1]:.3g},{sc[2]:.3g}) "
                f"block_voxels={int(block_voxels)} parts={len(raw_parts)} "
                f"target_vertices={target_vertices}",
                flush=True,
            )
            if not raw_parts:
                print(
                    "[labels] zmesh probe (blockwise) produced no geometry; trying next level "
                    f"{label_name} level=s{plan.level}",
                    flush=True,
                )
            return raw_parts, sc

    arr, (z0, y0, x0), sc, tr = _read_plan_array(vol, plan, pad=3)
    print(
        "[labels] zmesh probe "
        f"{label_name} ids={len(seg_ids)} "
        f"level=s{plan.level} stride={plan.stride} "
        f"scale_zyx_nm=({sc[0]:.3g},{sc[1]:.3g},{sc[2]:.3g}) "
        f"target_vertices={target_vertices} "
        f"surface_faces_estimate={plan.surface_faces_estimate} "
        f"surface_budget={plan.surface_budget}",
        flush=True,
    )
    keep = np.isin(arr, seg_ids)
    if not keep.any():
        print(
            "[labels] zmesh probe empty; trying finer level "
            f"{label_name} level=s{plan.level}",
            flush=True,
        )
        return [], sc
    labels = np.where(keep, arr, 0).astype(np.uint32, copy=False)
    labels = np.pad(labels, 1)

    mesher = zmesh.Mesher(tuple(float(s) for s in sc))
    mesher.mesh(labels)
    # Always mesh at full resolution (no in-mesher reduction); the optional
    # decimate_fraction handles lossy reduction uniformly afterward, so the single-read
    # and blockwise paths produce matching geometry.
    origin_zyx_nm = (
        (z0 - 1) * sc[0] + tr[0],
        (y0 - 1) * sc[1] + tr[1],
        (x0 - 1) * sc[2] + tr[2],
    )
    raw_parts = []
    available = set(int(s) for s in mesher.ids())
    for seg_id in seg_ids:
        if seg_id not in available:
            continue
        try:
            raw_mesh = mesher.get(
                seg_id,
                reduction_factor=0,
                max_error=None,
                voxel_centered=False,
            )
        except TypeError:
            raw_mesh = mesher.get(seg_id)
        part = _mesh_to_trimesh(raw_mesh, seg_id, origin_zyx_nm, colorize=colorize)
        if part is not None:
            raw_parts.append(part)
    if not raw_parts:
        print(
            "[labels] zmesh probe produced no geometry; trying finer level "
            f"{label_name} level=s{plan.level}",
            flush=True,
        )
    return raw_parts, sc


def generate_zmesh_auto(
    label_zarr_url: str,
    seg_ids,
    *,
    bbox_xyz_nm=None,
    target_voxels: int = 20_000_000,
    target_vertices: int | None = None,
    smooth_iters: int = 0,
    decimate_fraction: float = 0.0,
    blockwise: bool | str = "auto",
    block_voxels: int = 16_000_000,
    block_halo: int = 1,
    max_workers: int | None = None,
    colorize=None,
) -> trimesh.Trimesh:
    """Generate selected IDs with zmesh at the finest scale that fits the budget.

    `bbox_xyz_nm` lets callers pass a selected-object ROI from existing meshes. When
    absent, we scan a coarse label level to find a conservative ROI before selecting
    the final meshing scale. zmesh emits one mesh per label ID, so colors and object
    boundaries survive even when many IDs are meshed in one array read.

    `blockwise` reads and meshes the chosen ROI one cubic block at a time
    (`block_voxels` each, up to `max_workers` in parallel) and welds the per-block
    surfaces, instead of one whole-ROI read. ``"auto"`` (default) does this only when
    the single read would exceed a memory threshold (sparse-but-huge bboxes — e.g.
    thousands of scattered organelles at a fine scale — that would otherwise OOM);
    ``True``/``False`` force it. For a compact object whose read fits, the single read
    is faster, so auto leaves it alone. `target_vertices` caps how many raw vertices we
    LOAD (the planner picks the finest level whose raw count fits, never finer);
    `decimate_fraction` (0<f<1) then reduces the result to ~that fraction of its faces,
    so the final mesh lands BELOW the budget (e.g. load 15M, decimate 0.25 -> ~3.7M).
    `smooth_iters` applies Taubin smoothing. Decimation/smoothing run per segment so
    instance colors are preserved.
    """
    decimate_fraction = float(decimate_fraction or 0.0)
    decimate_active = 0.0 < decimate_fraction < 1.0
    seg_ids = [int(s) for s in seg_ids]
    if not seg_ids:
        raise ValueError("no segment ids")
    vol = get_volume(label_zarr_url)
    if bbox_xyz_nm is None:
        bbox_xyz_nm = _bbox_for_ids(label_zarr_url, seg_ids)

    plans = _candidate_plans(
        vol,
        bbox_xyz_nm,
        target_voxels=target_voxels,
        target_vertices=target_vertices,
        seg_ids=seg_ids,
    )
    best_mesh: trimesh.Trimesh | None = None
    best_plan: _ReadPlan | None = None
    best_vertices = 0
    best_faces = 0
    last_error: BaseException | None = None

    for plan in plans:
        try:
            plan_blockwise = _resolve_blockwise(vol, plan, blockwise)
            if blockwise == "auto":
                print(
                    "[labels] zmesh blockwise=auto -> "
                    f"{'blockwise' if plan_blockwise else 'single'} "
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                    f"planned_read={_planned_read_bytes(vol, plan) / 1e9:.2f}GB "
                    f"threshold={_blockwise_read_threshold_bytes() / 1e9:.2f}GB",
                    flush=True,
                )
            raw_parts, sc = _raw_parts_for_plan(
                vol,
                plan,
                seg_ids,
                colorize=colorize,
                blockwise=plan_blockwise,
                block_voxels=block_voxels,
                block_halo=block_halo,
                max_workers=max_workers,
                label_name=_label_source_name(label_zarr_url),
                target_vertices=target_vertices,
            )
            if not raw_parts:
                continue
            # Vertices we LOADED (pre-decimation) — the budget/refinement gate when
            # decimation is active, so target_vertices governs the read scale and the
            # decimation step reduces the final mesh from there.
            raw_vertices = sum(len(p.vertices) for p in raw_parts)

            parts = []
            for part in raw_parts:
                part_target = None
                min_decimate_faces = _MIN_DECIMATE_FACES
                if decimate_active:
                    # Decimate each segment to the keep-fraction (per segment so colors
                    # are preserved), flooring only at a small viable face count so the
                    # fraction holds even on many-small-segment layers.
                    part_target = max(_DECIMATE_MIN_FACES, int(len(part.faces) * decimate_fraction))
                    min_decimate_faces = _DECIMATE_MIN_FACES
                parts.append(
                    _postprocess_label_mesh(
                        part,
                        target_faces=part_target,
                        smooth_iters=smooth_iters,
                        min_decimate_faces=min_decimate_faces,
                    )
                )
            parts = [p for p in parts if len(p.faces) and len(p.vertices)]
            if not parts:
                print(
                    "[labels] zmesh probe empty after cleanup; trying finer level "
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level}",
                    flush=True,
                )
                continue
            mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
            mesh = _cleanup_mesh_lossless(mesh)
            vertices = len(mesh.vertices)
            faces = len(mesh.faces)
            # Gate refinement on the RAW (pre-decimation) vertex count, so
            # `target_vertices` caps what we LOAD — the planner never meshes a level
            # finer than the budget, and decimation then reduces the result BELOW the
            # limit. This is fast: it avoids meshing one level too fine just to discover
            # it overshoots. (Gating on the decimated count instead would chase the
            # budget by reading ~1/fraction more data — far slower for little gain.)
            # `vertices` is the delivered (decimated) count, logged for insight.
            gate_vertices = raw_vertices
            print(
                "[labels] zmesh probe result "
                f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                f"vertices={vertices} raw_vertices={raw_vertices} faces={faces} "
                f"target_vertices={target_vertices}",
                flush=True,
            )

            if target_vertices:
                accept_limit = int(np.floor(int(target_vertices) * _PLAN_NEXT_LEVEL_TOLERANCE))
            else:
                accept_limit = 0
            if target_vertices and gate_vertices > accept_limit:
                if best_mesh is not None:
                    print(
                        "[labels] zmesh probe exceeded budget; using previous coarser level "
                        f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                        f"vertices={vertices} target_vertices={target_vertices} "
                        f"accept_limit={accept_limit} "
                        f"chosen=s{best_plan.level if best_plan else '?'} "
                        f"chosen_vertices={best_vertices}",
                        flush=True,
                    )
                    break
                print(
                    "[labels] zmesh coarsest probe exceeds budget; using it anyway "
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                    f"vertices={vertices} target_vertices={target_vertices} "
                    f"accept_limit={accept_limit}",
                    flush=True,
                )
                best_mesh = mesh
                best_plan = plan
                best_vertices = vertices
                best_faces = faces
                break
            if target_vertices and gate_vertices > int(target_vertices):
                print(
                    "[labels] zmesh probe accepted within tolerance "
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                    f"vertices={vertices} target_vertices={target_vertices} "
                    f"accept_limit={accept_limit} "
                    f"tolerance={_PLAN_NEXT_LEVEL_TOLERANCE:g}",
                    flush=True,
                )

            best_mesh = mesh
            best_plan = plan
            best_vertices = vertices
            best_faces = faces

            if target_vertices:
                next_prediction = int(np.ceil(gate_vertices * _PLAN_NEXT_LEVEL_VERTEX_FACTOR))
                next_limit = int(np.floor(int(target_vertices) * _PLAN_NEXT_LEVEL_TOLERANCE))
                if next_prediction > next_limit:
                    print(
                        "[labels] zmesh next finer level predicted over budget; not trying finer level "
                        f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                        f"vertices={vertices} target_vertices={target_vertices} "
                        f"next_level_prediction={next_prediction} "
                        f"next_level_factor={_PLAN_NEXT_LEVEL_VERTEX_FACTOR:g} "
                        f"tolerance={_PLAN_NEXT_LEVEL_TOLERANCE:g}",
                        flush=True,
                    )
                    break
                print(
                    "[labels] zmesh next finer level predicted within budget; trying finer level "
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                    f"vertices={vertices} target_vertices={target_vertices} "
                    f"next_level_prediction={next_prediction} "
                    f"next_level_factor={_PLAN_NEXT_LEVEL_VERTEX_FACTOR:g} "
                    f"tolerance={_PLAN_NEXT_LEVEL_TOLERANCE:g}",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            if not _is_memory_error(e):
                raise
            last_error = e
            if best_mesh is not None:
                print(
                    "[labels] zmesh probe memory fallback; using previous coarser level "
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level} failed ({e}); "
                    f"chosen=s{best_plan.level if best_plan else '?'} "
                    f"chosen_vertices={best_vertices}",
                    flush=True,
                )
                break
            print(
                "[labels] zmesh probe memory fallback: "
                f"level={plan.level} stride={plan.stride} failed ({e}); "
                "trying next candidate"
            )

    if best_mesh is None:
        if last_error is not None:
            raise last_error
        raise ValueError("zmesh produced no geometry for the selected segments")
    print(
        "[labels] zmesh selected "
        f"{_label_source_name(label_zarr_url)} level=s{best_plan.level if best_plan else '?'} "
        f"vertices={best_vertices} faces={best_faces} target_vertices={target_vertices}",
        flush=True,
    )
    return best_mesh


def _vertex_colors(verts_world_zyx, normals, sc, tr, arr, colorize=None) -> np.ndarray:
    """Per-vertex RGBA by segment id.

    A marching-cubes vertex sits on the segment/background boundary, so sampling
    the label at the rounded voxel straddles into background (id 0) for ~half the
    vertices — which would tint each mesh with two colors. Instead we sample one
    voxel *inside* the surface (along the inward normal) so every vertex gets its
    real segment id.
    """
    sc = np.array(sc, dtype=float)
    tr = np.array(tr, dtype=float)
    shape = np.array(arr.shape)

    def sample(points):  # world (z,y,x) nm -> voxel index
        v = np.clip(np.round((points - tr) / sc).astype(int), 0, shape - 1)
        return arr[v[:, 0], v[:, 1], v[:, 2]]

    segs = sample(verts_world_zyx - normals * sc)      # one voxel inside (normals point out)
    zero = segs == 0
    if zero.any():                                     # fall back to the other side, then the vertex
        segs[zero] = sample(verts_world_zyx + normals * sc)[zero]
    zero = segs == 0
    if zero.any():
        segs[zero] = sample(verts_world_zyx)[zero]

    cf = colorize or (lambda s: seg_color(s))
    lut = {int(s): (cf(int(s)) if s != 0 else (0.6, 0.6, 0.6)) for s in np.unique(segs)}
    rgba = np.ones((len(segs), 4), dtype=np.float64)
    for s, c in lut.items():
        rgba[segs == s, :3] = c
    return (rgba * 255).astype(np.uint8)


def generate(
    label_zarr_url: str,
    seg_id: int,
    bbox_xyz_nm,
    target_voxels: int = 8_000_000,
    smooth_iters: int = 10,
    colorize=None,
) -> trimesh.Trimesh:
    """Marching-cubes mesh (nm, x/y/z) for `seg_id` within `bbox_xyz_nm`.

    bbox_xyz_nm = ((x0,y0,z0), (x1,y1,z1)). The bbox typically comes from the
    precomputed mesh (its global placement is correct); geometry comes from labels.
    """
    vol = get_volume(label_zarr_url)
    level = vol.pick_level_for_box(bbox_xyz_nm, target_voxels)
    sub, (z0, y0, x0), sc, tr = vol.read_box(bbox_xyz_nm, level, pad=2)

    mask = (sub == seg_id).astype(np.uint8)
    if not mask.any():
        raise ValueError(f"segment {seg_id} not present in label box")
    mask = np.pad(mask, 1)  # seal the surface at the box border

    # world_zyx = (voxel_index * scale) + translation; verts are in (z,y,x) nm
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=sc)
    origin_zyx_nm = np.array([(z0 - 1) * sc[0] + tr[0], (y0 - 1) * sc[1] + tr[1],
                              (x0 - 1) * sc[2] + tr[2]])
    verts_world_zyx = verts + origin_zyx_nm
    verts_xyz = verts_world_zyx[:, ::-1]  # z,y,x -> x,y,z

    vcolors = None
    if colorize is not None:  # uniform color for this single segment (NG color)
        r, g, b = colorize(int(seg_id))
        vcolors = np.tile(np.array([r, g, b, 1.0]) * 255, (len(verts_xyz), 1)).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces[:, ::-1], vertex_colors=vcolors, process=False)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    return mesh


def generate_union(
    label_zarr_url: str,
    seg_ids,
    target_voxels: int = 20_000_000,
    smooth_iters: int = 6,
    colorize=None,
) -> trimesh.Trimesh:
    """One mesh for ALL `seg_ids` at once — a single coarse whole-volume label
    read + marching cubes on the union mask. Cheap for hundreds/thousands of
    segments (e.g. "show every nucleus") vs. per-segment reads.
    """
    vol = get_volume(label_zarr_url)
    # finest whole-volume level whose total voxels fit the budget
    level = len(vol.level_scale_nm) - 1
    for lvl in range(len(vol.level_scale_nm)):
        shp = vol.level_shape_zyx(lvl)
        if shp[0] * shp[1] * shp[2] <= target_voxels:
            level = lvl
            break
    # Even the coarsest available level can blow the budget (e.g. a single-scale
    # volume with no downsampled levels). Subsample on read with an integer stride
    # so the materialized array — and thus peak memory — stays bounded regardless.
    shp = vol.level_shape_zyx(level)
    total = shp[0] * shp[1] * shp[2]
    stride = max(1, int(np.ceil((total / max(1, target_voxels)) ** (1 / 3))))
    arr = np.asarray(vol._open_level(level)[::stride, ::stride, ::stride].read().result())
    sc = [s * stride for s in vol.level_scale_nm[level]]   # nm/voxel after striding (z,y,x)
    tr = vol.level_translation_nm[level]    # z,y,x nm; world = voxel*scale + translation
    mask = np.isin(arr, list(seg_ids)).astype(np.uint8)
    if not mask.any():
        raise ValueError("none of the selected segments present in labels")
    mask = np.pad(mask, 1)
    verts, faces, normals, _ = measure.marching_cubes(mask, level=0.5, spacing=sc)
    verts_world_zyx = verts + np.array([tr[0] - sc[0], tr[1] - sc[1], tr[2] - sc[2]])  # undo pad + translation
    colors = _vertex_colors(verts_world_zyx, normals, sc, tr, arr, colorize)  # per-segment color
    mesh = trimesh.Trimesh(vertices=verts_world_zyx[:, ::-1], faces=faces[:, ::-1],
                           vertex_colors=colors, process=False)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    return mesh


def selected_region(label_zarr_url: str, seg_ids, target_voxels: int = 12_000_000):
    """Robust center + radius (xyz nm) of where the selected segments actually are,
    ignoring sparse outliers — so framing/slicing land on the dense cluster."""
    vol = get_volume(label_zarr_url)
    level = len(vol.level_scale_nm) - 1
    for lvl in range(len(vol.level_scale_nm)):
        shp = vol.level_shape_zyx(lvl)
        if shp[0] * shp[1] * shp[2] <= target_voxels:
            level = lvl
            break
    arr = np.asarray(vol._open_level(level)[:, :, :].read().result())
    sc = vol.level_scale_nm[level]          # z,y,x
    tr = vol.level_translation_nm[level]    # z,y,x
    zz, yy, xx = np.where(np.isin(arr, list(seg_ids)))
    if len(zz) == 0:
        raise ValueError("selected segments not found")
    # 5th–95th percentile bounds in nm, per axis (x,y,z); world = voxel*scale + translation
    lo = np.array([np.percentile(xx, 5) * sc[2] + tr[2], np.percentile(yy, 5) * sc[1] + tr[1],
                   np.percentile(zz, 5) * sc[0] + tr[0]])
    hi = np.array([np.percentile(xx, 95) * sc[2] + tr[2], np.percentile(yy, 95) * sc[1] + tr[1],
                   np.percentile(zz, 95) * sc[0] + tr[0]])
    center = (lo + hi) / 2
    radius = 0.5 * float(np.max(hi - lo))
    return center.tolist(), max(radius, 1000.0)
