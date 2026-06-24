"""Generate clean meshes for segments directly from OME-Zarr label volumes.

This bypasses the precomputed multilod-draco meshes (which cloud-volume decodes
with per-chunk gaps, producing a 'stippled' look). The preferred path uses zmesh
to generate separate meshes per selected label ID, preserving instance colors and
boundaries while choosing the finest label scale that fits the render budget.
"""
from __future__ import annotations

import colorsys
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
_FACES_PER_VERTEX_BUDGET = 2.1
_LOSSLESS_VERTEX_DEDUP_LIMIT = 8_000_000
_MIN_DECIMATE_FACES = 100_000
_PYFQMR_MISSING = False
_ZMESH_GENTLE_REDUCTION_FACTOR = 20
_ZMESH_BUDGET_REDUCTION_FACTOR = 50
_ZMESH_MAX_ERROR_VOXEL_FRACTION = 0.25


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


def _zmesh_simplify_params(simplify_budget_factor: float, scale_zyx_nm) -> tuple[int, float | None]:
    """Map UI simplification policy to zmesh's physical-error simplifier."""
    if simplify_budget_factor <= 0:
        return 0, None
    min_voxel_nm = min(float(s) for s in scale_zyx_nm)
    max_error = _ZMESH_MAX_ERROR_VOXEL_FRACTION * min_voxel_nm
    if simplify_budget_factor <= 1.25:
        return _ZMESH_BUDGET_REDUCTION_FACTOR, max_error
    return _ZMESH_GENTLE_REDUCTION_FACTOR, max_error


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


def _simplify_with_pyfqmr(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh | None:
    global _PYFQMR_MISSING
    if _PYFQMR_MISSING:
        return None
    try:
        import pyfqmr  # type: ignore
    except Exception:  # noqa: BLE001
        _PYFQMR_MISSING = True
        return None
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
        return None


def _postprocess_label_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_faces: int | None = None,
    smooth_iters: int = 0,
) -> trimesh.Trimesh:
    mesh = _cleanup_mesh_lossless(mesh)
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
        mesh = _cleanup_mesh_lossless(mesh)
    if target_faces is not None and len(mesh.faces) > max(_MIN_DECIMATE_FACES, int(target_faces)):
        simplified = _simplify_with_pyfqmr(mesh, int(target_faces))
        if simplified is not None:
            mesh = _cleanup_mesh_lossless(simplified)
    return mesh


def generate_zmesh_auto(
    label_zarr_url: str,
    seg_ids,
    *,
    bbox_xyz_nm=None,
    target_voxels: int = 20_000_000,
    target_vertices: int | None = None,
    smooth_iters: int = 0,
    simplify_budget_factor: float = 0.0,
    colorize=None,
) -> trimesh.Trimesh:
    """Generate selected IDs with zmesh at the finest scale that fits the budget.

    `bbox_xyz_nm` lets callers pass a selected-object ROI from existing meshes. When
    absent, we scan a coarse label level to find a conservative ROI before selecting
    the final meshing scale. zmesh emits one mesh per label ID, so colors and object
    boundaries survive even when many IDs are meshed in one array read.
    """
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
            arr, (z0, y0, x0), sc, tr = _read_plan_array(vol, plan, pad=3)
            print(
                "[labels] zmesh probe "
                f"{_label_source_name(label_zarr_url)} ids={len(seg_ids)} "
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
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level}",
                    flush=True,
                )
                continue
            labels = np.where(keep, arr, 0).astype(np.uint32, copy=False)
            labels = np.pad(labels, 1)

            mesher = zmesh.Mesher(tuple(float(s) for s in sc))
            mesher.mesh(labels)
            zmesh_reduction, zmesh_max_error = _zmesh_simplify_params(simplify_budget_factor, sc)
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
                        reduction_factor=zmesh_reduction,
                        max_error=zmesh_max_error,
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
                    f"{_label_source_name(label_zarr_url)} level=s{plan.level}",
                    flush=True,
                )
                continue

            target_faces_total = None
            if target_vertices and simplify_budget_factor > 0:
                target_faces_total = max(
                    1,
                    int(target_vertices * _FACES_PER_VERTEX_BUDGET * float(simplify_budget_factor)),
                )
            total_faces = sum(len(p.faces) for p in raw_parts)

            parts = []
            for part in raw_parts:
                part_target = None
                if target_faces_total is not None and total_faces > target_faces_total:
                    frac = len(part.faces) / max(1, total_faces)
                    part_target = max(256, int(target_faces_total * frac))
                parts.append(
                    _postprocess_label_mesh(
                        part,
                        target_faces=part_target,
                        smooth_iters=smooth_iters,
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
            print(
                "[labels] zmesh probe result "
                f"{_label_source_name(label_zarr_url)} level=s{plan.level} "
                f"vertices={vertices} faces={faces} target_vertices={target_vertices}",
                flush=True,
            )

            if target_vertices:
                accept_limit = int(np.floor(int(target_vertices) * _PLAN_NEXT_LEVEL_TOLERANCE))
            else:
                accept_limit = 0
            if target_vertices and vertices > accept_limit:
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
            if target_vertices and vertices > int(target_vertices):
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
                next_prediction = int(np.ceil(vertices * _PLAN_NEXT_LEVEL_VERTEX_FACTOR))
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
