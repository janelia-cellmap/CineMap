"""Blockwise label meshing: seams must weld so the assembled mesh matches a
single whole-ROI read, and decimation must reduce face count."""
import numpy as np
import trimesh
import zmesh

from cinemap.data import mesh_from_labels as m


class _Reader:
    def __init__(self, a):
        self._a = a

    def read(self):
        return self

    def result(self):
        return self._a


class _Level:
    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape

    def __getitem__(self, idx):
        return _Reader(self._arr[idx])


class _FakeVol:
    """Minimal stand-in for EMVolume exposing only what the blockwise path uses."""

    def __init__(self, arr, scale=(2.0, 2.0, 2.0), trans=(10.0, 20.0, 30.0)):
        self._arr = arr
        self.level_scale_nm = [list(scale)]
        self.level_translation_nm = [list(trans)]

    def _open_level(self, level):
        return _Level(self._arr)

    def level_shape_zyx(self, level):
        return self._arr.shape

    def _box_voxel_bounds(self, bbox, level, pad=2):
        s = self._arr.shape
        return (0, s[0]), (0, s[1]), (0, s[2])


def _solid_sphere(n=40, r=12, seg_id=7):
    c = n / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    mask = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= r ** 2
    arr = np.zeros((n, n, n), dtype=np.uint32)
    arr[mask] = seg_id
    return arr


def _single_read_mesh(arr, sc, tr, seg_id):
    """Replicate the non-blockwise path: one padded whole-array zmesh."""
    labels = np.where(arr == seg_id, arr, 0).astype(np.uint32)
    labels = np.pad(labels, 1)
    mesher = zmesh.Mesher(tuple(float(s) for s in sc))
    mesher.mesh(labels)
    raw = mesher.get(seg_id, reduction_factor=0, max_error=None, voxel_centered=False)
    origin = ((0 - 1) * sc[0] + tr[0], (0 - 1) * sc[1] + tr[1], (0 - 1) * sc[2] + tr[2])
    mesh = m._mesh_to_trimesh(raw, seg_id, origin)
    mesh.merge_vertices()
    return mesh


def _make_plan(arr):
    n = arr.shape[0]
    return m._ReadPlan(level=0, stride=1, bbox_xyz_nm=((0, 0, 0), (float(n),) * 3))


def test_blockwise_welds_seams_and_matches_single_read():
    seg_id = 7
    arr = _solid_sphere(n=40, r=12, seg_id=seg_id)
    sc = (2.0, 2.0, 2.0)
    tr = (10.0, 20.0, 30.0)
    vol = _FakeVol(arr, scale=sc, trans=tr)

    baseline = _single_read_mesh(arr, sc, tr, seg_id)

    # block side ~16 voxels -> the 40^3 ROI tiles into 3x3x3 = 27 blocks with seams.
    raw_parts, _ = m._blockwise_raw_parts(
        vol, _make_plan(arr), [seg_id], colorize=None,
        block_voxels=16 ** 3, max_workers=1,
    )
    assert len(raw_parts) == 1
    blockwise = raw_parts[0]

    # Seams welded => the assembled mesh is a single watertight shell, like the
    # single whole-ROI read. A broken weld would leave open seams (not watertight)
    # and split the surface into many connected components.
    assert blockwise.is_watertight, "blockwise seams did not weld (mesh not watertight)"
    assert blockwise.body_count == 1

    # Enclosed volume should match the single read to within a fraction of a voxel
    # worth of surface variation.
    rel = abs(blockwise.volume - baseline.volume) / baseline.volume
    assert rel < 0.02, f"volume mismatch vs single read: {rel:.4f}"

    # Same world placement (block offsets applied correctly).
    np.testing.assert_allclose(blockwise.bounds, baseline.bounds, atol=sc[0])


class _DtypeVol:
    """Vol stub for the auto-blockwise resolver: only needs level scale/shape/dtype."""

    def __init__(self, shape, scale, itemsize=2):
        self._shape = shape
        self.level_scale_nm = [list(scale)]
        self.level_translation_nm = [[0.0, 0.0, 0.0]]
        self._itemsize = itemsize

    def level_shape_zyx(self, level):
        return self._shape

    def _open_level(self, level):
        return type("L", (), {"dtype": np.dtype("uint16" if self._itemsize == 2 else "uint8"),
                              "shape": self._shape})()


def test_resolve_blockwise_auto_picks_by_read_size(monkeypatch):
    import cinemap.data.mesh_from_labels as mod
    monkeypatch.setattr(mod, "_blockwise_read_threshold_bytes", lambda: 4 * 1024 ** 3)

    vol = _DtypeVol((4000, 4000, 4000), (16.0, 16.0, 16.0), itemsize=2)
    # Explicit overrides always win.
    small = mod._ReadPlan(level=0, stride=1, bbox_xyz_nm=((0, 0, 0), (1600, 1600, 1600)))
    assert mod._resolve_blockwise(vol, small, True) is True
    assert mod._resolve_blockwise(vol, small, False) is False

    # auto: a compact bbox (~0.13 GB) stays single; a ~2k^3-voxel bbox (~16 GB) flips.
    assert mod._resolve_blockwise(vol, small, "auto") is False
    huge = mod._ReadPlan(level=0, stride=1, bbox_xyz_nm=((0, 0, 0), (32000, 32000, 32000)))
    assert mod._resolve_blockwise(vol, huge, "auto") is True

    # Strided plans can never go blockwise (welds would break).
    strided = mod._ReadPlan(level=0, stride=2, bbox_xyz_nm=((0, 0, 0), (32000, 32000, 32000)))
    assert mod._resolve_blockwise(vol, strided, "auto") is False


def test_available_ram_takes_min_of_caps(monkeypatch):
    """A memory-limited cluster job must size off its cgroup, not the node's free RAM."""
    import cinemap.data.mesh_from_labels as mod
    # Node reports 90 GB free, but the job's cgroup grants 16 GB with 2 GB used.
    monkeypatch.setattr(mod, "_meminfo_available_bytes", lambda: 90 * 1024 ** 3)
    monkeypatch.setattr(mod, "_cgroup_mem_limit_bytes", lambda: (16 * 1024 ** 3, 2 * 1024 ** 3))
    monkeypatch.setattr(mod, "_rlimit_as_bytes", lambda: None)
    # -> headroom 14 GB wins over the 90 GB node figure.
    assert mod._available_ram_bytes() == 14 * 1024 ** 3

    # No cgroup limit (local machine) -> just the machine free memory.
    monkeypatch.setattr(mod, "_cgroup_mem_limit_bytes", lambda: (None, None))
    assert mod._available_ram_bytes() == 90 * 1024 ** 3

    # Threshold scales down on a tight job (0.25 * 14 GB = 3.5 GB < 4 GB cap).
    monkeypatch.setattr(mod, "_cgroup_mem_limit_bytes", lambda: (16 * 1024 ** 3, 2 * 1024 ** 3))
    assert mod._blockwise_read_threshold_bytes() == int(0.25 * 14 * 1024 ** 3)


def test_block_read_axis_padding():
    # Box [10, 40) inside a length-100 volume, blocks of 10, halo 1.
    # Interior block [20,30): no low pad (neighbor owns the straddling cube),
    # +1 high overlap into the next block, no background seal.
    assert m._block_read_axis(20, 30, 10, 40, 100, 1) == (20, 31, 0, 0)
    # Low-edge block [10,20): reads one REAL voxel below the box (volume has room),
    # so no background seal; +1 high overlap.
    assert m._block_read_axis(10, 20, 10, 40, 100, 1) == (9, 21, 0, 0)
    # High-edge block [30,40): +halo real voxel above the box, no seal.
    assert m._block_read_axis(30, 40, 10, 40, 100, 1) == (30, 41, 0, 0)
    # Box flush with the volume edge -> background seal there (no real data beyond).
    assert m._block_read_axis(0, 10, 0, 40, 100, 1) == (0, 11, 1, 0)
    # High-edge block flush with the volume edge: low side is interior (neighbor owns
    # it, no pad), high side hits the volume edge so it's background-sealed.
    assert m._block_read_axis(90, 100, 60, 100, 100, 1) == (90, 100, 0, 1)


def test_blockwise_multi_segment_parallel():
    """Two segments meshed across blocks with several workers: each comes back as its
    own welded, watertight shell (no cross-segment bleed, no concurrency corruption)."""
    n = 48
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    arr = np.zeros((n, n, n), dtype=np.uint32)
    arr[(zz - 14) ** 2 + (yy - 14) ** 2 + (xx - 14) ** 2 <= 9 ** 2] = 3
    arr[(zz - 34) ** 2 + (yy - 34) ** 2 + (xx - 34) ** 2 <= 9 ** 2] = 5
    vol = _FakeVol(arr)

    def colorize(s):
        return (1.0, 0.0, 0.0) if s == 3 else (0.0, 0.0, 1.0)

    raw_parts, _ = m._blockwise_raw_parts(
        vol, _make_plan(arr), [3, 5], colorize=colorize,
        block_voxels=16 ** 3, max_workers=4,
    )
    assert len(raw_parts) == 2
    for part in raw_parts:
        assert part.is_watertight
        assert part.body_count == 1
        # one uniform color per segment survived the parallel block weld
        assert len(np.unique(np.asarray(part.visual.vertex_colors), axis=0)) == 1


def test_decimate_fraction_reduces_faces():
    mesh = trimesh.creation.icosphere(subdivisions=4)  # ~20k faces
    mesh.visual.vertex_colors = np.tile(
        np.array([200, 50, 50, 255], np.uint8), (len(mesh.vertices), 1)
    )
    before = len(mesh.faces)
    out = m._postprocess_label_mesh(
        mesh, target_faces=max(256, int(before * 0.25)), min_decimate_faces=256
    )
    assert len(out.faces) < before
    # ~quarter of the faces, within pyfqmr's tolerance.
    assert len(out.faces) <= before * 0.5
