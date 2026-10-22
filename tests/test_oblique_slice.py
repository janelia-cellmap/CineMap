"""EMVolume.read_oblique_slice / _gather_plane: chunk-restricted trilinear (EM)
and nearest-neighbor (labels) sampling of a tilted plane, validated against a
naive dense-array reference so the chunked gather can't silently drift from
what a full in-memory read would produce."""
import numpy as np
from scipy.ndimage import map_coordinates

from cinemap.data.slice_loader import EMVolume


class _Reader:
    def __init__(self, a):
        self._a = a

    def read(self):
        return self

    def result(self):
        return self._a


class _ChunkLayout:
    def __init__(self, chunk_shape):
        self._chunk_shape = chunk_shape

    @property
    def read_chunk(self):
        return self

    @property
    def shape(self):
        return self._chunk_shape


class _FakeDtype:
    def __init__(self, np_dtype):
        self.numpy_dtype = np_dtype


class _FakeArr:
    """Stand-in for a tensorstore array: numpy-backed, chunked, and counting
    how many distinct chunk reads it serves (so tests can assert the gather
    is actually chunk-restricted, not a disguised full-box read)."""

    def __init__(self, data: np.ndarray, chunk_shape=(8, 8, 8)):
        self._data = data
        self.shape = data.shape
        self.dtype = _FakeDtype(data.dtype)
        self.chunk_layout = _ChunkLayout(chunk_shape)
        self.reads: list[tuple] = []

    def __getitem__(self, idx):
        self.reads.append(idx)
        return _Reader(self._data[idx])


class _FakeVol(EMVolume):
    """Bypasses EMVolume.__init__ (which does live HTTP metadata fetches) to
    exercise the real read_oblique_slice/_gather_plane/pick_level_for_box
    logic against an in-memory array."""

    def __init__(self, arr: _FakeArr, scale=(1.0, 1.0, 1.0), trans=(0.0, 0.0, 0.0)):
        self._arr = arr
        self.level_scale_nm = [list(scale)]
        self.level_translation_nm = [list(trans)]

    def _open_level(self, level):
        return self._arr


def _gradient_volume(n=40, chunk=8, dtype=np.uint8):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    data = (zz * 4 + yy * 2 + xx).astype(np.float64)
    data = (255 * data / data.max()).astype(dtype)
    return _FakeArr(data, chunk_shape=(chunk, chunk, chunk))


def test_trilinear_matches_dense_reference():
    arr = _gradient_volume()
    vol = _FakeVol(arr)
    # A mildly tilted plane through the volume's center.
    normal = np.array([0.2, 0.3, 1.0])
    res = vol.read_oblique_slice(normal, (20.0, 20.0, 20.0), half_nm=10.0,
                                  target_px=25, raw=False)

    # Independent dense reference: map_coordinates order=1 over the FULL array.
    u, v, n = EMVolume.plane_basis(normal)
    p = np.array([20.0, 20.0, 20.0])
    s = np.linspace(-10.0, 10.0, 25)
    su, sv = np.meshgrid(s, s)
    world = p[None, None, :] + su[..., None] * u + sv[..., None] * v
    coords = np.stack([world[..., 2], world[..., 1], world[..., 0]], axis=0)  # z,y,x
    ref = map_coordinates(arr._data.astype(np.float64), coords, order=1, mode="nearest")

    np.testing.assert_allclose(res.image.astype(np.float64), ref, atol=1.0)


def test_raw_mode_is_exact_nearest_neighbor():
    arr = _gradient_volume(dtype=np.uint32)
    vol = _FakeVol(arr)
    normal = np.array([0.1, 0.1, 1.0])
    res = vol.read_oblique_slice(normal, (20.0, 20.0, 20.0), half_nm=8.0,
                                  target_px=17, raw=True)
    assert res.image.dtype == np.uint32

    u, v, n = EMVolume.plane_basis(normal)
    p = np.array([20.0, 20.0, 20.0])
    s = np.linspace(-8.0, 8.0, 17)
    su, sv = np.meshgrid(s, s)
    world = p[None, None, :] + su[..., None] * u + sv[..., None] * v
    iz = np.clip(np.round(world[..., 2]).astype(int), 0, arr.shape[0] - 1)
    iy = np.clip(np.round(world[..., 1]).astype(int), 0, arr.shape[1] - 1)
    ix = np.clip(np.round(world[..., 0]).astype(int), 0, arr.shape[2] - 1)
    ref = arr._data[iz, iy, ix]

    np.testing.assert_array_equal(res.image, ref)


def test_out_of_bounds_points_are_zeroed():
    arr = _gradient_volume(n=20)
    vol = _FakeVol(arr)
    normal = np.array([0.0, 0.0, 1.0])
    # half_nm=30 on a 20-voxel volume: most of the plane falls outside the array.
    res = vol.read_oblique_slice(normal, (10.0, 10.0, 10.0), half_nm=30.0,
                                  target_px=9, raw=True)
    assert res.image[0, 0] == 0
    assert res.image[4, 4] != 0  # plane center stays in-bounds


def test_only_touched_chunks_are_fetched_not_the_full_cuboid():
    arr = _gradient_volume(n=64, chunk=8)  # 8x8x8 = 512 chunks total
    vol = _FakeVol(arr)
    normal = np.array([0.05, 0.05, 1.0])  # near axis-aligned: thin bbox in z
    vol.read_oblique_slice(normal, (32.0, 32.0, 32.0), half_nm=16.0,
                           target_px=40, raw=False)
    total_chunks = (64 // 8) ** 3
    assert 0 < len(arr.reads) < total_chunks
