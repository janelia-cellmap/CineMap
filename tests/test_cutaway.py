import numpy as np

from cinemap.data.cutaway import _clip_arrays


def test_clip_arrays_returns_empty_mesh_when_plane_removes_everything():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    out = _clip_arrays(
        verts,
        faces,
        None,
        normal_xyz=np.array([0.0, 0.0, 1.0]),
        position_nm=-1.0,
        side=1.0,
    )

    assert out["v"].shape == (0, 3)
    assert out["f"].shape == (0, 3)
