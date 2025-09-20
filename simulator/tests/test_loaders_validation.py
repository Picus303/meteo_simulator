import numpy as np
import pytest

from simulator.grid.loaders import load_surface_static


def test_loaders_shape_and_range(tmp_path):
    ny, nx = 6, 8
    p = tmp_path
    np.save(p / "elevation.npy", np.zeros((ny, nx)))
    np.save(p / "mask_water.npy", np.zeros((ny, nx), dtype=bool))
    np.save(p / "albedo.npy", np.full((ny, nx), 0.5))
    np.save(p / "terrain_type.npy", np.zeros((ny, nx), dtype=np.int32))

    sf = load_surface_static(p, (ny, nx))
    assert sf.elevation.shape == (ny, nx)
    assert sf.mask_water.dtype == bool

    # Out-of-range albedo → ValueError
    np.save(p / "albedo.npy", np.full((ny, nx), 1.5))
    with pytest.raises(ValueError):
        _ = load_surface_static(p, (ny, nx))