import numpy as np
import pytest

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.staggering import to_u_centered, to_v_centered, to_c_from_u, to_c_from_v


@pytest.mark.parametrize("nx,ny", [(8,7), (17,14), (64,32)])
def test_staggering_roundtrip_constant(nx, ny):
    g = make_latlon_grid(nx, ny, 6.371e6, cap_deg=85.0)
    X = np.full((g.ny, g.nx), 2.718281828)

    U = to_u_centered(X)
    XcU = to_c_from_u(U)
    assert U.shape == (ny, nx + 1)
    assert XcU.shape == (ny, nx)
    assert np.allclose(XcU, X)

    V = to_v_centered(X)
    XcV = to_c_from_v(V)
    assert V.shape == (ny + 1, nx)
    assert XcV.shape == (ny, nx)
    assert np.allclose(XcV, X)