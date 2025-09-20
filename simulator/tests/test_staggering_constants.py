import numpy as np

from simulator.grid import make_latlon_grid
from simulator.grid.staggering import to_u_centered, to_v_centered, to_c_from_u, to_c_from_v

def test_constant_field_through_staggers():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    X = np.full((g.ny, g.nx), 3.14)

    U = to_u_centered(X)
    Xc_from_U = to_c_from_u(U)
    assert np.allclose(Xc_from_U, X)

    V = to_v_centered(X)
    Xc_from_V = to_c_from_v(V)
    # edges at first/last row are clamped; core should match exactly
    assert np.allclose(Xc_from_V[1:-1, :], X[1:-1, :])