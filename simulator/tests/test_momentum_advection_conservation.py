import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas
from simulator.grid.staggering import to_u_centered
from simulator.core.state import State
from simulator.advection.momentum import advect_momentum


def _area_u(grid) -> np.ndarray:
    A = cell_areas(grid)
    return 0.5 * (np.concatenate([A[:, :1], A], axis=1) + np.concatenate([A, A[:, -1:]], axis=1))


def test_domain_integral_MU_constant_in_zonal_1d():
    # 1D zonal ring: v=0, u varies with i only, M uniform.
    g = make_latlon_grid(40, 10, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)

    # Build u varying with longitude only
    M_u = to_u_centered(st.M)
    ii = np.arange(g.nx + 1)
    u = 10.0 + 2.0 * np.cos(2 * np.pi * ii / (g.nx + 1))[None, :]
    st.MU[:] = M_u * u
    st.MV[:] = 0.0

    k = advect_momentum(st, g)

    Au = _area_u(g)
    # Integral of d(Mu)/dt over domain should be ~0
    dI = float(np.sum(k.dMU * Au))
    assert abs(dI) < 1e-7