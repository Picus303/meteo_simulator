import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.advection.fluxform import mass_fluxes


def test_cap_zeroes_u_flux():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=82.0)
    st = State.zeros(g.ny, g.nx)
    st.MU[:] = 1.0
    st.MV[:] = 0.0

    Fu, Fv = mass_fluxes(st, g)
    # All U-flux on cap rows must be zero
    rows = np.where(g.cap_rows)[0]
    for j in rows:
        assert np.allclose(Fu[j, :], 0.0)