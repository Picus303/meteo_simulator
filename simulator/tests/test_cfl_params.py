import numpy as np
import pytest

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import dx_dy_on_centers
from simulator.core.state import State
from simulator.diagnostics.invariants import velocities_at_centers
from simulator.diagnostics.cfl import cfl_max


def test_cfl_gravity_requires_params():
    g = make_latlon_grid(8, 4, 6.371e6, cap_deg=85.0)
    dx, dy = dx_dy_on_centers(g)
    st = State.zeros(g.ny, g.nx)
    u_c, v_c = velocities_at_centers(st.MU, st.MV, st.M)

    with pytest.raises(ValueError):
        _ = cfl_max(st.M, u_c, v_c, dx, dy, dt=60.0, include_gravity=True)

    # With parameters provided, no error
    cfl = cfl_max(st.M, u_c, v_c, dx, dy, dt=60.0, include_gravity=True, g=9.81, rho_ref=1.2)
    assert cfl >= 0.0