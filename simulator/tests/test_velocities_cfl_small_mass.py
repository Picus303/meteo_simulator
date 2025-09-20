import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import dx_dy_on_centers
from simulator.core.state import State
from simulator.diagnostics.invariants import velocities_at_centers
from simulator.diagnostics.cfl import cfl_max


def test_velocities_and_cfl_finite_with_small_M():
    g = make_latlon_grid(40, 20, 6.371e6, cap_deg=85.0)
    dx, dy = dx_dy_on_centers(g)

    st = State.zeros(g.ny, g.nx)
    # Make a thin-mass band near the equator
    st.M[g.ny//2 - 1 : g.ny//2 + 1, :] = 1e-14

    u_c, v_c = velocities_at_centers(st.MU, st.MV, st.M)
    assert np.isfinite(u_c).all() and np.isfinite(v_c).all()

    cfl = cfl_max(st.M, u_c, v_c, dx, dy, dt=60.0, include_gravity=False)
    assert np.isfinite(cfl)