import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas, dx_dy_on_centers
from simulator.core.state import State
from simulator.diagnostics.invariants import velocities_at_centers, mass_total, energies_total
from simulator.diagnostics.cfl import cfl_max


def test_constants_and_energies():
    R = 6.371e6
    g = make_latlon_grid(32, 16, R, cap_deg=85.0)
    A = cell_areas(g)
    dx, dy = dx_dy_on_centers(g)

    st = State.zeros(g.ny, g.nx, T0=280.0)
    u_c, v_c = velocities_at_centers(st.MU, st.MV, st.M)

    # Mass total should equal sum of areas (since M=1)
    Mtot = mass_total(st.M, A)
    sphere_area = 4.0 * np.pi * R * R
    assert np.isclose(Mtot, sphere_area)

    # Kinetic energy is zero
    e = energies_total(st.M, u_c, v_c, A, g=9.81, rho_ref=1.2, potential_mode="none")
    assert np.isclose(e["Ekin"], 0.0)

    # Shallow potential energy > 0 and matches analytic value for M=1
    e2 = energies_total(st.M, u_c, v_c, A, g=9.81, rho_ref=1.2, potential_mode="shallow")
    expected_epot = 0.5 * 9.81 * np.sum((st.M / 1.2) ** 2 * A)
    assert np.isclose(e2["Epot"], expected_epot)

    # CFL: adv-only is zero, gravity term positive if included
    cfl_adv = cfl_max(st.M, u_c, v_c, dx, dy, dt=60.0, include_gravity=False)
    assert np.isclose(cfl_adv, 0.0)
    cfl_g = cfl_max(st.M, u_c, v_c, dx, dy, dt=60.0, g=9.81, rho_ref=1.2, include_gravity=True)
    assert cfl_g > 0.0