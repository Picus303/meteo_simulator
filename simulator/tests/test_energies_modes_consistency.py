import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas
from simulator.core.state import State
from simulator.diagnostics.invariants import velocities_at_centers, energies_total


def test_energy_modes_consistency_none_vs_shallow():
    g = make_latlon_grid(16, 8, 6.371e6, cap_deg=85.0)
    A = cell_areas(g)
    st = State.zeros(g.ny, g.nx)

    # Give a simple uniform velocity via MU on faces
    st.MU[:] = 1.0  # arbitrary momentum
    u_c, v_c = velocities_at_centers(st.MU, st.MV, st.M)

    e_none = energies_total(st.M, u_c, v_c, A, g=9.81, potential_mode="none")
    e_sh = energies_total(st.M, u_c, v_c, A, g=9.81, rho_ref=1.2, potential_mode="shallow")

    # Kinetic part must be identical across modes
    assert np.isclose(e_none["Ekin"], e_sh["Ekin"]) and e_sh["Epot"] >= 0.0