import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.core.steppers import rk3_ssp
from simulator.core.pipeline import RHSComposer
from simulator.operators.pressure import make_pressure_term
from simulator.diagnostics.energy_exchange import kinetic_energy, potential_energy


def test_pressure_only_energy_exchange_conserves_sum():
    g = make_latlon_grid(48, 24, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)

    # Small bump in mass to excite gravity waves; no initial momentum
    st.M += 1e-3 * np.cos(2 * np.pi * np.arange(g.nx)[None, :] / g.nx)

    term_p = make_pressure_term(g, g=9.81, rho_ref=1.2)
    rhs = RHSComposer([term_p])

    dt = 60.0
    n = 40

    E0 = kinetic_energy(st.MU, st.MV, st.M, g) + potential_energy(st.M, g, 9.81, 1.2)

    s = st.copy(); t = 0.0
    for _ in range(n):
        s = rk3_ssp(s, t, dt, rhs, post=None, enforce_at="final")
        t += dt

    E1 = kinetic_energy(s.MU, s.MV, s.M, g) + potential_energy(s.M, g, 9.81, 1.2)

    # Energy should be nearly conserved with the energy-safe pressure term
    # Allow a small tolerance for time-integration error
    rel = abs(E1 - E0) / max(1e-12, abs(E0))
    assert rel < 5e-3