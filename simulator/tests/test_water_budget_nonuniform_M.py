import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas
from simulator.core.state import State
from simulator.diagnostics.invariants import water_total
from simulator.diagnostics.water_budget import WaterBudgetTracker


def test_open_budget_matches_state_update_nonuniform_M():
    g = make_latlon_grid(24, 12, 6.371e6, cap_deg=85.0)
    A = cell_areas(g)
    dt = 120.0

    st = State.zeros(g.ny, g.nx)
    # Nonuniform mass (meridional gradient)
    st.M = (1.0 + 0.1 * (np.sin(g.latc2d)))

    W0 = water_total(st.M, st.qv, st.qc, st.qr, A)
    tracker = WaterBudgetTracker(area=A, dt=dt, W0=W0)

    E = 2e-6 * np.ones_like(st.M)
    P = 1e-6 * np.ones_like(st.M)

    Bw = tracker.step(E=E, P=P)

    dq = ((E - P) * dt) / st.M
    st2 = st.copy(); st2.qv = st2.qv + dq

    W1 = water_total(st2.M, st2.qv, st2.qc, st2.qr, A)
    assert np.isclose(W1, tracker.predicted())