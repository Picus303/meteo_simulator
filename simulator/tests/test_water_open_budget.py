import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas
from simulator.core.state import State
from simulator.diagnostics.invariants import water_total
from simulator.diagnostics.water_budget import WaterBudgetTracker


def test_open_budget_matches_state_update():
    R = 6.371e6
    g = make_latlon_grid(16, 8, R, cap_deg=85.0)
    A = cell_areas(g)
    dt = 300.0

    st = State.zeros(g.ny, g.nx)
    W0 = water_total(st.M, st.qv, st.qc, st.qr, A)

    tracker = WaterBudgetTracker(area=A, dt=dt, W0=W0)

    # Build synthetic sources: E=const, P=half of E, no eps, no Er → net = +0.5 E
    E = 1e-6 * np.ones_like(st.M)   # kg m^-2 s^-1
    P = 0.5e-6 * np.ones_like(st.M)
    eps = np.zeros_like(st.M)
    Er = np.zeros_like(st.M)

    # Advance budget
    Bw = tracker.step(E=E, P=P, eps=eps, Er=Er)

    # Construct a new state updated exactly by the same net source (mixing ratio change)
    net = (E - P - eps + Er)  # kg m^-2 s^-1
    dq = (net * dt) / st.M    # since M=1 everywhere here
    st2 = st.copy()
    st2.qv = st2.qv + dq  # put all into qv for this synthetic test

    W1 = water_total(st2.M, st2.qv, st2.qc, st2.qr, A)
    assert np.isclose(W1, tracker.predicted())
