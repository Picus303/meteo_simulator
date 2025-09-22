import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.operators.filters import make_shapiro_hook
from simulator.operators.massfix import make_water_rescale_hook
from simulator.grid.metrics import cell_areas


def test_shapiro_preserves_mean_T_qv():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    rng = np.random.default_rng(1)
    st.T = rng.standard_normal(st.T.shape)
    st.qv = rng.random(st.qv.shape)

    Tm0 = float(np.mean(st.T))
    qvm0 = float(np.mean(st.qv))

    hook = make_shapiro_hook(fields=("T", "qv"), passes_i=2, passes_j=2)
    s = hook(st)

    assert abs(np.mean(s.T) - Tm0) < 1e-14
    assert abs(np.mean(s.qv) - qvm0) < 1e-14


def test_massfix_rescales_total_water():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    rng = np.random.default_rng(2)
    st.qv = rng.random(st.qv.shape)

    A = cell_areas(g)
    target = float(np.sum(A * st.M * st.qv))

    # perturbe l'eau totale
    st.qv *= 1.01

    hook = make_water_rescale_hook(g, target=target, max_rel=1e-2)
    s = hook(st)

    total = float(np.sum(A * s.M * (s.qv + s.qc + s.qr)))
    rel = abs(total - target) / max(1e-12, abs(target))
    assert rel < 1e-3 + 1e-12