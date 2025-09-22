import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.core.steppers import rk3_ssp
from simulator.core.tendencies import Tend
from simulator.operators.diffusion import diffusion_tendencies, DiffusionConfig
from simulator.grid.metrics import cell_areas


def _rhs_diff_q(state, grid, K):
    def f(st: State, t: float) -> Tend:
        return diffusion_tendencies(st, grid, DiffusionConfig(K_T=0.0, K_q=K))
    return f


def test_scalar_diffusion_conserves_total_q():
    g = make_latlon_grid(73, 37, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)

    rng = np.random.default_rng(0)
    st.qv = rng.random(st.qv.shape)

    A = cell_areas(g)
    Q0 = float(np.sum(A * st.M * st.qv))

    f = _rhs_diff_q(st, g, K=5.0e3)
    s = rk3_ssp(st, 0.0, 300.0, f, post=None, enforce_at="final")

    Q1 = float(np.sum(A * s.M * s.qv))
    assert abs(Q1 - Q0) < 1e-10 * max(1.0, abs(Q0))