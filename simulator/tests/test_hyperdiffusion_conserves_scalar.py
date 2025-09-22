import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.core.steppers import rk3_ssp
from simulator.core.tendencies import Tend
from simulator.operators.diffusion import diffusion_tendencies, DiffusionConfig
from simulator.grid.metrics import cell_areas


def _rhs_hyperdiff_q(state, grid, K4):
    def f(st: State, t: float) -> Tend:
        return diffusion_tendencies(st, grid, DiffusionConfig(K_q=0.0, K4_q=K4))
    return f


def test_scalar_hyperdiffusion_conserves_total_q():
    g = make_latlon_grid(73, 37, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)

    rng = np.random.default_rng(1)
    st.qv = rng.random(st.qv.shape)

    A = cell_areas(g)
    Q0 = float(np.sum(A * st.M * st.qv))

    f = _rhs_hyperdiff_q(st, g, K4=1.0e10)
    s = rk3_ssp(st, 0.0, 300.0, f, post=None, enforce_at="final")

    Q1 = float(np.sum(A * s.M * s.qv))
    assert abs(Q1 - Q0) < 1e-10 * max(1.0, abs(Q0))
