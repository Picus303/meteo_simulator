import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.operators.coriolis import make_coriolis_rotation_hook, CoriolisRotateConfig
from simulator.core.steppers import rk3_ssp
from simulator.core.tendencies import Tend


def _zero_rhs(state, t):
    return Tend(
        dM=np.zeros_like(state.M), dT=np.zeros_like(state.M),
        dqv=np.zeros_like(state.M), dqc=np.zeros_like(state.M), dqr=np.zeros_like(state.M),
        dMU=np.zeros_like(state.MU), dMV=np.zeros_like(state.MV)
    )


def test_coriolis_rotation_respects_caps_freeze_unified():
    g = make_latlon_grid(50, 25, 6.371e6, cap_deg=82.0)
    st = State.zeros(g.ny, g.nx)

    st.MU[:] = 1.0
    st.MV[:] = 0.0

    hook = make_coriolis_rotation_hook(g, CoriolisRotateConfig(cap_zero_u=True, scheme="centers"))

    s = rk3_ssp(st, 0.0, 600.0, _zero_rhs, post=[hook], enforce_at="final")

    rows = np.where(g.cap_rows)[0]
    for j in rows:
        assert np.allclose(s.MU[j, :], st.MU[j, :])