import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.core.steppers import rk3_ssp
from simulator.operators.coriolis import make_coriolis_rotation_hook, CoriolisRotateConfig
from simulator.diagnostics.energy_exchange import kinetic_energy_faces, kinetic_energy
from simulator.grid.staggering import to_u_centered
from simulator.core.tendencies import Tend



def _zero_rhs(state, t):
    return Tend(
        dM=np.zeros_like(state.M), dT=np.zeros_like(state.M),
        dqv=np.zeros_like(state.M), dqc=np.zeros_like(state.M), dqr=np.zeros_like(state.M),
        dMU=np.zeros_like(state.MU), dMV=np.zeros_like(state.MV)
    )


def test_coriolis_preserves_center_energy():
    # Use nx odd to enable exact periodic inversion in longitude
    g = make_latlon_grid(75+1, 40, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)

    u0 = 12.0
    M_u = to_u_centered(st.M)
    st.MU[:] = M_u * u0
    st.MV[:] = 0.0

    hook = make_coriolis_rotation_hook(g, CoriolisRotateConfig(cap_zero_u=False, scheme="centers"))

    E0c = kinetic_energy(st.MU, st.MV, st.M, g)

    dt = 300.0
    s = st.copy(); t = 0.0
    for _ in range(40):
        s = rk3_ssp(s, t, dt, _zero_rhs, post=[hook], enforce_at="final")
        t += dt

    E1c = kinetic_energy(s.MU, s.MV, s.M, g)
    rel = abs(E1c - E0c) / max(1e-20, abs(E0c))
    assert rel < 1e-12