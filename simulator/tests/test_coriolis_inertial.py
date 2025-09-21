import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.core.steppers import rk3_ssp
from simulator.operators.coriolis import make_coriolis_rotation_hook, CoriolisRotateConfig
from simulator.grid.staggering import to_u_centered, to_v_centered
from simulator.core.tendencies import Tend


def _zero_rhs(state, t):
    return Tend(
        dM=np.zeros_like(state.M), dT=np.zeros_like(state.M),
        dqv=np.zeros_like(state.M), dqc=np.zeros_like(state.M), dqr=np.zeros_like(state.M),
        dMU=np.zeros_like(state.MU), dMV=np.zeros_like(state.MV)
    )


def test_inertial_oscillation_rotate():
    g = make_latlon_grid(75, 40, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)

    u0 = 10.0
    M_u = to_u_centered(st.M); M_v = to_v_centered(st.M)
    st.MU[:] = M_u * u0
    st.MV[:] = 0.0

    hook = make_coriolis_rotation_hook(g, CoriolisRotateConfig(cap_zero_u=False, scheme="centers"))

    dt = 300.0
    n = 12

    s = st.copy(); t = 0.0
    for _ in range(n):
        s = rk3_ssp(s, t, dt, _zero_rhs, post=[hook], enforce_at="final")
        t += dt

    # Choose a center row away from equator to avoid antisymmetric cancellation
    signs_V = np.sign(np.sin(g.lat_v))
    valid_js = np.where(signs_V[:-1] * signs_V[1:] > 0)[0]
    j = int(valid_js[len(valid_js)//2])

    omega = 7.292115e-5
    f_row = 0.5 * (2*omega*np.sin(g.lat_v[j]) + 2*omega*np.sin(g.lat_v[j+1]))
    theta = f_row * dt * n

    eps = 1e-12
    u_f = s.MU / np.maximum(to_u_centered(s.M), eps)
    v_f = s.MV / np.maximum(to_v_centered(s.M), eps)
    u_c = 0.5 * (u_f[:, 1:] + u_f[:, :-1])
    v_c = 0.5 * (v_f[1:, :] + v_f[:-1, :])
    um = u_c.mean(axis=1)[j]
    vm = v_c.mean(axis=1)[j]

    u_the = u0 * np.cos(theta)
    v_the = -u0 * np.sin(theta)

    err = np.hypot(um - u_the, vm - v_the) / max(1e-9, abs(u0))
    assert err < 5e-3