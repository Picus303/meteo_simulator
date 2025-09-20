import numpy as np

from simulator.core.state import State
from simulator.core.tendencies import Tend
from simulator.core.steppers import rk3_ssp


class LinearDecayT(Tend):
    pass


def rhs_linear_decay(lmbda: float):
    def _f(s: State, t: float) -> Tend:
        # Only temperature decays: dT/dt = -lambda T; others zero
        return Tend(
            dM=np.zeros_like(s.M),
            dT=-lmbda * s.T,
            dqv=np.zeros_like(s.M),
            dqc=np.zeros_like(s.M),
            dqr=np.zeros_like(s.M),
            dMU=np.zeros_like(s.MU),
            dMV=np.zeros_like(s.MV),
        )
    return _f


def test_rk3_global_order_linear_decay():
    ny, nx = 6, 7
    s0 = State.zeros(ny, nx, T0=1.0)
    lam = 0.7
    T_end = 1.0

    # Integrate to t_end with two step sizes dt and dt/2
    for dt in (T_end/40, T_end/80):
        s = s0.copy(); t=0.0
        f = rhs_linear_decay(lam)
        nsteps = int(np.round(T_end/dt))
        for k in range(nsteps):
            s = rk3_ssp(s, t, dt, f)
            t += dt
        exact = np.exp(-lam*T_end)
        err = float(np.max(np.abs(s.T - exact)))
        if dt == T_end/40:
            err_coarse = err
        else:
            err_fine = err
    # SSPRK3 is 3rd order: error should reduce ~ by 8 when halving dt. Allow margin.
    assert err_fine < err_coarse/6.0