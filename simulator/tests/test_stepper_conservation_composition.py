import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas
from simulator.core.state import State
from simulator.core.tendencies import Tend
from simulator.core.steppers import rk3_ssp
from simulator.core.pipeline import RHSComposer


def rhs_zero_mass_change():
    def _f(s: State, t: float) -> Tend:
        # No change to M; tiny linear change to T
        return Tend(
            dM=np.zeros_like(s.M),
            dT=0.1 * np.ones_like(s.T),
            dqv=np.zeros_like(s.M),
            dqc=np.zeros_like(s.M),
            dqr=np.zeros_like(s.M),
            dMU=np.zeros_like(s.MU),
            dMV=np.zeros_like(s.MV),
        )
    return _f


def rhs_linear_on_T(alpha: float):
    def _f(s: State, t: float) -> Tend:
        return Tend(
            dM=np.zeros_like(s.M),
            dT=alpha * s.T,
            dqv=np.zeros_like(s.M),
            dqc=np.zeros_like(s.M),
            dqr=np.zeros_like(s.M),
            dMU=np.zeros_like(s.MU),
            dMV=np.zeros_like(s.MV),
        )
    return _f


def test_mass_conservation_when_dM_zero():
    g = make_latlon_grid(12, 6, 6.371e6, cap_deg=85.0)
    A = cell_areas(g)
    s = State.zeros(g.ny, g.nx)
    s0_mass = float(np.sum(s.M * A))

    s1 = rk3_ssp(s, 0.0, 300.0, rhs_zero_mass_change())
    s1_mass = float(np.sum(s1.M * A))
    assert np.isclose(s1_mass, s0_mass)


def test_composition_equivalence_linear_T():
    s = State.zeros(4, 5, T0=2.0)
    dt = 0.1
    t0 = 0.0

    # Two terms
    f1 = rhs_linear_on_T(0.3)
    f2 = rhs_linear_on_T(-0.1)

    # Compose as one RHS vs. sum of RHS
    comp = RHSComposer([f1, f2])

    s_a = rk3_ssp(s, t0, dt, comp)

    def sum_rhs(s_in: State, t: float) -> Tend:
        k1 = f1(s_in, t); k2 = f2(s_in, t)
        return Tend(
            dM=k1.dM + k2.dM,
            dT=k1.dT + k2.dT,
            dqv=k1.dqv + k2.dqv,
            dqc=k1.dqc + k2.dqc,
            dqr=k1.dqr + k2.dqr,
            dMU=k1.dMU + k2.dMU,
            dMV=k1.dMV + k2.dMV,
        )

    s_b = rk3_ssp(s, t0, dt, sum_rhs)

    assert np.allclose(s_a.T, s_b.T)