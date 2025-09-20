import numpy as np

from simulator.core.state import State
from simulator.core.tendencies import Tend
from simulator.core.steppers import rk3_ssp
from simulator.core.pipeline import make_positivity_clip


def rhs_make_negative_qv():
    def _f(s: State, t: float) -> Tend:
        return Tend(
            dM=np.zeros_like(s.M),
            dT=np.zeros_like(s.T),
            dqv=-10.0 * np.ones_like(s.qv),  # will push qv < 0 in one step
            dqc=np.zeros_like(s.qc),
            dqr=np.zeros_like(s.qr),
            dMU=np.zeros_like(s.MU),
            dMV=np.zeros_like(s.MV),
        )
    return _f


def test_post_clip_applied_final():
    s = State.zeros(4, 4)
    s.qv[:] = 0.1
    s1 = rk3_ssp(s, 0.0, 1.0, rhs_make_negative_qv(), post=[make_positivity_clip()], enforce_at="final")
    assert (s1.qv >= 0.0).all()


def test_post_clip_applied_stage():
    s = State.zeros(4, 4)
    s.qv[:] = 0.0
    s1 = rk3_ssp(s, 0.0, 1.0, rhs_make_negative_qv(), post=[make_positivity_clip()], enforce_at="stage")
    print(s1.qv)
    assert (s1.qv >= 0.0).all()