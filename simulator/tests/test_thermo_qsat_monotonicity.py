import numpy as np
from simulator.physics.thermo import qsat


def test_qsat_monotone_T_and_p():
    T = np.linspace(250.0, 310.0, 61)
    p1 = 8.0e4 * np.ones_like(T)
    p2 = 9.5e4 * np.ones_like(T)

    q1 = qsat(T, p1)
    q2 = qsat(T, p2)

    # qsat increases with T
    assert np.all(np.diff(q1) >= -1e-12)
    # qsat decreases with p
    assert np.all(q2 <= q1 + 1e-12)