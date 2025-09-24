import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.physics.microphysics import microphysics_tendencies, MicrophysicsConfig
from simulator.physics.thermo import ThermoConfig, qsat
from simulator.grid.metrics import cell_areas


def test_total_water_conserved_without_fallout():
    g = make_latlon_grid(40, 20, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:] = 285.0
    p = 1.0e5
    st.qv[:] = qsat(st.T, np.full_like(st.M, p)) * 0.9
    st.qc[:] = 2.0e-3
    st.qr[:] = 0.0

    # Pas de conversion vers qr → aucune source de précip
    cfg = MicrophysicsConfig(
        thermo=ThermoConfig(p_ref=p, use_p_from_M=False),
        autoconv_rate=0.0, accretion_rate=0.0, tau_fall=1e12
    )
    tend, _ = microphysics_tendencies(st, cfg)

    A = cell_areas(g)
    dW = float(np.sum(A * st.M * (tend.dqv + tend.dqc + tend.dqr)))
    assert abs(dW) < 1e-12