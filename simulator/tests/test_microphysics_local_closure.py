import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.physics.microphysics import microphysics_tendencies, MicrophysicsConfig
from simulator.physics.thermo import ThermoConfig, qsat
from simulator.grid.metrics import cell_areas


def test_condensation_latent_heating_local():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    # Supersaturate uniformly
    st.T[:] = 280.0
    p = np.full_like(st.M, 9.0e4)
    qs = qsat(st.T, p)
    st.qv[:] = qs + 2e-3
    st.qc[:] = 1e-4

    cfg = MicrophysicsConfig(thermo=ThermoConfig(p_ref=9e4, use_p_from_M=False), tau_cond=300.0, tau_reevap=600.0,
                             qcrit=1e-3, autoconv_rate=0.0, accretion_rate=0.0, tau_fall=1e9)

    tend, _ = microphysics_tendencies(st, cfg)

    # Heating should be positive where condensation occurs
    assert np.all(tend.dT >= -1e-12)
    # Water phase change conserves total water in absence of autoconv/precip
    A = cell_areas(g)
    dW = float(np.sum(A * st.M * (tend.dqv + tend.dqc + tend.dqr)))
    assert abs(dW) < 1e-12