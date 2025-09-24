import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.physics.surface import surface_evaporation_tendencies, SurfaceInputs, SurfaceConfig
from simulator.grid.staggering import to_u_centered
from simulator.physics.thermo import qsat


def test_surface_evaporation_calm_no_flux():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:] = 290.0
    p = np.full_like(st.M, 1.0e5)

    # Air très sec, mais calme → pas d'évap sans vent
    st.qv[:] = qsat(st.T, p) * 0.2

    ocean = np.ones_like(st.M)
    sinp = SurfaceInputs(ocean_mask=ocean, qsat_func=qsat, p_field=p)
    cfg = SurfaceConfig(CE=2e-3, evap_heats_air=False)

    tend = surface_evaporation_tendencies(st, sinp, cfg)
    assert np.allclose(tend.dqv, 0.0)
    assert np.allclose(tend.dT, 0.0)


def test_surface_evaporation_positive_with_wind():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:] = 290.0
    p = np.full_like(st.M, 1.0e5)

    # Air sec + vent de fond uniforme → évap > 0
    st.qv[:] = qsat(st.T, p) * 0.2

    # Vent zonal uniforme: MU = M_u * u0 → U > 0 aux centres
    u0 = 5.0
    M_u = to_u_centered(st.M)
    st.MU[:] = M_u * u0

    ocean = np.ones_like(st.M)
    sinp = SurfaceInputs(ocean_mask=ocean, qsat_func=qsat, p_field=p)
    cfg = SurfaceConfig(CE=2e-3, evap_heats_air=False)

    tend = surface_evaporation_tendencies(st, sinp, cfg)
    assert np.all(tend.dqv > 0.0)
    # Convention actuelle: l'évap refroidit l'air (énergie prise au sol)
    assert (tend.dT < 0.0).all() or np.all(tend.dT <= 1e-12)
