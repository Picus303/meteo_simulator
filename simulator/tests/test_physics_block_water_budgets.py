import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.physics.block import physics_step_strang, PhysicsConfig, PhysicsInputs
from simulator.physics.orbit import KeplerOrbit, SpinConfig
from simulator.physics.radiation import RadiationConfig
from simulator.physics.surface import SurfaceConfig
from simulator.physics.microphysics import MicrophysicsConfig
from simulator.grid.metrics import cell_areas
from simulator.physics.thermo import qsat


def _default_inputs(st: State, g):
    albedo = np.zeros_like(st.M)
    ocean = np.ones_like(st.M)
    p = np.full_like(st.M, 1.0e5)
    return PhysicsInputs(albedo=albedo, ocean_mask=ocean, qsat_func=qsat, p_surf=p)


def test_column_no_precip_conserves_atmospheric_water():
    g = make_latlon_grid(48, 24, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:]  = 285.0
    p = np.full_like(st.M, 1.0e5)
    st.qv[:] = qsat(st.T, p) * 0.95
    st.qc[:] = 2.0e-3
    st.qr[:] = 0.0

    # Physics: no autoconv/accretion, no fallout → no precip sink
    pcfg = PhysicsConfig(
        radiation=RadiationConfig(use_clouds=False),
        surface=SurfaceConfig(CE=0.0),  # disable evaporation so only micro + radiation act
        micro=MicrophysicsConfig(autoconv_rate=0.0, accretion_rate=0.0, tau_fall=1e12),
    )

    orb = KeplerOrbit(e=0.0); spin = SpinConfig()
    pin = _default_inputs(st, g)

    dt = 600.0
    s1, diag = physics_step_strang(st, g, t_sec=0.0, dt=dt, orb_spin=(orb, spin), cfg=pcfg, pin=pin)

    A = cell_areas(g)
    # With no precipitation sources/sinks, total water should be conserved
    assert abs(diag.dW_atmos) < 1e-12 * np.sum(A) * st.M.mean()


def test_column_with_precip_budget_matches_fallout():
    g = make_latlon_grid(48, 24, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:]  = 285.0
    p = np.full_like(st.M, 1.0e5)
    st.qv[:] = qsat(st.T, p) * 0.95
    st.qc[:] = 2.0e-3
    st.qr[:] = 1.0e-3   # existing rain → fallout sink

    pcfg = PhysicsConfig(
        radiation=RadiationConfig(use_clouds=False),
        surface=SurfaceConfig(CE=0.0),
        micro=MicrophysicsConfig(autoconv_rate=0.0, accretion_rate=0.0, tau_fall=1800.0),
    )

    orb = KeplerOrbit(e=0.0); spin = SpinConfig()
    pin = _default_inputs(st, g)

    dt = 600.0
    s1, diag = physics_step_strang(st, g, t_sec=0.0, dt=dt, orb_spin=(orb, spin), cfg=pcfg, pin=pin)

    A = cell_areas(g)
    # Atmospheric water loss should equal accumulated precipitation
    assert np.isclose(-diag.dW_atmos, diag.global_precip_kg(A), rtol=1e-12, atol=1e-8)