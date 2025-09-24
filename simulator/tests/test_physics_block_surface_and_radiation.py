import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.core.state import State
from simulator.grid.staggering import to_u_centered
from simulator.physics.block import physics_step_strang, PhysicsConfig, PhysicsInputs
from simulator.physics.orbit import KeplerOrbit, SpinConfig
from simulator.physics.radiation import RadiationConfig
from simulator.physics.surface import SurfaceConfig
from simulator.physics.microphysics import MicrophysicsConfig
from simulator.physics.thermo import qsat


def _inputs_ocean(st: State):
    albedo = np.zeros_like(st.M)
    ocean = np.ones_like(st.M)
    p = np.full_like(st.M, 1.0e5)
    return PhysicsInputs(albedo=albedo, ocean_mask=ocean, qsat_func=qsat, p_surf=p)


def test_surface_evap_positive_with_background_wind():
    g = make_latlon_grid(40, 20, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:] = 290.0
    p = np.full_like(st.M, 1.0e5)
    st.qv[:] = qsat(st.T, p) * 0.3

    # impose uniform zonal wind on U-faces
    
    u0 = 5.0
    M_u = to_u_centered(st.M)
    st.MU[:] = M_u * u0

    pcfg = PhysicsConfig(
        radiation=RadiationConfig(use_clouds=False),
        surface=SurfaceConfig(CE=2e-3, evap_heats_air=False),
        micro=MicrophysicsConfig(autoconv_rate=0.0, accretion_rate=0.0, tau_fall=1e12),
    )

    orb = KeplerOrbit(e=0.0); spin = SpinConfig()
    pin = _inputs_ocean(st)

    s1, _ = physics_step_strang(st, g, t_sec=0.0, dt=600.0, orb_spin=(orb, spin), cfg=pcfg, pin=pin)

    assert (s1.qv > st.qv).any()  # some evaporation occurred
    # cooling or unchanged (depending on wind distribution); at least not positive overall
    assert (s1.T <= st.T + 1e-12).all()


def test_radiation_heating_on_dayside_with_Tref():
    g = make_latlon_grid(48, 24, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    # Set T = Tref so LW ≈ 0 and SW drives sign
    Tref = 260.0
    st.T[:] = Tref

    # Daylight via subsolar at (0,0), circular orbit, spin default
    pcfg = PhysicsConfig(radiation=RadiationConfig(use_clouds=False, Tref_lw=Tref, tau_sw=1.0, S0=1000.0))

    # Zero winds to isolate radiation
    st.MU[:] = 0.0; st.MV[:] = 0.0

    # Flat ocean/low albedo
    pin = PhysicsInputs(albedo=np.zeros_like(st.M), ocean_mask=np.ones_like(st.M), qsat_func=qsat, p_surf=np.full_like(st.M, 1e5))

    orb = KeplerOrbit(e=0.0); spin = SpinConfig()
    s1, _ = physics_step_strang(st, g, t_sec=0.0, dt=600.0, orb_spin=(orb, spin), cfg=pcfg, pin=pin)

    # Expect warming somewhere (dayside has μ0>0)
    assert (s1.T > st.T).any()