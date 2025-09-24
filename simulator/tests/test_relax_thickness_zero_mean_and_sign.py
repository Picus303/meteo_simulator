import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.physics.orbit import KeplerOrbit, SpinConfig, subsolar_lonlat
from simulator.physics.radiation import radiation_tendencies, RadiationConfig, RadiationInputs
from simulator.operators.pressure import pressure_tendencies_energy_safe, PressureConfig
from simulator.core.state import State
from simulator.grid.metrics import cell_areas


def test_thickness_relax_zero_mean_and_pressure_sign():
    g = make_latlon_grid(64, 32, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:] = 260.0  # LW term ~0 around Tref

    albedo = np.zeros((g.ny, g.nx), dtype=float)

    orb = KeplerOrbit(a=1.0, e=0.1, i_orb=0.0, Omega=0.0, omega=0.0, M0=0.0, year_length=365*86400)
    spin = SpinConfig(day_length=86400.0, theta0=0.0)
    rc = RadiationConfig(use_clouds=False, couple_to_M=True, tau_relax_hours=6.0, kappa_M=0.5)

    t = 0.0
    tend = radiation_tendencies(st, g, t, orb, spin, rc, RadiationInputs(albedo=albedo))

    # Zero-area-mean of dM
    A = cell_areas(g)
    total = float(np.sum(A * tend.dM))
    assert abs(total) < 1e-12 * (A.mean() * st.M.mean() * g.nx * g.ny)

    # Apply a small explicit step on M, then compute pressure tendency
    dt = 600.0
    st2 = st.copy()
    st2.M = st.M + dt * tend.dM

    pk = pressure_tendencies_energy_safe(st2, g, PressureConfig())

    # Find subsolar longitudes and select U-faces closest to ±45° from subsolar
    _, lam_s = subsolar_lonlat(t, orb, spin)
    # nearest U-face longitudes in [-pi,pi)
    lon_u = g.lon_u  # (nx+1,)
    def nearest_face(lam):
        d = np.angle(np.exp(1j*(lon_u - lam)))  # wrapped diff
        return int(np.argmin(np.abs(d)))

    i_pos = nearest_face(lam_s + 0.25*np.pi)
    i_neg = nearest_face(lam_s - 0.25*np.pi)

    j = g.ny // 2  # near-equator row

    # Expect opposite signs: pressure gradient points from thick (subsolar) to thin → eastward accel at +45°
    assert pk.dMU[j, i_pos] > 0.0
    assert pk.dMU[j, i_neg] < 0.0