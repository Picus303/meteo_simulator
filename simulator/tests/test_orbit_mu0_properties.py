import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.physics.orbit import KeplerOrbit, SpinConfig, mu0_field, subsolar_lonlat
from simulator.grid.metrics import cell_areas


def test_mu0_bounds_and_mean_quarter():
    g = make_latlon_grid(96, 48, 6.371e6, cap_deg=85.0)
    orb = KeplerOrbit(a=1.0, e=0.3, i_orb=np.radians(15.0), Omega=0.2, omega=1.0, M0=0.3, year_length=365*86400)
    spin = SpinConfig(day_length=86400.0, theta0=0.0, ra_pole=0.1, dec_pole=np.radians(23.0))

    t = 12345.0
    mu0 = mu0_field(g, t, orb, spin)

    assert np.all(mu0 >= -1e-14) and np.all(mu0 <= 1.0 + 1e-14)

    A = cell_areas(g)
    mu_avg = float(np.sum(A * mu0) / np.sum(A))
    assert abs(mu_avg - 0.25) < 5e-3


def test_mu0_diurnal_roll_consistency():
    g = make_latlon_grid(128, 64, 6.371e6, cap_deg=85.0)
    orb = KeplerOrbit(a=1.0, e=0.05, i_orb=0.0, Omega=0.0, omega=0.0, M0=0.0, year_length=365*86400)
    spin = SpinConfig(day_length=86400.0, theta0=0.0, ra_pole=0.0, dec_pole=np.radians(23.0))

    t = 10_000.0
    dt = 3600.0  # 1 hour
    mu0_t = mu0_field(g, t, orb, spin)
    mu0_tp = mu0_field(g, t + dt, orb, spin)

    # Use actual subsolar longitude drift instead of pure Ω_rot·dt (more robust)
    _, lam_t = subsolar_lonlat(t, orb, spin)
    _, lam_tp = subsolar_lonlat(t + dt, orb, spin)
    dlam = lam_tp - lam_t
    # wrap to [-pi, pi)
    dlam = (dlam + np.pi) % (2*np.pi) - np.pi

    dlon = 2*np.pi / g.nx
    shift = int(round(dlam / dlon))
    mu0_roll = np.roll(mu0_t, shift=shift, axis=1)

    # Allow small mismatch due to tiny change in subsolar latitude over dt
    assert np.allclose(mu0_tp, mu0_roll, atol=1e-2)