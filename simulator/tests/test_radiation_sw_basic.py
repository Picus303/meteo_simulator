import numpy as np
from simulator.grid.sphere import make_latlon_grid
from simulator.physics.orbit import KeplerOrbit, SpinConfig, orbit_radius
from simulator.physics.radiation import radiation_tendencies, RadiationConfig, RadiationInputs
from simulator.core.state import State
from simulator.grid.metrics import cell_areas


def test_qsw_scales_with_inverse_r2_global_mean():
    g = make_latlon_grid(96, 48, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    st.T[:] = 260.0  # so Q_LW ~ 0

    albedo = np.zeros((g.ny, g.nx), dtype=float)

    orb = KeplerOrbit(a=1.0, e=0.3, i_orb=0.0, Omega=0.0, omega=0.0, M0=0.0, year_length=400*86400)
    spin = SpinConfig(day_length=86400.0, theta0=0.0)
    rc = RadiationConfig(use_clouds=False)

    t1 = 0.0  # periapsis
    t2 = orb.year_length / 2.0  # apoapsis (by symmetry)

    tend1 = radiation_tendencies(st, g, t1, orb, spin, rc, RadiationInputs(albedo=albedo))
    tend2 = radiation_tendencies(st, g, t2, orb, spin, rc, RadiationInputs(albedo=albedo))

    A = cell_areas(g)
    # Extract Qsw global mean via dT * M * cp (since LW=0)
    cp = rc.cp
    Qsw1 = float(np.sum(A * st.M * tend1.dT) * cp)
    Qsw2 = float(np.sum(A * st.M * tend2.dT) * cp)

    r1 = orbit_radius(t1, orb); r2 = orbit_radius(t2, orb)
    ratio_the = (orb.a / r1) ** 2 / ((orb.a / r2) ** 2)
    ratio_mea = Qsw1 / max(1e-12, Qsw2)

    assert abs(ratio_mea - ratio_the) < 1e-9