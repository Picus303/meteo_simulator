import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.staggering import to_u_centered, to_v_centered
from simulator.core.state import State
from simulator.advection.momentum import advect_momentum


def test_solid_body_zonal_is_stationary():
    """
    On a sphere, solid-body rotation u(φ)=U0*cosφ, v=0 is stationary for pure advection.
    We build MU/MV accordingly and expect dMU≈0, dMV≈0.
    """
    g = make_latlon_grid(48, 24, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)

    # Build u(φ)=U0*cosφ on U faces, v=0 on V faces
    U0 = 20.0
    cos_lat_c = np.cos(g.lat_c)[:, None]                            # (ny,1)
    cos_lat_u = to_u_centered(np.repeat(cos_lat_c, g.nx, axis=1))   # (ny,nx+1)

    M_u = to_u_centered(st.M)

    st.MU[:] = M_u * (U0 * cos_lat_u)
    st.MV[:] = 0.0

    k = advect_momentum(st, g)

    assert np.allclose(k.dMU, 0.0, atol=1e-12)
    assert np.allclose(k.dMV, 0.0, atol=1e-12)


def test_no_normal_velocity_created_on_caps():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=82.0)
    st = State.zeros(g.ny, g.nx)

    # Put some random velocities
    rng = np.random.default_rng(0)
    M_u = to_u_centered(st.M)
    M_v = to_v_centered(st.M)
    st.MU[:] = M_u * (10.0 + rng.standard_normal(st.MU.shape))
    st.MV[:] = M_v * (5.0 + rng.standard_normal(st.MV.shape))

    k = advect_momentum(st, g)

    # On cap rows, dMU should be ~0 because Fu is zeroed there
    rows = np.where(g.cap_rows)[0]
    for j in rows:
        assert np.allclose(k.dMU[j, :], 0.0, atol=1e-12)