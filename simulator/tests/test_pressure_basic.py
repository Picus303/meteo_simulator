import numpy as np

from simulator.core.state import State
from simulator.grid.sphere import make_latlon_grid
from simulator.physics.pressure import pressure_tendencies


def test_pressure_zero_on_uniform_M():
    g = make_latlon_grid(20, 10, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    k = pressure_tendencies(st, g)
    assert np.allclose(k.dMU, 0.0) and np.allclose(k.dMV, 0.0)


def test_pressure_simple_zonal_grad_periodic():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    # Periodic variation in longitude: avoids a step at the wrap
    alpha = 1e-3
    lam = g.lon_c[None, :]
    st.M[...] = 1.0 + alpha * np.cos(lam)

    k = pressure_tendencies(st, g)

    # No meridional gradient ⇒ dMV ≈ 0 in the interior
    assert np.allclose(k.dMV[1:-1, :], 0.0, atol=1e-12)

    # Theory: ∂x(½ g h²) ∝ M ∂M/∂x ∝ −sin(λ) ⇒ dMU = −∂x(...) ∝ +sin(λ)
    dMU_mean = k.dMU.mean(axis=0)[:-1]  # average over j to smooth out cos(φ)
    target = np.sin(lam).ravel()
    corr = float(np.dot(dMU_mean, target))
    assert corr > 0.0


def test_pressure_cap_zero_u():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=82.0)
    st = State.zeros(g.ny, g.nx)
    st.M += 0.01 * np.random.default_rng(0).standard_normal(st.M.shape)

    k = pressure_tendencies(st, g)
    # All dMU on cap rows must be zero
    rows = np.where(g.cap_rows)[0]
    for j in rows:
        assert np.allclose(k.dMU[j, :], 0.0)