import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas, build_cap_row_weights, enforce_cap_mean, enforce_cap_fluxes


def test_cap_projection_conserves_and_flattens():
    g = make_latlon_grid(48, 24, 6.371e6, cap_deg=80.0)
    A = cell_areas(g)

    # random scalar field
    rng = np.random.default_rng(123)
    X = rng.standard_normal((g.ny, g.nx))

    W = build_cap_row_weights(g, A)
    total_before = float(np.sum(X * A))
    Xp = enforce_cap_mean(X, g, W)
    total_after = float(np.sum(Xp * A))

    assert np.isclose(total_before, total_after, atol=1e-9, rtol=1e-12)

    rows = np.where(g.cap_rows)[0]
    for j in rows:
        assert np.isclose(np.std(Xp[j, :]), 0.0)


def test_cap_enforce_zero_zonal_flux():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=82.0)
    Fu = np.ones((g.ny, g.nx + 1))
    Fv = np.ones((g.ny + 1, g.nx))

    Fu2, Fv2 = enforce_cap_fluxes(Fu, Fv, g)

    # On cap rows, Fu should be zero
    rows = np.where(g.cap_rows)[0]
    for j in rows:
        assert np.allclose(Fu2[j, :], 0.0)
    # Elsewhere unchanged
    for j in np.where(~g.cap_rows)[0]:
        assert np.allclose(Fu2[j, :], 1.0)
    # Fv untouched by this helper
    assert np.allclose(Fv2, 1.0)