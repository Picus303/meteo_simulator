import numpy as np

from simulator.grid import make_latlon_grid
from simulator.grid.metrics import cell_areas, build_cap_row_weights, enforce_cap_mean

def test_enforce_cap_mean_conserves_integral():
    R = 6.371e6
    g = make_latlon_grid(64, 32, R, cap_deg=80.0)
    A = cell_areas(g)
    W = build_cap_row_weights(g, A)

    # A random field with latitudinal gradient + noise
    rng = np.random.default_rng(0)
    X = (np.sin(g.latc2d) + 0.1 * rng.standard_normal((g.ny, g.nx))).astype(float)

    total_before = float(np.sum(X * A))
    Xp = enforce_cap_mean(X, g, W)
    total_after = float(np.sum(Xp * A))

    assert np.isfinite(total_before) and np.isfinite(total_after)
    # Cap projection should conserve the integral over the sphere
    assert abs(total_before - total_after) < 1e-10 * abs(total_before) + 1e-6