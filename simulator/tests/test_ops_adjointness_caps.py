import numpy as np

from simulator.grid.sphere import make_latlon_grid
from simulator.grid.metrics import cell_areas, _dx_dy_faces_from_centers
from simulator.numerics.operators import div_c_from_fluxes, grad_center_to_UV


def test_metric_adjointness_with_caps_masking():
    g = make_latlon_grid(32, 16, 6.371e6, cap_deg=82.0)
    rng = np.random.default_rng(2)
    A = rng.standard_normal((g.ny, g.nx))
    Fu = rng.standard_normal((g.ny, g.nx + 1))
    Fv = rng.standard_normal((g.ny + 1, g.nx))

    # Apply cap masking on zonal faces to mimic runtime behavior
    if np.any(g.cap_rows):
        Fu = Fu.copy(); Fu[g.cap_rows, :] = 0.0

    Fv = Fv.copy()
    Fv[0, :]  = 0.0   # no-normal-flow southern boundary
    Fv[-1, :] = 0.0   # no-normal-flow northern boundary

    Fu[:, 0] = Fu[:, -1]    # periodic in longitude

    area = cell_areas(g)
    dx_u, dy_v = _dx_dy_faces_from_centers(g)

    lhs = float(np.sum(area * div_c_from_fluxes(Fu, Fv, area) * A))
    Gx, Gy = grad_center_to_UV(A, g)
    # Mirror the masking on the gradient side to preserve discrete adjointness
    if np.any(g.cap_rows):
        Gx = Gx.copy(); Gx[g.cap_rows, :] = 0.0

    lhs_x = float(np.sum(A * (Fu[:, 1:] - Fu[:, :-1])))
    lhs_y = float(np.sum(A * (Fv[1:, :] - Fv[:-1, :])))
    rhs_x = -float(np.sum(Gx[:, 1:] * Fu[:, 1:] * dx_u[:, 1:]))
    rhs_y = -float(np.sum(Gy * Fv * dy_v))
    print(f"X: LHS {lhs_x:.6e} RHS {rhs_x:.6e}")
    print(f"Y: LHS {lhs_y:.6e} RHS {rhs_y:.6e}")

    rhs = -float(np.sum(Gx[:, 1:] * Fu[:, 1:] * dx_u[:, 1:]) + np.sum(Gy * Fv * dy_v))
    assert np.isclose(lhs, rhs, atol=1e-8)