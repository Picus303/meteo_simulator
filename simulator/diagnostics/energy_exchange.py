from __future__ import annotations

import numpy as np

from ..grid.metrics import cell_areas
from ..grid.sphere import Grid


def kinetic_energy(MU: np.ndarray, MV: np.ndarray, M: np.ndarray, grid: Grid) -> float:
    """Domain-integrated kinetic energy (J) using center velocities.

    We compute u_c,v_c by averaging face-normal momenta to centers, then ½ M (u_c²+v_c²).
    This shares the same philosophy as diagnostics.invariants.velocities_at_centers.
    """
    # Centered velocities
    # U→C average in i; V→C average in j (simple and consistent)
    u_c = 0.5 * (MU[:, 1:] + MU[:, :-1])
    v_c = 0.5 * (MV[1:, :] + MV[:-1, :])
    # Shapes (ny, nx)
    assert u_c.shape == M.shape and v_c.shape == M.shape
    A = cell_areas(grid)
    return float(0.5 * np.sum(M * (u_c * u_c + v_c * v_c) * A))


def potential_energy(M: np.ndarray, grid: Grid, g: float, rho_ref: float) -> float:
    """Shallow potential energy ∫ ½ g (M/ρ)² dA."""
    A = cell_areas(grid)
    return float(0.5 * g * np.sum((M / rho_ref) ** 2 * A))