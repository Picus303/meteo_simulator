from __future__ import annotations
import numpy as np
from ..grid.sphere import Grid
from ..grid.metrics import cell_areas
from ..numerics.operators import grad_center_to_UV, div_c_from_fluxes


def laplacian_center(phi: np.ndarray, grid: Grid) -> np.ndarray:
    """
    Metric-consistent Laplacian on centers: L = div_c( -grad(phi) ).
    Returns L(phi) in units of phi per unit length^2 (discrete).
    """
    A = cell_areas(grid)
    Gx, Gy = grad_center_to_UV(phi, grid)
    Fu = -Gx  # per-unit-length flux density on U faces
    Fv = -Gy  # per-unit-length flux density on V faces
    return div_c_from_fluxes(Fu, Fv, A)


def biharmonic_center(phi: np.ndarray, grid: Grid) -> np.ndarray:
    return laplacian_center(laplacian_center(phi, grid), grid)