from __future__ import annotations
import numpy as np
from ..grid.sphere import Grid
from ..grid.metrics import cell_areas
from .energy_exchange import kinetic_energy, potential_energy


def enthalpy(M: np.ndarray, T: np.ndarray, grid: Grid, cp: float) -> float:
    A = cell_areas(grid)
    return float(cp * np.sum(A * M * T))


def total_energy(MU, MV, M, T, grid: Grid, cp: float, g: float, rho_ref: float) -> float:
    return kinetic_energy(MU, MV, M, grid) + potential_energy(M, grid, g, rho_ref) + enthalpy(M, T, grid, cp)