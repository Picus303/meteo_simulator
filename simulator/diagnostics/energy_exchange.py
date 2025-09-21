from __future__ import annotations

import numpy as np

from ..grid.metrics import cell_areas, _dual_areas
from ..grid.sphere import Grid
from ..grid.staggering import to_u_centered, to_v_centered


def kinetic_energy(MU: np.ndarray, MV: np.ndarray, M: np.ndarray, grid: Grid) -> float:
    """Domain-integrated KE using center velocities.

    u_c = 0.5 * ((MU / max(M_u,eps))[:, 1:] + (MU / max(M_u,eps))[:, :-1])
    v_c = 0.5 * ((MV / max(M_v,eps))[1:, :] + (MV / max(M_v,eps))[:-1, :])
    KE = 0.5 * ∑ M * (u_c²+v_c²) * A
    """
    A = cell_areas(grid)
    eps = 1e-15 * float(np.mean(M) if M.size else 1.0)
    M_u = to_u_centered(M)
    M_v = to_v_centered(M)
    u = MU / np.maximum(M_u, eps)
    v = MV / np.maximum(M_v, eps)
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])
    v_c = 0.5 * (v[1:, :] + v[:-1, :])
    return float(0.5 * np.sum(M * (u_c * u_c + v_c * v_c) * A))


def kinetic_energy_faces(MU: np.ndarray, MV: np.ndarray, M: np.ndarray, grid: Grid) -> float:
    """Face-based KE, conserved exactly by the rotation hook.

    E = ½ ∑_U M_u u² Au  +  ½ ∑_V M_v v² Av
    """
    Au, Av = _dual_areas(grid)
    eps = 1e-15 * float(np.mean(M) if M.size else 1.0)
    M_u = to_u_centered(M)
    M_v = to_v_centered(M)
    u = MU / np.maximum(M_u, eps)
    v = MV / np.maximum(M_v, eps)
    EU = 0.5 * np.sum(M_u * (u * u) * Au)
    EV = 0.5 * np.sum(M_v * (v * v) * Av)
    return float(EU + EV)


def potential_energy(M: np.ndarray, grid: Grid, g: float, rho_ref: float) -> float:
    A = cell_areas(grid)
    return float(0.5 * g * np.sum((M / rho_ref) ** 2 * A))