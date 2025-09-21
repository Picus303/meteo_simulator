from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..grid.metrics import dx_dy_on_centers


@dataclass
class PressureConfig:
    g: float = 9.81
    rho_ref: float = 1.2  # consistent with diagnostics if you enabled shallow potential
    cap_zero_u: bool = True


def _center_to_u_avg(A: np.ndarray) -> np.ndarray:
    """Average a (ny,nx) field to U faces (ny,nx+1) by 1/2*(left+right) with periodic i."""
    return 0.5 * (np.concatenate([A[:, -1:], A], axis=1) + np.concatenate([A, A[:, :1]], axis=1))


def _center_to_v_avg(A: np.ndarray) -> np.ndarray:
    """Average a (ny,nx) field to V faces (ny+1,nx) by 1/2*(south+north) with clamped j."""
    out = np.empty((A.shape[0] + 1, A.shape[1]), dtype=A.dtype)
    # Correct explicit construction
    out[1:-1, :] = 0.5 * (A[:-1, :] + A[1:, :])
    out[0, :] = A[0, :]
    out[-1, :] = A[-1, :]
    return out


def _grad_half_g_h2_on_u(M: np.ndarray, dx_u: np.ndarray, *, g: float, rho_ref: float) -> np.ndarray:
    """Compute ∂x(½ g h²) at U faces using center values of M.

    Returns array (ny,nx+1) aligned with MU.
    """
    factor = 0.5 * g / (rho_ref * rho_ref)
    M2 = M * M
    # Periodic diff along i of center values (nx+1 faces: pairs [i→i+1], plus wrap [nx-1→0])
    right = np.concatenate([M2, M2[:, :1]], axis=1)
    left  = np.concatenate([M2[:, -1:], M2], axis=1)
    d = (right - left)  # (ny,nx+1)
    return factor * d / np.maximum(dx_u, 1e-16)


def _grad_half_g_h2_on_v(M: np.ndarray, dy_v: np.ndarray, *, g: float, rho_ref: float) -> np.ndarray:
    """Compute ∂y(½ g h²) at V faces using center values of M.

    Returns array (ny+1,nx) aligned with MV. We clamp gradients at j=0, ny to zero for simplicity.
    """
    factor = 0.5 * g / (rho_ref * rho_ref)
    M2 = M * M
    north = np.concatenate([M2[1:, :], M2[-1:, :]], axis=0)
    south = np.concatenate([M2[:1, :], M2[:-1, :]], axis=0)
    d_full = (north - south)  # (ny, nx) for interior if we centered, but we want faces
    # Build face-aligned gradient explicitly
    grad = np.zeros((M.shape[0] + 1, M.shape[1]), dtype=M.dtype)
    # interior faces 1..ny-1 use forward difference between centers j and j-1
    grad[1:-1, :] = (M2[1:, :] - M2[:-1, :]) / np.maximum(dy_v[1:-1, :], 1e-16)
    grad[0, :] = 0.0
    grad[-1, :] = 0.0
    return factor * grad


def _dx_dy_faces_from_centers(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Build face distances from center distances by simple averaging.

    dx_u: (ny,nx+1) from dx_centers (ny,nx) averaged to U faces (periodic in i)
    dy_v: (ny+1,nx) from dy_centers (ny,nx) averaged to V faces (clamped in j)
    """
    dx_c, dy_c = dx_dy_on_centers(grid)  # both (ny,nx)
    dx_u = _center_to_u_avg(dx_c)
    dy_v = np.empty((grid.ny + 1, grid.nx), dtype=dy_c.dtype)
    dy_v[1:-1, :] = 0.5 * (dy_c[:-1, :] + dy_c[1:, :])
    dy_v[0, :] = dy_c[0, :]
    dy_v[-1, :] = dy_c[-1, :]
    return dx_u, dy_v


def pressure_tendencies(state: State, grid: Grid, cfg: PressureConfig = PressureConfig()) -> Tend:
    """Return Tend with only pressure/grav contributions on MU, MV.

    dMU = - ∂x(½ g h²)   on U faces
    dMV = - ∂y(½ g h²)   on V faces

    dM,dT,dq* are zeros here; they are handled elsewhere (advection/physics).
    """
    dx_u, dy_v = _dx_dy_faces_from_centers(grid)

    dPhi_dx = _grad_half_g_h2_on_u(state.M, dx_u, g=cfg.g, rho_ref=cfg.rho_ref)
    dPhi_dy = _grad_half_g_h2_on_v(state.M, dy_v, g=cfg.g, rho_ref=cfg.rho_ref)

    dMU = -dPhi_dx
    dMV = -dPhi_dy

    if cfg.cap_zero_u and getattr(grid, "cap_rows", None) is not None and np.any(grid.cap_rows):
        dMU = dMU.copy()
        dMU[grid.cap_rows, :] = 0.0

    zc = np.zeros_like(state.M)
    zu = np.zeros_like(state.MU)
    zv = np.zeros_like(state.MV)
    return Tend(dM=zc, dT=zc, dqv=zc, dqc=zc, dqr=zc, dMU=dMU, dMV=dMV)


@dataclass
class PressureTerm:
    name: str
    grid: Grid
    cfg: PressureConfig

    def __call__(self, state: State, t: float) -> Tend:
        return pressure_tendencies(state, self.grid, self.cfg)


def make_pressure_term(grid: Grid, *, g: float = 9.81, rho_ref: float = 1.2, cap_zero_u: bool = True) -> PressureTerm:
    return PressureTerm(name="pressure", grid=grid, cfg=PressureConfig(g=g, rho_ref=rho_ref, cap_zero_u=cap_zero_u))