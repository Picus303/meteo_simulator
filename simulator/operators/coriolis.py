from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.state import State
from ..grid.sphere import Grid
from ..grid.staggering import to_u_centered, to_v_centered


@dataclass
class CoriolisRotateConfig:
    omega: float = 7.292115e-5
    cap_zero_u: bool = False
    scheme: str = "centers"


def _f_on_centers(grid: Grid, omega: float) -> np.ndarray:
    return 2.0 * omega * np.sin(grid.lat_c)  # (ny,)


def _reconstruct_u_faces_from_centers(u_c: np.ndarray) -> np.ndarray:
    """
    Given center u_c (ny, nx), construct U-face u_f (ny, nx+1) such that
    center average 0.5*(u_f[:,i] + u_f[:,i+1]) == u_c[:,i] for all i.

    Under periodic closure u_f[:,0] == u_f[:,-1], this system has a unique solution
    when nx is ODD. We implement the exact periodic inversion for odd nx.
    For even nx, we compute the recursive solution then enforce closure at the seam,
    which slightly alters the last equation; tests that require exactness should use nx odd.
    """
    ny, nx = u_c.shape
    u_f = np.empty((ny, nx + 1), dtype=u_c.dtype)
    # Vectorized computation per row: f[i+1] = 2*u_c[i] - f[i]
    # Choose f0 per-row to satisfy periodic closure when nx is odd.
    if nx % 2 == 1:
        # Compute f0 = S = sum_{k=0}^{nx-1} (-1)^{nx-1-k} u_c[:,k]
        # Note: for odd nx, (-1)^{nx-1-k} = (-1)^{-k} = (-1)^k
        alt = ((-1) ** np.arange(nx))[None, :]  # (1,nx)
        f0 = np.sum(alt * u_c, axis=1)  # (ny,)
    else:
        # Even nx: no exact periodic solution in general; pick f0 = 0 as a benign choice.
        f0 = np.zeros(ny, dtype=u_c.dtype)

    u_f[:, 0] = f0
    for i in range(nx):
        u_f[:, i + 1] = 2.0 * u_c[:, i] - u_f[:, i]

    # Enforce periodic closure explicitly
    u_f[:, -1] = u_f[:, 0]
    return u_f


def _reconstruct_v_faces_from_centers(v_c: np.ndarray) -> np.ndarray:
    """
    Given center v_c (ny, nx), construct V-face v_f (ny+1, nx) such that
    center average 0.5*(v_f[j] + v_f[j+1]) == v_c[j] for all j.
    Non-periodic in latitude: exact recursive solution with clamped ends.
    """
    ny, nx = v_c.shape
    v_f = np.empty((ny + 1, nx), dtype=v_c.dtype)
    v_f[0, :] = v_c[0, :]
    for j in range(ny):
        v_f[j + 1, :] = 2.0 * v_c[j, :] - v_f[j, :]
    return v_f


def make_coriolis_rotation_hook(grid: Grid, cfg: CoriolisRotateConfig):
    fC = _f_on_centers(grid, cfg.omega)  # (ny,)
    cap_rows = getattr(grid, "cap_rows", None)

    def rotate_centers(state: State, dt: float) -> State:
        eps = 1e-15 * float(np.mean(state.M) if state.M.size else 1.0)
        M = state.M
        M_u = to_u_centered(M)
        M_v = to_v_centered(M)
        # Face velocities
        u_f = state.MU / np.maximum(M_u, eps)   # (ny, nx+1)
        v_f = state.MV / np.maximum(M_v, eps)   # (ny+1, nx)
        # To centers
        u_c = 0.5 * (u_f[:, 1:] + u_f[:, :-1])  # (ny, nx)
        v_c = 0.5 * (v_f[1:, :] + v_f[:-1, :])  # (ny, nx)
        # Rotate at centers with row-dependent angle
        theta = fC[:, None] * dt
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        u_cp = cos_t * u_c + sin_t * v_c
        v_cp = -sin_t * u_c + cos_t * v_c
        # Restagger back to faces (exact in j; exact in i if nx odd)
        u_f_new = _reconstruct_u_faces_from_centers(u_cp)  # (ny, nx+1)
        v_f_new = _reconstruct_v_faces_from_centers(v_cp)  # (ny+1, nx)
        MU_new = M_u * u_f_new
        MV_new = M_v * v_f_new
        # Caps handling: optional freeze MU on cap rows
        if cfg.cap_zero_u and cap_rows is not None and np.any(cap_rows):
            MU_new = MU_new.copy(); MU_new[cap_rows, :] = state.MU[cap_rows, :]
        return State(M=state.M, T=state.T, qv=state.qv, qc=state.qc, qr=state.qr, MU=MU_new, MV=MV_new)

    def hook(state: State, dt: float | None = None) -> State:
        if dt is None:
            return state
        # unified: centers only
        return rotate_centers(state, dt)

    return hook