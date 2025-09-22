from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..grid.metrics import cell_areas
from ..grid.staggering import to_u_centered, to_v_centered
from ..numerics.operators import grad_center_to_UV, div_c_from_fluxes
from ..numerics.stencils import laplacian_center, biharmonic_center
from ..numerics.restagger import reconstruct_u_faces_from_centers, reconstruct_v_faces_from_centers


@dataclass
class DiffusionConfig:
    # Scalars (2nd order diffusion)
    K_T: float = 0.0   # thermal diffusivity (m^2/s) — conservative on MT
    K_q: float = 0.0   # moisture diffusivity (applies to qv,qc,qr on Mq)
    # Scalars (4th order hyperdiffusion)
    K4_T: float = 0.0  # thermal hyperdiffusivity (m^4/s) — conservative on MT
    K4_q: float = 0.0  # moisture hyperdiffusivity (m^4/s)
    # Velocity (center-based viscosity)
    nu_u: float = 0.0  # eddy viscosity (m^2/s) for u_c, v_c
    nu4_u: float = 0.0 # optional biharmonic coefficient (m^4/s)
    viscous_heating: bool = False
    cp: float = 1004.0 # J/kg/K


def _conservative_scalar_diffusion(M: np.ndarray, q: np.ndarray, K: float, grid: Grid) -> np.ndarray:
    """
    Return dq/dt such that d(Mq)/dt = -div( F ), F = -K * M_face * grad(q).
    Uses metric-aware grad and center divergence; imposes no-normal-flow at poles.
    """
    if K == 0.0:
        return np.zeros_like(q)
    A = cell_areas(grid)
    M_u = to_u_centered(M)
    M_v = to_v_centered(M)
    Gx, Gy = grad_center_to_UV(q, grid)    # (ny,nx+1), (ny+1,nx)
    Fu = -K * M_u * Gx
    Fv = -K * M_v * Gy
    # no-normal-flow at the top/bottom boundaries
    Fv = Fv.copy(); Fv[0, :] = 0.0; Fv[-1, :] = 0.0
    divF = div_c_from_fluxes(Fu, Fv, A)
    eps = 1e-15 * float(np.mean(M) if M.size else 1.0)
    return -divF / np.maximum(M, eps)


def _conservative_scalar_hyperdiffusion(M: np.ndarray, q: np.ndarray, K4: float, grid: Grid) -> np.ndarray:
    """
    Return dq/dt for **biharmonic** (hyperdiffusion) that conserves total Mq.
    Form: d(Mq)/dt = -div( F ), with F = -K4 * M_face * grad( laplacian(q) ).
    """
    if K4 == 0.0:
        return np.zeros_like(q)
    A = cell_areas(grid)
    M_u = to_u_centered(M)
    M_v = to_v_centered(M)
    Lq = laplacian_center(q, grid)         # (ny,nx)
    Gx, Gy = grad_center_to_UV(Lq, grid)   # gradients of Laplacian
    Fu = -K4 * M_u * Gx
    Fv = -K4 * M_v * Gy
    Fv = Fv.copy(); Fv[0, :] = 0.0; Fv[-1, :] = 0.0
    divF = div_c_from_fluxes(Fu, Fv, A)
    eps = 1e-15 * float(np.mean(M) if M.size else 1.0)
    return -divF / np.maximum(M, eps)


def diffusion_tendencies(state: State, grid: Grid, cfg: DiffusionConfig = DiffusionConfig()) -> Tend:
    """
    Conservative diffusion for scalars (T, q*) and viscous diffusion for u,v on centers.
    If viscous_heating=True, adds Q_visc/cp to dT to close ΔEk + c_p ΔH ≈ 0.
    """
    zc = np.zeros_like(state.M)

    # Scalars (conservative on Mq)
    dT = _conservative_scalar_diffusion(state.M, state.T, cfg.K_T, grid)
    dT += _conservative_scalar_hyperdiffusion(state.M, state.T, cfg.K4_T, grid)
    dqv = _conservative_scalar_diffusion(state.M, state.qv, cfg.K_q, grid)
    dqv += _conservative_scalar_hyperdiffusion(state.M, state.qv, cfg.K4_q, grid)
    dqc = _conservative_scalar_diffusion(state.M, state.qc, cfg.K_q, grid)
    dqc += _conservative_scalar_hyperdiffusion(state.M, state.qc, cfg.K4_q, grid)
    dqr = _conservative_scalar_diffusion(state.M, state.qr, cfg.K_q, grid)
    dqr += _conservative_scalar_hyperdiffusion(state.M, state.qr, cfg.K4_q, grid)

    # Velocity viscosity on centers
    dMU = np.zeros_like(state.MU)
    dMV = np.zeros_like(state.MV)

    if cfg.nu_u != 0.0 or cfg.nu4_u != 0.0 or cfg.viscous_heating:
        eps = 1e-15 * float(np.mean(state.M) if state.M.size else 1.0)
        M_u = to_u_centered(state.M); M_v = to_v_centered(state.M)
        u_f = state.MU / np.maximum(M_u, eps)
        v_f = state.MV / np.maximum(M_v, eps)
        u_c = 0.5 * (u_f[:, 1:] + u_f[:, :-1])
        v_c = 0.5 * (v_f[1:, :] + v_f[:-1, :])
        Lu = np.zeros_like(u_c); Lv = np.zeros_like(v_c)
        if cfg.nu_u != 0.0:
            Lu += cfg.nu_u * laplacian_center(u_c, grid)
            Lv += cfg.nu_u * laplacian_center(v_c, grid)
        if cfg.nu4_u != 0.0:
            Lu -= cfg.nu4_u * biharmonic_center(u_c, grid)  # sign chosen to damp grid-scale
            Lv -= cfg.nu4_u * biharmonic_center(v_c, grid)
        # Restagger increments exactly in center-average sense
        du_f = reconstruct_u_faces_from_centers(Lu)
        dv_f = reconstruct_v_faces_from_centers(Lv)
        dMU = M_u * du_f
        dMV = M_v * dv_f
        if cfg.viscous_heating:
            # dT += - (u_c*du_c + v_c*dv_c) / cp, where du_c = Lu, dv_c = Lv
            dT = dT - (u_c * Lu + v_c * Lv) / cfg.cp

    return Tend(dM=zc, dT=dT, dqv=dqv, dqc=dqc, dqr=dqr, dMU=dMU, dMV=dMV)