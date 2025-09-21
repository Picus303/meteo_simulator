from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..grid.metrics import cell_areas, enforce_cap_fluxes
from ..grid.staggering import to_u_centered, to_v_centered


@dataclass
class AdvecConfig:
    eps_mass_rel: float = 1e-12  # guard for division by small M
    cap_zero_u: bool = True      # zero U-flux on cap rows


# ── Geometry helpers (local to advection; rely on Grid 1D coords) ────────────

def _face_lengths(grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dy_u, dx_v) face lengths.

    dy_u: (ny, nx+1) meridional edge length at U faces j,* (constant along i)
          dy_u[j,:] = R * (lat_v[j+1] - lat_v[j])
    dx_v: (ny+1, nx) zonal edge length at V faces j,*
          dx_v[j,:] = R * cos(lat_v[j]) * (lon_u[i+1] - lon_u[i]) (varies with j)
    """
    R = float(grid.R)
    # Meridional span between V nodes
    dphi = np.diff(grid.lat_v)  # (ny,)
    dy_row = R * dphi  # (ny,)
    dy_u = np.repeat(dy_row[:, None], grid.nx + 1, axis=1)

    # Zonal span between U nodes at latitude lat_v[j]
    dlambda = np.diff(grid.lon_u)  # (nx,)
    # Ensure wrap is 2π total; lon_u should be periodic; dlambda likely constant
    dx_v = np.empty((grid.ny + 1, grid.nx), dtype=np.float64)
    for j in range(grid.ny + 1):
        dx_v[j, :] = R * np.cos(grid.lat_v[j]) * dlambda
    return dy_u, dx_v


# ── Flux builders ────────────────────────────────────────────────────────────

def _face_mass(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate cell-centered M to U and V faces by simple averages.
    Returns (M_u (ny,nx+1), M_v (ny+1,nx)).
    """
    return to_u_centered(M), to_v_centered(M)


def _face_velocity(MU: np.ndarray, MV: np.ndarray, M_u: np.ndarray, M_v: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    u = MU / np.maximum(M_u, eps)
    v = MV / np.maximum(M_v, eps)
    return u, v


def mass_fluxes(state: State, grid: Grid, cfg: AdvecConfig = AdvecConfig()) -> Tuple[np.ndarray, np.ndarray]:
    """Compute zonal/meridional **mass fluxes** across faces (kg s^-1 per face).

    Fu: (ny, nx+1) across U faces (vertical edges), positive eastward
    Fv: (ny+1, nx) across V faces (horizontal edges), positive northward
    """
    M_u, M_v = _face_mass(state.M)
    u, v = _face_velocity(state.MU, state.MV, M_u, M_v, cfg.eps_mass_rel * float(np.mean(state.M)))
    dy_u, dx_v = _face_lengths(grid)

    Fu = M_u * u * dy_u
    Fv = M_v * v * dx_v

    if cfg.cap_zero_u:
        Fu, Fv = enforce_cap_fluxes(Fu, Fv, grid)
    return Fu, Fv


def _divergence(Fu: np.ndarray, Fv: np.ndarray, area: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (DivU, DivV) centered divergences of the two components (without sign).
    Shapes:
      Fu: (ny, nx+1) → d/dx via Fu[:,1:] - Fu[:,:-1]
      Fv: (ny+1, nx) → d/dy via Fv[1:,:] - Fv[:-1,:]
    """
    dFx = Fu[:, 1:] - Fu[:, :-1]
    dFy = Fv[1:, :] - Fv[:-1, :]
    return dFx / area, dFy / area


def _upwind_u(phi: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Upwind reconstruction on U faces (ny, nx+1) from cell φ (ny,nx).
    Periodic in i.
    """
    left = np.concatenate([phi[:, -1:], phi], axis=1)   # donor if u>=0
    right = np.concatenate([phi, phi[:, :1]], axis=1)   # donor if u<0
    return np.where(u >= 0.0, left, right)


def _upwind_v(phi: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Upwind reconstruction on V faces (ny+1, nx) from cell φ (ny,nx).
    Clamped in j (replicate boundary rows).
    """
    south = np.concatenate([phi[:1, :], phi], axis=0)   # donor if v>=0 (from j-1)
    north = np.concatenate([phi, phi[-1:, :]], axis=0)  # donor if v<0 (from j)
    return np.where(v >= 0.0, south, north)


# ── Public driver ────────────────────────────────────────────────────────────

def advect_mass_and_tracers(state: State, grid: Grid, tracers: Dict[str, np.ndarray], cfg: AdvecConfig = AdvecConfig()) -> Tend:
    """Return conservative advection tendencies for M and the provided tracers.

    Parameters
    ----------
    state : State
        Provides M (centers) and MU/MV (faces) to infer velocities.
    grid : Grid
    tracers : dict name->array
        Mixing-ratio tracers at centers to advect with mass (T, qv, qc, qr...).
    cfg : AdvecConfig

    Returns
    -------
    Tend
        dM, dφ for each tracer; dMU,dMV are zero here (no momentum dynamics in Step 6).
    """
    area = cell_areas(grid)
    Fu, Fv = mass_fluxes(state, grid, cfg)
    # Mass divergence and tendency
    divU, divV = _divergence(Fu, Fv, area)
    DivM = divU + divV
    dM = -DivM

    # Prepare face data for tracers
    M_u, M_v = _face_mass(state.M)  # only for velocity; Fu,Fv already built
    eps = cfg.eps_mass_rel * float(np.mean(state.M))
    u, v = _face_velocity(state.MU, state.MV, M_u, M_v, eps)

    # Tracers tendencies
    dT = np.zeros_like(state.M); dqv = np.zeros_like(state.M); dqc = np.zeros_like(state.M); dqr = np.zeros_like(state.M)
    out_map = {"T": dT, "qv": dqv, "qc": dqc, "qr": dqr}

    for name, phi in tracers.items():
        # Upwind face values
        phi_u = _upwind_u(phi, u)
        phi_v = _upwind_v(phi, v)
        # Tracer mass fluxes use the precomputed mass fluxes multiplied by face value
        Fphi_u = phi_u * Fu
        Fphi_v = phi_v * Fv
        divU_phi, divV_phi = _divergence(Fphi_u, Fphi_v, area)
        DivMphi = divU_phi + divV_phi
        # Convert to mixing-ratio tendency: d(Mφ)/dt = -DivMφ  ⇒  dφ/dt = [ -DivMφ + φ DivM ] / M
        numer = -DivMphi + phi * DivM
        dphi = numer / np.maximum(state.M, eps)
        out_map[name][:] = dphi

    return Tend(dM=dM, dT=out_map["T"], dqv=out_map["qv"], dqc=out_map["qc"], dqr=out_map["qr"],
                dMU=np.zeros_like(state.MU), dMV=np.zeros_like(state.MV))