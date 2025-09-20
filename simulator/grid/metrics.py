from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .sphere import Grid


def cell_areas(grid: Grid) -> np.ndarray:
    """Cell areas A[j,i] = R^2 (sin φ_{j+1/2} - sin φ_{j-1/2}) Δλ.

    Returns (ny,nx) array.
    """
    R = grid.R
    dlon = grid.dlon
    # lat edges (ny+1,)
    lat_edges = grid.lat_v  # j±1/2 lives in lat_v
    sin_up = np.sin(lat_edges[1:])
    sin_dn = np.sin(lat_edges[:-1])
    band = (sin_up - sin_dn)[:, None]  # (ny,1)
    A = (R * R) * band * dlon  # broadcast to (ny,nx)
    return np.repeat(A, grid.nx, axis=1)


def dx_dy_on_centers(grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    """Return metric lengths Δx, Δy at cell centers (ny,nx).

    Δx = R cos φ Δλ,  Δy = R Δφ.
    """
    R = grid.R
    dlon, dlat = grid.dlon, grid.dlat
    cosφ = np.cos(grid.latc2d)
    dx = R * np.maximum(cosφ, 0.0) * dlon  # cosφ≥0 by definition here
    dy = R * dlat * np.ones_like(dx)
    return dx, dy


def edge_lengths_u(grid: Grid) -> np.ndarray:
    """
    Length of U edges (east-west oriented) at (ny, nx+1).
    L_λ = R cos φ Δλ, with φ from U lat lines, applied to each vertical face.
    """
    R = grid.R
    cosφ = np.cos(grid.latu2d)
    L = R * np.maximum(cosφ, 0.0) * grid.dlon
    return L  # (ny, nx+1)


def edge_lengths_v(grid: Grid) -> np.ndarray:
    """
    Length of V edges (north-south oriented) at (ny+1, nx).
    L_φ = R Δφ (independent of longitude).
    """
    R = grid.R
    L = R * grid.dlat * np.ones_like(grid.lonv2d)
    return L  # (ny+1, nx)


def coriolis_on_stag(grid: Grid, omega: float) -> Dict[str, np.ndarray]:
    """
    Coriolis parameter f on centers, U, and V staggers.
    f(φ) = 2 Ω sin φ.
    """
    fC = 2.0 * omega * np.sin(grid.latc2d)
    fU = 2.0 * omega * np.sin(grid.latu2d)
    fV = 2.0 * omega * np.sin(grid.latv2d)
    return {"fC": fC, "fU": fU, "fV": fV}


# ── Polar-cap helpers ────────────────────────────────────────────────────────

def build_cap_row_weights(grid: Grid, areas: np.ndarray) -> np.ndarray:
    """
    Compute per-row weights that sum to 1, using areas for conservation.
    Returns array W (ny,nx); W[j,:] = A[j,:] / sum(A[j,:]) if row is cap, else uniform 1/nx.
    """
    ny, nx = areas.shape
    W = np.full_like(areas, fill_value=1.0 / nx)
    rows = np.where(grid.cap_rows)[0]
    for j in rows:
        S = float(np.sum(areas[j, :]))
        if S > 0:
            W[j, :] = areas[j, :] / S
    return W


def enforce_cap_mean(field: np.ndarray, grid: Grid, weights: np.ndarray) -> np.ndarray:
    """
    Project scalar field on cap rows to its (area-weighted) row mean.
    Modifies a copy; returns the projected field. Shapes (ny,nx).
    """
    X = field.copy()
    rows = np.where(grid.cap_rows)[0]
    for j in rows:
        mean_val = float(np.sum(X[j, :] * weights[j, :]))  # scalar
        X[j, :] = mean_val
    return X


def enforce_cap_fluxes(FM_u: np.ndarray, FM_v: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply polar-cap rules to mass fluxes (or any zonal/meridional fluxes).

    Shapes:
      FM_u: (ny, nx+1)  # U faces (zonal)
      FM_v: (ny+1, nx)  # V faces (meridional)

    - On cap rows: zero the entire U-row (all nx+1 faces), since circumference → 0.
    - We leave FM_v untouched here; aggregation across the cap boundary is a separate diagnostic.
    """
    Fu = FM_u.copy()
    # Use 1D row mask to avoid shape mismatch (cap_mask2d is (ny,nx))
    if np.any(grid.cap_rows):
        Fu[grid.cap_rows, :] = 0.0
    Fv = FM_v.copy()
    return Fu, Fv


def aggregate_meridional_flux_row(FV_row: np.ndarray) -> float:
    """Return the aggregated meridional flux across a cap boundary row (sum over longitudes)."""
    return float(np.sum(FV_row))