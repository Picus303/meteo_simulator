from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(slots=True)
class Grid:
    R: float
    nx: int
    ny: int
    dlon: float  # radians
    dlat: float  # radians
    lon_c: np.ndarray  # (nx,) centers in [0, 2π)
    lat_c: np.ndarray  # (ny,) centers in [-π/2, π/2]
    lon_u: np.ndarray  # (nx+1,) edges for U (zonal faces)
    lat_u: np.ndarray  # (ny,) same as lat_c (U shares lat of cell centers)
    lon_v: np.ndarray  # (nx,) same as lon_c (V shares lon of cell centers)
    lat_v: np.ndarray  # (ny+1,) edges for V (meridional faces)

    # 2D versions (broadcasted for convenience)
    lonc2d: np.ndarray  # (ny, nx)
    latc2d: np.ndarray  # (ny, nx)
    lonu2d: np.ndarray  # (ny, nx+1)
    latu2d: np.ndarray  # (ny, nx+1)
    lonv2d: np.ndarray  # (ny+1, nx)
    latv2d: np.ndarray  # (ny+1, nx)

    # Polar cap masks & weights (mask rows with |lat|>=cap)
    cap_deg: Optional[float]
    cap_rows: np.ndarray  # (ny,) bool
    cap_mask2d: np.ndarray  # (ny, nx) bool
    # Area-weight weights per row for conservative row-mean projection (filled later)
    cap_row_weights: Optional[np.ndarray]  # (ny, nx) or None


def make_latlon_grid(
    nlon: int,
    nlat: int,
    R: float,
    *,
    cap_deg: float | None = 85.0,
) -> Grid:
    """Create a regular lat-lon C-grid with optional polar cap mask (Nx kept constant).

    Parameters
    ----------
    nlon, nlat : int
        Number of longitudes/latitudes (cell centers). Use nlat ≥ 4.
    R : float
        Planet radius (m).
    cap_deg : float | None
        If not None, rows with |lat| ≥ cap_deg are flagged as *cap rows*.
        Shapes remain (ny, nx). The cap handling is done via projection utilities.

    Returns
    -------
    Grid
        Grid dataclass with 1D & 2D arrays for centers/edges and cap masks.
    """
    assert nlon >= 4 and nlat >= 4, "Use at least 4x4 to define a meaningful grid"

    dlon = 2.0 * np.pi / nlon
    dlat = np.pi / nlat

    # Centers
    lon_c = (np.arange(nlon) + 0.5) * dlon  # [0, 2π)
    lat_c = -0.5 * np.pi + (np.arange(nlat) + 0.5) * dlat  # [-π/2, π/2]

    # Edges
    lon_u = np.arange(nlon + 1) * dlon  # zonal faces
    lat_u = lat_c.copy()                 # U shares lat lines with centers

    lon_v = lon_c.copy()                 # V shares lon with centers
    lat_v = -0.5 * np.pi + np.arange(nlat + 1) * dlat  # meridional faces

    # 2D broadcasts
    lonc2d, latc2d = np.meshgrid(lon_c, lat_c, indexing="xy")
    lonu2d, latu2d = np.meshgrid(lon_u, lat_u, indexing="xy")
    lonv2d, latv2d = np.meshgrid(lon_v, lat_v, indexing="xy")

    # Cap masks
    if cap_deg is None:
        cap_rows = np.zeros((nlat,), dtype=bool)
    else:
        cap_phi = np.deg2rad(cap_deg)
        cap_rows = (lat_c >= +cap_phi) | (lat_c <= -cap_phi)

    cap_mask2d = np.repeat(cap_rows[:, None], nlon, axis=1)

    grid = Grid(
        R=R,
        nx=nlon,
        ny=nlat,
        dlon=dlon,
        dlat=dlat,
        lon_c=lon_c,
        lat_c=lat_c,
        lon_u=lon_u,
        lat_u=lat_u,
        lon_v=lon_v,
        lat_v=lat_v,
        lonc2d=lonc2d,
        latc2d=latc2d,
        lonu2d=lonu2d,
        latu2d=latu2d,
        lonv2d=lonv2d,
        latv2d=latv2d,
        cap_deg=cap_deg,
        cap_rows=cap_rows,
        cap_mask2d=cap_mask2d,
        cap_row_weights=None,
    )
    return grid