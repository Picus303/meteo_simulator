from __future__ import annotations

import numpy as np


def to_u_centered(field_c: np.ndarray) -> np.ndarray:
    """Center-to-U interpolation: average in i (periodic wrap).

    field_c: (ny, nx)
    returns  : (ny, nx+1) — includes the wrapped face at i=0 and i=nx
    """
    ny, nx = field_c.shape
    left = field_c
    right = np.roll(field_c, shift=-1, axis=1)
    faces = 0.5 * (left + right)
    # append last face at i=nx using periodic wrap (same as face at i=0)
    last = faces[:, 0:1]
    return np.concatenate([faces, last], axis=1)


def to_v_centered(field_c: np.ndarray) -> np.ndarray:
    """Center-to-V interpolation: average in j to produce meridional *edges*.

    field_c: (ny, nx)
    returns : (ny+1, nx)

    Construction:
      edges[0   ] = field_c[0   ]      # south boundary
      edges[1:ny] = 0.5*(field_c[:-1] + field_c[1:])
      edges[ny  ] = field_c[ny-1]      # north boundary
    This preserves constants exactly across the whole column.
    """
    ny, nx = field_c.shape
    edges = np.empty((ny + 1, nx), dtype=field_c.dtype)
    edges[1:ny, :] = 0.5 * (field_c[:-1, :] + field_c[1:, :])
    edges[0, :] = field_c[0, :]
    edges[ny, :] = field_c[-1, :]
    return edges


def to_c_from_u(field_u: np.ndarray) -> np.ndarray:
    """U-to-center interpolation: average adjacent faces in i.

    field_u: (ny, nx+1)
    returns : (ny, nx)
    """
    core = 0.5 * (field_u[:, :-1] + field_u[:, 1:])
    return core


def to_c_from_v(field_v: np.ndarray) -> np.ndarray:
    """V-to-center interpolation: average adjacent faces in j.

    field_v: (ny+1, nx)
    returns : (ny, nx)
    """
    core = 0.5 * (field_v[:-1, :] + field_v[1:, :])
    return core