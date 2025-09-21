from __future__ import annotations

import numpy as np


def u_on_V(u: np.ndarray) -> np.ndarray:
    """
    Interpolate U-face velocity u (ny, nx+1) onto V faces (ny+1, nx).
    Periodic in i; clamped in j.
    """
    ny, nxp1 = u.shape
    nx = nxp1 - 1
    out = np.empty((ny + 1, nx), dtype=u.dtype)
    uh = 0.5 * (u[:, :nx] + u[:, 1:])   # (ny, nx)
    out[1:-1, :] = 0.5 * (uh[:-1, :] + uh[1:, :])
    out[0, :] = uh[0, :]
    out[-1, :] = uh[-1, :]
    return out


def v_on_U(v: np.ndarray) -> np.ndarray:
    """
    Interpolate V-face velocity v (ny+1, nx) onto U faces (ny, nx+1).
    Periodic in i; averaged in j.
    """
    nyp1, _ = v.shape
    ny = nyp1 - 1
    vv = 0.5 * (v[:ny, :] + v[1:, :])   # (ny, nx)
    left = np.concatenate([vv[:, -1:], vv], axis=1)
    right = np.concatenate([vv, vv[:, :1]], axis=1)
    return 0.5 * (left + right)         # (ny, nx+1)