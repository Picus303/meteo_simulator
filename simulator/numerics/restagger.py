from __future__ import annotations
import numpy as np


def reconstruct_u_faces_from_centers(u_c: np.ndarray) -> np.ndarray:
    """
    Given center u_c (ny, nx), construct U-face u_f (ny, nx+1) s.t.
    0.5*(u_f[:, i] + u_f[:, i+1]) == u_c[:, i] for all i, with periodic closure.
    Exact for nx odd; for nx even we apply the recursive solution and enforce closure.
    """
    ny, nx = u_c.shape
    u_f = np.empty((ny, nx + 1), dtype=u_c.dtype)
    if nx % 2 == 1:
        alt = ((-1) ** np.arange(nx))[None, :]
        f0 = np.sum(alt * u_c, axis=1)  # (ny,)
    else:
        f0 = np.zeros(ny, dtype=u_c.dtype)
    u_f[:, 0] = f0
    for i in range(nx):
        u_f[:, i + 1] = 2.0 * u_c[:, i] - u_f[:, i]
    u_f[:, -1] = u_f[:, 0]  # periodic closure
    return u_f


def reconstruct_v_faces_from_centers(v_c: np.ndarray) -> np.ndarray:
    """
    Given center v_c (ny, nx), construct V-face v_f (ny+1, nx) s.t.
    0.5*(v_f[j, :] + v_f[j+1, :]) == v_c[j, :] for all j. Non-periodic in j.
    """
    ny, nx = v_c.shape
    v_f = np.empty((ny + 1, nx), dtype=v_c.dtype)
    v_f[0, :] = v_c[0, :]
    for j in range(ny):
        v_f[j + 1, :] = 2.0 * v_c[j, :] - v_f[j, :]
    return v_f