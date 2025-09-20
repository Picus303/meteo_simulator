from __future__ import annotations

import numpy as np


def cfl_max(
    M: np.ndarray,
    u_c: np.ndarray,
    v_c: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dt: float,
    *,
    g: float | None = None,
    rho_ref: float | None = None,
    include_gravity: bool = True,
) -> float:
    """Return global CFL number.

    CFL = max( |u|/dx + |v|/dy + 1[min]*(c_g/min(dx,dy)) ) * dt
    where c_g = sqrt(g * h) with h = M/rho_ref if include_gravity.
    """
    adv = (np.abs(u_c) / np.maximum(dx, 1e-16)) + (np.abs(v_c) / np.maximum(dy, 1e-16))
    cfl = adv
    if include_gravity:
        if g is None or rho_ref is None:
            raise ValueError("g and rho_ref required when include_gravity=True")
        h = M / float(rho_ref)
        cg = np.sqrt(np.maximum(g * h, 0.0))
        inv_dmin = 1.0 / np.maximum(np.minimum(dx, dy), 1e-16)
        cfl = cfl + cg * inv_dmin
    return float(np.max(cfl) * float(dt))