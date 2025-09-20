from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.state import State
from ..grid.staggering import to_c_from_u, to_c_from_v


def velocities_at_centers(MU: np.ndarray, MV: np.ndarray, M: np.ndarray, *, eps_rel: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return u_c, v_c at cell centers via simple C-grid averaging, with safe division by M.

    MU: (ny, nx+1), MV: (ny+1, nx), M: (ny, nx)
    """
    ny, nx = M.shape
    u_c = to_c_from_u(MU)
    v_c = to_c_from_v(MV)
    M_safe = np.maximum(M, eps_rel * float(np.mean(M)))
    return u_c / M_safe, v_c / M_safe


def mass_total(M: np.ndarray, area: np.ndarray) -> float:
    return float(np.sum(M * area))


def water_total(M: np.ndarray, qv: np.ndarray, qc: np.ndarray, qr: np.ndarray, area: np.ndarray) -> float:
    return float(np.sum(M * (qv + qc + qr) * area))


def energies_total(
    M: np.ndarray,
    u_c: np.ndarray,
    v_c: np.ndarray,
    area: np.ndarray,
    *,
    g: float,
    rho_ref: Optional[float] = None,
    potential_mode: str = "none",
) -> Dict[str, float]:
    """Compute kinetic and (optionally) shallow-water potential energy (no orography).

    potential_mode: 'none' or 'shallow'. If 'shallow', rho_ref is required and
    E_pot = 0.5 * g * sum A * (M/rho_ref)^2.
    """
    ekin = 0.5 * np.sum(M * (u_c * u_c + v_c * v_c) * area)
    epot = 0.0
    if potential_mode == "shallow":
        if rho_ref is None:
            raise ValueError("rho_ref is required for potential_mode='shallow'")
        h = M / float(rho_ref)
        epot = 0.5 * g * np.sum((h * h) * area)
    elif potential_mode != "none":
        raise ValueError("potential_mode must be 'none' or 'shallow'")
    return {"Ekin": float(ekin), "Epot": float(epot), "Etot": float(ekin + epot)}


def extrema(state: State) -> Dict[str, Dict[str, float]]:
    d = {}
    for name in ("M", "T", "qv", "qc", "qr"):
        arr = getattr(state, name)
        d[name] = {"min": float(np.min(arr)), "max": float(np.max(arr))}
    d["MU"] = {"min": float(np.min(state.MU)), "max": float(np.max(state.MU))}
    d["MV"] = {"min": float(np.min(state.MV)), "max": float(np.max(state.MV))}
    return d


def nan_report(state: State) -> Dict[str, object]:
    bad = []
    for name in ("M", "T", "qv", "qc", "qr", "MU", "MV"):
        arr = getattr(state, name)
        if not np.isfinite(arr).all():
            bad.append(name)
    return {"has_nan": bool(bad), "fields": bad}