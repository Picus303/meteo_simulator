from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# Constants
EPS = 0.622                # epsilon = Mw/Md
P0 = 1.0e5                 # reference pressure [Pa]
CP = 1004.0                # specific heat dry air [J/kg/K]

@dataclass
class ThermoConfig:
    cp: float = CP
    p_ref: float = P0       # reference pressure if using constant p
    use_p_from_M: bool = False
    M_ref: float | None = None   # if None, use domain mean M passed by caller


def esat_tetens(T: np.ndarray) -> np.ndarray:
    """
    Saturation vapour pressure over liquid water (Pa), Tetens (approx, 0-50°C).
    T in K. Formula using T_C in °C: e_s(hPa)=6.112 * exp(17.67 T_C / (T_C+243.5)).
    Returned in Pa.
    """
    T_C = np.clip(T - 273.15, -80.0, 60.0)
    es_hPa = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    return 100.0 * es_hPa


def qsat(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Saturation specific humidity at (T,p) with epsilon correction.
    qsat = eps*es / (p - (1-eps) es). Clipped to [0, 0.5].
    """
    es = esat_tetens(T)
    denom = np.maximum(p - (1.0 - EPS) * es, 1.0)
    q = EPS * es / denom
    return np.clip(q, 0.0, 0.5)


def latent_heat_vapor(T: np.ndarray) -> np.ndarray:
    """
    Latent heat of vaporization (J/kg), weak T dependence.
    Simple linear approx: Lv(T) ≈ 2.501e6 - 2.3e3 (T-273.15).
    """
    return 2.501e6 - 2.3e3 * (T - 273.15)


def pressure_from_M(M: np.ndarray, cfg: ThermoConfig, M_mean: float | None = None) -> np.ndarray:
    """
    Very simple diagnostic pressure from layer mass: p = p_ref * (M / M_ref).
    If cfg.use_p_from_M is False, returns uniform p_ref.
    """
    if not cfg.use_p_from_M:
        return np.full_like(M, cfg.p_ref, dtype=float)
    if cfg.M_ref is not None:
        Mref = cfg.M_ref
    else:
        Mref = float(M_mean if M_mean is not None else np.mean(M))
    Mref = max(Mref, 1e-12)
    return cfg.p_ref * (M / Mref)