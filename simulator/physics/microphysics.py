from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from ..core.state import State
from ..core.tendencies import Tend
from .thermo import ThermoConfig, qsat, latent_heat_vapor, pressure_from_M


@dataclass
class MicrophysicsConfig:
    thermo: ThermoConfig = field(default_factory=ThermoConfig)
    tau_cond: float = 300.0        # s, relaxation time towards saturation when supersaturated
    tau_reevap: float = 600.0      # s, re-evap timescale when subsaturated and qc>0
    qcrit: float = 1.0e-3          # cloud water threshold for autoconversion
    autoconv_rate: float = 1.0e-3  # s^-1, (qc - qcrit)_+
    accretion_rate: float = 1.0e-3 # s^-1, qr*qc coefficient
    tau_fall: float = 3600.0       # s, rainfall fallout timescale (remove qr)
    limit_positive: bool = True


def microphysics_tendencies(state: State, cfg: MicrophysicsConfig) -> tuple[Tend, np.ndarray]:
    """Local Kessler-like 1-layer microphysics.
    Returns (Tend, precip_rate) where precip_rate is kg m^-2 s^-1 (column), i.e.,
    the mass flux of rain leaving the column (diagnostic for budgets).
    """
    # Thermo fields
    p = pressure_from_M(state.M, cfg.thermo)
    qs = qsat(state.T, p)
    Lv = latent_heat_vapor(state.T)

    epsM = 1e-15 * float(np.mean(state.M) if state.M.size else 1.0)
    M_safe = np.maximum(state.M, epsM)

    # Supersaturation/undersaturation
    supersat = np.maximum(state.qv - qs, 0.0)
    subsat = np.maximum(qs - state.qv, 0.0)

    # Phase-change rates (mixing ratio per second)
    cond_rate = supersat / max(cfg.tau_cond, 1e-9)
    # Re-evap limited by available cloud water
    reevap_potential = subsat / max(cfg.tau_reevap, 1e-9)
    reevap_rate = np.minimum(reevap_potential, state.qc / max(cfg.tau_reevap, 1e-9))

    # Autoconversion & accretion (to rain)
    autoconv = cfg.autoconv_rate * np.maximum(state.qc - cfg.qcrit, 0.0)
    accretion = cfg.accretion_rate * np.maximum(state.qc, 0.0) * np.maximum(state.qr, 0.0)

    # Fallout (remove qr from air column)
    fallout = state.qr / max(cfg.tau_fall, 1e-9)
    precip_rate = fallout * M_safe  # kg m^-2 s^-1

    # Tendencies on mixing ratios
    dqv = -cond_rate + reevap_rate
    dqc = +cond_rate - reevap_rate - autoconv - accretion
    dqr = +autoconv + accretion - fallout

    # Latent heating: +Lv for condensation, -Lv for re-evaporation
    dT = (Lv / cfg.thermo.cp) * (cond_rate - reevap_rate)

    # Positivity limiter on tendencies (optional soft-clip on dq to avoid driving below zero in one step)
    if cfg.limit_positive:
        # This limiter does not change conservation among phases except fallout
        dqv = np.where((state.qv <= 0.0) & (dqv < 0.0), 0.0, dqv)
        dqc = np.where((state.qc <= 0.0) & (dqc < 0.0), 0.0, dqc)
        dqr = np.where((state.qr <= 0.0) & (dqr < 0.0), 0.0, dqr)

    zc = np.zeros_like(state.M)
    zu = np.zeros_like(state.MU)
    zv = np.zeros_like(state.MV)
    tend = Tend(dM=zc, dT=dT, dqv=dqv, dqc=dqc, dqr=dqr, dMU=zu, dMV=zv)
    return tend, precip_rate