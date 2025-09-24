from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..grid.metrics import cell_areas
from .orbit import KeplerOrbit, SpinConfig, mu0_field, orbit_radius

SIGMA = 5.670374419e-8

@dataclass
class RadiationConfig:
    S0: float = 1361.0           # solar constant at r=a (W/m^2)
    tau_sw: float = 0.7          # shortwave transmissivity
    emissivity: float = 0.9      # LW emissivity
    Tref_lw: float = 260.0       # reference temperature for linearization (K)
    use_clouds: bool = True
    cloud_alpha: float = 0.5     # additional albedo factor of clouds
    cloud_beta: float = 1500.0   # sensitivity from qc to cloud fraction
    couple_to_M: bool = True     # enable thickness relaxation by insolation proxy
    tau_relax_hours: float = 24.0
    kappa_M: float = 0.2         # dimensionless amplitude for M_eq perturbation
    cp: float = 1004.0


def _cloud_fraction_from_qc(qc: np.ndarray, beta: float) -> np.ndarray:
    c = 1.0 - np.exp(-beta * np.clip(qc, 0.0, None))
    return np.clip(c, 0.0, 1.0)


def _q_sw(mu0: np.ndarray, albedo: np.ndarray, cloud_frac: np.ndarray, S_t: float, rcfg: RadiationConfig) -> np.ndarray:
    alpha_eff = albedo if not rcfg.use_clouds else np.clip(albedo + rcfg.cloud_alpha * cloud_frac * (1.0 - albedo), 0.0, 1.0)
    return S_t * mu0 * (1.0 - alpha_eff) * rcfg.tau_sw


def _q_lw(T: np.ndarray, rcfg: RadiationConfig) -> np.ndarray:
    lam = 4.0 * SIGMA * rcfg.emissivity * (rcfg.Tref_lw ** 3)
    return -lam * (T - rcfg.Tref_lw)


@dataclass
class RadiationInputs:
    albedo: np.ndarray  # (ny, nx)


def radiation_tendencies(state: State, grid: Grid, t_sec: float,
                          orb: KeplerOrbit, spin: SpinConfig,
                          rcfg: RadiationConfig, rin: RadiationInputs) -> Tend:
    """
    Compute radiative tendencies and optional thickness relaxation for general orbit/spin.
    Returns Tend with dT (radiative heating / cp) and dM if couple_to_M.
    """
    mu0 = mu0_field(grid, t_sec, orb, spin)            # (ny,nx)
    cloud_frac = _cloud_fraction_from_qc(state.qc, rcfg.cloud_beta) if rcfg.use_clouds else np.zeros_like(state.qc)

    r = orbit_radius(t_sec, orb)
    S_t = rcfg.S0 * (orb.a / r) ** 2

    Qsw = _q_sw(mu0, rin.albedo, cloud_frac, S_t, rcfg)
    Qlw = _q_lw(state.T, rcfg)
    Qnet = Qsw + Qlw

    eps = 1e-15 * float(np.mean(state.M) if state.M.size else 1.0)
    dT = Qnet / (np.maximum(state.M, eps) * rcfg.cp)

    # Thickness relaxation (zero-mean): dM = (κ M̄ (mu0 - mū)) / τ
    zM = np.zeros_like(state.M)
    if rcfg.couple_to_M and rcfg.tau_relax_hours > 0.0 and rcfg.kappa_M != 0.0:
        A = cell_areas(grid)
        Mbar = float(np.sum(A * state.M) / np.sum(A))
        mu_bar = float(np.sum(A * mu0) / np.sum(A))
        dM = rcfg.kappa_M * Mbar * (mu0 - mu_bar)
        tau = rcfg.tau_relax_hours * 3600.0
        dM = dM / tau
        dM = dM - (np.sum(A * dM) / np.sum(A))
    else:
        dM = zM

    zc = np.zeros_like(state.M)
    zu = np.zeros_like(state.MU)
    zv = np.zeros_like(state.MV)
    return Tend(dM=dM, dT=dT, dqv=zc, dqc=zc, dqr=zc, dMU=zu, dMV=zv)