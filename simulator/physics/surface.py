from __future__ import annotations
import numpy as np
from typing import Callable
from dataclasses import dataclass
from .thermo import CP, latent_heat_vapor
from ..core.state import State
from ..core.tendencies import Tend
from ..grid.staggering import to_u_centered, to_v_centered

@dataclass
class SurfaceConfig:
    CE: float = 1.5e-3            # bulk transfer coeff for moisture
    CH: float = 1.0e-3            # bulk transfer coeff for sensible heat (optional)
    use_sensible: bool = False
    evap_heats_air: bool = False  # if True, credit Lv*E to dT (usually False)
    cp: float = CP

@dataclass
class SurfaceInputs:
    ocean_mask: np.ndarray  # (ny, nx) 1 over ocean, 0 land
    qsat_func: Callable     # function (T,p)->qsat to allow reuse of thermo.qsat
    p_field: np.ndarray     # (ny, nx) pressure field to compute qsat at surface


def _center_wind_speed(state: State) -> np.ndarray:
    """Approximate 10m wind speed by center wind magnitude derived from face velocities.
    u_c ≈ average of neighboring U faces; same for v.
    """
    epsM = 1e-15 * float(np.mean(state.M) if state.M.size else 1.0)
    M_u = to_u_centered(state.M)
    M_v = to_v_centered(state.M)
    u = state.MU / np.maximum(M_u, epsM)  # (ny, nx+1)
    v = state.MV / np.maximum(M_v, epsM)  # (ny+1, nx)
    # to centers (simple averages)
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])
    v_c = 0.5 * (v[1:, :] + v[:-1, :])
    return np.sqrt(u_c * u_c + v_c * v_c)


def surface_evaporation_tendencies(state: State, sinp: SurfaceInputs, cfg: SurfaceConfig) -> Tend:
    """Compute evaporation (kg/kg/s on mixing ratios) and optional sensible heat.
    E (kg/m^2/s) ~ rho C_E U (q_sat - qv)_+ on ocean; here we operate in mixing-ratio tendencies:
    dqv = (E / M); dT optionally += +/- Lv * (E / (M cp)). By default, evaporation does NOT heat the air.
    """
    qsat_surf = sinp.qsat_func(state.T, sinp.p_field)
    U = _center_wind_speed(state)

    # Ensure shapes (ny,nx)
    assert U.shape == state.T.shape

    # Bulk flux as a simple proportionality (ρ absorbed in coefficient scaling)
    # We keep dimensions consistent by operating directly on mixing-ratio rates with an effective coefficient.
    # E/M ≈ C_E U (qsat - qv)_+ * mask
    evap_rate = cfg.CE * U * np.maximum(qsat_surf - state.qv, 0.0) * sinp.ocean_mask

    dqv = evap_rate
    dT = (-latent_heat_vapor(state.T) / cfg.cp) * evap_rate if not cfg.evap_heats_air else (+latent_heat_vapor(state.T) / cfg.cp) * evap_rate

    if cfg.use_sensible:
        # Simple sensible heat proportional to wind and (T_s - T). Here we don't have T_s; assume neutral (0) unless provided separately.
        # Placeholder: no sensible heat without a provided T_s; users can extend SurfaceInputs to include T_surf.
        pass

    zc = np.zeros_like(state.M)
    zu = np.zeros_like(state.MU)
    zv = np.zeros_like(state.MV)
    return Tend(dM=zc, dT=dT, dqv=dqv, dqc=zc, dqr=zc, dMU=zu, dMV=zv)