from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..numerics.operators import grad_half_g_h2_to_UV


@dataclass
class PressureConfig:
    g: float = 9.81
    rho_ref: float = 1.2
    cap_zero_u: bool = True


def pressure_tendencies_energy_safe(state: State, grid: Grid, cfg: PressureConfig = PressureConfig()) -> Tend:
    """Energy-friendly pressure gradient: dMU = -∂x(½gh²), dMV = -∂y(½gh²) on faces.

    Uses metric-aware gradients built from the same center spacings used in continuity.
    Applies cap-row masking consistently on U faces (circumference→0).
    """
    Gx_U, Gy_V = grad_half_g_h2_to_UV(state.M, grid, cfg.g, cfg.rho_ref)
    dMU = -Gx_U
    dMV = -Gy_V
    if cfg.cap_zero_u and getattr(grid, "cap_rows", None) is not None and np.any(grid.cap_rows):
        dMU = dMU.copy(); dMU[grid.cap_rows, :] = 0.0
    zc = np.zeros_like(state.M)
    zu = np.zeros_like(state.MU)
    zv = np.zeros_like(state.MV)
    return Tend(dM=zc, dT=zc, dqv=zc, dqc=zc, dqr=zc, dMU=dMU, dMV=dMV)


@dataclass
class PressureTerm:
    name: str
    grid: Grid
    cfg: PressureConfig

    def __call__(self, state: State, t: float) -> Tend:
        return pressure_tendencies_energy_safe(state, self.grid, self.cfg)


def make_pressure_term(grid: Grid, *, g: float = 9.81, rho_ref: float = 1.2, cap_zero_u: bool = True) -> PressureTerm:
    return PressureTerm(name="pressure", grid=grid, cfg=PressureConfig(g=g, rho_ref=rho_ref, cap_zero_u=cap_zero_u))