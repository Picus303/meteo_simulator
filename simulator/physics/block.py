from __future__ import annotations
import numpy as np
from typing import Callable
from dataclasses import dataclass, field
from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..grid.metrics import cell_areas
from ..diagnostics.water import PhysicsDiag
from .radiation import radiation_tendencies, RadiationConfig, RadiationInputs
from .surface import surface_evaporation_tendencies, SurfaceConfig, SurfaceInputs
from .microphysics import microphysics_tendencies, MicrophysicsConfig


@dataclass
class PhysicsConfig:
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    surface:   SurfaceConfig   = field(default_factory=SurfaceConfig)
    micro:     MicrophysicsConfig = field(default_factory=MicrophysicsConfig)
    clip_positive: bool = True


@dataclass
class PhysicsInputs:
    # Radiation
    albedo: np.ndarray  # (ny,nx)
    # Surface
    ocean_mask: np.ndarray  # (ny,nx)
    qsat_func: Callable
    p_surf: np.ndarray  # (ny,nx)


def _apply_tend_inplace(state: State, tend: Tend, dt: float, clip_positive: bool = True) -> None:
    """Forward-Euler apply tendencies to state in-place; optional positivity clip for q's."""
    state.M  = state.M  + dt * tend.dM
    state.T  = state.T  + dt * tend.dT
    state.qv = state.qv + dt * tend.dqv
    state.qc = state.qc + dt * tend.dqc
    state.qr = state.qr + dt * tend.dqr
    state.MU = state.MU + dt * tend.dMU
    state.MV = state.MV + dt * tend.dMV

    if clip_positive:
        state.qv = np.maximum(state.qv, 0.0)
        state.qc = np.maximum(state.qc, 0.0)
        state.qr = np.maximum(state.qr, 0.0)
        state.M  = np.maximum(state.M,  1e-30)  # ensure strictly positive mass


def physics_step_strang(state: State, grid: Grid, t_sec: float, dt: float,
                         orb_spin: tuple, cfg: PhysicsConfig, pin: PhysicsInputs,
                         diag: "PhysicsDiag | None" = None) -> tuple[State, "PhysicsDiag"]:
    """Advance physics by one full dt using Strang splitting.
    Returns (new_state, updated_diag). orb_spin is (orb, spin) for radiation.
    """
    orb, spin = orb_spin

    if diag is None:
        diag = PhysicsDiag.zeros_like(state)

    s = state.copy()

    # 1) ½ Surface
    sinp = SurfaceInputs(ocean_mask=pin.ocean_mask, qsat_func=pin.qsat_func, p_field=pin.p_surf)
    tend = surface_evaporation_tendencies(s, sinp, cfg.surface)
    _apply_tend_inplace(s, tend, 0.5 * dt, cfg.clip_positive)

    # 2) ½ Microphysics (collect precip diag)
    mtend, precip_rate = microphysics_tendencies(s, cfg.micro)
    _apply_tend_inplace(s, mtend, 0.5 * dt, cfg.clip_positive)
    diag.acc_precip[:, :] += precip_rate * (0.5 * dt)

    # 3) Radiation (full)
    rin = RadiationInputs(albedo=pin.albedo)
    rtend = radiation_tendencies(s, grid, t_sec, orb, spin, cfg.radiation, rin)
    _apply_tend_inplace(s, rtend, dt, cfg.clip_positive)

    # 4) ½ Microphysics (again)
    mtend2, precip_rate2 = microphysics_tendencies(s, cfg.micro)
    _apply_tend_inplace(s, mtend2, 0.5 * dt, cfg.clip_positive)
    diag.acc_precip[:, :] += precip_rate2 * (0.5 * dt)

    # 5) ½ Surface (again)
    tend2 = surface_evaporation_tendencies(s, sinp, cfg.surface)
    _apply_tend_inplace(s, tend2, 0.5 * dt, cfg.clip_positive)

    # Diagnostics: update global water change and precip
    A = cell_areas(grid)
    diag.update_water_budget(state, s, A)

    return s, diag