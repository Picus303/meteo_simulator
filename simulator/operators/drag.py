from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from ..grid.staggering import to_u_centered, to_v_centered


@dataclass
class DragConfig:
    r_u: float = 0.0
    r_v: float = 0.0
    heat: bool = False
    cp: float = 1004.0


def drag_tendencies(state: State, grid: Grid, cfg: DragConfig = DragConfig()) -> Tend:
    z = np.zeros_like(state.M)
    dMU = -cfg.r_u * state.MU
    dMV = -cfg.r_v * state.MV
    dT = np.zeros_like(state.M)
    if cfg.heat and (cfg.r_u != 0.0 or cfg.r_v != 0.0):
        eps = 1e-15 * float(np.mean(state.M) if state.M.size else 1.0)
        M_u = to_u_centered(state.M); M_v = to_v_centered(state.M)
        u_f = state.MU / np.maximum(M_u, eps)
        v_f = state.MV / np.maximum(M_v, eps)
        u_c = 0.5 * (u_f[:, 1:] + u_f[:, :-1])
        v_c = 0.5 * (v_f[1:, :] + v_f[:-1, :])
        dT = dT + (cfg.r_u * (u_c * u_c) + cfg.r_v * (v_c * v_c)) / cfg.cp
    return Tend(dM=z, dT=dT, dqv=z, dqc=z, dqr=z, dMU=dMU, dMV=dMV)