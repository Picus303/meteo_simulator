from __future__ import annotations

from dataclasses import dataclass

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from .momentum import advect_momentum, MomAdvConfig


@dataclass
class MomentumAdvectionTerm:
    name: str
    grid: Grid
    cfg: MomAdvConfig

    def __call__(self, state: State, t: float) -> Tend:
        return advect_momentum(state, self.grid, self.cfg)


def make_momentum_advection_term(grid: Grid, *, eps_mass_rel: float = 1e-12) -> MomentumAdvectionTerm:
    return MomentumAdvectionTerm(name="momentum_advection", grid=grid, cfg=MomAdvConfig(eps_mass_rel=eps_mass_rel))