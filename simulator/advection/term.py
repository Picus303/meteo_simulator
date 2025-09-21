from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.state import State
from ..core.tendencies import Tend
from ..grid.sphere import Grid
from .fluxform import advect_mass_and_tracers, AdvecConfig


@dataclass
class AdvectionTerm:
    name: str
    grid: Grid
    tracer_names: Sequence[str]
    cfg: AdvecConfig

    def __call__(self, state: State, t: float) -> Tend:
        tr_map = {k: getattr(state, k) for k in self.tracer_names}
        return advect_mass_and_tracers(state, self.grid, tr_map, self.cfg)


def make_advection_term(grid: Grid, tracer_names: Sequence[str] = ("T","qv","qc","qr"), *, eps_mass_rel: float = 1e-12, cap_zero_u: bool = True) -> AdvectionTerm:
    return AdvectionTerm(name="advection", grid=grid, tracer_names=tuple(tracer_names), cfg=AdvecConfig(eps_mass_rel=eps_mass_rel, cap_zero_u=cap_zero_u))