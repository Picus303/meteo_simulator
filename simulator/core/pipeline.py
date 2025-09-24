from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal

import numpy as np

from .state import State
from .tendencies import Tend
from ..grid.metrics import build_cap_row_weights, enforce_cap_mean, cell_areas
from .steppers import PostHook
from ..grid.sphere import Grid


class Term(Protocol):
    """RHS term interface: must provide __call__(state, t) -> Tend."""
    name: str
    def __call__(self, state: State, t: float) -> Tend: ...


@dataclass
class RHSComposer:
    """Compose multiple independent RHS terms by summation of tendencies."""
    terms: list[Term]

    def __call__(self, state: State, t: float) -> Tend:
        if not self.terms:
            return Tend.zeros_like(state)
        
        # Start with the first term's tendency
        total_tend = self.terms[0](state, t).copy()
        
        # Add all subsequent terms using iadd method
        for term in self.terms[1:]:
            total_tend.iadd(term(state, t))
        
        return total_tend


# ── Post-step hooks ─────────────────────────────────────────────────────────

def make_positivity_clip(eps_q: float = 0.0, eps_M_rel: float = 1e-12) -> PostHook:
    def _clip(s: State, _t: float, _dt: float, _when: Literal["stage", "final"]) -> State:
        out = s.copy()
        out.qv = np.maximum(out.qv, eps_q)
        out.qc = np.maximum(out.qc, eps_q)
        out.qr = np.maximum(out.qr, eps_q)
        meanM = float(np.mean(out.M))
        out.M = np.maximum(out.M, eps_M_rel * meanM)
        return out
    return _clip


def make_cap_projection(grid: Grid) -> PostHook:
    # Precompute weights once
    A = cell_areas(grid)
    W = build_cap_row_weights(grid, A)
    def _cap(s: State, _t: float, _dt: float, _when: Literal["stage", "final"]) -> State:
        out = s.copy()
        out.M  = enforce_cap_mean(out.M,  grid, W)
        out.T  = enforce_cap_mean(out.T,  grid, W)
        out.qv = enforce_cap_mean(out.qv, grid, W)
        out.qc = enforce_cap_mean(out.qc, grid, W)
        out.qr = enforce_cap_mean(out.qr, grid, W)
        return out
    return _cap