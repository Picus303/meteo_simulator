from __future__ import annotations
import numpy as np
from ..core.state import State
from ..grid.sphere import Grid
from ..grid.metrics import cell_areas


def make_water_rescale_hook(grid: Grid, target: float | None = None, max_rel: float = 1e-6):
    """
    Return a hook that rescales (qv,qc,qr) to preserve total water ∑ A M (qv+qc+qr).
    If target is None, the first call records the baseline target and uses it afterwards.
    Scaling factor alpha is clipped to [1-max_rel, 1+max_rel] to avoid shocks.
    """
    A = cell_areas(grid)
    stateful = {"target": target}

    def hook(state: State, *args):
        s = state.copy()
        total = float(np.sum(A * s.M * (s.qv + s.qc + s.qr)))
        if stateful["target"] is None:
            stateful["target"] = total
            return s
        if total == 0.0:
            return s
        alpha = np.clip(stateful["target"] / total, 1.0 - max_rel, 1.0 + max_rel)
        s.qv *= alpha; s.qc *= alpha; s.qr *= alpha
        return s

    return hook