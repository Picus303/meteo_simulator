from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WaterBudgetTracker:
    """Track open-system atmospheric water budget across steps.

    Keeps cumulative predicted atmospheric water W_hat(t) based on integrated
    net sources/sinks, and lets you compare to the actual water content.
    """

    area: np.ndarray  # (ny,nx)
    dt: float
    W0: float  # initial atmospheric water at t0

    def __post_init__(self) -> None:
        self.W_hat = float(self.W0)
        self.ticks = 0

    def step(
        self,
        E: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        eps: Optional[np.ndarray] = None,
        Er: Optional[np.ndarray] = None
    ) -> float:
        """
        Advance the predicted water budget by one step.
        E, P, eps, Er are per-cell rates in kg m^-2 s^-1 *on the column mass*. If your
        model uses mixing ratios tendencies, convert before passing here.
        Missing terms are treated as zeros.
        Returns the integrated net flux B_w for this step (kg s^-1 domain-integrated).
        """
        E_arr = 0.0 if E is None else E
        P_arr = 0.0 if P is None else P
        eps_arr = 0.0 if eps is None else eps
        Er_arr = 0.0 if Er is None else Er
        # rates are per unit area of column mass; multiply by area to get kg s^-1 per cell
        net = (E_arr - P_arr - eps_arr + Er_arr)
        Bw = float(np.sum(net * self.area))
        self.W_hat += Bw * float(self.dt)
        self.ticks += 1
        return Bw

    def predicted(self) -> float:
        return float(self.W_hat)
