from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.state import State
from ..grid.metrics import cell_areas


@dataclass
class PhysicsDiag:
    acc_precip: np.ndarray       # (ny,nx) accumulated precip (kg m^-2)
    dW_atmos: float = 0.0        # global change of atmospheric water (kg s^-1-equivalent over the step)

    @staticmethod
    def zeros_like(state: State) -> "PhysicsDiag":
        return PhysicsDiag(acc_precip=np.zeros_like(state.M))

    def reset_step(self):
        self.dW_atmos = 0.0

    def update_water_budget(self, s0: State, s1: State, A: np.ndarray) -> None:
        """Compute change of total atmospheric water (qv+qc+qr) times M and area.
        Store as absolute kg (not per second).
        """
        W0 = np.sum(A * s0.M * (s0.qv + s0.qc + s0.qr))
        W1 = np.sum(A * s1.M * (s1.qv + s1.qc + s1.qr))
        self.dW_atmos = float(W1 - W0)

    def global_precip_kg(self, A: np.ndarray) -> float:
        return float(np.sum(A * self.acc_precip))