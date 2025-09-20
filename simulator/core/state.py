from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class State:
    """Prognostic state on a C-grid (1-layer barotrope + moisture).

    Shapes follow Step 1 grid conventions:
      - Cell centers: (ny, nx)
      - U faces     : (ny, nx+1)
      - V faces     : (ny+1, nx)
    """

    # Centers
    M:  np.ndarray  # mass column (kg m^-2) or thickness h*ρref
    T:  np.ndarray  # temperature (K) or potential temperature
    qv: np.ndarray  # water vapor mixing ratio (kg/kg)
    qc: np.ndarray  # cloud water mixing ratio (kg/kg)
    qr: np.ndarray  # rain water mixing ratio  (kg/kg)

    # Faces (momenta on staggers)
    MU: np.ndarray  # (ny, nx+1)
    MV: np.ndarray  # (ny+1, nx)

    # Optional diagnostic reservoir (surface energy, etc.)
    Esfc: Optional[np.ndarray] = None  # (ny, nx)

    def copy(self) -> "State":
        return State(
            M=self.M.copy(), T=self.T.copy(), qv=self.qv.copy(), qc=self.qc.copy(), qr=self.qr.copy(),
            MU=self.MU.copy(), MV=self.MV.copy(), Esfc=None if self.Esfc is None else self.Esfc.copy(),
        )

    def is_finite(self) -> bool:
        arrs = [self.M, self.T, self.qv, self.qc, self.qr, self.MU, self.MV]
        if self.Esfc is not None: arrs.append(self.Esfc)
        return all(np.isfinite(a).all() for a in arrs)

    def is_physical(self, eps: float = 0.0) -> bool:
        ok = (self.M > eps).all() and (self.qv >= -1e-16).all() and (self.qc >= -1e-16).all() and (self.qr >= -1e-16).all()
        return bool(ok)

    @staticmethod
    def zeros(ny: int, nx: int, *, T0: float = 280.0) -> "State":
        M  = np.ones((ny, nx), dtype=np.float64)
        T  = np.full((ny, nx), T0, dtype=np.float64)
        qv = np.zeros((ny, nx), dtype=np.float64)
        qc = np.zeros((ny, nx), dtype=np.float64)
        qr = np.zeros((ny, nx), dtype=np.float64)
        MU = np.zeros((ny, nx + 1), dtype=np.float64)
        MV = np.zeros((ny + 1, nx), dtype=np.float64)
        return State(M=M, T=T, qv=qv, qc=qc, qr=qr, MU=MU, MV=MV, Esfc=None)
