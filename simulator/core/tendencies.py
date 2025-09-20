from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .state import State


@dataclass(slots=True)
class Tend:
    """Container of tendencies for each prognostic variable (same shapes as State)."""

    dM:  np.ndarray  # (ny, nx)
    dT:  np.ndarray  # (ny, nx)
    dqv: np.ndarray  # (ny, nx)
    dqc: np.ndarray  # (ny, nx)
    dqr: np.ndarray  # (ny, nx)
    dMU: np.ndarray  # (ny, nx+1)
    dMV: np.ndarray  # (ny+1, nx)

    @staticmethod
    def zeros_like(s: State) -> "Tend":
        zc = lambda: np.zeros_like(s.M)
        zu = lambda: np.zeros_like(s.MU)
        zv = lambda: np.zeros_like(s.MV)
        return Tend(dM=zc(), dT=zc(), dqv=zc(), dqc=zc(), dqr=zc(), dMU=zu(), dMV=zv())

    def copy(self) -> "Tend":
        return Tend(
            dM=self.dM.copy(), dT=self.dT.copy(), dqv=self.dqv.copy(), dqc=self.dqc.copy(), dqr=self.dqr.copy(),
            dMU=self.dMU.copy(), dMV=self.dMV.copy(),
        )

    def iadd(self, other: "Tend") -> "Tend":
        self.dM  += other.dM;  self.dT  += other.dT;  self.dqv += other.dqv
        self.dqc += other.dqc; self.dqr += other.dqr
        self.dMU += other.dMU; self.dMV += other.dMV
        return self

    def scaled(self, a: float) -> "Tend":
        return Tend(
            dM=a*self.dM, dT=a*self.dT, dqv=a*self.dqv, dqc=a*self.dqc, dqr=a*self.dqr,
            dMU=a*self.dMU, dMV=a*self.dMV,
        )


def sum_tendencies(terms: Iterable[Tend]) -> Tend:
    acc: Tend | None = None
    for t in terms:
        if acc is None:
            acc = t.copy()
        else:
            acc.iadd(t)
    assert acc is not None
    return acc


def advance_state(s: State, k: Tend, dt: float) -> State:
    """Return s + dt * k (pure, non in-place)."""
    return State(
        M=s.M + dt * k.dM,
        T=s.T + dt * k.dT,
        qv=s.qv + dt * k.dqv,
        qc=s.qc + dt * k.dqc,
        qr=s.qr + dt * k.dqr,
        MU=s.MU + dt * k.dMU,
        MV=s.MV + dt * k.dMV,
        Esfc=None if s.Esfc is None else (s.Esfc.copy()),
    )


def lincomb_states(a: float, sa: State, b: float, sb: State) -> State:
    """Return a*sa + b*sb (pure)."""
    return State(
        M=a*sa.M + b*sb.M,
        T=a*sa.T + b*sb.T,
        qv=a*sa.qv + b*sb.qv,
        qc=a*sa.qc + b*sb.qc,
        qr=a*sa.qr + b*sb.qr,
        MU=a*sa.MU + b*sb.MU,
        MV=a*sa.MV + b*sb.MV,
        Esfc=None if (sa.Esfc is None and sb.Esfc is None) else ((sa.Esfc if sa.Esfc is not None else 0.0) + (sb.Esfc if sb.Esfc is not None else 0.0)),
    )