import numpy as np
from ..core.state import State


def shapiro_121_i(A: np.ndarray) -> np.ndarray:
    left = np.roll(A, 1, axis=1)   # colonne i-1
    right = np.roll(A, -1, axis=1) # colonne i+1
    return 0.25 * left + 0.5 * A + 0.25 * right


def shapiro_121_j(A: np.ndarray) -> np.ndarray:
    top = np.vstack([A[0:1, :], A[:-1, :]])   # clamp bord haut
    bot = np.vstack([A[1:, :], A[-1:, :]])    # clamp bord bas
    return 0.25 * top + 0.5 * A + 0.25 * bot


def make_shapiro_hook(fields=("T", "qv", "qc", "qr"), passes_i=1, passes_j=1):
    def hook(state: State, *args):
        s = state.copy()
        for name in fields:
            A = getattr(s, name)
            for _ in range(passes_i):
                A = shapiro_121_i(A)
            for _ in range(passes_j):
                A = shapiro_121_j(A)
            setattr(s, name, A)
        return s
    return hook