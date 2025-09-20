from __future__ import annotations

import numpy as np


def wrap_i(i: int, nx: int) -> int:
    return i % nx


def neigh_i(i: int, nx: int) -> tuple[int, int, int]:
    return ((i - 1) % nx, i % nx, (i + 1) % nx)


def neigh_j(j: int, ny: int) -> tuple[int, int, int]:
    jm1 = max(j - 1, 0)
    jp1 = min(j + 1, ny - 1)
    return (jm1, j, jp1)


def pad_halo(arr: np.ndarray, halo: int = 1, periodic_i: bool = True) -> np.ndarray:
    if halo <= 0:
        return arr.copy()
    left = arr[:, -halo:] if periodic_i else arr[:, :1].repeat(halo, axis=1)
    right = arr[:, :halo] if periodic_i else arr[:, -1:].repeat(halo, axis=1)
    core = np.concatenate([left, arr, right], axis=1)
    top = core[0:1, :].repeat(halo, axis=0)
    bottom = core[-1:, :].repeat(halo, axis=0)
    return np.concatenate([top, core, bottom], axis=0)