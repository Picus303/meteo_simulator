from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SurfaceFields:
    elevation: np.ndarray       # (ny,nx) [m]
    mask_water: np.ndarray      # (ny,nx) bool
    albedo: np.ndarray          # (ny,nx) [0,1]
    terrain_type: np.ndarray    # (ny,nx) int


def load_surface_static(static_dir: str | Path, expect_shape: tuple[int, int]) -> SurfaceFields:
    p = Path(static_dir)
    if not p.exists():
        raise FileNotFoundError(f"Static dir not found: {p}")

    def _load(name: str) -> np.ndarray:
        f = p / f"{name}.npy"
        if not f.exists():
            raise FileNotFoundError(f"Missing static map: {f}")
        arr = np.load(f)
        if arr.shape != expect_shape:
            raise ValueError(f"{name}.npy shape {arr.shape} != expected {expect_shape}")
        return arr

    elevation = _load("elevation").astype(np.float64)
    mask_water = _load("mask_water").astype(bool)
    albedo = _load("albedo").astype(np.float64)
    terrain_type = _load("terrain_type").astype(np.int32)

    if np.any(~np.isfinite(elevation)):
        raise ValueError("elevation contains non-finite values")
    if np.any((albedo < 0.0) | (albedo > 1.0)):
        raise ValueError("albedo out of range [0,1]")

    return SurfaceFields(
        elevation=elevation,
        mask_water=mask_water,
        albedo=albedo,
        terrain_type=terrain_type,
    )