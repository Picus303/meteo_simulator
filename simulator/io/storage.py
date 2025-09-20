from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr
from numcodecs import Blosc

from ..core.state import State
from ..grid.sphere import Grid
from ..grid.loaders import SurfaceFields

ZARR_VERSION = 2  # for compatibility with xarray


# ── Helpers to build xarray Datasets ─────────────────────────────────────────

def _coords_for_grid(grid: Grid) -> Dict[str, xr.DataArray]:
    """
    Coordinate arrays *excluding* time (to avoid size conflicts on first append).
    We attach time with each snapshot; static coords are lat/lon and index axes.
    """
    coords = {
        "y": ("y", grid.lat_c * 0 + np.arange(grid.ny, dtype=np.int32)),
        "x": ("x", np.arange(grid.nx, dtype=np.int32)),
        "yv": ("yv", np.arange(grid.ny + 1, dtype=np.int32)),
        "xu": ("xu", np.arange(grid.nx + 1, dtype=np.int32)),
        "lat": ("y", grid.lat_c),
        "lon": ("x", grid.lon_c),
        "lat_v": ("yv", grid.lat_v),
        "lon_u": ("xu", grid.lon_u),
    }
    return {k: xr.DataArray(v[1], dims=(v[0],)) for k, v in coords.items()}


def _static_vars(static: SurfaceFields) -> Dict[str, xr.DataArray]:
    return {
        "elevation": xr.DataArray(static.elevation, dims=("y", "x")),
        "mask_water": xr.DataArray(static.mask_water.astype(np.int8), dims=("y", "x")),
        "albedo": xr.DataArray(static.albedo, dims=("y", "x")),
        "terrain_id": xr.DataArray(static.terrain_type.astype(np.int16), dims=("y", "x")),
    }


def _snapshot_from_state(state: State, time_value: np.datetime64) -> xr.Dataset:
    return xr.Dataset(
        {
            "M":  (("time", "y", "x"),  state.M[None, ...]),
            "T":  (("time", "y", "x"),  state.T[None, ...]),
            "qv": (("time", "y", "x"),  state.qv[None, ...]),
            "qc": (("time", "y", "x"),  state.qc[None, ...]),
            "qr": (("time", "y", "x"),  state.qr[None, ...]),
            "MU": (("time", "y", "xu"), state.MU[None, ...]),
            "MV": (("time", "yv", "x"), state.MV[None, ...]),
        },
        coords={"time": ("time", np.array([time_value], dtype="datetime64[ns]"))},
    )


# ── Storage backends ────────────────────────────────────────────────────────

@dataclass
class ZarrStorage:
    """
    Append-only Zarr writer for time-stepped States.
    Creates a Zarr directory (or reuses) with chunked arrays.
    The **first call** to `.append(state, time)` initializes the store by
    writing that snapshot with appropriate encoding; subsequent calls append
    along the `time` dimension.
    """

    path: Path
    grid: Grid
    static: SurfaceFields
    compressor: Optional[Blosc] = None
    chunks_xy: tuple[int, int] = (180, 360)  # tune per grid size

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        if self.compressor is None:
            self.compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    def _store_initialized(self) -> bool:
        return (self.path / ".zmetadata").exists() or (self.path / ".zarray").exists()

    def _encoding(self) -> dict:
        ny, nx = self.grid.ny, self.grid.nx
        chunks = {
            "time": 1,
            "y": self.chunks_xy[0],
            "x": self.chunks_xy[1],
            "xu": self.chunks_xy[1],  # same scale as x
            "yv": self.chunks_xy[0],  # same scale as y
        }
        enc = {v: {"chunks": (1, chunks["y"], chunks["x"]), "compressor": self.compressor}
               for v in ("M", "T", "qv", "qc", "qr")}
        enc["MU"] = {"chunks": (1, chunks["y"], chunks["xu"]), "compressor": self.compressor}
        enc["MV"] = {"chunks": (1, chunks["yv"], chunks["x"]), "compressor": self.compressor}
        return enc

    def append(self, state: State, time_value: np.datetime64) -> None:
        """Write one snapshot to Zarr. First call initializes the store."""
        ds = _snapshot_from_state(state, time_value)

        if not self._store_initialized():
            # First write: add coords and statics, and create the store schema via this snapshot
            coords = _coords_for_grid(self.grid)
            ds = ds.assign_coords(coords).assign(_static_vars(self.static))

            t0 = np.array(ds["time"].values[0], dtype="datetime64[ns]")
            units = f"hours since {str(t0)[:19]}"

            enc = self._encoding()
            enc["time"] = {"units": units, "dtype": "int64"}

            ds.to_zarr(str(self.path), mode="w", compute=True, encoding=enc, zarr_format=ZARR_VERSION)
            return

        # Subsequent appends
        ds.to_zarr(str(self.path), mode="a", append_dim="time", zarr_format=ZARR_VERSION)


@dataclass
class NPZCheckpoint:
    """
    Write compressed NPZ snapshots for restart.
    Keeps only the last K snapshots to limit disk usage.
    """

    outdir: Path
    keep_last: int = 4

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def save(self, state: State, step: int) -> Path:
        path = self.outdir / f"state_{step:08d}.npz"
        np.savez_compressed(
            path,
            M=state.M, T=state.T, qv=state.qv, qc=state.qc, qr=state.qr,
            MU=state.MU, MV=state.MV,
        )
        self._gc()
        return path

    def _gc(self) -> None:
        files = sorted(self.outdir.glob("state_*.npz"))
        if len(files) > self.keep_last:
            for f in files[:-self.keep_last]:
                try: f.unlink()
                except Exception: pass