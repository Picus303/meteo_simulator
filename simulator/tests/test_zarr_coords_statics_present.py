import numpy as np
import xarray as xr

from simulator.core.state import State
from simulator.grid.sphere import make_latlon_grid
from simulator.grid.loaders import SurfaceFields
from simulator.io.storage import ZarrStorage


def _dummy_static(ny, nx):
    elev = np.zeros((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    alb  = np.full((ny, nx), 0.2)
    terr = np.zeros((ny, nx), dtype=np.int32)
    return SurfaceFields(elevation=elev, mask_water=mask, albedo=alb, terrain_type=terr)


def test_zarr_has_coords_and_static_vars(tmp_path):
    g = make_latlon_grid(12, 6, 6.371e6, cap_deg=85.0)
    zs = ZarrStorage(tmp_path / "out.zarr", g, _dummy_static(g.ny, g.nx))
    st = State.zeros(g.ny, g.nx)

    zs.append(st, np.datetime64("2025-01-01T00:00:00"))
    ds = xr.open_zarr(tmp_path / "out.zarr", chunks=None)

    # Coordinates present
    for c in ("lat", "lon", "lat_v", "lon_u"):
        assert c in ds.coords

    # Static maps present with expected dtypes
    assert "terrain_id" in ds.variables
    assert str(ds["terrain_id"].dtype) in ("int16", ">i2", "<i2")
    assert "mask_water" in ds.variables
    assert str(ds["mask_water"].dtype) in ("int8", ">i1", "<i1")