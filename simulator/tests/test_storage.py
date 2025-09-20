import numpy as np
import xarray as xr

from simulator.core.state import State
from simulator.grid.sphere import make_latlon_grid
from simulator.grid.loaders import SurfaceFields
from simulator.io.storage import ZarrStorage, NPZCheckpoint


def _dummy_static(ny, nx):
    import numpy as np
    elev = np.zeros((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    alb  = np.full((ny, nx), 0.2)
    terr = np.zeros((ny, nx), dtype=np.int32)
    return SurfaceFields(elevation=elev, mask_water=mask, albedo=alb, terrain_type=terr)


def test_zarr_append(tmp_path):
    g = make_latlon_grid(16, 8, 6.371e6, cap_deg=85.0)
    st = State.zeros(g.ny, g.nx)
    zs = ZarrStorage(tmp_path / "data.zarr", g, _dummy_static(g.ny, g.nx))

    zs.append(st, np.datetime64("2025-01-01T00:00:00"))
    st2 = st.copy(); st2.T += 1.0
    zs.append(st2, np.datetime64("2025-01-01T01:00:00"))

    ds = xr.open_zarr(tmp_path / "data.zarr", chunks=None)
    assert ds.sizes["time"] == 2
    assert np.isclose(ds["T"].isel(time=1).mean().item(), 281.0)


def test_npz_checkpoint(tmp_path):
    st = State.zeros(4, 5)
    cp = NPZCheckpoint(tmp_path / "ckpt", keep_last=2)
    p1 = cp.save(st, step=1)
    st2 = st.copy(); st2.M += 1.0
    p2 = cp.save(st2, step=2)
    st3 = st.copy(); st3.M += 2.0
    p3 = cp.save(st3, step=3)

    # only last 2 kept
    files = sorted((tmp_path / "ckpt").glob("state_*.npz"))
    assert len(files) == 2 and files[0].name.endswith("00000002.npz") and files[1].name.endswith("00000003.npz")