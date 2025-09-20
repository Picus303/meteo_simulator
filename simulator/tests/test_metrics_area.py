import numpy as np

from simulator.grid import make_latlon_grid, cell_areas

def test_area_sums_to_4piR2():
    R = 6.371e6
    g = make_latlon_grid(nlon=48, nlat=24, R=R, cap_deg=85.0)
    A = cell_areas(g)
    S = np.sum(A)
    assert np.isfinite(S)
    target = 4.0 * np.pi * R * R
    rel = abs(S - target) / target
    # Discrete exact for a regular lat-lon with our formula; tolerate tiny fp error
    assert rel < 1e-14