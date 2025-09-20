import numpy as np

from simulator.physics import build_surface_params


def _catalog():
    return {
        "id_to_class": {
            "0": "ocean",
            "1": "ice",
            "2": "snow",
            "3": "forest",
            "4": "grass",
            "5": "desert",
            "6": "alpine",
        },
        "classes": {
            "ocean":  {"C_E": 1.2e-3, "qs_mode": "water", "qs_scale": 1.0},
            "ice":    {"C_E": 8.0e-4, "qs_mode": "ice",   "qs_scale": 1.0},
            "snow":   {"C_E": 6.0e-4, "qs_mode": "ice",   "qs_scale": 1.0},
            "forest": {"C_E": 7.0e-4, "qs_mode": "water", "qs_scale": 1.0},
            "grass":  {"C_E": 6.0e-4, "qs_mode": "water", "qs_scale": 1.0},
            "desert": {"C_E": 3.0e-4, "qs_mode": "dry",   "qs_scale": 0.3},
            "alpine": {"C_E": 4.0e-4, "qs_mode": "ice",   "qs_scale": 0.8},
            "land_generic": {"C_E": 5.0e-4, "qs_mode": "water", "qs_scale": 1.0},
        },
        "default_class": "land_generic",
        "ocean_class": "ocean",
    }


def test_surface_params_shapes_and_values():
    ny, nx = 4, 5
    terrain = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 3, 4, 0],
        [4, 3, 2, 1, 5],
        [6, 6, 5, 4, 3],
    ], dtype=np.int32)
    mask_water = (terrain == 0)

    params = build_surface_params(terrain, mask_water, _catalog())

    assert params.C_E.shape == (ny, nx)
    assert params.qs_mode.shape == (ny, nx)
    assert params.qs_scale.shape == (ny, nx)

    # Known checks
    assert np.isclose(params.C_E[0,0], 1.2e-3)  # ocean
    assert np.isclose(params.C_E[1,0], 3.0e-4)  # desert
    assert np.isclose(params.C_E[1,1], 4.0e-4)  # alpine
    # Ocean override wherever mask_water True (here only where terrain==0)
    assert np.allclose(params.C_E[mask_water], 1.2e-3)


def test_unknown_id_uses_default():
    terrain = np.array([[9, 9, 9],[0, 1, 2]], dtype=np.int32)
    mask_water = (terrain == 0)

    cat = _catalog()
    params = build_surface_params(terrain, mask_water, cat)

    # unknown id=9 should fall back to default_class (land_generic → C_E=5e-4)
    assert np.allclose(params.C_E[0,:], 5.0e-4)
    # known ids keep their values
    assert np.isclose(params.C_E[1,0], 1.2e-3)  # ocean
    assert np.isclose(params.C_E[1,1], 8.0e-4)  # ice