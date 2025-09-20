import numpy as np
import pytest

from simulator.physics.surface_registry import build_surface_params


def _catalog_min():
    return {
        "id_to_class": {"0": "ocean"},
        "classes": {
            "ocean": {"C_E": 1.0e-3, "qs_mode": "water", "qs_scale": 1.0},
            "land_generic": {"C_E": 5.0e-4, "qs_mode": "water", "qs_scale": 1.0},
        },
        "default_class": "land_generic",
        "ocean_class": "ocean",
    }


def test_unknown_id_falls_back_to_default():
    terr = np.array([[9, 0], [9, 9]], dtype=np.int32)
    params = build_surface_params(terr, mask_water=None, catalog=_catalog_min())
    # default C_E = 5e-4 on id=9, and ocean on id=0
    assert np.isclose(params.C_E[0, 1], 1.0e-3)
    assert np.allclose(params.C_E[[0,1],[0,0]], 5.0e-4)


def test_invalid_qs_mode_raises():
    cat = _catalog_min()
    cat["classes"]["ocean"]["qs_mode"] = "invalid"
    terr = np.array([[0]])
    with pytest.raises(ValueError):
        _ = build_surface_params(terr, None, cat)