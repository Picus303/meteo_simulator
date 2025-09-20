from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

# Internal enum for q*_surf mode (kept simple & compact)
_QS_MODE = {"water": 0, "ice": 1, "dry": 2}


@dataclass(slots=True)
class SurfaceParams:
    """Per-cell surface parameters derived from integer terrain IDs.

    Attributes
    ----------
    C_E : (ny,nx) float64
        Bulk latent exchange coefficient (nondimensional).
    qs_mode : (ny,nx) uint8
        Encodes which saturation to use at the surface: 0=water, 1=ice, 2=dry.
    qs_scale : (ny,nx) float64
        Optional multiplicative factor on q*_surf (default 1.0).
    terrain_id : (ny,nx) int32
        Original terrain ID map, for diagnostics.
    """

    C_E: np.ndarray
    qs_mode: np.ndarray
    qs_scale: np.ndarray
    terrain_id: np.ndarray


def _resolve_class_params(classes: Mapping[str, Any], cls: str) -> Dict[str, Any]:
    p = dict(classes.get(cls, {}))
    if not p:
        raise KeyError(f"Surface class '{cls}' missing in catalog 'classes'.")
    # Defaults
    p.setdefault("C_E", 5.0e-4)
    p.setdefault("qs_mode", "water")
    p.setdefault("qs_scale", 1.0)
    # Validate
    if p["qs_mode"] not in _QS_MODE:
        raise ValueError(f"qs_mode must be one of {list(_QS_MODE)}, got {p['qs_mode']!r}")
    return p


def build_surface_params(
    terrain_id: np.ndarray,
    mask_water: Optional[np.ndarray],
    catalog: Mapping[str, Any],
) -> SurfaceParams:
    """Vectorized mapping from terrain_id → (C_E, qs_mode, qs_scale).

    Parameters
    ----------
    terrain_id : (ny,nx) int array
        Integer codes as provided by your static .npy (e.g., 0..6).
    mask_water : (ny,nx) bool array or None
        If provided, cells with True are forced to the *ocean* class.
    catalog : dict
        Loaded from YAML: {
          'id_to_class': { '0': 'ocean', '1': 'ice', ... },
          'classes': { 'ocean': {...}, 'ice': {...}, ... },
          'default_class': 'land_generic' (optional),
          'ocean_class': 'ocean' (optional)
        }

    Returns
    -------
    SurfaceParams
        Per-cell rasters ready for physics modules.
    """
    if terrain_id.ndim != 2:
        raise ValueError("terrain_id must be 2D (ny,nx)")

    id_to_class: Dict[int, str] = {}
    raw_map = catalog.get("id_to_class", {})
    for k, v in raw_map.items():
        id_to_class[int(k)] = str(v)

    classes: Mapping[str, Any] = catalog.get("classes", {})
    if not classes:
        raise KeyError("Catalog missing 'classes' section")

    default_cls: Optional[str] = catalog.get("default_class")
    ocean_cls: str = str(catalog.get("ocean_class", "ocean"))
    if ocean_cls not in classes:
        # Not fatal, only matters if mask_water is provided
        pass

    ny, nx = terrain_id.shape
    C_E = np.empty((ny, nx), dtype=np.float64)
    qs_mode = np.empty((ny, nx), dtype=np.uint8)
    qs_scale = np.empty((ny, nx), dtype=np.float64)

    # Start with default class everywhere (if provided), else use first class available
    if default_cls is not None:
        p_def = _resolve_class_params(classes, default_cls)
    else:
        # Pick the first class deterministically
        first_cls = sorted(classes.keys())[0]
        p_def = _resolve_class_params(classes, first_cls)

    C_E.fill(float(p_def["C_E"]))
    qs_mode.fill(_QS_MODE[p_def["qs_mode"]])
    qs_scale.fill(float(p_def["qs_scale"]))

    # Apply each known id mapping
    for k_id, cls_name in id_to_class.items():
        mask = (terrain_id == k_id)
        if not np.any(mask):
            continue
        p = _resolve_class_params(classes, cls_name)
        C_E[mask] = float(p["C_E"]) 
        qs_mode[mask] = _QS_MODE[p["qs_mode"]]
        qs_scale[mask] = float(p["qs_scale"]) 

    # Ocean override by mask_water (if provided and ocean exists)
    if mask_water is not None and ocean_cls in classes:
        ow = mask_water.astype(bool)
        if np.any(ow):
            p_o = _resolve_class_params(classes, ocean_cls)
            C_E[ow] = float(p_o["C_E"]) 
            qs_mode[ow] = _QS_MODE[p_o["qs_mode"]]
            qs_scale[ow] = float(p_o["qs_scale"]) 

    return SurfaceParams(C_E=C_E, qs_mode=qs_mode, qs_scale=qs_scale, terrain_id=terrain_id.astype(np.int32))