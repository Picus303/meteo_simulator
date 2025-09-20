from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np

from .config import RunConfig
from .logging_utils import get_logger, log_kv
from .grid.loaders import load_surface_static
from .physics import build_surface_params


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="meteo",
        description="Simplified weather simulation sandbox (Step 0)",
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (constants, numerics, physics, outputs)",
    )
    p.add_argument(
        "--staticdir",
        type=str,
        required=True,
        help="Path to directory with static .npy maps (elevation, mask_water, albedo, terrain_type)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for run artifacts",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logger = get_logger("meteo")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig.load(args.config)
    log_kv(logger, "config_loaded", path=str(args.config))

    staticdir = Path(args.staticdir)
    if not staticdir.exists():
        raise FileNotFoundError(f"Static data dir not found: {staticdir}")
    log_kv(logger, "static_loaded", path=str(staticdir))

    # Determine expected shape from elevation.npy for convenience
    elev_path = Path(args.staticdir) / "elevation.npy"
    if not elev_path.exists():
        raise FileNotFoundError(f"Missing elevation.npy in {args.staticdir}")
    expect_shape = tuple(np.load(elev_path).shape)

    # Load static maps with shape checks
    static = load_surface_static(args.staticdir, expect_shape)
    log_kv(logger, "static_shapes", shape=list(expect_shape))

    # Build per-cell surface parameters from catalog (if provided)
    catalog = cfg.physics.get("surface_catalog", {}) if cfg else {}
    if catalog:
        surf_params = build_surface_params(static.terrain_type, static.mask_water, catalog)
        # Minimal diagnostic: report a few stats
        log_kv(
            logger,
            "surface_params_built",
            C_E_min=float(np.min(surf_params.C_E)),
            C_E_max=float(np.max(surf_params.C_E)),
        )
    else:
        log_kv(logger, "surface_params_skipped", reason="no surface_catalog in YAML")

    # Placeholder run
    log_kv(
        logger,
        "run_start",
        outdir=str(outdir),
        staticdir=str(staticdir),
    )

    # Example: metadata file to prove run executed
    meta_path = outdir / "run_meta.json"
    meta_path.write_text(
        '{"status":"initialized"}',
        encoding="utf-8",
    )

    log_kv(logger, "run_end", outputs=[str(meta_path)])


if __name__ == "__main__":
    main()