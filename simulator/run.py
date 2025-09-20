from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunConfig
from .logging_utils import get_logger, log_kv


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