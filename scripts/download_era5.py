"""CLI entry point: download_era5.

Requests the ERA5 input frames one case needs from the CDS. Frames are cached
by timestamp in a directory shared across cases, so the frame the two Krovanh
cases have in common is only retrieved once.

Usage:
    uv run python scripts/download_era5.py --config configs/krovanh.yaml --case init-2026-08-31T18
    uv run python scripts/download_era5.py --all-cases
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wn2_typhoon.config import get_case, load_raw
from wn2_typhoon.data.era5_download import download_case
from wn2_typhoon.model_spec import load_spec
from wn2_typhoon.utils.logs import configure

logger = configure("download_era5")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="case id from the config")
    group.add_argument(
        "--all-cases", action="store_true", help="every case in the config"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/era5"))
    parser.add_argument(
        "--force", action="store_true", help="re-download frames already present"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="keep third-party INFO logs"
    )
    return parser.parse_args()


def main() -> None:
    """Download the ERA5 frames for one case or for all of them."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    cfg = load_raw(args.config)
    spec = load_spec(cfg["model"]["name"])

    case_ids = (
        [entry["id"] for entry in cfg["cases"]] if args.all_cases else [args.case]
    )
    for case_id in case_ids:
        case = get_case(cfg, case_id)
        logger.info("Case %s, init %s", case.id, case.init_time)
        paths = download_case(
            case.init_time,
            spec,
            args.out_dir,
            step_hours=cfg["forecast"]["step_hours"],
            force=args.force,
        )
        for path in paths:
            size = path.stat().st_size / 1024**2 if path.exists() else 0
            logger.info("  %s  %.0f MiB", path.name, size)


if __name__ == "__main__":
    main()
