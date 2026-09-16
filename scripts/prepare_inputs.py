"""CLI entry point: prepare_inputs.

Converts the downloaded ERA5 frames into the two-frame NetCDF the pod reads.
The 40 NaN target frames the rollout needs are added there, not here: they
would add 17 GB to a 0.7 GB file.

Usage:
    uv run python scripts/prepare_inputs.py --config configs/krovanh.yaml --case init-2026-08-31T18
    uv run python scripts/prepare_inputs.py --all-cases
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wn2_typhoon.config import get_case, load_raw
from wn2_typhoon.data.era5_download import build_requests
from wn2_typhoon.data.era5_to_wn2 import build_inputs
from wn2_typhoon.model_spec import load_spec
from wn2_typhoon.utils.logs import configure

logger = configure("prepare_inputs")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="case id from the config")
    group.add_argument("--all-cases", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/era5"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--verbose", action="store_true", help="keep third-party INFO logs"
    )
    return parser.parse_args()


def main() -> None:
    """Build the model input file for one case or for all of them."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    cfg = load_raw(args.config)
    spec = load_spec(cfg["model"]["name"])
    step_hours = cfg["forecast"]["step_hours"]

    case_ids = (
        [entry["id"] for entry in cfg["cases"]] if args.all_cases else [args.case]
    )
    for case_id in case_ids:
        case = get_case(cfg, case_id)
        files = [
            item.target
            for item in build_requests(case.init_time, spec, args.raw_dir, step_hours)
        ]
        missing = [path.name for path in files if not path.exists()]
        if missing:
            raise SystemExit(
                f"{case.id}: missing ERA5 files {missing}; run make download-era5 first"
            )

        dataset = build_inputs(files, case.init_time, spec, step_hours)
        target = args.out_dir / case.id / "inputs.nc"
        target.parent.mkdir(parents=True, exist_ok=True)
        encoding = {name: {"zlib": True, "complevel": 4} for name in dataset.data_vars}
        dataset.to_netcdf(target, encoding=encoding)
        logger.info(
            "%s: %s  %.0f MiB  dims=%s",
            case.id, target, target.stat().st_size / 1024**2, dict(dataset.sizes),
        )
        for name in sorted(dataset.data_vars):
            logger.info("    %s %s", name, dataset[name].dims)


if __name__ == "__main__":
    main()
