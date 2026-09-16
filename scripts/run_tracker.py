"""CLI entry point: run_tracker.

Re-runs the cyclone tracker over members already written by run_inference,
for trying a different seed or a different tracker setting without paying for
the GPU again.

It reads the stored regional crops and pads them back onto a global grid,
because the tracker interpolates with longitude wraparound and rejects any
grid that is not the full [0, 360). So it sees the storm only while the storm
is inside the configured region; the tracks run_inference wrote from the real
global field stay authoritative, and results go to tracks-retracked.csv rather
than overwriting them.

Usage:
    uv run python scripts/run_tracker.py --config configs/krovanh.yaml \\
        --case init-2026-08-31T18
    uv run python scripts/run_tracker.py --case init-2026-08-31T18 --seed-position 22.6 131.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wn2_typhoon.config import get_case, load_raw
from wn2_typhoon.inference.tracker import (
    build_tracker,
    initial_storms_from_positions,
    pad_to_global,
    track_member,
    tracker_variables,
)
from wn2_typhoon.utils.logs import configure

logger = configure("run_tracker")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    parser.add_argument("--case", required=True, help="case id from the config")
    parser.add_argument("--out-dir", type=Path, help="default outputs/<case>")
    parser.add_argument(
        "--seed-position",
        type=float,
        nargs=2,
        metavar=("LAT", "LON"),
        help="override the case's seed; omit for cyclogenesis mode",
    )
    parser.add_argument(
        "--cyclogenesis-only",
        action="store_true",
        help="ignore any seed position and let the tracker find the storm",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="keep third-party INFO logs"
    )
    return parser.parse_args()


def main() -> None:
    """Re-track every stored member of one case."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    cfg = load_raw(args.config)
    case = get_case(cfg, args.case)
    out_dir = args.out_dir or Path("outputs") / case.id

    stores = sorted(out_dir.glob("member-*.zarr"))
    if not stores:
        raise SystemExit(f"no member stores under {out_dir}; run run_inference first")

    position = args.seed_position or case.seed_position
    if args.cyclogenesis_only:
        position = None
    init_time = np.datetime64(case.init_time)
    initial_storms = (
        None
        if position is None
        else initial_storms_from_positions(
            [(case.id, float(position[0]), float(position[1]))], init_time
        )
    )
    logger.info(
        "Case %s, init %s, seed %s", case.id, case.init_time, position or "none"
    )

    tracker = build_tracker()
    all_tracks = []
    for store in stores:
        member = int(store.stem.split("-")[-1])
        forecast = xr.open_zarr(store).load()
        kept = tracker_variables(forecast)
        resolution = float(abs(forecast.lat.values[1] - forecast.lat.values[0]))
        padded = pad_to_global(forecast[kept], resolution)
        logger.info(
            "%s: %d tracker fields, crop %s padded to %s",
            store.name, len(kept), dict(forecast.sizes), dict(padded.sizes),
        )
        tracks = track_member(
            padded,
            init_time,
            None if initial_storms is None else initial_storms.copy(),
            tracker=tracker,
        )
        tracks.insert(0, "member", member)
        all_tracks.append(tracks)
        logger.info("  %d track rows", len(tracks))

    target = out_dir / "tracks-retracked.csv"
    pd.concat(all_tracks, ignore_index=True).to_csv(target, index=False)
    logger.info("wrote %s", target)


if __name__ == "__main__":
    main()
