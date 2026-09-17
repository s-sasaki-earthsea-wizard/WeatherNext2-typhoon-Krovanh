"""CLI entry point: evaluate.

Compares one case's forecast tracks with the JMA reference track and writes
the tables and figures. Needs no GPU and no forecast fields: it reads
``outputs/<case>/tracks.csv`` and ``data/interim/besttrack.csv`` only.

Results are written beside the tracks, under ``outputs/<case>/analysis/``,
which is on the NAS through the symlink. The NAS is not always mounted, and
that alone is enough to make this fail.

The track map draws Natural Earth coastlines by default; ``--basemap osm``
puts OpenStreetMap tiles under the tracks instead, which needs the network on
the first run and adds the tiles' attribution to the credit line.

The reference track is the preliminary JMA table until the post-analysis CSV
reaches storm 2624, expected around the turn of the year. Re-running this
command after ``make fetch-besttrack`` picks the new one up with no other
change, and every figure carries which release it used.

Usage:
    uv run python scripts/evaluate.py --config configs/krovanh.yaml --case init-2026-08-31T18
    uv run python scripts/evaluate.py --all-cases
    uv run python scripts/evaluate.py --case init-2026-09-01T00 --basemap osm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from wn2_typhoon.analysis.plot import (
    BASEMAPS,
    credit_line,
    plot_error_vs_lead,
    plot_pressure,
    plot_tracks,
)
from wn2_typhoon.analysis.track_error import (
    ensemble_summary,
    genesis_report,
    initial_state_error,
    lifetime_report,
    position_errors,
    select_storm,
)
from wn2_typhoon.config import get_case, load_raw
from wn2_typhoon.data.jma_besttrack import load_track
from wn2_typhoon.utils.logs import configure

logger = configure("evaluate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="case id from the config")
    group.add_argument("--all-cases", action="store_true")
    parser.add_argument(
        "--best-track", type=Path, default=Path("data/interim/besttrack.csv")
    )
    parser.add_argument("--inputs-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--out-dir", type=Path, help="default outputs/<case>/analysis")
    parser.add_argument(
        "--basemap", choices=BASEMAPS, default="natural-earth",
        help="map background for the track figure",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="write the tables only"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="keep third-party INFO logs"
    )
    return parser.parse_args()


def selection_table(selections) -> pd.DataFrame:
    """Turn the per-member selection decisions into a writable table.

    Args:
        selections: The list returned by ``track_error.select_storm``.

    Returns:
        One row per member, including members where nothing matched.
    """
    return pd.DataFrame(
        [
            {
                "member": s.member,
                "found": s.found,
                "track_id": s.track_id,
                "anchor_time": s.anchor_time,
                "distance_km": s.distance_km,
                "runner_up_km": s.runner_up_km,
                "separation_ratio": (
                    s.runner_up_km / s.distance_km if s.distance_km else None
                ),
            }
            for s in selections
        ]
    )


def report_initial_state(case, inputs_dir: Path) -> dict | None:
    """Measure and log the initial-state error, when a position exists for it.

    Args:
        case: The case being evaluated.
        inputs_dir: Directory holding ``<case>/inputs.nc``.

    Returns:
        The measurement, or None when the case has no analysed position or the
        input file is not on this machine.
    """
    if case.observed_position is None:
        logger.info(
            "No analysed position at init time, so no initial-state error: "
            "the preliminary JMA table starts at formation"
        )
        return None
    inputs_path = inputs_dir / case.id / "inputs.nc"
    if not inputs_path.exists():
        logger.info("No %s, skipping the initial-state error", inputs_path)
        return None

    lat, lon = case.observed_position
    with xr.open_dataset(inputs_path) as inputs:
        measured = initial_state_error(inputs, lat, lon)
    logger.info(
        "Initial state: ERA5 has the centre at %.2fN %.2fE %.1f hPa, "
        "%.0f km from the JMA %.1fN %.1fE. Errors of this size at short lead "
        "describe the analysis, not the model.",
        measured["era5_lat"], measured["era5_lon"] % 360,
        measured["era5_pressure_hpa"], measured["distance_km"], lat, lon,
    )
    return measured


def evaluate_case(case, cfg: dict, args: argparse.Namespace) -> None:
    """Run the whole comparison for one case and write its outputs.

    Args:
        case: The case to evaluate.
        cfg: The parsed configuration.
        args: Parsed command-line arguments.

    Raises:
        SystemExit: If the case has no tracker output yet.
    """
    tracks_path = Path("outputs") / case.id / "tracks.csv"
    if not tracks_path.exists():
        raise SystemExit(
            f"no {tracks_path}; run the forecast for {case.id} first, "
            "or mount the NAS if it holds the results"
        )
    out_dir = args.out_dir or Path("outputs") / case.id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_track = load_track(args.best_track)
    tracks = pd.read_csv(tracks_path)
    logger.info(
        "Case %s, init %s: %d track rows, reference %s (%d rows, %s to %s)",
        case.id, case.init_time, len(tracks),
        best_track["source"].iloc[0], len(best_track),
        best_track["time"].min(), best_track["time"].max(),
    )

    report_initial_state(case, args.inputs_dir)

    selected, selections = select_storm(tracks, best_track, case.init_time)
    chosen = selection_table(selections)
    found = int(chosen["found"].sum())
    logger.info(
        "Selected the storm in %d of %d members; nearest %.0f-%.0f km, "
        "runner-up at least %.0f km",
        found, len(chosen),
        chosen.loc[chosen["found"], "distance_km"].min() if found else float("nan"),
        chosen.loc[chosen["found"], "distance_km"].max() if found else float("nan"),
        chosen.loc[chosen["found"], "runner_up_km"].min() if found else float("nan"),
    )
    if found == 0:
        raise SystemExit(
            f"no member of {case.id} produced a storm within the selection "
            "threshold; re-run run_tracker.py --keep-short-tracks to see "
            "whether short-lived tracks were filtered out"
        )

    errors = position_errors(selected, best_track, case.init_time)
    summary = ensemble_summary(errors)
    genesis = genesis_report(
        selected, selections,
        cfg["storm"]["formation_time"], case.init_time,
        cfg["forecast"]["step_hours"],
    )
    lifetime = lifetime_report(
        selected, selections, cfg["storm"]["depression_time"]
    )

    for name, table in [
        ("selection", chosen), ("errors", errors), ("summary", summary),
        ("genesis", genesis), ("lifetime", lifetime), ("track", selected),
    ]:
        table.to_csv(out_dir / f"{name}.csv", index=False)
    logger.info("wrote 6 tables to %s", out_dir)

    if not summary.empty:
        for lead in (24, 48, 72, 120):
            row = summary.loc[summary["lead_hours"] == lead]
            if not row.empty:
                logger.info(
                    "  %3d h: mean %.0f km, ensemble mean %.0f km, spread %.0f km, "
                    "%d members",
                    lead, row["mean_error_km"].iloc[0],
                    row["ensemble_mean_error_km"].iloc[0],
                    row["spread_km"].iloc[0], int(row["n_members"].iloc[0]),
                )
    if not genesis.empty and bool(genesis["resolvable"].iloc[0]) is False:
        logger.info(
            "Genesis timing is not resolvable for this case: the tracker "
            "cannot report a storm before lead %d h, which is not earlier "
            "than the observed formation.", cfg["forecast"]["step_hours"],
        )

    if args.no_figures:
        return
    source = str(best_track["source"].iloc[0])
    credit = credit_line(source)
    label = f"Krovanh (T{cfg['storm']['jma_number']}), init {case.init_time} UTC"
    plot_tracks(selected, best_track, out_dir / "tracks.png",
                title=f"{label} -- ensemble tracks",
                credit=credit_line(source, args.basemap), basemap=args.basemap)
    plot_error_vs_lead(errors, summary, out_dir / "error-vs-lead.png",
                       title=f"{label} -- position error", credit=credit)
    plot_pressure(selected, best_track, out_dir / "pressure.png",
                  title=f"{label} -- central pressure", credit=credit)
    logger.info("wrote 3 figures to %s", out_dir)


def main() -> None:
    """Evaluate one case or every case in the config."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    cfg = load_raw(args.config)

    case_ids = (
        [entry["id"] for entry in cfg["cases"]] if args.all_cases else [args.case]
    )
    if args.all_cases and args.out_dir:
        raise SystemExit("--out-dir applies to one case; drop it with --all-cases")

    for case_id in case_ids:
        case = get_case(cfg, case_id)
        if args.all_cases and not (Path("outputs") / case.id / "tracks.csv").exists():
            logger.info("Skipping %s: no tracker output yet", case.id)
            continue
        evaluate_case(case, cfg, args)


if __name__ == "__main__":
    main()
