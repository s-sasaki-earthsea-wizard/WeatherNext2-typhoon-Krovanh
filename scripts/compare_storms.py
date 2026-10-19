"""CLI entry point: compare_storms.

Puts the storms side by side: each storm's cases pooled by lead time, the
position error split into the distance to the observed path and the clock
along it, each member's shape against the path and its clock, how directed
the members' motion was against the storm's (a Péclet-like ratio over 18-36 h
windows), when the tracks ended, and spread against skill. Reads only the
per-case tables that ``scripts/evaluate.py`` wrote and the reference tracks,
so it needs no forecast fields and reruns in seconds.

Results go to ``outputs/storms/`` by default: seven tables and five figures.

Usage:
    uv run python scripts/compare_storms.py
    uv run python scripts/compare_storms.py --configs configs/krovanh.yaml configs/dujuan.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wn2_typhoon.analysis.compare import WITHIN_KM
from wn2_typhoon.analysis.directedness import directedness_by_time, directedness_lag
from wn2_typhoon.analysis.plot import credit_line
from wn2_typhoon.analysis.plot_storms import (
    plot_directedness,
    plot_error_by_storm,
    plot_road_and_clock,
    plot_shape_and_clock,
    plot_track_ends,
)
from wn2_typhoon.analysis.storms import (
    CLOCK_LEADS,
    load_storm,
    member_shape_and_clock,
    road_and_clock,
    spread_skill_by_storm,
    track_end_summary,
    track_ends,
)
from wn2_typhoon.config import across_storms_dir, load_raw
from wn2_typhoon.utils.logs import configure

logger = configure("compare_storms")

# Krovanh first: it is the subject, the other two are its controls (issue #12).
DEFAULT_CONFIGS = [
    Path("configs/krovanh.yaml"), Path("configs/dujuan.yaml"), Path("configs/peipah.yaml")
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS,
        help="one config per storm, in the order of the figures' colours",
    )
    parser.add_argument("--out-dir", type=Path, help="default outputs/storms")
    parser.add_argument(
        "--within-km", type=float, default=WITHIN_KM,
        help="radius for 'at the observed position' and 'on the observed path'",
    )
    parser.add_argument(
        "--clock-lead", type=int, default=72, choices=CLOCK_LEADS,
        help="lead of the single-lead clock panel of shape-and-clock.png",
    )
    parser.add_argument("--no-figures", action="store_true", help="tables only")
    parser.add_argument("--verbose", action="store_true", help="keep third-party INFO logs")
    return parser.parse_args()


def credit(storms) -> str:
    """Attribution naming each storm's reference release."""
    releases = {}
    for storm in storms:
        source = str(storm.best_track["source"].iloc[0])
        releases.setdefault(source, []).append(f"T{storm.jma_number}")
    return credit_line(
        "; ".join(f"{source} {', '.join(numbers)}" for source, numbers in releases.items())
    )


def main() -> None:
    """Compare the storms and write the tables and figures."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    storms = []
    for path in args.configs:
        storm = load_storm(load_raw(path))
        if len(storm.cases) < storm.n_cases:
            logger.info("%s: %d of %d cases evaluated", storm.label, len(storm.cases),
                        storm.n_cases)
        if not storm.cases:
            raise SystemExit(f"{storm.label} has no evaluated case; run make evaluate-all")
        storms.append(storm)
    out_dir = args.out_dir or across_storms_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    table = road_and_clock(storms, args.within_km)
    members = member_shape_and_clock(storms, args.within_km)
    directedness = directedness_by_time(storms)
    lags = directedness_lag(directedness)
    ends = track_ends(storms)
    tables = {
        "road-and-clock-by-lead": table,
        "member-shape-and-clock": members,
        "directedness-by-time": directedness,
        "directedness-lag": lags,
        "track-ends": ends,
        "track-end-by-storm": track_end_summary(ends, storms),
        "spread-skill-by-storm": spread_skill_by_storm(storms),
    }
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    logger.info("wrote %d tables to %s", len(tables), out_dir)

    for lead in (48, 72, 96, 120):
        rows = table.loc[table["lead_hours"] == lead]
        if rows.empty:
            continue
        logger.info(
            "  %3d h: %s", lead,
            "; ".join(
                f"{r['storm']} {r['mean_error_km']:.0f} km, "
                f"{r['n_on_path']}/{r['n_present'] - r['n_beyond']} on the path, "
                f"clock {r['clock_median_hours']:+.0f} h"
                for _, r in rows.iterrows()
            ),
        )

    for storm in storms:
        mine = members.loc[members["storm"] == storm.label]
        best = lags.loc[(lags["storm"] == storm.label) & lags["best"]].drop_duplicates(
            "window_hours"
        )
        logger.info(
            "  %s: shape %.0f km, clock median %+.1f h, trend %+.2f h/h (medians over "
            "members); best lag %s",
            storm.label, mine["chamfer_km"].median(), mine["clock_median_hours"].median(),
            mine["clock_trend"].median(),
            ", ".join(f"{r.window_hours} h window {r.lag_hours:+d} h" for r in best.itertuples())
            or "n/a",
        )

    if args.no_figures:
        return
    line = credit(storms)
    names = ", ".join(storm.name.title() for storm in storms)
    plot_error_by_storm(table, storms, out_dir / "error-by-storm.png", args.within_km,
                        title=f"{names} -- position error, each storm's cases pooled",
                        credit=line)
    plot_road_and_clock(table, storms, out_dir / "road-and-clock.png", args.within_km,
                        title=f"{names} -- on the observed path, and early or late on it",
                        credit=line)
    plot_track_ends(ends, storms, out_dir / "track-ends.png",
                    title=f"{names} -- when the member tracks ended", credit=line)
    plot_shape_and_clock(members, storms, out_dir / "shape-and-clock.png",
                         clock_lead=args.clock_lead,
                         title=f"{names} -- each member's shape against the observed "
                               "path, and its clock on it (+ = ahead)",
                         credit=line)
    plot_directedness(directedness, lags, storms, out_dir / "directedness.png",
                      title=f"{names} -- how directed the members moved, against the "
                            "storm at the same time",
                      credit=line)
    logger.info("wrote 5 figures to %s", out_dir)


if __name__ == "__main__":
    main()
