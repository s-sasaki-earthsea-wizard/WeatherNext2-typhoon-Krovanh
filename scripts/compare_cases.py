"""CLI entry point: compare_cases.

Puts the evaluated cases side by side: skill against initialization time,
spread against skill, genesis timing, and how long and how deep each member
kept the storm. Reads only the per-case tables that ``scripts/evaluate.py``
wrote under ``outputs/<case>/analysis/``, so it needs neither the forecast
fields nor a GPU and reruns in seconds.

Results go to ``outputs/comparison/`` by default: nine tables, two GeoJSON
files holding every case's tracks for a GIS, and five figures.

Usage:
    uv run python scripts/compare_cases.py --config configs/krovanh.yaml
    uv run python scripts/compare_cases.py --cases init-2026-08-31T18 init-2026-09-01T00
    uv run python scripts/compare_cases.py --basemap osm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wn2_typhoon.analysis.compare import (
    FATES,
    WITHIN_KM,
    case_labels,
    fate_counts,
    genesis_by_case,
    lifetime_by_case,
    load_case,
    member_fate,
    observed_box,
    pivot,
    skill_by_lead,
    spread_skill,
    within_column,
)
from wn2_typhoon.analysis.export import write_tracks
from wn2_typhoon.analysis.plot import BASEMAPS, credit_line
from wn2_typhoon.analysis.plot_compare import (
    plot_case_tracks,
    plot_error_vs_lead,
    plot_error_vs_valid_time,
    plot_lifetime,
    plot_spread_vs_error,
)
from wn2_typhoon.config import get_case, load_raw
from wn2_typhoon.data.jma_besttrack import load_track
from wn2_typhoon.utils.logs import configure

logger = configure("compare_cases")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    parser.add_argument(
        "--cases", nargs="+", metavar="CASE",
        help="case ids to compare; default: every case with an analysis directory",
    )
    parser.add_argument(
        "--best-track", type=Path, default=Path("data/interim/besttrack.csv")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument(
        "--basemap", choices=BASEMAPS, default="natural-earth",
        help="map background for the track figure",
    )
    parser.add_argument(
        "--within-km", type=float, default=WITHIN_KM,
        help="radius for the count of members near the observed centre",
    )
    parser.add_argument("--no-figures", action="store_true", help="tables only")
    parser.add_argument("--verbose", action="store_true", help="keep third-party INFO logs")
    return parser.parse_args()


def load_cases(cfg: dict, case_ids: list[str] | None) -> list:
    """Load the evaluated cases, in config order.

    Args:
        cfg: The parsed configuration.
        case_ids: Explicit case ids, or None for every evaluated case.

    Returns:
        The loaded cases.

    Raises:
        SystemExit: If a requested case has no analysis, or fewer than two
            cases are available.
    """
    wanted = case_ids or [entry["id"] for entry in cfg["cases"]]
    cases = []
    for case_id in wanted:
        case = get_case(cfg, case_id)
        analysis_dir = Path("outputs") / case.id / "analysis"
        if not (analysis_dir / "summary.csv").exists():
            if case_ids:
                raise SystemExit(f"{case.id} has no analysis; run make evaluate CASE={case.id}")
            logger.info("Skipping %s: not evaluated yet", case.id)
            continue
        cases.append(load_case(case.id, case.init_time, analysis_dir))
    if len(cases) < 2:
        raise SystemExit("need at least two evaluated cases to compare")
    return sorted(cases, key=lambda case: case.init_time)


def main() -> None:
    """Compare the evaluated cases and write the tables and figures."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    cfg = load_raw(args.config)
    storm = cfg["storm"]
    step_hours = int(cfg["forecast"]["step_hours"])

    cases = load_cases(cfg, args.cases)
    best_track = load_track(args.best_track)
    labels = case_labels(cases, storm["formation_time"])
    logger.info(
        "Comparing %d cases: %s; reference %s",
        len(cases),
        ", ".join(f"{case.case_id} ({labels[case.case_id]})" for case in cases),
        best_track["source"].iloc[0],
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    observed_min_pressure = float(
        best_track["pressure_hpa"].min()
    )
    skill = skill_by_lead(cases, args.within_km)
    near = within_column(args.within_km)
    fate = member_fate(cases, best_track)
    ends = pd.concat(
        [case["lifetime"].assign(case=case.case_id)[["case", "member", "end_error_hours"]]
         for case in cases],
        ignore_index=True,
    )
    fate = fate.merge(ends, on=["case", "member"], how="left")
    counts = fate_counts(fate, cases)
    box = observed_box(best_track)

    tables = {
        "skill-by-lead": skill,
        "error-by-lead": pivot(skill, "mean_error_km", cases),
        "error-by-valid-time": pivot(skill, "mean_error_km", cases, index="valid_time"),
        f"members-within-{args.within_km:.0f}km-by-valid-time": pivot(
            skill, near, cases, index="valid_time"
        ),
        "genesis-by-case": genesis_by_case(cases, step_hours),
        "lifetime-by-case": lifetime_by_case(cases, observed_min_pressure),
        "spread-skill-by-case": spread_skill(cases),
        "member-fate": fate,
        "member-fate-by-case": counts,
    }
    for name, table in tables.items():
        index = name in ("error-by-lead", "error-by-valid-time") or name.startswith(
            "members-within"
        )
        table.to_csv(out_dir / f"{name}.csv", index=index)
    lines_path, points_path = write_tracks(
        [(case.case_id, case.init_time, case["track"]) for case in cases],
        best_track, out_dir,
    )
    logger.info("wrote %d tables and %s, %s to %s", len(tables),
                lines_path.name, points_path.name, out_dir)

    for lead in (48, 96, 120):
        row = skill.loc[skill["lead_hours"] == lead]
        if row.empty:
            continue
        logger.info(
            "  %3d h mean member error: %s", lead,
            "; ".join(
                f"{labels[r['case']]} {r['mean_error_km']:.0f} km ({r[near]}/{r['n_members']} near)"
                for _, r in row.sort_values("init_time").iterrows()
            ),
        )
    logger.info(
        "Observed track box %.1f-%.1fN %.1f-%.1fE (padded); members ending outside it: %s",
        box["south"], box["north"], box["west"], box["east"],
        "; ".join(
            f"{labels[r['case']]} " + ", ".join(
                f"{int(r[f])} {f}" for f in FATES if r[f]
            )
            for _, r in counts.iterrows()
        ),
    )

    if args.no_figures:
        return
    credit = credit_line(str(best_track["source"].iloc[0]), args.basemap)
    plain_credit = credit_line(str(best_track["source"].iloc[0]))
    label = f"Krovanh (T{storm['jma_number']}), {len(cases)} initialization times"
    events = {
        "formation": pd.Timestamp(storm["formation_time"]),
        "minimum pressure": pd.Timestamp(storm["peak_time"]),
        "depression": pd.Timestamp(storm["depression_time"]),
    }
    plot_error_vs_lead(skill, cases, labels, out_dir / "error-vs-lead.png", args.within_km,
                       title=f"{label} -- error against lead time", credit=plain_credit)
    plot_error_vs_valid_time(skill, cases, labels, events,
                             out_dir / "error-vs-valid-time.png", args.within_km,
                             title=f"{label} -- error against valid time",
                             credit=plain_credit)
    plot_spread_vs_error(skill, cases, labels, out_dir / "spread-vs-error.png",
                         title=f"{label} -- spread against skill", credit=plain_credit)
    region = cfg["output"]["region"]
    plot_case_tracks(cases, best_track, labels, events, out_dir / "tracks.png",
                     basemap=args.basemap,
                     extent=[*region["lon"], *region["lat"]],
                     title=f"{label} -- member tracks (stored region)", credit=credit)
    plot_lifetime(fate, cases, labels, observed_min_pressure, out_dir / "lifetime.png",
                  title=f"{label} -- track end and minimum pressure", credit=plain_credit)
    logger.info("wrote 5 figures to %s", out_dir)


if __name__ == "__main__":
    main()
