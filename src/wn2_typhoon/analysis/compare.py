"""Compare the cases with each other.

The per-case tables under ``outputs/<case>/analysis/`` answer "how good was
this forecast". Read side by side they answer the question the five
initialization times were run for: does skill depend on how much of the storm
the analysis had seen, and how does the ensemble's spread relate to its error.

Everything here is built from those tables. Nothing reads the tracker output
or the forecast fields, so the comparison reruns in seconds after any change
to the per-case evaluation.

Two views of the same errors
----------------------------
Indexed by **lead time**, the curves compare the model against itself: a 48 h
forecast from one initialization against a 48 h forecast from another.
Indexed by **valid time**, they compare what was known about the same moment:
the forecast for 3 September from the run 12 h before formation against the
one from 12 h after, which is how a forecaster meets them. Both are written,
because a case can look better in one view and worse in the other.

The ensemble mean is not a track
--------------------------------
On these cases the members split into a group that recurves over Japan and a
group that drifts toward the continent, and the mean of the two sits between
them, near the observed loop, on a path no member took. A small error of the
ensemble mean is therefore always reported next to the number of members
within a fixed radius of the observed centre, and every member's end point is
classified against the observed track's box, so that agreement of the mean is
never mistaken for a member that got it right.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wn2_typhoon.analysis.track_error import TRACK_PRESSURE, TRACK_TIME

# The per-case tables evaluate.py writes, and the columns to parse as dates.
CASE_TABLES = {
    "summary": [],
    "errors": ["valid_time"],
    "genesis": ["model_genesis", "observed_genesis"],
    "lifetime": ["track_end", "observed_end", "min_pressure_time"],
    "track": [TRACK_TIME],
    "selection": ["anchor_time"],
}

# "Near the observed centre" for the member count. About the size of the
# short-lead errors, so it separates members that stayed with the storm from
# members that left it, not good members from better ones.
WITHIN_KM = 200.0

# Padding around the observed track's bounding box for the fate classes.
FATE_MARGIN_DEG = 1.5
FATES = ("stayed", "north", "west", "east", "south")


@dataclass(frozen=True)
class CaseResult:
    """One case's evaluation tables, as written by ``scripts/evaluate.py``.

    Attributes:
        case_id: Case identifier from the config.
        init_time: Initialization time.
        tables: The tables keyed as in :data:`CASE_TABLES`.
    """

    case_id: str
    init_time: pd.Timestamp
    tables: dict[str, pd.DataFrame]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]


def load_case(case_id: str, init_time, analysis_dir: Path) -> CaseResult:
    """Read one case's tables from its analysis directory.

    Args:
        case_id: Case identifier.
        init_time: Initialization time, UTC ISO 8601.
        analysis_dir: Directory holding ``summary.csv`` and the others.

    Returns:
        The loaded :class:`CaseResult`.

    Raises:
        FileNotFoundError: If a table is missing; ``make evaluate`` writes them.
    """
    analysis_dir = Path(analysis_dir)
    tables = {}
    for name, date_columns in CASE_TABLES.items():
        path = analysis_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run 'make evaluate CASE={case_id}' first"
            )
        tables[name] = pd.read_csv(path, parse_dates=date_columns)
    return CaseResult(case_id, pd.Timestamp(str(init_time)), tables)


def case_labels(cases: list[CaseResult], formation_time) -> dict[str, str]:
    """Short labels giving each case's initialization relative to formation.

    Args:
        cases: The cases being compared.
        formation_time: Observed formation time, UTC ISO 8601.

    Returns:
        ``{case_id: "-12 h"}`` style labels; ``"0 h"`` is formation itself.
    """
    formation = pd.Timestamp(str(formation_time))
    labels = {}
    for case in cases:
        offset = (case.init_time - formation).total_seconds() / 3600.0
        labels[case.case_id] = f"{offset:+.0f} h" if offset else "0 h"
    return labels


def offset_label(init_time, formation_time) -> str:
    """Describe an initialization relative to formation, in words.

    For figure titles, where "-12 h" alone would leave the reader to work out
    what it is relative to.

    Args:
        init_time: Initialization time, UTC ISO 8601 or Timestamp.
        formation_time: Observed formation time, UTC ISO 8601 or Timestamp.

    Returns:
        "12 h before formation", "at formation" or "6 h after formation".
    """
    hours = (
        pd.Timestamp(str(init_time)) - pd.Timestamp(str(formation_time))
    ).total_seconds() / 3600.0
    if hours == 0:
        return "at formation"
    return f"{abs(hours):.0f} h {'before' if hours < 0 else 'after'} formation"


def within_column(within_km: float) -> str:
    """Name of the member-count column for a radius."""
    return f"n_within_{within_km:.0f}km"


def skill_by_lead(
    cases: list[CaseResult], within_km: float = WITHIN_KM
) -> pd.DataFrame:
    """Stack every case's per-lead summary into one long table.

    Args:
        cases: The cases being compared.
        within_km: Radius for the count of members near the observed centre.

    Returns:
        One row per case and lead time: ``case``, ``init_time``,
        ``valid_time``, every column of the per-case summary, and
        ``n_within_<r>km``, the number of members whose error at that lead is
        at most ``within_km``.
    """
    frames = []
    for case in cases:
        summary = case["summary"].copy()
        errors = case["errors"]
        near = (
            errors.assign(near=errors["error_km"] <= within_km)
            .groupby("lead_hours")["near"]
            .sum()
        )
        summary[within_column(within_km)] = (
            summary["lead_hours"].map(near).fillna(0).astype(int)
        )
        summary.insert(
            0, "valid_time",
            case.init_time + pd.to_timedelta(summary["lead_hours"], unit="h"),
        )
        summary.insert(0, "init_time", case.init_time)
        summary.insert(0, "case", case.case_id)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def pivot(
    skill: pd.DataFrame, column: str, cases: list[CaseResult], index: str = "lead_hours"
) -> pd.DataFrame:
    """One column of :func:`skill_by_lead` as a lead-by-case (or time-by-case) table.

    Args:
        skill: Output of :func:`skill_by_lead`.
        column: Column to tabulate.
        cases: The cases, in the column order wanted.
        index: ``"lead_hours"`` or ``"valid_time"``.

    Returns:
        Rows are leads or valid times, columns are case ids; missing where a
        case has no members at that row.
    """
    table = skill.pivot(index=index, columns="case", values=column).sort_index()
    return table.reindex(columns=[case.case_id for case in cases])


def genesis_by_case(cases: list[CaseResult], step_hours: int) -> pd.DataFrame:
    """Collapse each case's genesis report onto one row.

    ``n_at_earliest`` counts members that reported the storm at the first
    rollout step, which is the earliest the tracker can. A case where every
    member sits there has hit the floor, and its timing error is a bound, not
    a measurement.

    Args:
        cases: The cases being compared.
        step_hours: Model time step.

    Returns:
        One row per case: ``case``, ``init_time``, ``resolvable``,
        ``earliest_reportable``, ``n_members``, ``n_found``,
        ``n_at_earliest``, and the median, min and max of
        ``lead_error_hours`` over the members that produced the storm.
    """
    rows = []
    for case in cases:
        genesis = case["genesis"]
        found = genesis.loc[genesis["found"]]
        earliest = case.init_time + pd.Timedelta(hours=step_hours)
        rows.append(
            {
                "case": case.case_id,
                "init_time": case.init_time,
                "resolvable": bool(genesis["resolvable"].iloc[0]) if len(genesis) else False,
                "earliest_reportable": earliest,
                "n_members": len(genesis),
                "n_found": len(found),
                "n_at_earliest": int((found["model_genesis"] == earliest).sum()),
                "median_lead_error_hours": found["lead_error_hours"].median(),
                "min_lead_error_hours": found["lead_error_hours"].min(),
                "max_lead_error_hours": found["lead_error_hours"].max(),
            }
        )
    return pd.DataFrame(rows)


def lifetime_by_case(
    cases: list[CaseResult], observed_min_pressure_hpa: float
) -> pd.DataFrame:
    """Collapse each case's lifetime report onto one row.

    Args:
        cases: The cases being compared.
        observed_min_pressure_hpa: The storm's observed minimum pressure.

    Returns:
        One row per case: ``case``, ``init_time``, ``n_found``, the median,
        min and max of ``end_error_hours``, ``n_outliving_observed`` (members
        still tracked after the observed weakening), ``median_duration_hours``,
        ``median_min_pressure_hpa``, ``deepest_hpa`` and
        ``n_deeper_than_observed``.
    """
    rows = []
    for case in cases:
        lifetime = case["lifetime"]
        found = lifetime.loc[lifetime["found"]]
        rows.append(
            {
                "case": case.case_id,
                "init_time": case.init_time,
                "n_found": len(found),
                "median_end_error_hours": found["end_error_hours"].median(),
                "min_end_error_hours": found["end_error_hours"].min(),
                "max_end_error_hours": found["end_error_hours"].max(),
                "n_outliving_observed": int((found["end_error_hours"] > 0).sum()),
                "median_duration_hours": found["duration_hours"].median(),
                "median_min_pressure_hpa": found["min_pressure_hpa"].median(),
                "deepest_hpa": found["min_pressure_hpa"].min(),
                "n_deeper_than_observed": int(
                    (found["min_pressure_hpa"] < observed_min_pressure_hpa).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def spread_skill(cases: list[CaseResult]) -> pd.DataFrame:
    """How the spread tracks the error, per case.

    Only leads at which every member is still present are used: once tracks
    start ending, both the spread and the mean are taken over a shrinking
    sample and the relation between them says less.

    Args:
        cases: The cases being compared.

    Returns:
        One row per case: ``case``, ``init_time``, ``n_leads``,
        ``max_lead_hours``, ``corr_spread_error`` (Pearson correlation of the
        spread with the error of the ensemble mean across leads) and
        ``mean_error_to_spread`` (mean over leads of the mean member error
        divided by the spread; above 1 the ensemble is under-dispersive).
    """
    rows = []
    for case in cases:
        summary = case["summary"]
        full = summary.loc[summary["n_members"] == summary["n_members"].max()]
        full = full.loc[full["spread_km"] > 0]
        rows.append(
            {
                "case": case.case_id,
                "init_time": case.init_time,
                "n_leads": len(full),
                "max_lead_hours": int(full["lead_hours"].max()) if len(full) else 0,
                "corr_spread_error": (
                    full["spread_km"].corr(full["ensemble_mean_error_km"])
                    if len(full) > 2 else np.nan
                ),
                "mean_error_to_spread": (
                    (full["mean_error_km"] / full["spread_km"]).mean()
                    if len(full) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def observed_box(best_track: pd.DataFrame, margin_deg: float = FATE_MARGIN_DEG) -> dict:
    """Bounding box of the observed track, padded.

    Args:
        best_track: Reference track.
        margin_deg: Padding on every side.

    Returns:
        ``{"south", "north", "west", "east"}`` in degrees.
    """
    return {
        "south": float(best_track["lat"].min()) - margin_deg,
        "north": float(best_track["lat"].max()) + margin_deg,
        "west": float(best_track["lon"].min()) - margin_deg,
        "east": float(best_track["lon"].max()) + margin_deg,
    }


def member_fate(
    cases: list[CaseResult],
    best_track: pd.DataFrame,
    margin_deg: float = FATE_MARGIN_DEG,
) -> pd.DataFrame:
    """Where each member's track ended, relative to the observed track's box.

    A member that ends inside the padded box "stayed", as the observed storm
    did. One that ends outside is classed by the side it left through, north
    first (the recurving group), then west (toward the continent), then east
    and south. The box is a description of this storm, not a physical
    boundary, so the margin is a parameter and is reported with the table.

    Args:
        cases: The cases being compared.
        best_track: Reference track.
        margin_deg: Padding around the observed track's bounding box.

    Returns:
        One row per member: ``case``, ``init_time``, ``member``,
        ``track_end``, ``end_lat``, ``end_lon``, ``max_lat``, ``min_lon``,
        ``min_pressure_hpa``, ``fate``.
    """
    box = observed_box(best_track, margin_deg)
    rows = []
    for case in cases:
        track = case["track"]
        for member, group in track.groupby("member", sort=True):
            group = group.sort_values(TRACK_TIME)
            last = group.iloc[-1]
            lat, lon = float(last["lat"]), float(last["lon"])
            if box["south"] <= lat <= box["north"] and box["west"] <= lon <= box["east"]:
                fate = "stayed"
            elif lat > box["north"]:
                fate = "north"
            elif lon < box["west"]:
                fate = "west"
            elif lon > box["east"]:
                fate = "east"
            else:
                fate = "south"
            rows.append(
                {
                    "case": case.case_id,
                    "init_time": case.init_time,
                    "member": int(member),
                    "track_end": last[TRACK_TIME],
                    "end_lat": lat,
                    "end_lon": lon,
                    "max_lat": float(group["lat"].max()),
                    "min_lon": float(group["lon"].min()),
                    "min_pressure_hpa": (
                        float(group[TRACK_PRESSURE].min())
                        if group[TRACK_PRESSURE].notna().any() else np.nan
                    ),
                    "fate": fate,
                }
            )
    return pd.DataFrame(rows)


def fate_counts(fate: pd.DataFrame, cases: list[CaseResult]) -> pd.DataFrame:
    """Count members per fate class, per case.

    Args:
        fate: Output of :func:`member_fate`.
        cases: The cases, in row order.

    Returns:
        One row per case with a column per class in :data:`FATES` and
        ``n_members``.
    """
    counts = (
        fate.groupby(["case", "fate"]).size().unstack("fate", fill_value=0)
        .reindex(index=[case.case_id for case in cases], columns=list(FATES), fill_value=0)
    )
    counts["n_members"] = counts[list(FATES)].sum(axis=1)
    return counts.reset_index()
