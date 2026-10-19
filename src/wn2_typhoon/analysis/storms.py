"""Compare the storms with each other.

``compare`` puts one storm's five initialization times side by side. This
module pools each storm's cases and puts the storms side by side, which is
what the control experiment of issue #12 asks: is Krovanh's scatter a
property of the model or of the situation?

Pooled by lead time
-------------------
Each storm's cases are pooled at equal lead, 40 member forecasts at most
(five cases of eight). Leads that only some cases reach, because a later
initialization meets the end of the tropical phase sooner, are tabulated
with their case count, and the figures draw only the leads every case
reaches, so a curve never changes its sample halfway along.

Shares are taken over the members still tracking the storm at that lead,
not over the members run. Peipah's first case selected 3 of 8 members
because the tracker's 60 h minimum duration deleted broken tracks of the
other five, which is a tracker artifact; counting them as misses would
charge it to the model. The members run are reported alongside.

Km are not comparable across storms
-----------------------------------
A storm moving at 50 km/h turns a six-hour timing error into 300 km.
Dujuan and Peipah run that fast after recurving, Krovanh crawled and
stalled, so equal km errors do not mean equally good forecasts. The
position error is therefore reported next to its two parts from
:mod:`path_offset`: the distance to the observed path and the clock along it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wn2_typhoon.analysis.compare import WITHIN_KM, CaseResult, load_case, spread_skill
from wn2_typhoon.analysis.path_offset import chamfer_km, path_offsets
from wn2_typhoon.config import (
    OUTPUTS_DIR,
    TropicalEnd,
    besttrack_path,
    case_dir,
    tropical_end,
)
from wn2_typhoon.data.jma_besttrack import load_track, tropical_phase

#: Leads at which each member's clock is also read on its own, for a summary
#: that compares members at the same point of their forecast.
CLOCK_LEADS = (48, 72, 96)
#: On-road positions with a readable clock needed for a clock trend.
MIN_TREND_ROWS = 3


@dataclass(frozen=True)
class StormResult:
    """One storm's evaluated cases, pooled for the comparison across storms.

    Attributes:
        name: Storm name as JMA writes it, e.g. "KROVANH".
        jma_number: JMA number, e.g. "2624".
        cases: The evaluated cases, in initialization order.
        n_cases: Cases in the config, evaluated or not.
        n_members: Members run per case.
        best_track: Reference track cut to the tropical phase.
        end: When and how the tropical phase ended.
        dissipation_time: When the storm dissipated, where the config records
            it; None otherwise.
        clock_until: Last valid time at which the clock is read
            (``loop_start_time`` in the config); None for the whole path.
        offsets: Every case's per-member errors with the columns of
            ``path_offset.path_offsets`` and ``case``, ``init_time``.
    """

    name: str
    jma_number: str
    cases: list[CaseResult]
    n_cases: int
    n_members: int
    best_track: pd.DataFrame
    end: TropicalEnd
    dissipation_time: pd.Timestamp | None
    clock_until: pd.Timestamp | None
    offsets: pd.DataFrame

    @property
    def label(self) -> str:
        """Short name for legends, e.g. "Krovanh (T2624)"."""
        return f"{self.name.title()} (T{self.jma_number})"


def load_storm(cfg: dict, outputs_dir: Path = OUTPUTS_DIR) -> StormResult:
    """Read one storm's evaluated cases and measure them against its path.

    Args:
        cfg: The storm's parsed configuration.
        outputs_dir: Root of the results.

    Returns:
        The pooled :class:`StormResult`. Cases without an analysis are left
        out; ``n_cases`` still counts them.
    """
    storm = cfg["storm"]
    end = tropical_end(cfg)
    loop_start = storm.get("loop_start_time")
    clock_until = pd.Timestamp(str(loop_start)) if loop_start else None
    best_track = tropical_phase(load_track(besttrack_path(cfg)), end.time)
    cases, frames = [], []
    for entry in cfg["cases"]:
        analysis_dir = case_dir(cfg, entry["id"], outputs_dir) / "analysis"
        if not (analysis_dir / "summary.csv").exists():
            continue
        case = load_case(entry["id"], entry["init_time"], analysis_dir)
        cases.append(case)
        offsets = path_offsets(
            case["errors"], best_track, case.init_time, clock_until
        )
        offsets.insert(0, "init_time", case.init_time)
        offsets.insert(0, "case", case.case_id)
        frames.append(offsets)
    dissipation = storm.get("dissipation_time")
    return StormResult(
        name=str(storm["name"]),
        jma_number=str(storm["jma_number"]),
        cases=sorted(cases, key=lambda case: case.init_time),
        n_cases=len(cfg["cases"]),
        n_members=int(cfg["forecast"]["num_members"]),
        best_track=best_track,
        end=end,
        dissipation_time=pd.Timestamp(str(dissipation)) if dissipation else None,
        clock_until=clock_until,
        offsets=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
    )


def _cases_reaching(storm: StormResult, lead_hours: int) -> tuple[int, int]:
    """Cases scored at this lead, and how many of them have a readable clock.

    A case is scored where its valid time has a reference row; its clock is
    readable where that time is also no later than ``clock_until``.
    """
    times = set(storm.best_track["time"])
    valid = [case.init_time + pd.Timedelta(hours=lead_hours) for case in storm.cases]
    scored = [when for when in valid if when in times]
    clocked = [when for when in scored
               if storm.clock_until is None or when <= storm.clock_until]
    return len(scored), len(clocked)


def road_and_clock(storms: list[StormResult], within_km: float = WITHIN_KM) -> pd.DataFrame:
    """Pool each storm's member forecasts by lead time.

    Args:
        storms: The storms being compared.
        within_km: Radius for "at the observed position" and "on the
            observed path".

    Returns:
        One row per storm and lead: ``storm``, ``lead_hours``, ``n_cases``
        (cases scored at this lead), ``n_run`` (members run in them),
        ``n_present`` (members still tracking the storm), ``mean_error_km``,
        ``n_at_position`` (error within ``within_km``), ``n_beyond`` (behind
        the path's start or past its end), ``n_on_path`` (within
        ``within_km`` of the path and not beyond it), ``n_clock_cases``
        (cases whose valid time is within the clock's window), ``n_clock``
        (on the path with a readable clock), the quartiles of ``clock_hours`` over
        those (``clock_q25_hours``, ``clock_median_hours``,
        ``clock_q75_hours``) and ``n_ahead`` (clock above zero).
    """
    rows = []
    for storm in storms:
        for lead, group in storm.offsets.groupby("lead_hours", sort=True):
            n_cases, n_clock_cases = _cases_reaching(storm, int(lead))
            beyond = group["before_start"] | group["past_end"]
            on_path = group.loc[~beyond & (group["path_km"] <= within_km)]
            clock = on_path["clock_hours"].dropna()
            rows.append(
                {
                    "storm": storm.label,
                    "lead_hours": int(lead),
                    "n_cases": n_cases,
                    "n_run": n_cases * storm.n_members,
                    "n_present": len(group),
                    "mean_error_km": group["error_km"].mean(),
                    "n_at_position": int((group["error_km"] <= within_km).sum()),
                    "n_beyond": int(beyond.sum()),
                    "n_on_path": len(on_path),
                    "n_clock_cases": n_clock_cases,
                    "n_clock": len(clock),
                    "clock_q25_hours": clock.quantile(0.25) if len(clock) else np.nan,
                    "clock_median_hours": clock.median() if len(clock) else np.nan,
                    "clock_q75_hours": clock.quantile(0.75) if len(clock) else np.nan,
                    "n_ahead": int((clock > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def track_ends(storms: list[StormResult]) -> pd.DataFrame:
    """When each member's track ended, against the storm's milestones.

    The tracker follows a vortex, not a tropical storm: it carries on
    through a weakening to a depression and through an extratropical
    transition, until the vortex fades or leaves its criteria. Comparing
    the track's end with the end of the tropical phase alone therefore
    reads as the model keeping the storm alive for days. Where the config
    records when the storm dissipated, the end is compared with that too.

    Args:
        storms: The storms being compared.

    Returns:
        One row per member that found the storm: ``storm``, ``case``,
        ``member``, ``track_end``, ``after_tropical_end_hours``,
        ``ended_while_tropical`` (the track stopped before the observed storm
        left the tropical phase: the member lost it),
        ``after_dissipation_hours`` (NaN without a dissipation time) and
        ``lost_before_forecast_end``.
    """
    rows = []
    for storm in storms:
        end = pd.Timestamp(storm.end.time)
        for case in storm.cases:
            lifetime = case["lifetime"].loc[case["lifetime"]["found"].astype(bool)]
            for _, member in lifetime.iterrows():
                track_end = pd.Timestamp(member["track_end"])
                rows.append(
                    {
                        "storm": storm.label,
                        "case": case.case_id,
                        "member": int(member["member"]),
                        "track_end": track_end,
                        "after_tropical_end_hours": (track_end - end) / pd.Timedelta(hours=1),
                        "ended_while_tropical": track_end < end,
                        "after_dissipation_hours": (
                            (track_end - storm.dissipation_time) / pd.Timedelta(hours=1)
                            if storm.dissipation_time is not None else np.nan
                        ),
                        "lost_before_forecast_end": member.get("lost_before_forecast_end"),
                    }
                )
    return pd.DataFrame(rows)


def track_end_summary(ends: pd.DataFrame, storms: list[StormResult]) -> pd.DataFrame:
    """Collapse :func:`track_ends` onto one row per storm.

    Returns:
        ``storm``, ``n_run``, ``n_found``, ``n_ended_while_tropical``, the
        quartiles of ``after_tropical_end_hours`` and the median of
        ``after_dissipation_hours`` (NaN without a dissipation time).
    """
    rows = []
    for storm in storms:
        mine = ends.loc[ends["storm"] == storm.label]
        after = mine["after_tropical_end_hours"]
        rows.append(
            {
                "storm": storm.label,
                "n_run": len(storm.cases) * storm.n_members,
                "n_found": len(mine),
                "n_ended_while_tropical": int(mine["ended_while_tropical"].sum()),
                "after_tropical_end_q25_hours": after.quantile(0.25),
                "after_tropical_end_median_hours": after.median(),
                "after_tropical_end_q75_hours": after.quantile(0.75),
                "after_dissipation_median_hours": mine["after_dissipation_hours"].median(),
            }
        )
    return pd.DataFrame(rows)


def spread_skill_by_storm(storms: list[StormResult]) -> pd.DataFrame:
    """``compare.spread_skill`` for every storm, stacked.

    Returns:
        Its columns, with ``storm`` in front.
    """
    return pd.concat(
        [spread_skill(storm.cases).assign(storm=storm.label) for storm in storms],
        ignore_index=True,
    )[["storm", "case", "init_time", "n_leads", "max_lead_hours",
       "corr_spread_error", "mean_error_to_spread"]]


def _departure(valid_time: pd.Series, on_road: np.ndarray, beyond: np.ndarray):
    """When a member left the observed road for good.

    The first position, among those not beyond the path's ends, after which
    no later such position is on the road. NaT for a member on the road to
    its end and for one never on it.
    """
    inside = np.where(~beyond)[0]
    if not on_road[inside].any():
        return pd.NaT
    last_on = inside[on_road[inside]][-1]
    after = inside[inside > last_on]
    return pd.Timestamp(valid_time.iloc[after[0]]) if len(after) else pd.NaT


def member_shape_and_clock(
    storms: list[StormResult],
    within_km: float = WITHIN_KM,
    clock_leads=CLOCK_LEADS,
) -> pd.DataFrame:
    """One row per member: how far its track is from the observed road as a
    shape, and how early or late it ran on it.

    The two are reported side by side rather than folded into one number:
    the position error is their mixture. The shape is the Chamfer distance
    between the member's track and the observed path (``path_offset``),
    without the member's positions beyond either end of the path. The clock
    is read only at positions on the road, and is summarised three ways,
    since no one summary serves every storm: Dujuan's clock is zero before
    its turn and +14 h after, so a median over the whole forecast dilutes
    what a reading at one lead shows, while a reading at one lead does not
    exist for every case.

    Args:
        storms: The storms being compared.
        within_km: Radius for "on the road".
        clock_leads: Leads at which the clock is also read alone.

    Returns:
        ``storm``, ``case``, ``member``, ``n_rows``, ``n_beyond``,
        ``mean_error_km``, ``chamfer_km``, ``n_clock`` (on-road positions with
        a readable clock), ``clock_median_hours``, ``clock_at_<L>h`` for each
        lead (NaN where the member is off the road, beyond the path, or past
        ``clock_until`` there), ``clock_trend`` (slope of the clock against
        lead time, hours per hour; for a member moving along the road at a
        steady fraction k of the storm's speed it is k - 1), ``departure``
        (when the member left the road for good, NaT if it did not) and
        ``on_road_to_end``.
    """
    rows = []
    for storm in storms:
        for case in storm.cases:
            reference = storm.best_track.loc[storm.best_track["time"] >= case.init_time]
            mine = storm.offsets.loc[storm.offsets["case"] == case.case_id]
            for member, group in mine.groupby("member", sort=True):
                group = group.sort_values("valid_time").reset_index(drop=True)
                beyond = (group["before_start"] | group["past_end"]).to_numpy()
                on_road = ~beyond & (group["path_km"] <= within_km).to_numpy()
                readable = on_road & group["clock_hours"].notna().to_numpy()
                clock = group.loc[readable, ["lead_hours", "clock_hours"]]
                inside = group.loc[~beyond]
                shape = chamfer_km(
                    pd.DataFrame({"time": inside["valid_time"], "lat": inside["fcst_lat"],
                                  "lon": inside["fcst_lon"]}),
                    reference,
                )
                at = clock.set_index("lead_hours")["clock_hours"]
                trend = (
                    float(np.polyfit(clock["lead_hours"], clock["clock_hours"], 1)[0])
                    if len(clock) >= MIN_TREND_ROWS else np.nan
                )
                departure = _departure(group["valid_time"], on_road, beyond)
                rows.append(
                    {
                        "storm": storm.label,
                        "case": case.case_id,
                        "member": int(member),
                        "n_rows": len(group),
                        "n_beyond": int(beyond.sum()),
                        "mean_error_km": group["error_km"].mean(),
                        "chamfer_km": shape,
                        "n_clock": len(clock),
                        "clock_median_hours": clock["clock_hours"].median(),
                        **{f"clock_at_{lead}h": at.get(lead, np.nan) for lead in clock_leads},
                        "clock_trend": trend,
                        "departure": departure,
                        "on_road_to_end": bool(on_road.any()) and pd.isna(departure),
                    }
                )
    return pd.DataFrame(rows)
