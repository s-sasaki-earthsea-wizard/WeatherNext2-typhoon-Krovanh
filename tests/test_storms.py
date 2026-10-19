"""Tests for pooling the cases of each storm for the comparison across storms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.analysis.compare import CaseResult
from wn2_typhoon.analysis.storms import (
    StormResult,
    member_shape_and_clock,
    road_and_clock,
    track_ends,
)
from wn2_typhoon.config import TropicalEnd

INIT = "2026-09-01T00:00"


def _northward(rows: int = 5) -> pd.DataFrame:
    """A reference moving north one degree every six hours along 130E."""
    times = pd.date_range(INIT, periods=rows, freq="6h")
    return pd.DataFrame({"time": times, "lat": 20.0 + np.arange(rows), "lon": 130.0})


def _storm(offsets: pd.DataFrame, inits, lifetimes=None, dissipation=None) -> StormResult:
    cases = [
        CaseResult(f"init-{i}", pd.Timestamp(i), {"lifetime": (lifetimes or {}).get(i)})
        for i in inits
    ]
    return StormResult(
        name="TESTSTORM", jma_number="2699", cases=cases, n_cases=len(cases),
        n_members=2, best_track=_northward(), end=TropicalEnd("2026-09-02T00:00", "depression"),
        dissipation_time=dissipation, clock_until=None, offsets=offsets,
    )


def test_pooling_counts_cases_reaching_the_lead_and_leaves_out_the_beyond() -> None:
    offsets = pd.DataFrame({
        "case": ["a", "a", "b", "b"],
        "lead_hours": [6, 6, 6, 6],
        "error_km": [50.0, 400.0, 100.0, 120.0],
        "path_km": [10.0, 300.0, 20.0, 0.0],
        "before_start": [False, False, False, False],
        "past_end": [False, False, False, True],
        "clock_hours": [2.0, -10.0, -1.0, np.nan],
    })
    # The second case starts 18 h later: its lead 6 h is still on the reference.
    storm = _storm(offsets, ["2026-09-01 00:00", "2026-09-01 18:00"])
    row = road_and_clock([storm], within_km=200.0).iloc[0]
    assert (row["n_cases"], row["n_run"], row["n_present"]) == (2, 4, 4)
    assert (row["n_at_position"], row["n_beyond"], row["n_on_path"]) == (3, 1, 2)
    assert (row["n_clock"], row["n_ahead"], row["clock_median_hours"]) == (2, 1, 0.5)

    late = _storm(offsets, ["2026-09-01 00:00", "2026-09-02 00:00"])
    assert road_and_clock([late]).iloc[0]["n_cases"] == 1


def test_track_ends_read_against_the_tropical_end_and_the_dissipation() -> None:
    lifetime = pd.DataFrame({
        "member": [0, 1, 2], "found": [True, True, False],
        "track_end": pd.to_datetime(["2026-09-01 18:00", "2026-09-05 00:00", None]),
        "lost_before_forecast_end": [True, False, None],
    })
    storm = _storm(pd.DataFrame(), [INIT], {INIT: lifetime},
                   dissipation=pd.Timestamp("2026-09-05 00:00"))
    ends = track_ends([storm])
    assert list(ends["member"]) == [0, 1]
    assert list(ends["after_tropical_end_hours"]) == [-6.0, 72.0]
    assert list(ends["ended_while_tropical"]) == [True, False]
    assert list(ends["after_dissipation_hours"]) == [-78.0, 0.0]


def test_one_member_on_the_road_then_off_it() -> None:
    """On the road and running ahead by 0, 3, 6 h, then 500 km off it."""
    valid = pd.date_range("2026-09-01 06:00", periods=4, freq="6h")
    offsets = pd.DataFrame({
        "case": "init-" + INIT, "member": 0, "valid_time": valid,
        "lead_hours": [6, 12, 18, 24],
        "fcst_lat": [21.0, 22.1, 23.2, 23.0], "fcst_lon": [130.0, 130.0, 130.0, 135.0],
        "error_km": [0.0, 10.0, 20.0, 500.0],
        "path_km": [0.0, 0.0, 0.0, 500.0],
        "before_start": False, "past_end": False,
        "clock_hours": [0.0, 3.0, 6.0, 40.0],
    })
    row = member_shape_and_clock([_storm(offsets, [INIT])], clock_leads=(12, 24)).iloc[0]
    assert row["n_clock"] == 3
    assert row["clock_median_hours"] == 3.0
    assert row["clock_at_12h"] == 3.0
    assert np.isnan(row["clock_at_24h"])
    assert row["clock_trend"] == pytest.approx(0.5)
    assert row["departure"] == valid[3]
    assert not row["on_road_to_end"]
    assert row["chamfer_km"] > 0


def test_a_member_past_the_end_has_not_left_the_road() -> None:
    valid = pd.date_range("2026-09-01 06:00", periods=3, freq="6h")
    offsets = pd.DataFrame({
        "case": "init-" + INIT, "member": 0, "valid_time": valid, "lead_hours": [6, 12, 18],
        "fcst_lat": [21.0, 22.0, 26.0], "fcst_lon": 130.0, "error_km": [0.0, 0.0, 400.0],
        "path_km": [0.0, 0.0, 220.0], "before_start": False, "past_end": [False, False, True],
        "clock_hours": [0.0, 0.0, np.nan],
    })
    row = member_shape_and_clock([_storm(offsets, [INIT])]).iloc[0]
    assert pd.isna(row["departure"]) and row["on_road_to_end"]
    assert row["n_beyond"] == 1
    assert np.isnan(row["clock_trend"])
