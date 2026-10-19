"""Tests for the drift-against-scatter ratio and its comparison with the observed storm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.analysis.compare import CaseResult
from wn2_typhoon.analysis.directedness import (
    directedness_by_time,
    directedness_lag,
    peclet,
    step_vectors_km,
    synoptic,
)
from wn2_typhoon.analysis.storms import StormResult
from wn2_typhoon.config import TropicalEnd

INIT = pd.Timestamp("2026-09-01 00:00")
DEGREE_KM = 111.19


def _track(lat, lon, start=INIT, column="time") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(lat), freq="6h")
    return pd.DataFrame({column: times, "lat": lat, "lon": lon})


def test_steps_are_km_east_and_north() -> None:
    steps = step_vectors_km([0.0, 1.0, 1.0], [0.0, 0.0, 1.0])
    assert steps[0] == pytest.approx([0.0, DEGREE_KM], rel=1e-3)
    assert steps[1] == pytest.approx([DEGREE_KM * np.cos(np.radians(1.0)), 0.0], rel=1e-3)


def test_steady_drift_is_unbounded_and_out_and_back_is_zero() -> None:
    straight = peclet(_track([0.0, 0.1, 0.2, 0.3, 0.4], 0.0), 24)
    assert straight.iloc[0] > 1e6
    shuttle = peclet(_track([0.0, 0.1, 0.0, 0.1, 0.0], 0.0), 24)
    assert shuttle.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_a_right_angle_turn_keeps_most_of_the_drift() -> None:
    """Two steps east, two north: net (2, 2), scatter 4 x 0.5, Pe = 8 / 2."""
    turn = peclet(_track([0.0, 0.0, 0.0, 0.1, 0.2], [0.0, 0.1, 0.2, 0.2, 0.2]), 24)
    assert turn.iloc[0] == pytest.approx(4.0, rel=1e-3)
    assert turn.index[0] == INIT + pd.Timedelta(hours=12)


def test_a_window_across_a_missing_row_is_skipped() -> None:
    track = _track([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 0.0).drop(index=3)
    assert peclet(track, 18).empty


def test_synoptic_drops_the_off_hour_rows() -> None:
    track = pd.DataFrame({"time": pd.to_datetime(
        ["2026-09-01 00:00", "2026-09-01 03:00", "2026-09-01 06:00", "2026-09-01 07:00"])})
    assert list(synoptic(track)["time"].dt.hour) == [0, 6]


def _storm(best_track, member_tracks) -> StormResult:
    tracks = pd.concat(
        [track.assign(member=member) for member, track in enumerate(member_tracks)],
        ignore_index=True,
    )
    case = CaseResult("init-a", INIT, {"track": tracks})
    return StormResult(
        name="TESTSTORM", jma_number="2699", cases=[case], n_cases=1,
        n_members=len(member_tracks), best_track=best_track,
        end=TropicalEnd("2026-09-03T00:00", "depression"), dissipation_time=None,
        clock_until=None, offsets=pd.DataFrame(),
    )


def test_members_that_stall_where_the_storm_drifts_read_below_it() -> None:
    rows = 9
    drifting = _track(0.1 * np.arange(rows) + 0.01 * (np.arange(rows) % 2), 0.0)
    shuttle = 0.05 * (np.arange(rows) % 2)
    members = [_track(shuttle, 0.001 * k, column="valid_time") for k in range(4)]
    table = directedness_by_time([_storm(drifting, members)], window_hours=(24,))
    assert len(table) == rows - 4
    assert (table["log_ratio_median"] < 0).all()
    assert (table["share_below"] == 1.0).all()


def test_the_lag_finds_members_running_late() -> None:
    centres = pd.date_range(INIT, periods=16, freq="6h")
    signal = np.sin(np.arange(16) / 2.0)
    late = np.sin((np.arange(16) - 2) / 2.0)  # the same curve, two steps later
    table = pd.DataFrame({
        "storm": "T", "window_hours": 24, "centre": centres,
        "pe_observed": 10 ** signal, "median_log_pe": late,
    })
    profile = directedness_lag(table, min_overlap=6)
    best = profile.loc[profile["best"]].iloc[0]
    assert best["lag_hours"] == 12
    assert best["correlation"] == pytest.approx(1.0)
    assert len(profile) > 1
