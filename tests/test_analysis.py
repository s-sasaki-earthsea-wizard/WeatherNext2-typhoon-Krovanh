"""Tests for the track comparison.

The selection rule and the lead-0 exclusion are the two places where a quiet
mistake would produce a plausible-looking but wrong skill number, so both are
tested against constructed input rather than only against the real output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.analysis.track_error import (
    SELECTION_THRESHOLD_KM,
    ensemble_summary,
    genesis_report,
    lifetime_report,
    position_errors,
    select_storm,
)
from wn2_typhoon.utils.geo import (
    along_cross_track_km,
    haversine_km,
    initial_bearing_deg,
    mean_position,
    track_bearing_deg,
)

INIT = "2026-09-01T00:00"


def _best_track(times, lats, lons, pressures=None):
    """Build a reference track in the loader's schema."""
    return pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "lat": lats,
            "lon": lons,
            "pressure_hpa": pressures if pressures is not None else [990.0] * len(times),
            "wind_kt": [40.0] * len(times),
            "grade": [np.nan] * len(times),
            "remark": [""] * len(times),
            "source": ["preliminary"] * len(times),
        }
    )


def _tracks(rows):
    """Build tracker output from ``(member, track_id, time, lat, lon, mslp)``."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "member", "track_id", "valid_time", "lat", "lon",
            "minimum_sea_level_pressure_hpa",
        ],
    )
    frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    frame["maximum_sustained_wind_speed_knots"] = 45.0
    return frame


def test_haversine_matches_known_distances() -> None:
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.02)
    assert haversine_km(35.68, 139.77, 34.69, 135.50) == pytest.approx(403, abs=3)


def test_haversine_wraps_across_the_dateline() -> None:
    """A one-degree step across 180 must not come out as 359 degrees."""
    assert haversine_km(0, 179.5, 0, -179.5) == pytest.approx(111.19, abs=0.02)
    assert haversine_km(0, 179.5, 0, 180.5) == pytest.approx(111.19, abs=0.02)


def test_bearings_point_the_expected_way() -> None:
    assert initial_bearing_deg(0, 0, 1, 0) == pytest.approx(0.0, abs=1e-6)
    assert initial_bearing_deg(0, 0, 0, 1) == pytest.approx(90.0, abs=1e-6)


def test_along_track_is_positive_when_the_forecast_is_ahead() -> None:
    """A storm moving north, forecast placed north of it, is running fast."""
    along, cross = along_cross_track_km(25.0, 130.0, 25.9, 130.0, 0.0)
    assert along == pytest.approx(100, abs=2)
    assert cross == pytest.approx(0, abs=1)


def test_cross_track_is_positive_to_the_right_of_motion() -> None:
    along, cross = along_cross_track_km(25.0, 130.0, 25.0, 131.0, 0.0)
    assert cross == pytest.approx(101, abs=3)
    assert along == pytest.approx(0, abs=2)


def test_track_bearing_of_a_northward_track_is_north() -> None:
    bearings = track_bearing_deg([20.0, 21.0, 22.0], [130.0, 130.0, 130.0])
    assert np.allclose(bearings, 0.0, atol=1e-6)


def test_mean_position_averages_on_the_sphere() -> None:
    lat, lon = mean_position([0.0, 0.0], [179.0, -179.0])
    assert lon == pytest.approx(180.0, abs=1e-6)
    assert lat == pytest.approx(0.0, abs=1e-6)


def test_selection_takes_the_nearest_track_not_the_lowest_id() -> None:
    """Ids are not stable between members, so only distance may decide."""
    best = _best_track(["2026-09-01 06:00"], [22.3], [131.4])
    tracks = _tracks(
        [
            (0, 7, "2026-09-01 06:00", 22.4, 131.5, 995.0),
            (0, 0, "2026-09-01 06:00", 10.0, 200.0, 1000.0),
        ]
    )
    selected, selections = select_storm(tracks, best, INIT)
    assert selections[0].track_id == "7"
    assert selections[0].distance_km < 50
    assert selections[0].runner_up_km > 5000
    assert set(selected["track_id"].astype(str)) == {"7"}


def test_selection_ignores_lead_zero() -> None:
    """A row at init time is the tracker's input, not a prediction.

    Output produced before seeding was dropped carries the observed position at
    lead 0, so anchoring there would match a track to itself at zero distance
    and could pick the wrong storm.
    """
    best = _best_track(
        ["2026-09-01 00:00", "2026-09-01 06:00"], [22.6, 22.3], [131.9, 131.4]
    )
    tracks = _tracks(
        [
            # A decoy sitting exactly on the observed position at lead 0 only.
            (0, 1, "2026-09-01 00:00", 22.6, 131.9, 1000.0),
            (0, 1, "2026-09-01 06:00", 40.0, 150.0, 1000.0),
            (0, 2, "2026-09-01 06:00", 22.35, 131.45, 990.0),
        ]
    )
    _, selections = select_storm(tracks, best, INIT)
    assert selections[0].track_id == "2"
    assert selections[0].anchor_time == pd.Timestamp("2026-09-01 06:00")


def test_selection_records_a_miss_rather_than_taking_the_nearest() -> None:
    best = _best_track(["2026-09-01 06:00"], [22.3], [131.4])
    tracks = _tracks([(0, 0, "2026-09-01 06:00", 40.0, 160.0, 1000.0)])
    selected, selections = select_storm(tracks, best, INIT)
    assert selections[0].found is False
    assert selections[0].distance_km > SELECTION_THRESHOLD_KM
    assert selected.empty


def test_position_errors_drop_lead_zero_and_keep_the_rest() -> None:
    best = _best_track(
        ["2026-09-01 00:00", "2026-09-01 06:00", "2026-09-01 12:00"],
        [22.6, 22.3, 22.4],
        [131.9, 131.4, 131.7],
    )
    tracks = _tracks(
        [
            (0, 1, "2026-09-01 00:00", 22.6, 131.9, np.nan),
            (0, 1, "2026-09-01 06:00", 22.3, 131.4, 985.0),
            (0, 1, "2026-09-01 12:00", 22.4, 131.7, 980.0),
        ]
    )
    errors = position_errors(tracks, best, INIT)
    assert list(errors["lead_hours"]) == [6, 12]
    assert errors["error_km"].max() < 1.0
    assert list(errors["dp_hpa"]) == [-5.0, -10.0]


def test_ensemble_summary_separates_the_two_ensemble_errors() -> None:
    """Members either side of the truth cancel, so the mean position wins.

    Reporting only the error of the ensemble mean would flatter the ensemble,
    and reporting only the mean of member errors would hide that they cancel.
    """
    best = _best_track(["2026-09-01 06:00"], [22.0], [131.0])
    tracks = _tracks(
        [
            (0, 1, "2026-09-01 06:00", 23.0, 131.0, 990.0),
            (1, 1, "2026-09-01 06:00", 21.0, 131.0, 990.0),
        ]
    )
    summary = ensemble_summary(position_errors(tracks, best, INIT))
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_members"] == 2
    assert row["mean_error_km"] == pytest.approx(111, abs=2)
    assert row["ensemble_mean_error_km"] == pytest.approx(0, abs=1)
    assert row["spread_km"] == pytest.approx(111, abs=2)


def test_genesis_is_not_resolvable_when_the_case_starts_too_late() -> None:
    """The tracker cannot report a storm before the first rollout step."""
    best = _best_track(["2026-09-01 06:00"], [22.3], [131.4])
    tracks = _tracks([(0, 1, "2026-09-01 06:00", 22.3, 131.4, 990.0)])
    selected, selections = select_storm(tracks, best, INIT)
    report = genesis_report(selected, selections, "2026-09-01T00:00", INIT, 6)
    assert bool(report["resolvable"].iloc[0]) is False

    early = genesis_report(selected, selections, "2026-09-01T00:00",
                           "2026-08-31T12:00", 6)
    assert bool(early["resolvable"].iloc[0]) is True


def test_lifetime_reports_the_end_and_the_deepest_pressure() -> None:
    best = _best_track(
        ["2026-09-01 06:00", "2026-09-01 12:00"], [22.3, 22.4], [131.4, 131.7]
    )
    tracks = _tracks(
        [
            (0, 1, "2026-09-01 06:00", 22.3, 131.4, 990.0),
            (0, 1, "2026-09-01 12:00", 22.4, 131.7, 975.0),
        ]
    )
    selected, selections = select_storm(tracks, best, INIT)
    report = lifetime_report(selected, selections, "2026-09-01T18:00")
    row = report.iloc[0]
    assert row["track_end"] == pd.Timestamp("2026-09-01 12:00")
    assert row["end_error_hours"] == pytest.approx(-6.0)
    assert row["min_pressure_hpa"] == pytest.approx(975.0)
    assert row["duration_hours"] == pytest.approx(6.0)


def test_a_member_that_never_matched_still_appears_in_the_reports() -> None:
    """A miss is a result. Dropping it would silently shrink the ensemble."""
    best = _best_track(["2026-09-01 06:00"], [22.3], [131.4])
    tracks = _tracks(
        [
            (0, 1, "2026-09-01 06:00", 22.3, 131.4, 990.0),
            (1, 1, "2026-09-01 06:00", 45.0, 170.0, 1000.0),
        ]
    )
    selected, selections = select_storm(tracks, best, INIT)
    genesis = genesis_report(selected, selections, "2026-09-01T00:00", INIT, 6)
    lifetime = lifetime_report(selected, selections, "2026-09-01T18:00")
    assert list(genesis["member"]) == [0, 1]
    assert list(genesis["found"]) == [True, False]
    assert list(lifetime["found"]) == [True, False]
    # pandas turns the missing end into NaT, since the column holds Timestamps.
    assert pd.isna(lifetime.loc[lifetime["member"] == 1, "track_end"].iloc[0])
    assert pd.isna(genesis.loc[genesis["member"] == 1, "lead_error_hours"].iloc[0])
