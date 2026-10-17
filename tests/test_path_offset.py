"""Tests for the distance to the observed path and the clock along it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.analysis.path_offset import dense_path, path_offsets

INIT = "2026-09-01T00:00"
# One degree of latitude.
DEGREE_KM = 111.19


def _northward(rows: int = 5, start: str = INIT) -> pd.DataFrame:
    """A reference moving north one degree every six hours along 130E."""
    times = pd.date_range(start, periods=rows, freq="6h")
    return pd.DataFrame({"time": times, "lat": 20.0 + np.arange(rows), "lon": 130.0})


def _errors(*positions) -> pd.DataFrame:
    """Forecast positions as ``(valid_time, lat, lon)``."""
    return pd.DataFrame(
        [{"valid_time": pd.Timestamp(t), "fcst_lat": lat, "fcst_lon": lon}
         for t, lat, lon in positions]
    )


def test_a_member_on_the_path_six_hours_ahead() -> None:
    out = path_offsets(_errors(("2026-09-01 06:00", 22.0, 130.0)), _northward(), INIT)
    assert out["path_km"].iloc[0] == pytest.approx(0.0, abs=0.5)
    assert out["clock_hours"].iloc[0] == pytest.approx(6.0)
    assert not out["past_end"].iloc[0] and not out["before_start"].iloc[0]


def test_a_member_beside_the_path_on_time() -> None:
    out = path_offsets(_errors(("2026-09-01 12:00", 22.0, 131.0)), _northward(), INIT)
    assert out["path_km"].iloc[0] == pytest.approx(DEGREE_KM * np.cos(np.radians(22)), rel=0.01)
    assert out["clock_hours"].iloc[0] == pytest.approx(0.0, abs=0.2)


def test_a_member_past_either_end_has_no_clock() -> None:
    out = path_offsets(
        _errors(("2026-09-01 18:00", 26.0, 130.0), ("2026-09-01 06:00", 19.0, 130.0)),
        _northward(), INIT,
    )
    assert list(out["past_end"]) == [True, False]
    assert list(out["before_start"]) == [False, True]
    assert out["clock_hours"].isna().all()


def test_the_path_starts_at_init() -> None:
    """A member back where the storm was before init is behind the start."""
    out = path_offsets(_errors(("2026-09-01 12:00", 20.0, 130.0)), _northward(),
                       "2026-09-01T06:00")
    assert out["before_start"].iloc[0]


def test_the_clock_stops_where_the_path_turns_back() -> None:
    """North for a day, then back south 0.2 degrees to the east.

    Read along the whole path, a member at 21N on the way up is nearer the
    return leg and gets a clock of days. Cut at the turn it reads the way up.
    """
    up = _northward(5)
    down = pd.DataFrame({
        "time": pd.date_range("2026-09-02 06:00", periods=4, freq="6h"),
        "lat": [23.0, 22.0, 21.0, 20.0], "lon": 130.2,
    })
    track = pd.concat([up, down], ignore_index=True)
    member = _errors(("2026-09-01 06:00", 21.0, 130.15), ("2026-09-02 12:00", 22.0, 130.2))

    loose = path_offsets(member, track, INIT)
    assert loose["clock_hours"].iloc[0] > 24
    cut = path_offsets(member, track, INIT, clock_until="2026-09-02T00:00")
    assert cut["clock_hours"].iloc[0] == pytest.approx(0.0, abs=0.5)
    assert np.isnan(cut["clock_hours"].iloc[1])
    assert cut["path_km"].iloc[1] == pytest.approx(0.0, abs=0.5)


def test_the_dense_path_keeps_its_ends_and_crosses_the_dateline() -> None:
    track = pd.DataFrame({
        "time": pd.date_range(INIT, periods=2, freq="6h"), "lat": [40.0, 41.0],
        "lon": [179.0, -179.0],
    })
    path = dense_path(track, points_per_segment=4)
    assert path["time"].iloc[-1] == track["time"].iloc[-1]
    assert np.allclose(np.diff(path["lon"]), 0.5)
