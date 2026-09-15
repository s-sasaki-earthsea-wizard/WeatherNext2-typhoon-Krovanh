"""Tests for the initial-storm table handed to the bundled cyclone tracker."""

import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.inference.tracker import (
    INITIAL_STORM_DTYPES,
    TRACKER_OVERRIDES,
    build_tracker,
    empty_initial_storms,
    initial_storms_from_positions,
)


def test_empty_table_keeps_the_schema() -> None:
    """The reason None is not passed through: the tracker indexes by column."""
    frame = empty_initial_storms()
    assert frame.empty
    assert list(frame.columns) == list(INITIAL_STORM_DTYPES)


def test_positions_are_typed_and_wrapped() -> None:
    frame = initial_storms_from_positions(
        [("2624", 22.6, -228.1)], np.datetime64("2026-09-01T00:00")
    )
    assert frame.loc[0, "lon"] == 131.9
    assert frame.loc[0, "lat"] == 22.6
    assert frame.loc[0, "valid_time"] == pd.Timestamp("2026-09-01T00:00")
    assert list(frame.columns) == list(INITIAL_STORM_DTYPES)


def test_positive_longitudes_are_left_alone() -> None:
    frame = initial_storms_from_positions(
        [("2624", 22.6, 131.9)], np.datetime64("2026-09-01T00:00")
    )
    assert frame.loc[0, "lon"] == 131.9


@pytest.mark.parametrize("name,value", TRACKER_OVERRIDES.items())
def test_overrides_reach_the_tracker(name: str, value: object) -> None:
    pytest.importorskip("weathernext.cyclones.direct_tracker")
    assert getattr(build_tracker(), name) == value


def test_overrides_can_be_put_back() -> None:
    pytest.importorskip("weathernext.cyclones.direct_tracker")
    tracker = build_tracker(enforce_physically_consistent_quadrants_and_winds=True)
    assert tracker.enforce_physically_consistent_quadrants_and_winds is True
