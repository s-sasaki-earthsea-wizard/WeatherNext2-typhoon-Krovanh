"""Tests for the CDS request builder. No network access."""

from pathlib import Path

import pytest

from wn2_typhoon.data.era5_download import build_requests, input_frames
from wn2_typhoon.model_spec import (
    CDS_PRESSURE_LEVELS,
    CDS_SINGLE_LEVELS,
    load_spec,
)

pytest.importorskip("weathernext.utils.fiddle_config_io")

OUT = Path("data/raw/era5")


@pytest.fixture(scope="module")
def spec():
    return load_spec("WeatherNext2")


def test_frames_are_the_step_apart() -> None:
    # Naive datetimes: everything in this project is UTC by convention.
    assert [m.isoformat() for m in input_frames("2026-09-01T00:00")] == [
        "2026-08-31T18:00:00",
        "2026-09-01T00:00:00",
    ]


def test_one_request_per_frame_and_dataset(spec) -> None:
    requests = build_requests("2026-09-01T00:00", spec, OUT)
    # Two frames x (pressure levels + single levels), plus the statics.
    assert len(requests) == 5
    assert sum(r.dataset == CDS_PRESSURE_LEVELS for r in requests) == 2
    assert sum(r.dataset == CDS_SINGLE_LEVELS for r in requests) == 3


def test_a_frame_selects_exactly_one_timestamp(spec) -> None:
    """The CDS expands year/month/day/time as a cross product."""
    for request in build_requests("2026-09-01T00:00", spec, OUT):
        for key in ("year", "month", "day", "time"):
            assert len(request.request[key]) == 1, key


def test_frames_that_straddle_midnight_stay_exact(spec) -> None:
    requests = build_requests("2026-09-01T00:00", spec, OUT)
    stamps = {
        (
            r.request["year"][0],
            r.request["month"][0],
            r.request["day"][0],
            r.request["time"][0],
        )
        for r in requests
        if r.dataset == CDS_PRESSURE_LEVELS
    }
    assert stamps == {("2026", "08", "31", "18:00"), ("2026", "09", "01", "00:00")}


def test_cases_share_the_frame_they_have_in_common(spec) -> None:
    """Both Krovanh cases use 2026-08-31 18 UTC, so it is fetched once."""
    early = {r.target.name for r in build_requests("2026-08-31T18:00", spec, OUT)}
    late = {r.target.name for r in build_requests("2026-09-01T00:00", spec, OUT)}
    shared = early & late
    assert "pressure_levels_20260831T18.nc" in shared
    assert "single_levels_20260831T18.nc" in shared
    assert "static.nc" in shared


def test_levels_and_variables_come_from_the_spec(spec) -> None:
    pressure = next(
        r for r in build_requests("2026-09-01T00:00", spec, OUT)
        if r.dataset == CDS_PRESSURE_LEVELS
    )
    assert pressure.request["pressure_level"] == [
        str(level) for level in spec.pressure_levels
    ]
    assert pressure.request["variable"] == [
        spec.era5_source(v).variable for v in spec.pressure_level_vars
    ]


def test_statics_are_requested_once_and_separately(spec) -> None:
    static = next(
        r for r in build_requests("2026-09-01T00:00", spec, OUT)
        if r.target.name == "static.nc"
    )
    assert static.request["variable"] == ["geopotential", "land_sea_mask"]
