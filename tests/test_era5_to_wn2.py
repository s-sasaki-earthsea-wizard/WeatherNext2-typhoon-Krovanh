"""Tests for the ERA5 to WeatherNext 2 conversion.

The conversion itself needs downloaded ERA5 and is exercised by
`make prepare-inputs`; what is pinned here is the contract check that guards
it, because that is what stops a layout mistake reaching the pod.
"""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("weathernext.utils.fiddle_config_io")

from wn2_typhoon.data.era5_to_wn2 import validate_inputs
from wn2_typhoon.model_spec import load_spec

INIT = "2026-08-31T18:00"
STEP = 6


@pytest.fixture(scope="module")
def spec():
    return load_spec("WeatherNext2")


def _valid(spec) -> xr.Dataset:
    """A tiny dataset shaped exactly like a real input file."""
    lat = np.linspace(-90, 90, 5, dtype="float32")
    lon = np.array([0.0, 90.0, 180.0, 270.0], dtype="float32")
    levels = np.array(spec.pressure_levels, dtype="int32")
    times = np.array([0, STEP * 3600], dtype="int64").astype("timedelta64[s]")
    datetimes = np.array(
        [[np.datetime64("2026-08-31T12:00", "ns"), np.datetime64(INIT, "ns")]]
    )

    data = {}
    for name in spec.pressure_level_vars:
        data[name] = (
            ("batch", "time", "level", "lat", "lon"),
            np.zeros((1, 2, len(levels), len(lat), len(lon)), dtype="float32"),
        )
    for name in spec.single_level_vars:
        data[name] = (
            ("batch", "time", "lat", "lon"),
            np.zeros((1, 2, len(lat), len(lon)), dtype="float32"),
        )
    for name in spec.static_vars:
        data[name] = (("lat", "lon"), np.zeros((len(lat), len(lon)), dtype="float32"))

    return xr.Dataset(
        data,
        coords={
            "lat": lat,
            "lon": lon,
            "level": levels,
            "time": times.astype("timedelta64[ns]"),
            "datetime": (("batch", "time"), datetimes),
        },
    )


def test_a_correct_file_passes(spec) -> None:
    validate_inputs(_valid(spec), spec, INIT, STEP)


def test_descending_latitude_is_caught(spec) -> None:
    """The CDS delivers latitude north to south; forgetting to flip it is the
    easiest mistake to make and the hardest to notice."""
    dataset = _valid(spec).isel(lat=slice(None, None, -1))
    with pytest.raises(ValueError, match="lat must ascend"):
        validate_inputs(dataset, spec, INIT, STEP)


def test_shifted_longitude_is_caught(spec) -> None:
    dataset = _valid(spec).assign_coords(lon=_valid(spec).lon - 180)
    with pytest.raises(ValueError, match=r"lon must ascend"):
        validate_inputs(dataset, spec, INIT, STEP)


def test_missing_variable_is_caught(spec) -> None:
    dataset = _valid(spec).drop_vars("100m_u_component_of_wind")
    with pytest.raises(ValueError, match="100m_u_component_of_wind"):
        validate_inputs(dataset, spec, INIT, STEP)


def test_wrong_level_order_is_caught(spec) -> None:
    dataset = _valid(spec).isel(level=slice(None, None, -1))
    with pytest.raises(ValueError, match="do not match the checkpoint"):
        validate_inputs(dataset, spec, INIT, STEP)


def test_static_with_a_time_dim_is_caught(spec) -> None:
    dataset = _valid(spec)
    dataset["land_sea_mask"] = dataset["land_sea_mask"].expand_dims(time=2)
    with pytest.raises(ValueError, match="must not carry a time dim"):
        validate_inputs(dataset, spec, INIT, STEP)


def test_last_frame_must_be_the_init_time(spec) -> None:
    with pytest.raises(ValueError, match="not the init time"):
        validate_inputs(_valid(spec), spec, "2026-09-01T00:00", STEP)


def test_nan_is_only_allowed_in_sea_surface_temperature(spec) -> None:
    dataset = _valid(spec)
    dataset["sea_surface_temperature"][:] = np.nan
    validate_inputs(dataset, spec, INIT, STEP)

    dataset["2m_temperature"][:] = np.nan
    with pytest.raises(ValueError, match="2m_temperature contains NaN"):
        validate_inputs(dataset, spec, INIT, STEP)
