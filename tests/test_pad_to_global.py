"""The regional crop has to become a global grid before the tracker sees it."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from wn2_typhoon.inference.tracker import pad_to_global


def _crop(resolution: float) -> xr.Dataset:
    lat = np.arange(15.0, 50.0 + resolution / 2, resolution, dtype="float32")
    lon = np.arange(115.0, 150.0 + resolution / 2, resolution, dtype="float32")
    values = np.ones((len(lat), len(lon)), dtype="float32")
    return xr.Dataset(
        {"cyclone_exists_gaussian_unit_mode": (("lat", "lon"), values)},
        coords={"lat": lat, "lon": lon},
    )


@pytest.mark.parametrize("resolution", [0.25, 1.0])
def test_grid_covers_the_whole_globe(resolution: float) -> None:
    padded = pad_to_global(_crop(resolution), resolution)
    assert float(padded.lat.min()) == -90.0
    assert float(padded.lat.max()) == 90.0
    # The tracker requires lon.min() == 0 and lon.max() < 360.
    assert float(padded.lon.min()) == 0.0
    assert float(padded.lon.max()) < 360.0


def test_outside_the_crop_is_zero_not_nan() -> None:
    """Zero existence is below the cyclogenesis threshold; NaN would trip the
    tracker's NaN checks."""
    padded = pad_to_global(_crop(1.0), 1.0)
    field = padded["cyclone_exists_gaussian_unit_mode"]
    assert not np.isnan(field.values).any()
    assert float(field.sel(lat=0.0, lon=0.0)) == 0.0


def test_the_crop_itself_is_untouched() -> None:
    crop = _crop(1.0)
    padded = pad_to_global(crop, 1.0)
    inside = padded.sel(lat=crop.lat, lon=crop.lon)
    xr.testing.assert_equal(
        inside["cyclone_exists_gaussian_unit_mode"],
        crop["cyclone_exists_gaussian_unit_mode"],
    )
