"""Tests for the stored crop's zarr chunking.

Chunking is not cosmetic here. Left to xarray's inference the crop came out as
624 files in 532 directories for 0.27 GB, and that file count is paid on the
pod's MooseFS volume, on the rsync off the pod and again on the SMB write to
the NAS.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from wn2_typhoon.inference.rollout import MAX_CHUNK_BYTES, chunking_for

# The crop as configured: 40 steps, 13 levels, 141 x 141 at 0.25 degrees.
CROP_4D = (40, 13, 141, 141)
CROP_3D = (40, 141, 141)


def _array(shape, dims):
    return xr.DataArray(np.zeros(shape, "float32"), dims=dims)


def test_the_configured_crop_is_one_chunk_per_variable() -> None:
    """41 MB is under the cap, so nothing is split."""
    array = _array(CROP_4D, ("time", "level", "lat", "lon"))
    assert chunking_for(array) == CROP_4D


def test_surface_fields_are_one_chunk_too() -> None:
    array = _array(CROP_3D, ("time", "lat", "lon"))
    assert chunking_for(array) == CROP_3D


def test_a_much_wider_region_splits_along_time_and_respects_the_cap() -> None:
    """Widening output.region must not produce chunks too big to read."""
    shape = (40, 13, 400, 400)
    array = _array(shape, ("time", "level", "lat", "lon"))
    chunks = chunking_for(array)
    assert chunks[1:] == shape[1:], "only the time axis may be split"
    assert 0 < chunks[0] < shape[0]
    assert int(np.prod(chunks)) * 4 <= MAX_CHUNK_BYTES


def test_an_array_without_time_is_left_whole() -> None:
    array = _array((141, 141), ("lat", "lon"))
    assert chunking_for(array) == (141, 141)


def test_a_single_step_larger_than_the_cap_still_gives_one_step() -> None:
    """A chunk of one time step is the floor; it must not come out as zero."""
    array = _array((4, 13, 2000, 2000), ("time", "level", "lat", "lon"))
    chunks = chunking_for(array)
    assert chunks[0] == 1


def test_chunking_survives_a_round_trip(tmp_path) -> None:
    """The store must read back byte for byte, and in far fewer files."""
    from wn2_typhoon.inference.rollout import save_zarr

    rng = np.random.default_rng(0)
    data = rng.normal(size=(8, 3, 20, 20)).astype("float32")
    dataset = xr.Dataset(
        {
            "mean_sea_level_pressure": (("time", "lat", "lon"), data[:, 0]),
            "temperature": (("time", "level", "lat", "lon"), data),
        },
        coords={
            "time": np.arange(8),
            "level": np.array([500, 700, 850]),
            "lat": np.linspace(15, 20, 20),
            "lon": np.linspace(120, 125, 20),
        },
    )
    store = save_zarr(dataset, tmp_path / "member-00.zarr")
    read_back = xr.open_zarr(store)
    for name in dataset.data_vars:
        assert np.array_equal(read_back[name].values, dataset[name].values)
        assert read_back[name].encoding["chunks"] == dataset[name].shape
