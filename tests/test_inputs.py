"""Check the target template against the upstream splitting it stands in for.

The public 1 deg sample carries real target frames. Taking only its two input
frames, rebuilding the targets as a NaN template and splitting that must give
inputs and forcings identical to splitting the untouched sample, and a target
template of the same shape. If it does, the pod can be handed a two-frame file
instead of a 42-frame one.

The sample is cached by ``make smoke-mini``; without it these are skipped.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("weathernext.utils.data_utils")

from weathernext.utils import data_utils

from wn2_typhoon.inference.inputs import (
    extend_with_target_template,
    split_for_rollout,
    target_times,
)
from wn2_typhoon.model_spec import load_spec, load_task_config

SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data/cache"
    / "source-hres_forecast_init-2024-10-07 00:00:00_res-1.0_levels-13_steps-04.nc"
)
MODEL = "WeatherNextCyclones_Mini"
STEP_HOURS = 6
# The sample holds two input frames and four target frames.
LEAD_HOURS = 4 * STEP_HOURS


@pytest.fixture(scope="module")
def sample() -> xr.Dataset:
    if not SAMPLE.exists():
        pytest.skip(f"run `make smoke-mini` to cache {SAMPLE.name}")
    return xr.load_dataset(SAMPLE)


@pytest.fixture(scope="module")
def spec():
    return load_spec(MODEL)


@pytest.fixture(scope="module")
def task_config():
    return load_task_config(MODEL)


def _frames_only(sample: xr.Dataset, spec) -> xr.Dataset:
    """The two input frames, as build_inputs would have stored them."""
    kept = [v for v in sample.data_vars if v in set(spec.downloaded_vars)]
    frames = sample[kept].isel(time=slice(0, 2))
    # Statics lose their time dim in storage; the sample already has them that
    # way, so this only guards against a change in the sample.
    return frames


def test_target_times_continue_the_input_spacing(sample, spec) -> None:
    frames = _frames_only(sample, spec)
    times = target_times(frames, num_steps=4, step_hours=STEP_HOURS)
    expected = sample.time.values[2:]
    assert np.array_equal(times, expected)


def test_template_is_lazy(sample, spec) -> None:
    frames = _frames_only(sample, spec)
    extended = extend_with_target_template(frames, spec, 4, STEP_HOURS)
    chunked = [
        name for name, da in extended.data_vars.items() if da.chunks is not None
    ]
    assert chunked, "the template must stay dask-backed or it costs 17 GB"
    assert extended.sizes["time"] == 6
    assert set(spec.target_vars) <= set(extended.data_vars)


def test_datetime_is_filled_for_every_frame(sample, spec) -> None:
    frames = _frames_only(sample, spec)
    extended = extend_with_target_template(frames, spec, 4, STEP_HOURS)
    assert not np.isnat(extended.datetime.values).any()
    assert np.array_equal(extended.datetime.values, sample.datetime.values)


def test_split_matches_the_upstream_split(sample, spec, task_config) -> None:
    reference_inputs, reference_targets, reference_forcings = (
        data_utils.extract_inputs_targets_forcings(
            sample.copy(deep=True),
            target_lead_times=slice(f"{STEP_HOURS}h", f"{LEAD_HOURS}h"),
            **dataclasses.asdict(task_config),
        )
    )
    inputs, targets, forcings = split_for_rollout(
        _frames_only(sample, spec), spec, LEAD_HOURS, STEP_HOURS, task_config
    )

    xr.testing.assert_allclose(inputs, reference_inputs)
    xr.testing.assert_allclose(forcings, reference_forcings)
    assert dict(targets.sizes) == dict(reference_targets.sizes)
    assert list(targets.data_vars) == list(reference_targets.data_vars)
