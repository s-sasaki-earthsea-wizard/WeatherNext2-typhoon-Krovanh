"""Tests for the model input contract.

These read the fiddle configs bundled with weathernext, so they need the
`model` dependency group but neither weights nor a GPU.
"""

from dataclasses import dataclass, field

import pytest

from wn2_typhoon import model_spec

weathernext_variables = pytest.importorskip("weathernext.utils.variables")

BUNDLED_MODELS = ["WeatherNext2", "WeatherNextCyclones", "WeatherNextCyclones_Mini"]


@dataclass
class FakeTaskConfig:
    """Minimal stand-in for a checkpoint task config."""

    input_variables: list[str]
    target_variables: list[str] = field(default_factory=list)
    forcing_variables: list[str] = field(default_factory=list)
    pressure_levels: list[int] = field(default_factory=list)
    input_duration: str = "12h"


@pytest.fixture(scope="module", params=BUNDLED_MODELS)
def spec(request) -> model_spec.ModelSpec:
    """A spec for each checkpoint bundled with weathernext."""
    return model_spec.load_spec(request.param)


def test_every_input_is_downloaded_or_computed(spec: model_spec.ModelSpec) -> None:
    # load_spec raises on an input it cannot place, so reaching here already
    # proves coverage; this pins the partition itself.
    groups = (
        spec.pressure_level_vars
        + spec.single_level_vars
        + spec.static_vars
        + spec.computed_vars
    )
    assert len(set(groups)) == len(groups)


def test_two_input_frames(spec: model_spec.ModelSpec) -> None:
    assert spec.input_duration == "12h"


def test_thirteen_weatherbench_levels(spec: model_spec.ModelSpec) -> None:
    assert spec.pressure_levels == weathernext_variables.PRESSURE_LEVELS_WEATHERBENCH_13


def test_classification_agrees_with_weathernext(spec: model_spec.ModelSpec) -> None:
    """ERA5_SOURCES must not contradict weathernext's own variable taxonomy.

    That taxonomy does not cover every input (it predates the 100 m winds
    WeatherNext2 wants), so it is used as a one-way check.
    """
    for variable in spec.pressure_level_vars:
        assert variable not in weathernext_variables.ALL_SURFACE_VARS
        assert variable not in weathernext_variables.STATIC_VARS
    for variable in spec.single_level_vars:
        assert variable not in weathernext_variables.ALL_ATMOSPHERIC_VARS
        assert variable not in weathernext_variables.STATIC_VARS
    assert set(spec.static_vars) <= set(weathernext_variables.STATIC_VARS)
    assert set(spec.computed_vars) <= set(weathernext_variables.TIME_FORCING_VARS)


def test_no_external_forcing_needed(spec: model_spec.ModelSpec) -> None:
    """WeatherNext 2 needs no solar radiation input, unlike GraphCast."""
    inputs = set(spec.downloaded_vars) | set(spec.computed_vars)
    assert inputs.isdisjoint(weathernext_variables.EXTERNAL_FORCING_VARS)


def test_precipitation_is_output_only(spec: model_spec.ModelSpec) -> None:
    assert "total_precipitation_6hr" in spec.target_vars
    assert "total_precipitation_6hr" not in spec.downloaded_vars


def test_checkpoints_disagree_on_inputs() -> None:
    """The reason the lists are derived rather than hardcoded."""
    wn2 = model_spec.load_spec("WeatherNext2")
    cyclones = model_spec.load_spec("WeatherNextCyclones")
    assert set(wn2.single_level_vars) - set(cyclones.single_level_vars) == {
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
    }


def test_surface_geopotential_maps_to_a_different_cds_variable() -> None:
    spec = model_spec.load_spec("WeatherNext2")
    surface = spec.era5_source("geopotential_at_surface")
    upper_air = spec.era5_source("geopotential")
    assert surface.variable == "geopotential"
    assert surface.dataset == model_spec.CDS_SINGLE_LEVELS
    assert upper_air.dataset == model_spec.CDS_PRESSURE_LEVELS


def test_unknown_input_variable_is_fatal() -> None:
    task = FakeTaskConfig(input_variables=["temperature", "snow_depth"])
    with pytest.raises(KeyError, match="snow_depth"):
        model_spec.spec_from_task_config(task, "Fake")
