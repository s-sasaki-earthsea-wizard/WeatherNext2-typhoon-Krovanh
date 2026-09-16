"""The model input contract, derived from the checkpoint task config.

Nothing here hardcodes *which* variables a checkpoint wants. The lists come
from ``config.task`` of the fiddle config bundled with ``weathernext``
(``weathernext/weathernext2/configs/<name>.json``), which is why they can be
read on any machine without weights, GPUs or GCS access.

What *is* tabulated here is :data:`ERA5_SOURCES`: how each model variable name
maps onto a CDS dataset and variable. That mapping is unavoidable, so it is
kept in one place and every model input is required to resolve through it --
an unknown variable raises instead of silently dropping out of the download.

The three bundled checkpoints do not agree on their inputs: WeatherNext2 wants
100 m winds, WeatherNextCyclones and its Mini variant do not. Deriving the list
per checkpoint is therefore not academic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

# CDS dataset identifiers.
CDS_PRESSURE_LEVELS = "reanalysis-era5-pressure-levels"
CDS_SINGLE_LEVELS = "reanalysis-era5-single-levels"


@dataclass(frozen=True)
class Era5Source:
    """Where one model variable comes from in the CDS catalogue.

    Attributes:
        dataset: CDS dataset identifier.
        variable: CDS variable name used in the request, which is not always
            the model's name.
        short_name: Name the variable carries inside the NetCDF the CDS
            returns. It comes from the underlying GRIB and matches neither of
            the other two, so the conversion needs it spelled out.
        static: True for time-invariant fields, downloaded once for a single
            arbitrary timestamp rather than per input frame.
    """

    dataset: str
    variable: str
    short_name: str
    static: bool = False


# Model variable name -> CDS source. Covers every non-computed input of the
# three checkpoints bundled with weathernext 0.3.0.
ERA5_SOURCES: dict[str, Era5Source] = {
    # Pressure levels.
    "temperature": Era5Source(CDS_PRESSURE_LEVELS, "temperature", "t"),
    "geopotential": Era5Source(CDS_PRESSURE_LEVELS, "geopotential", "z"),
    "u_component_of_wind": Era5Source(
        CDS_PRESSURE_LEVELS, "u_component_of_wind", "u"
    ),
    "v_component_of_wind": Era5Source(
        CDS_PRESSURE_LEVELS, "v_component_of_wind", "v"
    ),
    "vertical_velocity": Era5Source(CDS_PRESSURE_LEVELS, "vertical_velocity", "w"),
    "specific_humidity": Era5Source(CDS_PRESSURE_LEVELS, "specific_humidity", "q"),
    # Single level, time varying.
    "2m_temperature": Era5Source(CDS_SINGLE_LEVELS, "2m_temperature", "t2m"),
    "mean_sea_level_pressure": Era5Source(
        CDS_SINGLE_LEVELS, "mean_sea_level_pressure", "msl"
    ),
    "10m_u_component_of_wind": Era5Source(
        CDS_SINGLE_LEVELS, "10m_u_component_of_wind", "u10"
    ),
    "10m_v_component_of_wind": Era5Source(
        CDS_SINGLE_LEVELS, "10m_v_component_of_wind", "v10"
    ),
    "100m_u_component_of_wind": Era5Source(
        CDS_SINGLE_LEVELS, "100m_u_component_of_wind", "u100"
    ),
    "100m_v_component_of_wind": Era5Source(
        CDS_SINGLE_LEVELS, "100m_v_component_of_wind", "v100"
    ),
    # NaN over land in ERA5; the model was trained with that convention.
    "sea_surface_temperature": Era5Source(
        CDS_SINGLE_LEVELS, "sea_surface_temperature", "sst"
    ),
    # Static. Surface geopotential is plain "geopotential" in the single-level
    # dataset, i.e. a different field from the pressure-level variable above,
    # and it arrives under the same short name. They never share a file.
    "geopotential_at_surface": Era5Source(
        CDS_SINGLE_LEVELS, "geopotential", "z", static=True
    ),
    "land_sea_mask": Era5Source(
        CDS_SINGLE_LEVELS, "land_sea_mask", "lsm", static=True
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    """The input contract of one checkpoint.

    Attributes:
        config_name: Fiddle config name, e.g. "weathernext2/configs/WeatherNext2".
        pressure_levels: Pressure levels in hPa, in the order the model expects.
        input_duration: Span covered by the input frames, e.g. "12h" for two
            frames at t-6h and t.
        pressure_level_vars: Inputs downloaded from the pressure-level dataset.
        single_level_vars: Time-varying inputs from the single-level dataset.
        static_vars: Time-invariant inputs from the single-level dataset.
        computed_vars: Inputs generated locally rather than downloaded; these
            are the task's forcing variables, produced by
            ``weathernext.utils.data_utils.add_derived_vars``.
        target_vars: What the model predicts, including the cyclone fields the
            direct tracker consumes.
    """

    config_name: str
    pressure_levels: tuple[int, ...]
    input_duration: str
    pressure_level_vars: tuple[str, ...]
    single_level_vars: tuple[str, ...]
    static_vars: tuple[str, ...]
    computed_vars: tuple[str, ...]
    target_vars: tuple[str, ...]

    @property
    def downloaded_vars(self) -> tuple[str, ...]:
        """Every input that has to come from the CDS."""
        return self.pressure_level_vars + self.single_level_vars + self.static_vars

    def era5_source(self, variable: str) -> Era5Source:
        """Return the CDS source of one downloaded variable.

        Args:
            variable: Model variable name.

        Returns:
            Its entry in :data:`ERA5_SOURCES`.

        Raises:
            KeyError: If the variable is not downloaded from the CDS.
        """
        return ERA5_SOURCES[variable]


def config_name_for(model_name: str) -> str:
    """Build the fiddle config name for a model.

    Args:
        model_name: "WeatherNext2", "WeatherNextCyclones" or
            "WeatherNextCyclones_Mini".

    Returns:
        The name understood by ``fiddle_config_io.get_fiddle_config_by_name``.
    """
    return f"weathernext2/configs/{model_name}"


def load_task_config(model_name: str) -> Any:
    """Read ``config.task`` of a bundled checkpoint config.

    Importing ``weathernext`` is deferred so that modules which only need the
    dataclasses above stay importable in an environment without the model.

    Args:
        model_name: Model name, see :func:`config_name_for`.

    Returns:
        The task config dataclass of the checkpoint.
    """
    from weathernext.utils import fiddle_config_io

    config = fiddle_config_io.get_fiddle_config_by_name(config_name_for(model_name))
    return config.task


def spec_from_task_config(task_config: Any, model_name: str) -> ModelSpec:
    """Classify a task config into the download groups.

    Args:
        task_config: Value returned by :func:`load_task_config`.
        model_name: Model name, used only to label the result.

    Returns:
        The corresponding :class:`ModelSpec`.

    Raises:
        KeyError: If an input variable is neither a forcing nor present in
            :data:`ERA5_SOURCES`. That means the checkpoint asks for something
            this repository does not know how to fetch, and is deliberately
            fatal rather than a silently missing field.
    """
    fields = dataclasses.asdict(task_config)
    forcings = tuple(fields["forcing_variables"])

    pressure_level, single_level, static, computed = [], [], [], []
    for variable in fields["input_variables"]:
        if variable in forcings:
            computed.append(variable)
            continue
        try:
            source = ERA5_SOURCES[variable]
        except KeyError:
            raise KeyError(
                f"{model_name} needs input variable {variable!r}, which has no "
                "ERA5 source in wn2_typhoon.model_spec.ERA5_SOURCES. Add it "
                "there before running this checkpoint."
            ) from None
        if source.static:
            static.append(variable)
        elif source.dataset == CDS_PRESSURE_LEVELS:
            pressure_level.append(variable)
        else:
            single_level.append(variable)

    return ModelSpec(
        config_name=config_name_for(model_name),
        pressure_levels=tuple(fields["pressure_levels"]),
        input_duration=str(fields["input_duration"]),
        pressure_level_vars=tuple(pressure_level),
        single_level_vars=tuple(single_level),
        static_vars=tuple(static),
        computed_vars=tuple(computed),
        target_vars=tuple(fields["target_variables"]),
    )


def load_spec(model_name: str) -> ModelSpec:
    """Read and classify a checkpoint's task config in one call.

    Args:
        model_name: Model name, see :func:`config_name_for`.

    Returns:
        The corresponding :class:`ModelSpec`.
    """
    return spec_from_task_config(load_task_config(model_name), model_name)
