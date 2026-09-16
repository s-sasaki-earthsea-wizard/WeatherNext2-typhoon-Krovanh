"""Convert raw ERA5 NetCDF into the xarray layout WeatherNext 2 expects.

The target layout mirrors the sample datasets under ``gs://dm_graphcast/
weathernext2/dataset/``, which this repository checks against in
``make smoke-mini``:

    * dims ``batch, time, lat, lon, level``
    * ``lat`` float32 ascending from -90 to 90, ``lon`` float32 from 0 to
      359.75, ``level`` int32 ascending in hPa
    * ``time`` a timedelta coordinate; only its spacing matters, because
      ``data_utils.extract_input_target_times`` re-origins it so that the last
      frame of the dataset is the last target
    * a ``datetime`` coordinate of dims ``(batch, time)`` carrying the absolute
      times, which ``data_utils.add_derived_vars`` needs to build the forcings
    * static fields carry dims ``(lat, lon)`` only, with no batch or time
    * ``sea_surface_temperature`` is NaN over land, the convention the model
      was trained with, so the NaNs are kept

What the CDS returns does not match any of that. Its NetCDF uses
``valid_time``, ``pressure_level``, ``latitude`` and ``longitude``, orders
latitude from north to south, and names variables by their GRIB short names
(``t``, ``z``, ``u10`` and so on), which is why
:class:`~wn2_typhoon.model_spec.Era5Source` records a short name per variable.
Note that surface geopotential and pressure-level geopotential are both ``z``;
they are told apart by the file they arrive in.

Only the two input frames are built here. The 40 NaN target frames the rollout
needs are added on the pod by
:func:`wn2_typhoon.inference.inputs.split_for_rollout`, because materialising
them would add 17 GB to a 0.72 GB file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from wn2_typhoon.data.era5_download import input_frames
from wn2_typhoon.model_spec import ModelSpec

logger = logging.getLogger(__name__)

COORD_RENAMES = {"latitude": "lat", "longitude": "lon", "pressure_level": "level"}
# Scalar coordinates the CDS adds that the model has no use for. "expver"
# distinguishes final ERA5 (0001) from preliminary ERA5T (0005); this project
# knowingly runs on ERA5T, and the value is logged rather than kept.
DROP_COORDS = ("number", "expver")


def _open_group(path: Path, variables: tuple[str, ...], spec: ModelSpec) -> xr.Dataset:
    """Open one CDS file and rename it into the model's vocabulary.

    Args:
        path: NetCDF written by :mod:`wn2_typhoon.data.era5_download`.
        variables: Model variable names expected in this file.
        spec: Input contract, used to look up the NetCDF short names.

    Returns:
        A dataset holding exactly ``variables``, with renamed coordinates.

    Raises:
        KeyError: If a variable the spec expects is not in the file.
    """
    dataset = xr.open_dataset(path)
    if "expver" in dataset.coords:
        logger.info("%s: ERA5 expver=%s", path.name, dataset.expver.values)

    renames = {spec.era5_source(v).short_name: v for v in variables}
    missing = sorted(set(renames) - set(dataset.data_vars))
    if missing:
        raise KeyError(
            f"{path} is missing {missing}; it holds {sorted(dataset.data_vars)}. "
            "The short names in model_spec.ERA5_SOURCES and the request that "
            "produced this file have drifted apart."
        )

    dataset = dataset[list(renames)].rename(renames)
    dataset = dataset.drop_vars(
        [c for c in DROP_COORDS if c in dataset.coords], errors="ignore"
    )
    return dataset.rename({k: v for k, v in COORD_RENAMES.items() if k in dataset.dims})


def _normalise_grid(dataset: xr.Dataset, spec: ModelSpec) -> xr.Dataset:
    """Put the horizontal and vertical coordinates in the model's order.

    Args:
        dataset: Dataset with ``lat``, ``lon`` and optionally ``level``.
        spec: Input contract, which fixes the level order.

    Returns:
        The dataset with ascending latitude, longitudes in [0, 360) and levels
        in the order the checkpoint lists them.
    """
    if "lon" in dataset.coords and float(dataset.lon.min()) < 0:
        dataset = dataset.assign_coords(lon=(dataset.lon % 360))
    if "lon" in dataset.coords:
        dataset = dataset.sortby("lon")
    if "lat" in dataset.coords:
        dataset = dataset.sortby("lat")
    if "level" in dataset.coords:
        dataset = dataset.sel(level=list(spec.pressure_levels))
        dataset = dataset.assign_coords(level=dataset.level.astype("int32"))
    for axis in ("lat", "lon"):
        if axis in dataset.coords:
            dataset = dataset.assign_coords({axis: dataset[axis].astype("float32")})
    return dataset


def validate_inputs(dataset: xr.Dataset, spec: ModelSpec, init_time: str,
                    step_hours: int = 6) -> None:
    """Check a built input file against the checkpoint's contract.

    This runs before the file ever reaches the pod, because a layout mistake
    that only shows up after the weights are loaded costs GPU time. The
    expectations come from the public sample dataset; see the module docstring.

    Args:
        dataset: Output of :func:`build_inputs`.
        spec: Input contract of the checkpoint being run.
        init_time: Initialization time, UTC ISO 8601.
        step_hours: Model time step.

    Raises:
        ValueError: On the first discrepancy, naming it.
    """
    problems: list[str] = []

    present = set(dataset.data_vars)
    expected = set(spec.downloaded_vars)
    if present != expected:
        problems.append(
            f"variables differ: missing {sorted(expected - present)}, "
            f"unexpected {sorted(present - expected)}"
        )

    lat, lon = dataset.lat.values, dataset.lon.values
    if not np.all(np.diff(lat) > 0):
        problems.append("lat must ascend; the CDS delivers it north to south")
    if not (lat[0] == -90 and lat[-1] == 90):
        problems.append(f"lat must span -90..90, found {lat[0]}..{lat[-1]}")
    if not np.all(np.diff(lon) > 0) or lon[0] != 0 or lon[-1] >= 360:
        problems.append(f"lon must ascend within [0, 360), found {lon[0]}..{lon[-1]}")

    levels = tuple(int(v) for v in dataset.level.values)
    if levels != tuple(spec.pressure_levels):
        problems.append(f"levels {levels} do not match the checkpoint's {spec.pressure_levels}")

    expected_frames = 2
    if dataset.sizes.get("time") != expected_frames:
        problems.append(f"expected {expected_frames} input frames, found {dataset.sizes.get('time')}")
    else:
        step = np.timedelta64(step_hours * 3600, "s").astype("timedelta64[ns]")
        if dataset.time.values[1] - dataset.time.values[0] != step:
            problems.append(f"input frames must be {step_hours} h apart")
        last = dataset.datetime.values[0][-1]
        if last != np.datetime64(init_time, "ns"):
            problems.append(f"last frame is {last}, not the init time {init_time}")

    for name in spec.static_vars:
        if name in dataset and "time" in dataset[name].dims:
            problems.append(f"{name} is static and must not carry a time dim")
    for name in spec.pressure_level_vars:
        if name in dataset and "level" not in dataset[name].dims:
            problems.append(f"{name} must carry a level dim")
    for name in spec.single_level_vars:
        if name in dataset and "level" in dataset[name].dims:
            problems.append(f"{name} must not carry a level dim")

    # ERA5 masks sea surface temperature over land; nothing else may be NaN.
    for name in sorted(present & expected):
        if name == "sea_surface_temperature":
            continue
        if bool(np.isnan(dataset[name].values).any()):
            problems.append(f"{name} contains NaN")

    if problems:
        raise ValueError(
            "the built inputs do not match the checkpoint contract:\n  - "
            + "\n  - ".join(problems)
        )


def build_inputs(era5_files: list[Path], init_time: str, spec: ModelSpec,
                 step_hours: int = 6) -> xr.Dataset:
    """Assemble the two model input frames for one case.

    Args:
        era5_files: NetCDF files produced by
            :func:`wn2_typhoon.data.era5_download.download_case`, in the order
            that function returns them.
        init_time: Initialization time, UTC ISO 8601.
        spec: Input contract of the checkpoint being run.
        step_hours: Model time step.

    Returns:
        Dataset with dims ``batch, time, lat, lon, level`` holding the two
        input frames, a ``datetime`` coordinate and the static fields. The
        forcing variables are not added here: they are derived on the pod for
        the target times as well as the input ones.
    """
    frames = input_frames(init_time, step_hours)
    by_name = {path.name: path for path in era5_files}

    per_frame = []
    for moment in frames:
        stamp = moment.strftime("%Y%m%dT%H")
        pressure = _open_group(
            by_name[f"pressure_levels_{stamp}.nc"], spec.pressure_level_vars, spec
        )
        single = _open_group(
            by_name[f"single_levels_{stamp}.nc"], spec.single_level_vars, spec
        )
        merged = xr.merge([pressure, single], join="exact")
        per_frame.append(merged.squeeze("valid_time", drop=True))

    dataset = xr.concat(per_frame, dim="time")
    dataset = dataset.expand_dims(batch=1)

    statics = _open_group(by_name["static.nc"], spec.static_vars, spec)
    statics = statics.squeeze("valid_time", drop=True)
    dataset = xr.merge([dataset, statics], join="exact")

    dataset = _normalise_grid(dataset, spec)

    # Only the spacing matters; extract_input_target_times re-origins it.
    offsets = np.array(
        [(moment - frames[0]).total_seconds() for moment in frames], dtype="int64"
    )
    dataset = dataset.assign_coords(
        time=offsets.astype("timedelta64[s]").astype("timedelta64[ns]")
    )
    dataset = dataset.assign_coords(
        datetime=(
            ("batch", "time"),
            np.array([[np.datetime64(moment, "ns") for moment in frames]]),
        )
    )
    validate_inputs(dataset, spec, init_time, step_hours)
    return dataset
