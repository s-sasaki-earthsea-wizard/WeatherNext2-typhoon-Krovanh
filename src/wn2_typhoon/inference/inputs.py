"""Turn the two stored input frames into what the rollout expects.

``data_utils.extract_inputs_targets_forcings`` wants a single dataset holding
the input frames *and* every target frame, because it slices both out of one
time axis and re-origins that axis so the last frame is the last target. For a
240 h forecast that is 40 extra frames of every target variable: 17 GB at
0.25 deg, which is why they are not stored alongside the inputs.

They never have to exist in memory either. ``rollout`` takes one chunk of the
template at a time and calls ``.compute()`` on it, so a dask-backed template
chunked along time costs one frame rather than forty.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from wn2_typhoon.model_spec import ModelSpec

logger = logging.getLogger(__name__)


def target_times(frames: xr.Dataset, num_steps: int, step_hours: int) -> np.ndarray:
    """Return the time coordinates of the NaN target frames.

    Args:
        frames: The stored input frames.
        num_steps: Number of forecast steps.
        step_hours: Model time step.

    Returns:
        ``num_steps`` timedeltas continuing the input frames' spacing.
    """
    last = frames.time.values[-1]
    step = np.timedelta64(step_hours * 3600, "s").astype("timedelta64[ns]")
    return np.array([last + step * (i + 1) for i in range(num_steps)])


def _nan_like(dims: tuple[str, ...], shape: tuple[int, ...],
              dtype: Any) -> xr.Variable:
    """Build a lazy all-NaN variable chunked one frame at a time."""
    import dask.array as dask_array

    chunks = tuple(1 if dim == "time" else size for dim, size in zip(dims, shape))
    return xr.Variable(
        dims, dask_array.full(shape, np.nan, dtype=dtype, chunks=chunks)
    )


def extend_with_target_template(frames: xr.Dataset, spec: ModelSpec,
                                num_steps: int, step_hours: int) -> xr.Dataset:
    """Append NaN target frames to the stored input frames.

    Args:
        frames: Output of :func:`wn2_typhoon.data.era5_to_wn2.build_inputs`.
        spec: Input contract of the checkpoint being run.
        num_steps: Number of forecast steps.
        step_hours: Model time step.

    Returns:
        A dataset whose time axis covers the input frames followed by
        ``num_steps`` NaN frames, with every target variable present.

    Raises:
        KeyError: If a target variable is absent from the inputs and is an
            upper-air field, which would make its shape ambiguous.
    """
    from weathernext.utils import variables as wn_variables

    times = np.concatenate([frames.time.values, target_times(frames, num_steps,
                                                             step_hours)])
    extended = frames.chunk({"time": 1}).reindex(time=times)

    # reindex fills with NaN but promotes float32 to float64 while doing it.
    for name, array in extended.data_vars.items():
        if name in frames and array.dtype != frames[name].dtype:
            extended[name] = array.astype(frames[name].dtype)

    sizes = frames.sizes
    for name in spec.target_vars:
        if name in extended:
            continue
        if name in wn_variables.ALL_ATMOSPHERIC_VARS:
            raise KeyError(
                f"target variable {name!r} is an upper-air field but is not "
                "among the inputs, so its level coordinate is unknown"
            )
        dims = ("batch", "time", "lat", "lon")
        shape = tuple(sizes[d] if d != "time" else len(times) for d in dims)
        extended[name] = _nan_like(dims, shape, np.float32)

    # reindex leaves datetime as NaT on the new frames; add_derived_vars needs
    # real absolute times there, since the forcings are what they are made of.
    start = frames.datetime.values[0][0]
    step = np.timedelta64(step_hours * 3600, "s").astype("timedelta64[ns]")
    absolute = np.array([[start + step * i for i in range(len(times))]])
    return extended.assign_coords(datetime=(("batch", "time"), absolute))


def split_for_rollout(frames: xr.Dataset, spec: ModelSpec, lead_hours: int,
                      step_hours: int, task_config: Any) -> tuple:
    """Split the stored frames into inputs, a target template and forcings.

    Args:
        frames: Output of :func:`wn2_typhoon.data.era5_to_wn2.build_inputs`.
        spec: Input contract of the checkpoint being run.
        lead_hours: Forecast length.
        step_hours: Model time step.
        task_config: The checkpoint's task config, passed straight through to
            ``data_utils.extract_inputs_targets_forcings``.

    Returns:
        ``(inputs, targets_template, forcings)``. The inputs and forcings are
        materialised; the template stays lazy.
    """
    import dataclasses

    from weathernext.utils import data_utils

    num_steps = lead_hours // step_hours
    extended = extend_with_target_template(frames, spec, num_steps, step_hours)

    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        extended,
        target_lead_times=slice(f"{step_hours}h", f"{lead_hours}h"),
        **dataclasses.asdict(task_config),
    )
    logger.info(
        "inputs %s, targets %s, forcings %s",
        dict(inputs.sizes), dict(targets.sizes), dict(forcings.sizes),
    )
    return inputs.compute(), targets, forcings.compute()
