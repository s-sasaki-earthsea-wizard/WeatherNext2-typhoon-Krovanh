"""Autoregressive ensemble rollout.

Memory strategy. A 10-day 0.25 deg forecast is 103 fields over 40 steps, or
17.1 GB per member, and it is never held whole. The generator yields one step
at a time and :func:`stream_member` reduces each step immediately to the two
things that are wanted:

    * the variables the cyclone tracker reads, globally, because the storm may
      be anywhere -- 17 of the 103 fields, so 2.8 GB over the rollout
    * the regional crop from the config, all variables and levels, 0.33 GB

Everything else is dropped as it arrives, which keeps a member near 3 GB
rather than 17 GB and lets the pod run with ordinary host memory.

Member ``i`` uses ``jax.random.fold_in(PRNGKey(seed), i)`` so runs are
reproducible and the first N members stay identical when the ensemble grows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr


def member_rng(seed: int, member: int) -> Any:
    """Return the RNG key for one ensemble member.

    Args:
        seed: Base seed from the config.
        member: Member index.

    Returns:
        A jax PRNG key.
    """
    import jax

    return jax.random.fold_in(jax.random.PRNGKey(seed), member)


def crop(dataset: xr.Dataset, region: dict) -> xr.Dataset:
    """Cut a dataset down to the configured region.

    Args:
        dataset: A forecast, or one chunk of one, on ascending lat and lon.
        region: ``{"lat": [min, max], "lon": [min, max]}`` in degrees, with
            longitudes east.

    Returns:
        The cropped dataset.
    """
    return dataset.sel(
        lat=slice(*region["lat"]), lon=slice(*region["lon"])
    )


def stream_member(predictor_fn: Any, inputs: xr.Dataset,
                  targets_template: xr.Dataset, forcings: xr.Dataset,
                  member: int, seed: int, region: dict,
                  global_vars: list[str] | None = None,
                  steps_per_chunk: int = 1) -> tuple[xr.Dataset, xr.Dataset]:
    """Roll out one member, reducing each step as it arrives.

    Args:
        predictor_fn: Pmapped forward function from
            ``inference.load_model.build_predictor``.
        inputs: The two input frames.
        targets_template: NaN-filled template defining the forecast steps. It
            may be dask-backed; the generator computes one chunk at a time.
        forcings: Forcing variables over the target times.
        member: Member index, which selects the RNG key.
        seed: Base RNG seed.
        region: Crop passed to :func:`crop`.
        global_vars: Variables to retain globally. Defaults to the ones the
            cyclone tracker reads.
        steps_per_chunk: Rollout steps per generator chunk.

    Returns:
        ``(global_subset, regional_crop)``, both with the ``batch`` and
        ``sample`` dimensions dropped.
    """
    import jax
    import numpy as np
    from weathernext.utils import rollout as wn_rollout

    from wn2_typhoon.inference.tracker import tracker_variables

    rngs = np.stack([member_rng(seed, member)], axis=0)
    global_chunks: list[xr.Dataset] = []
    crop_chunks: list[xr.Dataset] = []

    for chunk in wn_rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=predictor_fn,
        rngs=rngs,
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
        num_steps_per_chunk=steps_per_chunk,
        num_samples=1,
        pmap_devices=jax.local_devices(),
    ):
        chunk = chunk.isel(batch=0, sample=0, drop=True)
        names = global_vars if global_vars is not None else tracker_variables(chunk)
        global_chunks.append(chunk[names])
        crop_chunks.append(crop(chunk, region))
        del chunk

    return (
        xr.concat(global_chunks, dim="time"),
        xr.concat(crop_chunks, dim="time"),
    )


# Upper bound on one uncompressed zarr chunk. Chosen so that the crop as
# configured today lands in a single chunk per variable (the largest is 40 x 13
# x 141 x 141 float32, about 41 MB) while a much wider `output.region` still
# splits along time instead of producing chunks too big to read comfortably.
MAX_CHUNK_BYTES = 64 * 1024 * 1024


def chunking_for(array: xr.DataArray, max_bytes: int = MAX_CHUNK_BYTES) -> tuple[int, ...]:
    """Choose a zarr chunk shape: whole fields, split along time if too large.

    The default chunking xarray infers is far too fine for this crop. Left
    alone it wrote ``(time 10, level 4, lat 71, lon 71)`` for arrays that are
    only ``(40, 13, 141, 141)``, which is 624 files in 532 directories for
    0.27 GB, an average of 458 KB per file. Every one of those files is paid
    for three times: in MooseFS's directory accounting on the pod, in rsync's
    per-file protocol overhead on the pull, and in SMB round trips on the write
    to the NAS.

    Measured on 2026-09-17 for one member, the two chunkings are the same size
    on disk to within 0.1 per cent, and whole-field chunks cut the file count
    from 624 to 71 and the Mac-to-NAS write from 210 s to 76 s, a factor of
    2.8. Reading a time series at one point, a single field at one time, and
    all 17 cyclone fields over all steps took under 0.1 s either way, so the
    partial-read cost the fine chunking was buying does not exist at this size.

    Args:
        array: The variable to be written.
        max_bytes: Largest uncompressed chunk to allow.

    Returns:
        A chunk shape, in the array's own dimension order.
    """
    shape = tuple(int(size) for size in array.shape)
    whole = array.dtype.itemsize
    for size in shape:
        whole *= size
    if whole <= max_bytes or "time" not in array.dims or not shape:
        return shape

    axis = array.dims.index("time")
    per_step = max(whole // max(shape[axis], 1), 1)
    steps = max(1, min(shape[axis], max_bytes // per_step))
    return shape[:axis] + (steps,) + shape[axis + 1:]


def save_zarr(forecast: xr.Dataset, path: Path) -> Path:
    """Write a forecast subset as compressed zarr.

    Only the regional crop is written. Keeping the global cyclone fields too
    would be convenient for re-tracking, but they do not compress: unlike the
    training targets, which are 94 per cent NaN and 99.6 per cent exact zero,
    the model predicts dense small values everywhere, measured at a ratio of
    1.09. That is 2.6 GB per member and 21 GB per case. The crop already
    carries all 17 cyclone fields, so re-tracking pads it back to a global grid
    instead; see ``tracker.pad_to_global``.

    Chunking is set explicitly rather than inferred; see :func:`chunking_for`
    for the measurement behind that.

    Args:
        forecast: A subset from :func:`stream_member`.
        path: Destination store.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"compressors": "auto", "chunks": chunking_for(forecast[name])}
        for name in forecast.data_vars
    }
    forecast.to_zarr(path, mode="w", encoding=encoding, consolidated=True)
    return path
