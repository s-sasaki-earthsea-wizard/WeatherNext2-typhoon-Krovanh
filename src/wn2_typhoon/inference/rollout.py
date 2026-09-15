"""Autoregressive ensemble rollout.

Memory strategy: a 10-day, 0.25 deg forecast with all variables is ~16 GiB
per member in host RAM, so members are generated one at a time. For each
member: rollout -> tracker -> save subsets (regional crop of all fields,
global surface fields) -> free memory. Member ``i`` uses
``jax.random.fold_in(PRNGKey(seed), i)`` so runs are reproducible and the
first N members stay identical when the ensemble grows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr


def predict(
    predictor_fn: Any,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    num_members: int,
    seed: int = 0,
    steps_per_chunk: int = 1,
) -> xr.Dataset:
    """Run the autoregressive rollout for a block of ensemble members.

    Args:
        predictor_fn: Pmapped forward function from
            ``inference.load_model.build_predictor``.
        inputs: The two input frames.
        targets_template: NaN-filled template defining the forecast steps.
        forcings: Forcing variables over the target times.
        num_members: Number of members, which must be a multiple of the number
            of local devices.
        seed: Base RNG seed. Member ``i`` uses ``fold_in(PRNGKey(seed), i)`` so
            the first N members are unchanged when the ensemble grows.
        steps_per_chunk: Rollout steps per generator chunk.

    Returns:
        Forecast dataset with a ``sample`` dimension of length ``num_members``.
    """
    import jax
    import numpy as np
    from weathernext.utils import rollout as wn_rollout

    rng = jax.random.PRNGKey(seed)
    rngs = np.stack(
        [jax.random.fold_in(rng, i) for i in range(num_members)], axis=0
    )
    chunks = list(
        wn_rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=predictor_fn,
            rngs=rngs,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            num_steps_per_chunk=steps_per_chunk,
            num_samples=num_members,
            pmap_devices=jax.local_devices(),
        )
    )
    return xr.combine_by_coords(chunks)


def save_subsets(forecast: xr.Dataset, out_dir: Path, region: dict, surface_vars: list[str]) -> None:
    """Persist the retained subsets of one member's forecast as zarr.

    Args:
        forecast: Output of :func:`run_member`.
        out_dir: ``outputs/<case>/member-XX/``.
        region: ``{"lat": [min, max], "lon": [min, max]}``.
        surface_vars: Global surface variables to keep.
    """
    raise NotImplementedError("TODO: crop + compressed zarr")
