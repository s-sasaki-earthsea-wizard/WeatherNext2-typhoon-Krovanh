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

import xarray as xr


def run_member(member: int, inputs: xr.Dataset, num_steps: int, seed: int) -> xr.Dataset:
    """Produce the full forecast for one ensemble member.

    Args:
        member: Member index.
        inputs: Dataset from ``data.era5_to_wn2.build_inputs``.
        num_steps: Number of 6 h steps (40 for 10 days).
        seed: Base RNG seed.

    Returns:
        Forecast dataset with a ``time`` dimension of length ``num_steps``.
    """
    raise NotImplementedError("TODO: rollout.chunked_prediction_generator_multiple_runs")


def save_subsets(forecast: xr.Dataset, out_dir: Path, region: dict, surface_vars: list[str]) -> None:
    """Persist the retained subsets of one member's forecast as zarr.

    Args:
        forecast: Output of :func:`run_member`.
        out_dir: ``outputs/<case>/member-XX/``.
        region: ``{"lat": [min, max], "lon": [min, max]}``.
        surface_vars: Global surface variables to keep.
    """
    raise NotImplementedError("TODO: crop + compressed zarr")
