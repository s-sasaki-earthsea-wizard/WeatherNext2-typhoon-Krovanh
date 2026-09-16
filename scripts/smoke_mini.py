"""CLI entry point: smoke_mini.

Exercises the whole inference path on the Mac before any GPU time is spent:
config -> sample data -> input extraction -> rollout -> cyclone tracker. It
runs the 1 deg Mini checkpoint on the jax CPU backend, so it validates the
plumbing rather than the science; WeatherNext2 at 0.25 deg needs an H100.

Usage:
    uv run python scripts/smoke_mini.py --config configs/krovanh.yaml
    uv run python scripts/smoke_mini.py --steps 2 --members 1
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from wn2_typhoon.config import load_raw
from wn2_typhoon.inference.gcs import download_blob
from wn2_typhoon.inference.load_model import (
    attention_type_for_backend,
    download_checkpoint,
)
from wn2_typhoon.inference.rollout import stream_member
from wn2_typhoon.inference.tracker import initial_storms_from_positions, track_member
from wn2_typhoon.model_spec import (
    config_name_for,
    load_task_config,
    spec_from_task_config,
)

DATASET_PREFIX = "weathernext2/dataset"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    parser.add_argument("--steps", type=int, default=2, help="rollout steps of 6 h")
    parser.add_argument(
        "--members",
        type=int,
        default=1,
        help="ensemble members; must be a multiple of the local device count",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return parser.parse_args()


@contextmanager
def stage(title: str):
    """Print a timed section header.

    Args:
        title: Section name.
    """
    print(f"\n== {title}")
    start = time.monotonic()
    yield
    print(f"   ({time.monotonic() - start:.1f} s)")


def sample_blob_name(smoke: dict) -> str:
    """Build the GCS path of the public sample forecast.

    Args:
        smoke: The ``smoke`` block of the experiment config.

    Returns:
        Object path within the bucket.
    """
    return (
        f"{DATASET_PREFIX}/source-hres_forecast_init-{smoke['sample_init']}"
        f"_res-{smoke['resolution']}_levels-13_steps-{smoke['sample_steps']}.nc"
    )


def seed_from_sample(example_batch) -> list[tuple[str, float, float]]:
    """Seed one storm from the strongest cyclone signal in the analysis frame.

    The real runs seed from an observed JMA position; here the sample dataset
    is its own reference, which keeps the check self-contained.

    Args:
        example_batch: The public sample forecast, with the analysis at
            ``time`` index 1.

    Returns:
        A single ``(track_id, lat, lon)`` entry.
    """
    import numpy as np

    field = example_batch["cyclone_exists_gaussian_unit_mode"].isel(batch=0, time=1)
    values = field.values
    lat_idx, lon_idx = np.unravel_index(np.nanargmax(values), values.shape)
    return [
        (
            "smoke-0",
            float(field.lat[lat_idx]),
            float(field.lon[lon_idx]),
        )
    ]


def main() -> None:
    """Run the local pipeline check."""
    # Stage headers should appear while the slow stages run, not at the end.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    cfg = load_raw(args.config)
    smoke = cfg["smoke"]
    bucket = cfg["model"]["gcs_bucket"]

    import jax
    import xarray as xr
    from weathernext.utils import data_utils

    from wn2_typhoon.inference.load_model import build_predictor

    backend = jax.default_backend()
    with stage("Backend"):
        print(f"   jax {jax.__version__} on {backend}: {jax.local_devices()}")
        print(f"   attention_type override: {attention_type_for_backend(backend)}")

    with stage(f"Input contract of {smoke['name']}"):
        task_config = load_task_config(smoke["name"])
        spec = spec_from_task_config(task_config, smoke["name"])
        print(f"   {len(spec.downloaded_vars)} downloaded inputs, "
              f"{len(spec.computed_vars)} computed, "
              f"{len(spec.target_vars)} targets")

    with stage("Fetch sample forecast and weights"):
        data_path = download_blob(bucket, sample_blob_name(smoke), args.cache_dir)
        ckpt_path = download_checkpoint(
            bucket, smoke["name"], smoke["split"], smoke["checkpoint"], args.cache_dir
        )
        print(f"   {data_path.name}  {data_path.stat().st_size / 1024**2:.0f} MiB")
        print(f"   {ckpt_path.name}  {ckpt_path.stat().st_size / 1024**2:.0f} MiB")

    with stage("Validate the sample against the contract"):
        example_batch = xr.load_dataset(data_path).compute()
        present = set(example_batch.data_vars) | set(example_batch.coords)
        missing = [v for v in spec.downloaded_vars if v not in present]
        if missing:
            raise SystemExit(f"sample dataset is missing model inputs: {missing}")
        computed_present = [v for v in spec.computed_vars if v in present]
        print(f"   dims: {dict(example_batch.sizes)}")
        print(f"   all {len(spec.downloaded_vars)} downloaded inputs present")
        print(f"   forcings already in the sample: {len(computed_present)}"
              f"/{len(spec.computed_vars)}")

    with stage(f"Extract inputs, targets and forcings for {args.steps} step(s)"):
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{6 * args.steps}h"),
            **dataclasses.asdict(task_config),
        )
        print(f"   inputs   {dict(inputs.sizes)}")
        print(f"   targets  {dict(targets.sizes)}")
        print(f"   forcings {dict(forcings.sizes)}")

    with stage("Build the predictor and load weights"):
        _, predictor_fn = build_predictor(
            config_name_for(smoke["name"]), ckpt_path, backend
        )

    with stage(f"Rollout: {args.members} member(s), {args.steps} step(s)"):
        # The same streaming path the pod uses: each step is reduced to the
        # tracker variables globally plus the regional crop, and the full
        # global field is never held.
        region = cfg["output"]["region"]
        members = [
            stream_member(
                predictor_fn,
                inputs,
                targets * float("nan"),
                forcings,
                member=index,
                seed=cfg["forecast"]["seed"],
                region=region,
            )
            for index in range(args.members)
        ]
        global_ds, crop_ds = members[0]
        print(f"   tracker fields kept globally: {len(global_ds.data_vars)}"
              f" of {len(spec.target_vars)}, {dict(global_ds.sizes)}")
        print(f"   regional crop {dict(crop_ds.sizes)}")

    with stage("Cyclone tracker, seeded from an observed centre"):
        init_time = example_batch.isel(batch=0, time=1).datetime.values
        positions = seed_from_sample(example_batch)
        print(f"   init time {init_time}")
        print(f"   seed: {positions[0][0]} at "
              f"{positions[0][1]:.1f}N {positions[0][2]:.1f}E")
        seeded = track_member(
            global_ds, init_time, initial_storms_from_positions(positions, init_time)
        )
        print(f"   track rows: {len(seeded)}")
        if len(seeded):
            columns = ["lead_time", "lat", "lon", "minimum_sea_level_pressure_hpa"]
            print(seeded[[c for c in columns if c in seeded]].to_string(index=False))

    with stage("Cyclone tracker, cyclogenesis only"):
        # Exercises the path the pre-formation case uses. Tracks shorter than
        # 2.5 days are dropped by the v1 config, so a short rollout finding
        # nothing is the expected outcome; not crashing is the point.
        found = track_member(global_ds, init_time)
        print(f"   track rows: {len(found)}")

    print("\nsmoke run complete")


if __name__ == "__main__":
    main()
