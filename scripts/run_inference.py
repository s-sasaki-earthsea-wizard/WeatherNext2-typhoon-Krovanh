"""CLI entry point: run_inference.

Rolls out the ensemble for one case and tracks each member as it is produced.
Tracking happens here rather than in run_tracker because it needs the global
cyclone fields, which are never written to disk: 2.8 GB per member would be
22 GB for a case, against 2.6 GB for the regional crops that are kept.

Members are independent and each writes its own files, so an interrupted run
resumes with --member-start.

Usage (on the pod):
    uv run python scripts/run_inference.py --config configs/krovanh.yaml \\
        --case init-2026-09-01T00
    uv run python scripts/run_inference.py --case init-2026-09-01T00 --members 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wn2_typhoon.config import Case, get_case, load_raw
from wn2_typhoon.inference.inputs import split_for_rollout
from wn2_typhoon.inference.load_model import build_predictor, download_checkpoint
from wn2_typhoon.inference.rollout import save_zarr, stream_member
from wn2_typhoon.inference.tracker import (
    build_tracker,
    initial_storms_from_positions,
    track_member,
)
from wn2_typhoon.model_spec import config_name_for, load_spec, load_task_config
from wn2_typhoon.utils.logs import configure

logger = configure("run_inference")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    parser.add_argument("--case", required=True, help="case id from the config")
    parser.add_argument("--inputs", type=Path, help="default data/interim/<case>/inputs.nc")
    parser.add_argument("--out-dir", type=Path, help="default outputs/<case>")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--members", type=int, help="default forecast.num_members")
    parser.add_argument(
        "--member-start", type=int, default=0, help="resume from this member"
    )
    parser.add_argument(
        "--lead-hours", type=int, help="shorten the forecast, for timing runs"
    )
    parser.add_argument("--model", help="override model.name")
    parser.add_argument("--split", help="override model.split")
    parser.add_argument("--checkpoint", help="override model.checkpoint")
    parser.add_argument(
        "--verbose", action="store_true", help="keep third-party INFO logs"
    )
    return parser.parse_args()


def seed_storms(case: Case) -> pd.DataFrame | None:
    """Build the tracker's initial-storm table for a case.

    Args:
        case: The case being run.

    Returns:
        A one-row table at the observed centre, or None when no observed
        position exists and the tracker has to find the storm itself.
    """
    if case.seed_position is None:
        logger.info("No seed position: the tracker runs in cyclogenesis mode")
        return None
    lat, lon = case.seed_position
    logger.info("Seeding the tracker at %.1fN %.1fE", lat, lon)
    return initial_storms_from_positions(
        [(case.id, lat, lon)], np.datetime64(case.init_time)
    )


def main() -> None:
    """Run the ensemble for one case."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    cfg = load_raw(args.config)
    case = get_case(cfg, args.case)

    model_name = args.model or cfg["model"]["name"]
    split = args.split or cfg["model"]["split"]
    checkpoint = args.checkpoint if args.model else cfg["model"]["checkpoint"]
    step_hours = cfg["forecast"]["step_hours"]
    lead_hours = args.lead_hours or cfg["forecast"]["lead_hours"]
    members = args.members if args.members is not None else cfg["forecast"]["num_members"]
    region = cfg["output"]["region"]

    inputs_path = args.inputs or Path("data/interim") / case.id / "inputs.nc"
    out_dir = args.out_dir or Path("outputs") / case.id
    out_dir.mkdir(parents=True, exist_ok=True)

    import jax

    backend = jax.default_backend()
    logger.info("Case %s, init %s", case.id, case.init_time)
    logger.info("Model %s <%s %s on %s", model_name, split, checkpoint, backend)
    logger.info("%d members, %d h lead (%d steps)", members, lead_hours,
                lead_hours // step_hours)

    spec = load_spec(model_name)
    task_config = load_task_config(model_name)
    frames = xr.open_dataset(inputs_path)
    model_inputs, targets_template, forcings = split_for_rollout(
        frames, spec, lead_hours, step_hours, task_config
    )

    ckpt_path = download_checkpoint(
        cfg["model"]["gcs_bucket"], model_name, split, checkpoint, args.cache_dir
    )
    _, predictor_fn = build_predictor(
        config_name_for(model_name), ckpt_path, backend
    )

    init_time = np.datetime64(case.init_time)
    initial_storms = seed_storms(case)
    tracker = build_tracker()

    for member in range(args.member_start, args.member_start + members):
        started = time.monotonic()
        global_ds, crop_ds = stream_member(
            predictor_fn,
            model_inputs,
            targets_template,
            forcings,
            member=member,
            seed=cfg["forecast"]["seed"],
            region=region,
        )
        rolled = time.monotonic()

        tracks = track_member(
            global_ds,
            init_time,
            None if initial_storms is None else initial_storms.copy(),
            tracker=tracker,
        )
        tracks.insert(0, "member", member)
        tracks.to_csv(out_dir / f"tracks-member-{member:02d}.csv", index=False)

        del global_ds

        store = save_zarr(crop_ds, out_dir / f"member-{member:02d}.zarr")
        del crop_ds

        logger.info(
            "member %02d: rollout %.0f s, track %.0f s, %d track rows -> %s",
            member, rolled - started, time.monotonic() - rolled, len(tracks),
            store.name,
        )

    combined = sorted(out_dir.glob("tracks-member-*.csv"))
    if combined:
        frames_out = [pd.read_csv(path) for path in combined]
        pd.concat(frames_out, ignore_index=True).to_csv(
            out_dir / "tracks.csv", index=False
        )
        logger.info("wrote %s from %d members", out_dir / "tracks.csv", len(combined))


if __name__ == "__main__":
    main()
