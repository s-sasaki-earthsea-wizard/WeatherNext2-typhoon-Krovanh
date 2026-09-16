"""Download ERA5 (ERA5T) fields from the Copernicus Climate Data Store.

WeatherNext 2 consumes two input frames spaced 6 h apart (t-6h, t) on a
0.25 deg global grid with 13 pressure levels. This module requests exactly
those frames and writes NetCDF under ``data/raw/era5/<case>/``.

Notes:
    * ``cdsapi`` reads credentials from ``~/.cdsapirc``.
    * ARCO-ERA5 on GCS lags by months, so CDS is the only source for
      September 2026. Data younger than about three months is ERA5T
      (preliminary), which is what this project uses.
    * Variables and levels come from :mod:`wn2_typhoon.model_spec`, which reads
      them from the checkpoint task config. Nothing here is hardcoded, and a
      variable the spec cannot map to a CDS name is a hard error there.
    * No accumulated field is needed: precipitation is a target only and
      WeatherNext 2 uses no solar radiation input.

One request is issued per frame and dataset rather than one request covering
both frames. The CDS expands year/month/day/time as a cross product, so a pair
of frames that straddles midnight (as the 2026-09-01 00 UTC case does) would
otherwise return four frames and download twice what is needed.

Files are named after the frame they hold and live in one directory shared by
every case, not under a per-case directory. The two cases overlap: the
2026-08-31 18 UTC frame is the second input of one and the first input of the
other, so sharing saves a request and a few hundred megabytes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from wn2_typhoon.model_spec import CDS_PRESSURE_LEVELS, CDS_SINGLE_LEVELS, ModelSpec

logger = logging.getLogger(__name__)

# ERA5 is archived on a regular 0.25 deg grid, which is what the model wants.
GRID = [0.25, 0.25]


@dataclass(frozen=True)
class Era5Request:
    """One CDS retrieval.

    Attributes:
        dataset: CDS dataset identifier.
        request: Request body passed to ``cdsapi.Client.retrieve``.
        target: Where the NetCDF is written.
    """

    dataset: str
    request: dict[str, Any] = field(compare=False)
    target: Path


def input_frames(init_time: str, step_hours: int = 6) -> list[datetime]:
    """Return the valid times of the model's input frames.

    Args:
        init_time: Initialization time, UTC ISO 8601 (e.g. "2026-08-31T18:00").
        step_hours: Model time step; the earlier frame sits this far back.

    Returns:
        ``[init - step, init]``, in chronological order.
    """
    init = datetime.fromisoformat(init_time)
    return [init - timedelta(hours=step_hours), init]


def _stamp(moment: datetime) -> str:
    """Format a frame time for use in file names."""
    return moment.strftime("%Y%m%dT%H")


def _time_fields(moment: datetime) -> dict[str, list[str]]:
    """Return the year/month/day/time selectors for a single frame."""
    return {
        "year": [moment.strftime("%Y")],
        "month": [moment.strftime("%m")],
        "day": [moment.strftime("%d")],
        "time": [moment.strftime("%H:00")],
    }


def _base_request(moment: datetime, variables: tuple[str, ...]) -> dict[str, Any]:
    """Build the parts of a request that every dataset shares."""
    return {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        **_time_fields(moment),
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": GRID,
    }


def build_requests(init_time: str, spec: ModelSpec, out_dir: Path,
                   step_hours: int = 6) -> list[Era5Request]:
    """Build every CDS request one case needs.

    Two requests per input frame (pressure levels and single levels) plus one
    for the time-invariant fields, which are fetched at the initialization
    time and later broadcast over both frames.

    Args:
        init_time: Initialization time, UTC ISO 8601.
        spec: Input contract of the checkpoint being run.
        out_dir: Directory the NetCDF files are written to.
        step_hours: Model time step.

    Returns:
        The requests, in the order they should be issued.
    """
    frames = input_frames(init_time, step_hours)
    requests: list[Era5Request] = []

    for moment in frames:
        cds_names = tuple(
            spec.era5_source(v).variable for v in spec.pressure_level_vars
        )
        requests.append(
            Era5Request(
                dataset=CDS_PRESSURE_LEVELS,
                request={
                    **_base_request(moment, cds_names),
                    "pressure_level": [str(level) for level in spec.pressure_levels],
                },
                target=out_dir / f"pressure_levels_{_stamp(moment)}.nc",
            )
        )
        cds_names = tuple(
            spec.era5_source(v).variable for v in spec.single_level_vars
        )
        requests.append(
            Era5Request(
                dataset=CDS_SINGLE_LEVELS,
                request=_base_request(moment, cds_names),
                target=out_dir / f"single_levels_{_stamp(moment)}.nc",
            )
        )

    static_names = tuple(spec.era5_source(v).variable for v in spec.static_vars)
    requests.append(
        Era5Request(
            dataset=CDS_SINGLE_LEVELS,
            request=_base_request(frames[-1], static_names),
            target=out_dir / "static.nc",
        )
    )
    return requests


def download_case(init_time: str, spec: ModelSpec, out_dir: Path,
                  step_hours: int = 6, force: bool = False) -> list[Path]:
    """Download the ERA5 frames required to initialize one forecast case.

    Existing non-empty targets are reused: CDS queue time is the slowest part
    of the pipeline and a re-run should not pay for it twice.

    Args:
        init_time: Initialization time, UTC ISO 8601 (e.g. "2026-08-31T18:00").
        spec: Input contract of the checkpoint being run.
        out_dir: Directory where NetCDF files are written.
        step_hours: Model time step.
        force: Re-download even when the target already exists.

    Returns:
        Paths of the downloaded files, in request order.
    """
    import cdsapi

    out_dir.mkdir(parents=True, exist_ok=True)
    requests = build_requests(init_time, spec, out_dir, step_hours)

    client = None
    for item in requests:
        if item.target.exists() and item.target.stat().st_size > 0 and not force:
            logger.info("Using cached %s", item.target)
            continue
        client = client or cdsapi.Client()
        logger.info("Retrieving %s -> %s", item.dataset, item.target)
        partial = item.target.with_suffix(item.target.suffix + ".part")
        client.retrieve(item.dataset, item.request, str(partial))
        partial.replace(item.target)

    return [item.target for item in requests]
