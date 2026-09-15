"""Download ERA5 (ERA5T) fields from the Copernicus Climate Data Store.

WeatherNext 2 consumes two input frames spaced 6 h apart (t-6h, t) on a
0.25 deg global grid with 13 pressure levels. This module requests exactly
the frames one case needs and writes NetCDF under ``data/raw/era5/``.

Notes:
    * ``cdsapi`` reads credentials from ``~/.cdsapirc``.
    * ARCO-ERA5 on GCS lags by months, so CDS is the only source for
      September 2026. Data younger than ~3 months is ERA5T (preliminary).
    * Variables and levels come from :mod:`wn2_typhoon.model_spec`, which reads
      them from the checkpoint task config. Nothing here is hardcoded, and a
      variable the spec cannot map to a CDS name is a hard error there.
    * No accumulated field is needed: precipitation is a target only and
      WeatherNext 2 uses no solar radiation input.
"""

from __future__ import annotations

from pathlib import Path

from wn2_typhoon.model_spec import ModelSpec


def download_case(init_time: str, spec: ModelSpec, out_dir: Path) -> list[Path]:
    """Download the ERA5 frames required to initialize one forecast case.

    Three requests: pressure levels and single levels for both frames, plus one
    single-level request at an arbitrary timestamp for the static fields.

    Args:
        init_time: Initialization time, UTC ISO 8601 (e.g. "2026-08-31T18:00").
        spec: Input contract of the checkpoint being run.
        out_dir: Directory where NetCDF files are written.

    Returns:
        Paths of the downloaded files (pressure levels, single levels, static).
    """
    raise NotImplementedError("TODO: build CDS requests for t-6h and t")
