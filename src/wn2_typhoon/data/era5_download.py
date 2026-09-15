"""Download ERA5 (ERA5T) fields from the Copernicus Climate Data Store.

WeatherNext 2 consumes two input frames spaced 6 h apart (t-6h, t) on a
0.25 deg global grid with 13 pressure levels. This module requests exactly
the frames one case needs and writes NetCDF under ``data/raw/era5/``.

Notes:
    * ``cdsapi`` reads credentials from ``~/.cdsapirc``.
    * ARCO-ERA5 on GCS lags by months, so CDS is the only source for
      September 2026. Data younger than ~3 months is ERA5T (preliminary).
    * Variable lists must match the model task config; derive them from the
      checkpoint config at runtime rather than hardcoding (see
      ``inference.load_model``).
"""

from __future__ import annotations

from pathlib import Path

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def download_case(init_time: str, out_dir: Path) -> list[Path]:
    """Download the ERA5 frames required to initialize one forecast case.

    Args:
        init_time: Initialization time, UTC ISO 8601 (e.g. "2026-08-31T18:00").
        out_dir: Directory where NetCDF files are written.

    Returns:
        Paths of the downloaded files (pressure levels, single levels, static).
    """
    raise NotImplementedError("TODO: build CDS requests for t-6h and t")
