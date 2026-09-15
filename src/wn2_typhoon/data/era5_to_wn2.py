"""Convert raw ERA5 NetCDF into the xarray layout WeatherNext 2 expects.

The target layout mirrors the sample datasets in ``gs://dm_graphcast/
weathernext2/dataset/`` (dims: batch, time, lat, lon, level; GraphCast-style
variable names such as ``2m_temperature``).

The variable names themselves are settled; see :mod:`wn2_typhoon.model_spec`.
The forcing variables are not built here either: ``data_utils.add_derived_vars``
computes them from the ``datetime`` and ``lon`` coordinates.

Conventions read off the public 1 deg sample (``make smoke-mini`` caches it
under ``data/cache/``):

    * dims ``batch, time, lat, lon, level``
    * ``lat`` ascending from -90 to 90, ``lon`` from 0 to 359, ``level``
      ascending in hPa. CDS delivers latitude descending, so it has to be
      flipped.
    * ``time`` is a timedelta relative to the initialization time, and a
      separate ``datetime`` coordinate carries the absolute times
    * ``sea_surface_temperature`` is NaN over land (about a third of the grid)
    * static fields are broadcast over ``time`` like any other variable, and
      the surface geopotential is renamed (the CDS calls it ``geopotential``
      as well, but in the single-level dataset)
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def build_inputs(era5_files: list[Path], init_time: str, lead_hours: int, step_hours: int) -> xr.Dataset:
    """Assemble the model input dataset for one case.

    Args:
        era5_files: NetCDF files produced by :mod:`era5_download`.
        init_time: Initialization time, UTC ISO 8601.
        lead_hours: Forecast length; used to size the NaN target template.
        step_hours: Model time step (6 h).

    Returns:
        Dataset with the two input frames and a NaN-filled target template.
    """
    raise NotImplementedError("TODO: rename, coordinate conventions, statics")
