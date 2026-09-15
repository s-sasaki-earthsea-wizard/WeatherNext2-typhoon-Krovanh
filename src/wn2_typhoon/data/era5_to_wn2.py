"""Convert raw ERA5 NetCDF into the xarray layout WeatherNext 2 expects.

The target layout mirrors the sample datasets in ``gs://dm_graphcast/
weathernext2/dataset/`` (dims: batch, time, lat, lon, level; GraphCast-style
variable names such as ``2m_temperature``). Open questions to settle against
the checkpoint task config:

    * exact input / forcing / static variable names (e.g. 100 m wind,
      6-hourly accumulated precipitation, sea surface temperature)
    * latitude ordering and longitude range (0..360 vs -180..180)
    * datetime coordinate and ``time`` as timedelta relative to init
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
    raise NotImplementedError("TODO: rename, regrid checks, accumulations, forcings")
