"""Compare forecast storm centres with the JMA best track.

Metrics (per lead time):
    * great-circle position error [km] for each member and the ensemble mean
    * ensemble spread [km]
    * central pressure difference [hPa]
    * genesis timing (pre-formation case only)

Maximum wind is reported for reference only: JMA uses 10-minute means,
the tracker reports gridded 10 m wind maxima.
"""

from __future__ import annotations

import pandas as pd


def position_errors(forecast_tracks: pd.DataFrame, best_track: pd.DataFrame) -> pd.DataFrame:
    """Compute position error per member and lead time.

    Args:
        forecast_tracks: Concatenated tracker output with a ``member`` column.
        best_track: JMA best track from ``data.jma_besttrack.load_track``.

    Returns:
        Long-format table: member, lead_hours, error_km, dp_hpa.
    """
    raise NotImplementedError("TODO: align on valid time, haversine")
