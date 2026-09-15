"""Storm-centre tracking with the tracker bundled in ``weathernext``.

``weathernext.cyclones.direct_tracker_6h_v1_config`` builds a tracker that
takes a gridded forecast plus a table of storms present at init time and
returns a track DataFrame. For the pre-formation case there is no storm
yet, so ``do_cyclogenesis=True`` lets the tracker spin the storm up; for
the at-formation case the JMA analysis position seeds ``initial_storms_df``.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr


def track_member(forecast: xr.Dataset, init_time: str, initial_storms: pd.DataFrame | None) -> pd.DataFrame:
    """Run the tracker on one member's forecast.

    Args:
        forecast: Full-field forecast for one member.
        init_time: Initialization time, UTC ISO 8601.
        initial_storms: Storms at init time in the tracker's schema, or None
            to rely on cyclogenesis detection only.

    Returns:
        Track table: valid time, lat, lon, min MSLP, max 10 m wind, storm id.
    """
    raise NotImplementedError("TODO: preprocess_gridded_ds + tracker call")
