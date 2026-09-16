"""Storm-centre tracking with the tracker bundled in ``weathernext``.

WeatherNext 2 predicts the cyclone fields directly (17 of its 31 targets are
``cyclone_*`` variables), so ``weathernext.cyclones.direct_tracker`` reads
centres out of the forecast rather than deriving them from MSLP minima.

``DirectTracker`` documents ``initial_storms_df=None`` as "pure cyclogenesis
mode", but that path is broken in weathernext 0.3.0: it assigns a bare
``pd.DataFrame()`` to ``predictions_df`` and then indexes it by column name on
the first loop iteration, which raises ``KeyError: lead_time``. Passing an
empty frame that *has* the columns takes the working branch and gives the same
semantics, so :func:`empty_initial_storms` is used instead of None.

Note also that the v1 config discards cyclogenesis tracks shorter than 2.5
days, so rollouts under ten steps produce an empty table by design.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# Minimum schema the tracker reads from initial_storms_df. It adds
# prob_cyclone_exists, track_merge and lead_time itself.
INITIAL_STORM_DTYPES = {
    "track_id": "string",
    "valid_time": "datetime64[ns]",
    "lat": "float64",
    "lon": "float64",
}

# Applied on top of direct_tracker_6h_v1_config.
#
# enforce_physically_consistent_quadrants_and_winds: the upstream helper calls
# DataFrame.idxmax(axis=1) over the quadrant-radius columns and relies on it
# returning NaN for an all-NaN row. pandas raises "Encountered all NA values"
# there from 2.1 onwards, and a storm seeded from an observed position has no
# quadrant radii, so the tracker dies on its own t=0 row. Pinning pandas below
# 2.1 is not available to us (numpy 2). The helper only rewrites the radius of
# maximum winds and the quadrant radii -- never lat, lon, mean sea level
# pressure or maximum sustained wind -- so switching it off leaves the
# storm-centre comparison, which is all this project makes, untouched. The
# price is that wind-radius columns may violate rmw <= r64 <= r50 <= r34.
TRACKER_OVERRIDES = {"enforce_physically_consistent_quadrants_and_winds": False}


def tracker_variables(dataset: xr.Dataset) -> list[str]:
    """Return the variables of a forecast the direct tracker actually reads.

    The tracker declares 159 candidate names covering every warning centre;
    WeatherNext 2 predicts 17 of them. Knowing which lets the rollout keep only
    those globally and crop everything else, which is the difference between
    2.8 GB and 17.1 GB of host memory per member.

    Args:
        dataset: A forecast, or one chunk of one.

    Returns:
        The intersection, in the dataset's own order.
    """
    from weathernext.cyclones import direct_tracker

    wanted = set(direct_tracker.DIRECT_TRACKER_CYCLONE_VARIABLES)
    return [name for name in dataset.data_vars if name in wanted]


def pad_to_global(dataset: xr.Dataset, resolution: float) -> xr.Dataset:
    """Place a regional crop back on a global grid, padded with zeros.

    ``tracker_utils.bilinear_interpolation_with_lon_wraparound`` rejects any
    grid that is not the full [0, 360) in longitude, so a stored crop cannot be
    handed to the tracker directly. Padding with zeros restores a grid it
    accepts: zero existence probability is below the cyclogenesis threshold, so
    nothing is invented outside the region.

    What this cannot do is see a storm outside the crop, and a track that
    reaches the edge will interpolate against the zeros beyond it. The tracks
    ``run_inference`` writes come from the real global field and stay the
    authoritative ones.

    Args:
        dataset: A regional crop with ascending ``lat`` and ``lon``.
        resolution: Grid spacing in degrees.

    Returns:
        The same data on a global grid, zero outside the crop.
    """
    lat = np.arange(-90.0, 90.0 + resolution / 2, resolution, dtype="float32")
    lon = np.arange(0.0, 360.0, resolution, dtype="float32")
    return dataset.reindex(lat=lat, lon=lon, fill_value=0.0)


def build_tracker(**overrides: Any) -> Any:
    """Construct the 6-hourly direct tracker from its bundled config.

    Args:
        **overrides: Tracker keyword arguments to override, applied after
            :data:`TRACKER_OVERRIDES`.

    Returns:
        A ``DirectTracker`` instance.
    """
    from weathernext.cyclones import direct_tracker_6h_v1_config

    config = direct_tracker_6h_v1_config.get_config()
    kwargs = {**config.tracker_kwargs, **TRACKER_OVERRIDES, **overrides}
    return config.tracker_constructor(**kwargs)


def empty_initial_storms() -> pd.DataFrame:
    """Return a typed, empty initial-storm table.

    Equivalent to running with no observed storms, but without tripping the
    upstream None handling described in the module docstring.

    Returns:
        An empty DataFrame with the tracker's initial-storm schema.
    """
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in INITIAL_STORM_DTYPES.items()})


def initial_storms_from_positions(
    positions: list[tuple[str, float, float]],
    valid_time: np.datetime64,
) -> pd.DataFrame:
    """Build an initial-storm table from analysed storm centres.

    This is how an observed position -- a JMA analysis for Krovanh, say --
    seeds a track instead of waiting for the model to spin one up.

    Args:
        positions: One ``(track_id, lat, lon)`` per storm present at
            ``valid_time``. Longitudes are wrapped to [0, 360) because the
            tracker rejects negative values.
        valid_time: The initialization time the positions refer to.

    Returns:
        A DataFrame with the tracker's initial-storm schema.
    """
    frame = pd.DataFrame(
        [
            {
                "track_id": str(track_id),
                "valid_time": pd.Timestamp(valid_time),
                "lat": float(lat),
                "lon": float(lon) % 360,
            }
            for track_id, lat, lon in positions
        ]
    )
    return frame.astype(INITIAL_STORM_DTYPES)


def track_member(
    forecast: xr.Dataset,
    init_time: np.datetime64,
    initial_storms: pd.DataFrame | None = None,
    tracker: Any | None = None,
) -> pd.DataFrame:
    """Run the tracker on one member's forecast.

    Args:
        forecast: Full-field forecast for one member, with no ``batch`` or
            ``sample`` dimension left.
        init_time: Initialization time as a numpy datetime.
        initial_storms: Storms present at init time, from
            :func:`initial_storms_from_positions`. None means cyclogenesis
            detection only, via :func:`empty_initial_storms`.
        tracker: A tracker from :func:`build_tracker`; built on demand if
            omitted.

    Returns:
        Track table with longitudes wrapped to [0, 360).
    """
    from weathernext.cyclones import constants as cyclone_constants

    tracker = tracker or build_tracker()
    if initial_storms is None:
        initial_storms = empty_initial_storms()

    gridded = tracker.preprocess_gridded_ds(forecast)
    gridded = gridded.expand_dims(forecast_datetime=[init_time])
    gridded = gridded.assign_coords(
        lead_time_secs=gridded.time.astype("timedelta64[s]").astype(int)
    )
    gridded = gridded.assign_coords(
        date_time=gridded.forecast_datetime.data + gridded.time
    )

    tracks = tracker(
        gridded_ds=gridded.as_numpy(),
        initial_storms_df=initial_storms,
        do_cyclogenesis=True,
    )
    tracks[cyclone_constants.LON] = tracks[cyclone_constants.LON] % 360
    return tracks
