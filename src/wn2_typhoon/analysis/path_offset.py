"""Measure a forecast position against the observed path: off it, and early or late on it.

The position error compares a forecast centre with where the storm was at the
same moment. It mixes two failures that mean different things. A member can
follow the observed path exactly and run half a day ahead of the storm, which
on a storm moving at 50 km/h is a 600 km error; or it can go another way
altogether. The first is a wrong clock on the right road, the second a wrong
road.

The usual answer, along-track and cross-track components, resolves the error
against the observed direction of motion at that moment. On Krovanh that
direction rotates through the loop, so the along-track component changes sign
from one lead to the next and means nothing. This module asks the question
without a direction:

    * **distance to the path**: the shortest distance from the forecast centre
      to the observed path, at any time. Small means the member is on the
      road, however late or early.
    * **clock**: the time at which the observed storm passed the nearest point
      of the path, minus the forecast's valid time. Positive means the member
      is where the storm would be later, i.e. ahead.

The path is the reference track from the first row at or after the case's
initialization to the end of the tropical phase, interpolated linearly
between its rows.

Two places where the nearest point says nothing:

    * **Past either end of the path.** A member that runs ahead of the storm
      beyond the last reference row, or lags behind the first, has the end
      point as its nearest point. Its distance to the path then includes how
      far ahead or behind it is, and its clock is clipped. Such positions are
      flagged and left out of both counts; they are reported as their own
      number.
    * **Where the path comes back on itself.** Krovanh turned back south at
      its northernmost point and crossed its own path two days later. Near
      the crossing two passes of the storm, days apart, are equally near, and
      a member that recurved north past the top of the loop is matched to the
      day the storm turned there. The clock is therefore read only up to
      ``loop_start_time`` in the config, against the path cut at that time;
      the distance to the path still uses the whole path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wn2_typhoon.utils.geo import along_cross_track_km, haversine_km, track_bearing_deg

# Points per interval between reference rows when the path is interpolated.
# The rows are 3 or 6 h apart, so the points are a few km apart at most,
# well under anything read off the result.
POINTS_PER_SEGMENT = 60

_HOUR = pd.Timedelta(hours=1)


def dense_path(
    track: pd.DataFrame, points_per_segment: int = POINTS_PER_SEGMENT
) -> pd.DataFrame:
    """Interpolate a track linearly in time, latitude and longitude.

    Args:
        track: Rows with ``time``, ``lat`` and ``lon``, in any order.
        points_per_segment: Points per interval between rows, the row itself
            included.

    Returns:
        ``time``, ``lat``, ``lon`` along the track, ending on its last row.
        Longitude is unwrapped, so it may leave [0, 360) on a track that
        crosses the dateline; distances are unaffected.
    """
    track = track.sort_values("time")
    start = track["time"].iloc[0]
    hours = ((track["time"] - start) / _HOUR).to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    lon = np.unwrap(track["lon"].to_numpy(dtype=float), period=360.0)
    fractions = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)

    def fill(values: np.ndarray) -> np.ndarray:
        steps = values[:-1, None] + fractions[None, :] * np.diff(values)[:, None]
        return np.concatenate([steps.ravel(), values[-1:]])

    return pd.DataFrame(
        {
            "time": start + pd.to_timedelta(fill(hours), unit="h"),
            "lat": fill(lat),
            "lon": fill(lon),
        }
    )


def _nearest(lat, lon, path: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest path point for each position, and its distance."""
    distance = haversine_km(
        np.asarray(lat, dtype=float)[:, None], np.asarray(lon, dtype=float)[:, None],
        path["lat"].to_numpy()[None, :], path["lon"].to_numpy()[None, :],
    )
    index = distance.argmin(axis=1)
    return index, distance[np.arange(len(index)), index]


def _beyond(
    lat, lon, path: pd.DataFrame, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Whether a position lies behind the path's start or past its end.

    Only a position whose nearest point is the first or last one can be;
    it is beyond when it lies ahead of the end (behind the start) along the
    observed motion there.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    before = np.zeros(len(index), dtype=bool)
    past = np.zeros(len(index), dtype=bool)
    if len(path) < 2:
        return before, past
    bearing = track_bearing_deg(path["lat"].to_numpy(), path["lon"].to_numpy())
    for flags, point, sign in ((before, 0, -1.0), (past, len(path) - 1, 1.0)):
        at = index == point
        if at.any():
            along, _ = along_cross_track_km(
                path["lat"].iloc[point], path["lon"].iloc[point],
                lat[at], lon[at], bearing[point],
            )
            flags[at] = sign * along > 0
    return before, past


def path_offsets(
    errors: pd.DataFrame,
    best_track: pd.DataFrame,
    init_time,
    clock_until=None,
) -> pd.DataFrame:
    """Measure each forecast position of one case against the observed path.

    Args:
        errors: One case's per-member errors from
            ``track_error.position_errors``; needs ``valid_time``,
            ``fcst_lat`` and ``fcst_lon``.
        best_track: Reference track, already cut to the tropical phase.
        init_time: The case's initialization; the path starts at the first
            reference row at or after it.
        clock_until: Last valid time at which the clock is read, and where
            the path it is read against ends (``loop_start_time``); None
            reads it along the whole path.

    Returns:
        ``errors`` with four columns added:

        * ``path_km``: distance to the nearest point of the path.
        * ``before_start`` / ``past_end``: the position lies behind the
          path's first point or beyond its last, so ``path_km`` partly
          measures how far behind or ahead it is.
        * ``clock_hours``: when the observed storm passed the nearest point
          of the path, minus ``valid_time``; positive is ahead. NaN after
          ``clock_until`` and for a position beyond either end of the path
          the clock is read against.
    """
    init = pd.Timestamp(str(init_time))
    path = dense_path(best_track.loc[best_track["time"] >= init])
    out = errors.copy()
    lat, lon = out["fcst_lat"].to_numpy(), out["fcst_lon"].to_numpy()

    index, distance = _nearest(lat, lon, path)
    out["path_km"] = distance
    out["before_start"], out["past_end"] = _beyond(lat, lon, path, index)

    valid = pd.to_datetime(out["valid_time"])
    readable = np.ones(len(out), dtype=bool)
    clock_rows = best_track.loc[best_track["time"] >= init]
    if clock_until is not None:
        until = pd.Timestamp(str(clock_until))
        readable &= (valid <= until).to_numpy()
        clock_rows = clock_rows.loc[clock_rows["time"] <= until]
    clock = np.full(len(out), np.nan)
    if readable.any() and len(clock_rows):
        clock_path = dense_path(clock_rows)
        index, _ = _nearest(lat[readable], lon[readable], clock_path)
        before, past = _beyond(lat[readable], lon[readable], clock_path, index)
        hours = ((clock_path["time"].to_numpy()[index] - valid.to_numpy()[readable])
                 / np.timedelta64(1, "h"))
        clock[readable] = np.where(before | past, np.nan, hours)
    out["clock_hours"] = clock
    return out


def chamfer_km(track: pd.DataFrame, reference: pd.DataFrame) -> float:
    """How far apart two tracks are as shapes, whatever the timing.

    The mean distance from each point of one track to the nearest point of the
    other, taken both ways and averaged. Both tracks are interpolated between
    their rows first, so the result does not depend on how often either was
    sampled: compared row by row, a member of a storm moving 300 km in six
    hours would sit up to 150 km from the path midway between its own
    positions even when it lies exactly on it.

    Unlike a time-warped alignment (DTW), nothing is paired with anything, so
    a member that leaves the observed road produces no artefact downstream of
    where it left: the part of the path it never visited simply counts as far
    away. Pass the member without its positions beyond either end of the path
    (``before_start``, ``past_end``), which are timing, not shape.

    Args:
        track: The member, with ``time``, ``lat`` and ``lon``.
        reference: The observed path, with ``time``, ``lat`` and ``lon``.

    Returns:
        The symmetric mean nearest distance [km]; NaN if either is empty.
    """
    if track.empty or reference.empty:
        return float("nan")
    a, b = dense_path(track), dense_path(reference)
    distance = haversine_km(
        a["lat"].to_numpy()[:, None], a["lon"].to_numpy()[:, None],
        b["lat"].to_numpy()[None, :], b["lon"].to_numpy()[None, :],
    )
    return float(0.5 * (distance.min(axis=1).mean() + distance.min(axis=0).mean()))
