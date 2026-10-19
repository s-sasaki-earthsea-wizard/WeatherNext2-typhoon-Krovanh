"""How directed a storm's motion is: drift against scatter, member against observation.

A storm that stalls has not stopped; it wanders. Over a window of n
six-hourly steps its motion can be read the way a drifting Brownian particle
is: the net displacement is the drift, the scatter of the steps about their
mean is the diffusion. Their ratio,

    Pe = |sum of steps|^2 / sum |step - mean step|^2,

is a Péclet number for the track: advection against diffusion. It is
dimensionless and needs no diffusion coefficient from outside, because the
scatter in the denominator is measured from the same steps. Large means the
storm is being carried somewhere; small means its net progress is lost in its
own fluctuations.

No threshold is put on it. Two choices would each have added an assumption:

    * **The reference.** Steps of an unbiased random walk give Pe near
      n/(n-1), but a cyclone is not a random walk. The reference is instead
      the observed storm over the same window: ``log10(Pe_member /
      Pe_observed)`` says whether the members were more or less directed than
      the storm at that moment.
    * **The window.** A stall lasts about a day and a loop about two, and each
      shows at its own scale. Every window from 18 to 36 h is computed and
      shown together, as a scalogram, rather than one being picked. 18 h is
      the shortest window whose scatter rests on more than two steps.

What it does not tell apart: a stall and a loop both lower Pe, the first by
shrinking the net displacement, the second by turning it round. The position
jitter of the tracker and of the analysis (tens of km) adds to the scatter,
so a weak, slow storm reads less directed than its motion is.

The reference uses only its rows at 00, 06, 12 and 18 UTC, the same times as
the members: a preliminary table's 3-hourly rows, or an extra row at a
landfall, would otherwise change the step length between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wn2_typhoon.analysis.storms import StormResult
from wn2_typhoon.analysis.track_error import TRACK_TIME
from wn2_typhoon.utils.geo import EARTH_RADIUS_KM

STEP_HOURS = 6
#: Windows of the scalogram [h]: 3 to 6 steps.
WINDOW_HOURS = (18, 24, 30, 36)
#: Share of the members run that must have a value before a window is shown.
MIN_MEMBER_SHARE = 0.5
#: How far the member curve is shifted against the observed one when looking
#: for the lag of best agreement [h].
MAX_LAG_HOURS = 36
#: Overlapping windows needed for a lag to be reported.
MIN_LAG_OVERLAP = 8


def step_vectors_km(lat, lon) -> np.ndarray:
    """Steps between consecutive positions, east and north, in km.

    Each step is projected on the plane tangent at its own midpoint, which
    for steps of a few hundred km is exact to well under a km.

    Args:
        lat: Latitudes [deg], in time order.
        lon: Longitudes [deg], in time order.

    Returns:
        Array of shape (len - 1, 2).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    mid = np.radians((lat[:-1] + lat[1:]) / 2)
    east = np.radians((np.diff(lon) + 180.0) % 360.0 - 180.0) * EARTH_RADIUS_KM * np.cos(mid)
    north = np.radians(np.diff(lat)) * EARTH_RADIUS_KM
    return np.column_stack([east, north])


def peclet(
    track: pd.DataFrame,
    window_hours: int,
    time_column: str = "time",
    step_hours: int = STEP_HOURS,
) -> pd.Series:
    """Pe over every window of a track that covers ``window_hours`` without a gap.

    Args:
        track: Positions with ``lat``, ``lon`` and a time column, every
            ``step_hours``; windows across a missing row are skipped.
        window_hours: Window length, a multiple of ``step_hours``.
        time_column: Name of the time column.
        step_hours: Spacing of the rows.

    Returns:
        Pe indexed by the window's centre time. A window whose steps are all
        equal has no scatter and gives inf, or a very large value where
        rounding leaves a trace of one.
    """
    n = window_hours // step_hours
    track = track.sort_values(time_column)
    times = pd.to_datetime(track[time_column]).reset_index(drop=True)
    steps = step_vectors_km(track["lat"], track["lon"])
    span = pd.Timedelta(hours=window_hours)
    values = {}
    for i in range(len(steps) - n + 1):
        if times[i + n] - times[i] != span:
            continue
        window = steps[i:i + n]
        drift = float((window.sum(axis=0) ** 2).sum())
        scatter = float(((window - window.mean(axis=0)) ** 2).sum())
        values[times[i] + span / 2] = drift / scatter if scatter > 0 else np.inf
    return pd.Series(values, dtype=float)


def synoptic(best_track: pd.DataFrame, step_hours: int = STEP_HOURS) -> pd.DataFrame:
    """The reference rows at multiples of ``step_hours`` after 00 UTC."""
    times = best_track["time"]
    on_step = (times.dt.minute == 0) & (times.dt.hour % step_hours == 0)
    return best_track.loc[on_step].sort_values("time")


def _member_tracks(storm: StormResult):
    """Each member's selected track, cut at the end of the tropical phase."""
    end = pd.Timestamp(storm.end.time)
    for case in storm.cases:
        for member, track in case["track"].groupby("member", sort=True):
            yield case.case_id, int(member), track.loc[track[TRACK_TIME] <= end]


def directedness_by_time(
    storms: list[StormResult],
    window_hours=WINDOW_HOURS,
    min_member_share: float = MIN_MEMBER_SHARE,
) -> pd.DataFrame:
    """Members' Pe against the observed storm's, per window and valid time.

    Args:
        storms: The storms being compared.
        window_hours: Window lengths of the scalogram.
        min_member_share: Share of the members run (evaluated cases times
            members) that must have a value for the window to be kept, so a
            window resting on a handful of members is not shown.

    Returns:
        One row per storm, window and centre time at which the observed Pe
        exists: ``storm``, ``window_hours``, ``centre``, ``pe_observed``,
        ``n_members``, ``median_log_pe`` (members), the quartiles of
        ``log_ratio`` = log10(Pe_member / Pe_observed) (``log_ratio_q25``,
        ``log_ratio_median``, ``log_ratio_q75``) and ``share_below``, the
        share of members less directed than the storm.
    """
    rows = []
    for storm in storms:
        reference = synoptic(storm.best_track)
        n_run = len(storm.cases) * storm.n_members
        tracks = list(_member_tracks(storm))
        for hours in window_hours:
            observed = peclet(reference, hours)
            members = pd.concat(
                [peclet(track, hours, TRACK_TIME).rename(f"{case}/{member}")
                 for case, member, track in tracks],
                axis=1,
            )
            for centre, pe_observed in observed.items():
                if centre not in members.index:
                    continue
                values = members.loc[centre].dropna()
                if len(values) < min_member_share * n_run:
                    continue
                # A Pe of 0 or inf gives an infinite log; the ratio of two of
                # them is undefined and left out of the quantiles.
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_pe = np.log10(values.to_numpy())
                    ratio = log_pe - np.log10(pe_observed)
                rows.append(
                    {
                        "storm": storm.label,
                        "window_hours": hours,
                        "centre": centre,
                        "pe_observed": pe_observed,
                        "n_members": len(values),
                        "median_log_pe": float(np.median(log_pe)),
                        "log_ratio_q25": float(np.nanquantile(ratio, 0.25)),
                        "log_ratio_median": float(np.nanmedian(ratio)),
                        "log_ratio_q75": float(np.nanquantile(ratio, 0.75)),
                        "share_below": float((values < pe_observed).mean()),
                    }
                )
    return pd.DataFrame(rows)


def directedness_lag(
    table: pd.DataFrame,
    max_lag_hours: int = MAX_LAG_HOURS,
    step_hours: int = STEP_HOURS,
    min_overlap: int = MIN_LAG_OVERLAP,
) -> pd.DataFrame:
    """How well the members' directedness lines up with the storm's, by shift.

    Correlates the observed log10 Pe with the members' median, shifting the
    members by whole steps. A positive lag means the members' slowing and
    speeding up comes that many hours after the storm's. It needs no
    threshold on Pe at all.

    The whole profile is returned, not only its best shift: on Krovanh the
    correlation at 0 and +6 h differs by 0.02, and how the members' median is
    taken decides which wins. The curve as a whole is dominated by its
    longest, most directed stretch, so the lag says when the members' motion
    as a whole tracks the storm's, not when they enter the stall.

    Args:
        table: Output of :func:`directedness_by_time`.
        max_lag_hours: Largest shift tried either way.
        step_hours: Shift increment.
        min_overlap: Windows that must overlap at a shift for it to count.

    Returns:
        One row per storm, window and shift with enough overlap: ``storm``,
        ``window_hours``, ``lag_hours``, ``correlation``, ``n_overlap`` and
        ``best`` (the shift of highest correlation for that storm and window).
    """
    rows = []
    for (storm, hours), group in table.groupby(["storm", "window_hours"], sort=False):
        with np.errstate(divide="ignore"):
            observed = pd.Series(np.log10(group["pe_observed"].to_numpy()),
                                 index=pd.to_datetime(group["centre"]))
        members = pd.Series(group["median_log_pe"].to_numpy(),
                            index=pd.to_datetime(group["centre"]))
        for lag in range(-max_lag_hours, max_lag_hours + 1, step_hours):
            shifted = members.copy()
            shifted.index = shifted.index - pd.Timedelta(hours=lag)
            common = observed.index.intersection(shifted.index)
            pair = pd.DataFrame({"o": observed[common], "m": shifted[common]})
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < min_overlap:
                continue
            rows.append(
                {
                    "storm": storm,
                    "window_hours": int(hours),
                    "lag_hours": lag,
                    "correlation": float(pair["o"].corr(pair["m"])),
                    "n_overlap": len(pair),
                }
            )
    profile = pd.DataFrame(
        rows, columns=["storm", "window_hours", "lag_hours", "correlation", "n_overlap"]
    )
    if profile.empty:
        return profile.assign(best=pd.Series(dtype=bool))
    top = profile.groupby(["storm", "window_hours"])["correlation"].transform("max")
    return profile.assign(best=profile["correlation"] == top)
