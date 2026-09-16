"""Compare forecast storm centres with the JMA reference track.

Metrics, per lead time:

    * great-circle position error [km] for each member, for the ensemble mean
      position, and the spread about that mean
    * the same error resolved into along-track and cross-track parts, which
      separates "too fast or too slow" from "off to one side"
    * central pressure difference [hPa]
    * genesis timing and the time the storm stops being tracked

Maximum wind is reported for reference only: JMA publishes a 10-minute mean and
the tracker reports a gridded maximum, so the two are not the same quantity.

Selecting the storm
-------------------
The tracker returns every cyclone it finds on the global field, five to eight
per member, so a case's ``tracks.csv`` is not one track. No case is seeded
(docs/requirements.md decision 12), so Krovanh is identified by position: at
the earliest time a member's tracks and the reference track share, take the
track nearest the observed centre.

That is safe here rather than merely convenient. Measured over the sixteen
members run on 2026-09-16, the nearest track sat 29-134 km from the observed
position while the runner-up was at least 1879 km away, so any threshold
between roughly 200 and 1800 km selects the same track. The threshold exists to
notice when the rule stops working, not to do the work.

Track ids are not stable between members, so selection is never by id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

from wn2_typhoon.utils.geo import (
    along_cross_track_km,
    haversine_km,
    mean_position,
    track_bearing_deg,
)

# Safety valve for the selection rule described in the module docstring. A
# member whose nearest track is further away than this is recorded as a miss
# rather than being matched to whatever happened to be closest.
SELECTION_THRESHOLD_KM = 500.0

# Column names in the tracker's output.
TRACK_TIME = "valid_time"
TRACK_PRESSURE = "minimum_sea_level_pressure_hpa"
TRACK_WIND = "maximum_sustained_wind_speed_knots"


@dataclass(frozen=True)
class Selection:
    """Which track was taken for one member, and how clear the choice was.

    Attributes:
        member: Ensemble member index.
        track_id: Id of the chosen track, or None when nothing was near enough.
        anchor_time: Time at which the choice was made.
        distance_km: Distance from the observed centre at that time.
        runner_up_km: Distance of the next nearest track, NaN when the member
            produced only one. The ratio to ``distance_km`` says whether the
            rule discriminated or guessed.
    """

    member: int
    track_id: str | None
    anchor_time: pd.Timestamp | None
    distance_km: float
    runner_up_km: float

    @property
    def found(self) -> bool:
        """Whether a track was matched at all."""
        return self.track_id is not None


def _anchor_time(member_tracks: pd.DataFrame, best_track: pd.DataFrame, init_time):
    """Earliest time after lead 0 present in both a member's tracks and the reference.

    Lead 0 is excluded deliberately. The model predicts nothing there, so any
    row at that time came from the tracker's own input, and on output produced
    before seeding was dropped it is the observed position echoed back. Letting
    it anchor the selection would match a track to itself at zero distance.
    """
    shared = np.intersect1d(
        member_tracks[TRACK_TIME].unique(), best_track["time"].unique()
    )
    shared = shared[shared > np.datetime64(pd.Timestamp(str(init_time)))]
    return None if shared.size == 0 else pd.Timestamp(shared.min())


def select_member_track(
    member_tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    init_time: str | np.datetime64,
    threshold_km: float = SELECTION_THRESHOLD_KM,
) -> Selection:
    """Pick the track belonging to the reference storm for one member.

    Args:
        member_tracks: One member's rows from the tracker output.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        init_time: Initialization time of the case, so that lead 0 is excluded.
        threshold_km: Distance beyond which the match is rejected.

    Returns:
        A :class:`Selection`, with ``track_id`` None when no track was near
        enough at the anchor time.
    """
    anchor = _anchor_time(member_tracks, best_track, init_time)
    member = int(member_tracks["member"].iloc[0])
    if anchor is None:
        return Selection(member, None, None, np.nan, np.nan)

    observed = best_track.loc[best_track["time"] == anchor].iloc[0]
    at_anchor = member_tracks.loc[member_tracks[TRACK_TIME] == anchor].copy()
    at_anchor["distance_km"] = haversine_km(
        at_anchor["lat"], at_anchor["lon"], observed["lat"], observed["lon"]
    )
    at_anchor = at_anchor.sort_values("distance_km")

    nearest = at_anchor.iloc[0]
    runner_up = (
        float(at_anchor["distance_km"].iloc[1]) if len(at_anchor) > 1 else np.nan
    )
    if nearest["distance_km"] > threshold_km:
        return Selection(member, None, anchor, float(nearest["distance_km"]), runner_up)
    return Selection(
        member,
        str(nearest["track_id"]),
        anchor,
        float(nearest["distance_km"]),
        runner_up,
    )


def select_storm(
    tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    init_time: str | np.datetime64,
    threshold_km: float = SELECTION_THRESHOLD_KM,
) -> tuple[pd.DataFrame, list[Selection]]:
    """Reduce a case's tracker output to one track per member.

    Args:
        tracks: A case's concatenated tracker output, with a ``member`` column.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        init_time: Initialization time of the case, so that lead 0 is excluded.
        threshold_km: Distance beyond which a match is rejected.

    Returns:
        ``(selected, selections)``. ``selected`` holds the chosen rows for
        every member that matched, in time order. ``selections`` records the
        decision for every member including the misses, so a case can report
        how many members produced the storm at all.
    """
    tracks = tracks.copy()
    tracks[TRACK_TIME] = pd.to_datetime(tracks[TRACK_TIME])
    selections, chosen = [], []
    for member, member_tracks in tracks.groupby("member", sort=True):
        selection = select_member_track(
            member_tracks, best_track, init_time, threshold_km
        )
        selections.append(selection)
        if selection.found:
            rows = member_tracks.loc[
                member_tracks["track_id"].astype(str) == selection.track_id
            ]
            chosen.append(rows.sort_values(TRACK_TIME))
    selected = (
        pd.concat(chosen, ignore_index=True)
        if chosen
        else tracks.iloc[0:0].reset_index(drop=True)
    )
    return selected, selections


def position_errors(
    forecast_tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    init_time: str | np.datetime64,
) -> pd.DataFrame:
    """Compute position and pressure error per member and lead time.

    The reference track is 3-hourly and the forecast 6-hourly, so the join is
    exact and nothing is interpolated. Times present in only one of the two are
    dropped, which is what ends the comparison when the storm stops being
    tracked or when the reference table runs out.

    Args:
        forecast_tracks: One track per member, from :func:`select_storm`.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        init_time: Initialization time of the case.

    Returns:
        Long-format table with one row per member and valid time: ``member``,
        ``valid_time``, ``lead_hours``, ``error_km``, ``along_track_km``,
        ``cross_track_km``, ``dp_hpa``, plus the two positions. Lead 0 is
        excluded, because the model predicts nothing there.
    """
    reference = best_track.loc[:, ["time", "lat", "lon", "pressure_hpa"]].copy()
    reference["motion_bearing_deg"] = track_bearing_deg(
        reference["lat"].to_numpy(), reference["lon"].to_numpy()
    )
    reference = reference.rename(
        columns={"lat": "obs_lat", "lon": "obs_lon", "pressure_hpa": "obs_pressure_hpa"}
    )

    forecast = forecast_tracks.loc[
        :, ["member", TRACK_TIME, "lat", "lon", TRACK_PRESSURE, TRACK_WIND]
    ].copy()
    forecast[TRACK_TIME] = pd.to_datetime(forecast[TRACK_TIME])
    forecast = forecast.rename(
        columns={
            TRACK_TIME: "valid_time",
            "lat": "fcst_lat",
            "lon": "fcst_lon",
            TRACK_PRESSURE: "fcst_pressure_hpa",
            TRACK_WIND: "fcst_wind_kt",
        }
    )

    merged = forecast.merge(reference, left_on="valid_time", right_on="time")
    if merged.empty:
        return merged.assign(lead_hours=[], error_km=[]).drop(columns=["time"])
    merged = merged.drop(columns=["time"])

    start = pd.Timestamp(str(init_time))
    merged["lead_hours"] = (
        (merged["valid_time"] - start).dt.total_seconds() / 3600.0
    ).astype(int)
    merged = merged.loc[merged["lead_hours"] > 0]

    merged["error_km"] = haversine_km(
        merged["obs_lat"], merged["obs_lon"], merged["fcst_lat"], merged["fcst_lon"]
    )
    along, cross = along_cross_track_km(
        merged["obs_lat"],
        merged["obs_lon"],
        merged["fcst_lat"],
        merged["fcst_lon"],
        merged["motion_bearing_deg"],
    )
    merged["along_track_km"] = along
    merged["cross_track_km"] = cross
    merged["dp_hpa"] = merged["fcst_pressure_hpa"] - merged["obs_pressure_hpa"]
    return merged.sort_values(["member", "lead_hours"]).reset_index(drop=True)


def ensemble_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-member errors onto one row per lead time.

    Two different quantities are both called "the ensemble error" in the
    literature and they are not equal, so both are reported. The mean of the
    members' errors measures how well a typical member did; the error of the
    ensemble-mean position measures how well the ensemble did as one forecast,
    and is usually the smaller of the two because member errors partly cancel.

    Args:
        errors: Output of :func:`position_errors`.

    Returns:
        One row per lead time: ``lead_hours``, ``n_members``,
        ``mean_error_km``, ``ensemble_mean_error_km``, ``spread_km``,
        ``min_error_km``, ``max_error_km``, ``mean_along_track_km``,
        ``mean_cross_track_km``, ``mean_dp_hpa``.
    """
    if errors.empty:
        return pd.DataFrame(
            columns=[
                "lead_hours", "n_members", "mean_error_km",
                "ensemble_mean_error_km", "spread_km", "min_error_km",
                "max_error_km", "mean_along_track_km", "mean_cross_track_km",
                "mean_dp_hpa",
            ]
        )

    rows = []
    for lead, group in errors.groupby("lead_hours", sort=True):
        centre_lat, centre_lon = mean_position(group["fcst_lat"], group["fcst_lon"])
        observed = group.iloc[0]
        rows.append(
            {
                "lead_hours": int(lead),
                "n_members": int(group["member"].nunique()),
                "mean_error_km": float(group["error_km"].mean()),
                "ensemble_mean_error_km": float(
                    haversine_km(
                        observed["obs_lat"], observed["obs_lon"], centre_lat, centre_lon
                    )
                ),
                "spread_km": float(
                    np.sqrt(
                        np.mean(
                            haversine_km(
                                group["fcst_lat"], group["fcst_lon"],
                                centre_lat, centre_lon,
                            )
                            ** 2
                        )
                    )
                ),
                "min_error_km": float(group["error_km"].min()),
                "max_error_km": float(group["error_km"].max()),
                "mean_along_track_km": float(group["along_track_km"].mean()),
                "mean_cross_track_km": float(group["cross_track_km"].mean()),
                "mean_dp_hpa": float(group["dp_hpa"].mean()),
            }
        )
    return pd.DataFrame(rows)


def genesis_report(
    forecast_tracks: pd.DataFrame,
    selections: list[Selection],
    observed_genesis: str,
    init_time: str,
    step_hours: int = 6,
) -> pd.DataFrame:
    """When each member first produced the storm, against the observed time.

    A caveat is attached to every row rather than left to the reader. The
    tracker cannot report a storm before the first rollout step, because the
    model emits no cyclone fields for the initial state. A case initialized
    less than ``step_hours`` before the observed genesis therefore cannot
    resolve an early genesis at all, and one initialized after it cannot
    measure genesis timing in any direction.

    Args:
        forecast_tracks: One track per member, from :func:`select_storm`.
        selections: The decisions from :func:`select_storm`, so that members
            which never produced the storm appear as misses.
        observed_genesis: Observed formation time, UTC ISO 8601.
        init_time: Initialization time of the case, UTC ISO 8601.
        step_hours: Model time step.

    Returns:
        One row per member: ``member``, ``found``, ``model_genesis``,
        ``observed_genesis``, ``lead_error_hours`` (negative is early), and
        ``resolvable``, which is False when the case cannot measure the sign.
    """
    observed = pd.Timestamp(observed_genesis)
    earliest = pd.Timestamp(init_time) + pd.Timedelta(hours=step_hours)
    resolvable = earliest < observed

    starts = {}
    if not forecast_tracks.empty:
        times = pd.to_datetime(forecast_tracks[TRACK_TIME])
        starts = (
            forecast_tracks.assign(**{TRACK_TIME: times})
            .groupby("member")[TRACK_TIME]
            .min()
            .to_dict()
        )

    rows = []
    for selection in selections:
        start = starts.get(selection.member)
        rows.append(
            {
                "member": selection.member,
                "found": selection.found,
                "model_genesis": start,
                "observed_genesis": observed,
                "lead_error_hours": (
                    None
                    if start is None
                    else (start - observed).total_seconds() / 3600.0
                ),
                "resolvable": resolvable,
            }
        )
    return pd.DataFrame(rows)


def lifetime_report(
    forecast_tracks: pd.DataFrame,
    selections: list[Selection],
    observed_end: str,
) -> pd.DataFrame:
    """When each member stopped tracking the storm, against the observed time.

    The end of a track is the tracker's dissipation criterion, not a JMA grade
    change, so this answers "how long did the model keep a cyclone here" rather
    than "when did it weaken to a depression". Read the two together with the
    minimum pressure, which is also reported here.

    Args:
        forecast_tracks: One track per member, from :func:`select_storm`.
        selections: The decisions from :func:`select_storm`.
        observed_end: Observed time the storm ceased to be a typhoon, UTC.

    Returns:
        One row per member: ``member``, ``found``, ``track_end``,
        ``observed_end``, ``end_error_hours`` (negative is early),
        ``duration_hours``, ``min_pressure_hpa``, ``min_pressure_time``.
    """
    observed = pd.Timestamp(observed_end)
    rows = []
    for selection in selections:
        row = {
            "member": selection.member,
            "found": selection.found,
            "track_end": None,
            "observed_end": observed,
            "end_error_hours": None,
            "duration_hours": None,
            "min_pressure_hpa": None,
            "min_pressure_time": None,
        }
        if selection.found and not forecast_tracks.empty:
            member_rows = forecast_tracks.loc[
                forecast_tracks["member"] == selection.member
            ].copy()
            member_rows[TRACK_TIME] = pd.to_datetime(member_rows[TRACK_TIME])
            end = member_rows[TRACK_TIME].max()
            row["track_end"] = end
            row["end_error_hours"] = (end - observed).total_seconds() / 3600.0
            row["duration_hours"] = (
                end - member_rows[TRACK_TIME].min()
            ).total_seconds() / 3600.0
            if member_rows[TRACK_PRESSURE].notna().any():
                deepest = member_rows.loc[member_rows[TRACK_PRESSURE].idxmin()]
                row["min_pressure_hpa"] = float(deepest[TRACK_PRESSURE])
                row["min_pressure_time"] = deepest[TRACK_TIME]
        rows.append(row)
    return pd.DataFrame(rows)


def initial_state_error(
    inputs: xr.Dataset,
    observed_lat: float,
    observed_lon: float,
    search_km: float = 300.0,
) -> dict[str, float]:
    """How far the initial conditions put the storm from the analysed centre.

    This is the floor under any forecast error worth reporting. ERA5 and the
    JMA analysis do not place the storm identically, and 0.25 degree reanalysis
    is known to under-deepen tropical cyclones, so a short-lead error of this
    size says more about the initial state than about the model.

    The centre is taken as the mean-sea-level-pressure minimum within
    ``search_km`` of the analysed position in the last input frame, which is
    the one valid at initialization.

    Args:
        inputs: A case's model input file, as written by
            ``scripts/prepare_inputs.py``.
        observed_lat: Analysed latitude at init time [deg].
        observed_lon: Analysed longitude at init time [deg east].
        search_km: Radius to search for the pressure minimum.

    Returns:
        ``era5_lat``, ``era5_lon``, ``era5_pressure_hpa``, ``distance_km``.

    Raises:
        ValueError: If no grid point falls inside ``search_km``.
    """
    field = inputs["mean_sea_level_pressure"].isel(time=-1)
    for dim in ("batch", "level"):
        if dim in field.dims:
            field = field.isel({dim: 0})

    lat_grid, lon_grid = np.meshgrid(
        field["lat"].to_numpy(), field["lon"].to_numpy(), indexing="ij"
    )
    distance = haversine_km(lat_grid, lon_grid, observed_lat, observed_lon % 360)
    inside = distance <= search_km
    if not inside.any():
        raise ValueError(
            f"no grid point within {search_km} km of "
            f"{observed_lat}N {observed_lon}E"
        )

    values = np.where(inside, field.to_numpy(), np.inf)
    row, column = np.unravel_index(np.argmin(values), values.shape)
    pressure = float(field.to_numpy()[row, column])
    # ERA5 mean sea level pressure arrives in Pa; the JMA table is in hPa.
    if pressure > 10000.0:
        pressure /= 100.0
    return {
        "era5_lat": float(lat_grid[row, column]),
        "era5_lon": float(lon_grid[row, column]),
        "era5_pressure_hpa": pressure,
        "distance_km": float(distance[row, column]),
    }
