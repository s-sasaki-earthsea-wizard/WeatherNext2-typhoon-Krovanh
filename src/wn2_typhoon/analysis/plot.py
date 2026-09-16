"""Figures: track map, error against lead time, and central pressure.

Every figure is drawn from the tables in :mod:`track_error`, never from the
forecast fields, so plotting needs neither the crops nor a GPU.

Matplotlib's Agg backend is selected on import because these run from scripts
with no display attached.

Attribution. Any figure published from here has to credit WeatherNext 2
(Google DeepMind, CC BY 4.0), ERA5 (Copernicus) and the JMA position table;
:func:`credit_line` builds that string, and the plotting functions put it on
the figure by default so it cannot be forgotten. See docs/license-notes.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wn2_typhoon.utils.geo import mean_position

MEMBER_STYLE = {"color": "#7f7f7f", "linewidth": 0.9, "alpha": 0.75, "zorder": 2}
ENSEMBLE_STYLE = {"color": "#d62728", "linewidth": 2.2, "zorder": 4}
OBSERVED_STYLE = {"color": "#111111", "linewidth": 2.2, "zorder": 5}


def credit_line(source: str) -> str:
    """Return the attribution required on a published figure.

    Args:
        source: Value of the reference track's ``source`` column, either
            "preliminary" or "post-analysis".

    Returns:
        A one-line credit naming the model, the initial conditions and the
        reference track, with the reference's release stated so a preliminary
        figure is never mistaken for a final one.
    """
    release = {
        "preliminary": "JMA preliminary position table",
        "post-analysis": "JMA post-analysis position table",
    }.get(source, f"JMA position table ({source})")
    return (
        "WeatherNext 2 (Google DeepMind, CC BY 4.0) "
        "| initial conditions ERA5 (Copernicus) "
        f"| reference {release}"
    )


def _finish(fig, out_path: Path, credit: str | None) -> Path:
    """Add the credit line, save and close.

    Args:
        fig: The figure to write.
        out_path: PNG destination; parent directories are created.
        credit: Attribution text, or None to leave it off.

    Returns:
        ``out_path``.
    """
    if credit:
        fig.text(0.01, 0.01, credit, fontsize=6.5, color="#555555", ha="left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately not bbox_inches="tight". On a cartopy GeoAxes, combining it
    # with a figure-level text artist collapses the saved bounding box to the
    # text alone, and the map is silently dropped: the track figure came out as
    # a 9 KB image holding nothing but this credit line. The figures set their
    # own size and use the default margins instead, which leave the bottom
    # strip this line sits in.
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _ensemble_mean_track(forecast_tracks: pd.DataFrame) -> pd.DataFrame:
    """Mean position at each valid time, over whichever members still exist.

    The member count falls with lead time as tracks end, so the mean is over a
    shrinking sample and the curve is not a forecast of one storm. It is drawn
    to show the centre of the spread, not as a track in its own right.

    Args:
        forecast_tracks: One track per member, from ``track_error.select_storm``.

    Returns:
        Columns ``valid_time``, ``lat``, ``lon``, ``n_members``.
    """
    rows = []
    for valid_time, group in forecast_tracks.groupby("valid_time", sort=True):
        lat, lon = mean_position(group["lat"], group["lon"])
        rows.append(
            {
                "valid_time": valid_time,
                "lat": lat,
                "lon": lon,
                "n_members": int(group["member"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def plot_tracks(
    forecast_tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    margin_deg: float = 3.0,
    credit: str | None = None,
) -> Path:
    """Draw the ensemble tracks over the reference track on a map.

    Args:
        forecast_tracks: One track per member, from ``track_error.select_storm``.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        out_path: PNG destination.
        title: Figure title.
        margin_deg: Padding around the drawn tracks, in degrees.
        credit: Attribution line; built from the reference's ``source`` column
            when omitted. Pass an empty string to suppress it.

    Returns:
        ``out_path``.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if credit is None:
        credit = credit_line(str(best_track["source"].iloc[0]))

    lats = np.concatenate([forecast_tracks["lat"].to_numpy(), best_track["lat"].to_numpy()])
    lons = np.concatenate([forecast_tracks["lon"].to_numpy(), best_track["lon"].to_numpy()])
    extent = [
        float(lons.min()) - margin_deg,
        float(lons.max()) + margin_deg,
        float(lats.min()) - margin_deg,
        float(lats.max()) + margin_deg,
    ]

    fig = plt.figure(figsize=(8.5, 7.5))
    axes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    axes.set_extent(extent, crs=ccrs.PlateCarree())
    axes.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f0eb", zorder=0)
    axes.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eaf1f7", zorder=0)
    axes.add_feature(
        cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="#888888", zorder=1
    )
    gridlines = axes.gridlines(draw_labels=True, linewidth=0.3, color="#cccccc")
    gridlines.top_labels = gridlines.right_labels = False

    for member, group in forecast_tracks.groupby("member", sort=True):
        group = group.sort_values("valid_time")
        axes.plot(
            group["lon"], group["lat"],
            transform=ccrs.PlateCarree(),
            label="ensemble members" if member == forecast_tracks["member"].min() else None,
            **MEMBER_STYLE,
        )

    # Only while every member is still present. Past that the mean is taken
    # over a shrinking, diverging sample and wanders off along a path no member
    # took: on the formation case it ran northwest to 30N 123E after the
    # members began to end, which is not a forecast of anything.
    mean_track = _ensemble_mean_track(forecast_tracks)
    complete = mean_track.loc[mean_track["n_members"] == mean_track["n_members"].max()]
    if not complete.empty:
        last = complete["valid_time"].max()
        axes.plot(
            complete["lon"], complete["lat"],
            transform=ccrs.PlateCarree(),
            label=f"ensemble mean (all {int(complete['n_members'].max())} members)",
            **ENSEMBLE_STYLE,
        )
        axes.plot(
            complete["lon"].iloc[-1], complete["lat"].iloc[-1],
            marker="|", markersize=9, markeredgewidth=2.2,
            color=ENSEMBLE_STYLE["color"], linestyle="none",
            transform=ccrs.PlateCarree(), zorder=ENSEMBLE_STYLE["zorder"],
            label=f"first member ends ({last:%m-%d %HZ})",
        )

    best = best_track.sort_values("time")
    axes.plot(
        best["lon"], best["lat"],
        transform=ccrs.PlateCarree(), label="JMA reference", **OBSERVED_STYLE,
    )
    axes.plot(
        best["lon"].iloc[0], best["lat"].iloc[0],
        marker="o", markersize=11, markerfacecolor="none",
        markeredgecolor="#111111", markeredgewidth=2.0,
        transform=ccrs.PlateCarree(), zorder=6, linestyle="none",
        label="observed formation",
    )

    # A GeoAxes keeps a fixed aspect, so it shrinks inside its subplot box and
    # an axes title floats far above the map. Put it on the figure instead.
    if title:
        fig.suptitle(title, y=0.92)
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return _finish(fig, out_path, credit)


def plot_error_vs_lead(
    errors: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Plot position error against lead time, with the along and cross parts.

    The upper panel carries the member envelope, the mean of the member errors
    and the error of the ensemble-mean position. Those last two differ because
    member errors partly cancel when the positions are averaged, and reporting
    only the smaller one would flatter the ensemble.

    The lower panel splits the error into along-track, where positive means the
    forecast is ahead of the observed storm, and cross-track, where positive is
    to the right of its direction of travel. A curve that sits on one side of
    zero is a systematic bias rather than scatter.

    Args:
        errors: Output of ``track_error.position_errors``.
        summary: Output of ``track_error.ensemble_summary``.
        out_path: PNG destination.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(8.5, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )

    upper.fill_between(
        summary["lead_hours"], summary["min_error_km"], summary["max_error_km"],
        color="#7f7f7f", alpha=0.18, label="member range",
    )
    for _, group in errors.groupby("member", sort=True):
        upper.plot(group["lead_hours"], group["error_km"], **MEMBER_STYLE)
    upper.plot(
        summary["lead_hours"], summary["mean_error_km"],
        label="mean of member errors", **ENSEMBLE_STYLE,
    )
    upper.plot(
        summary["lead_hours"], summary["ensemble_mean_error_km"],
        color="#1f77b4", linewidth=2.2, label="error of the ensemble mean", zorder=4,
    )
    upper.plot(
        summary["lead_hours"], summary["spread_km"],
        color="#2ca02c", linewidth=1.4, linestyle="--", label="spread", zorder=3,
    )
    upper.set_ylabel("position error [km]")
    upper.set_ylim(bottom=0)
    upper.grid(alpha=0.25)
    upper.legend(fontsize=8, loc="upper left")
    if title:
        upper.set_title(title)

    lower.axhline(0.0, color="#111111", linewidth=0.8)
    lower.plot(
        summary["lead_hours"], summary["mean_along_track_km"],
        color="#9467bd", linewidth=2.0, label="along-track (+ = forecast ahead)",
    )
    lower.plot(
        summary["lead_hours"], summary["mean_cross_track_km"],
        color="#ff7f0e", linewidth=2.0, label="cross-track (+ = right of motion)",
    )
    lower.set_xlabel("lead time [h]")
    lower.set_ylabel("mean component [km]")
    lower.grid(alpha=0.25)
    lower.legend(fontsize=8, loc="upper left")

    return _finish(fig, out_path, credit)


def plot_pressure(
    forecast_tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Plot central pressure against valid time, forecast against reference.

    Pressure is plotted rather than wind because the two sources define wind
    differently: JMA publishes a 10-minute mean and the tracker reports a
    gridded maximum.

    Args:
        forecast_tracks: One track per member, from ``track_error.select_storm``.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        out_path: PNG destination.
        title: Figure title.
        credit: Attribution line; built from the reference when omitted.

    Returns:
        ``out_path``.
    """
    if credit is None:
        credit = credit_line(str(best_track["source"].iloc[0]))

    fig, axes = plt.subplots(figsize=(8.5, 4.6))
    for member, group in forecast_tracks.groupby("member", sort=True):
        group = group.sort_values("valid_time")
        axes.plot(
            group["valid_time"], group["minimum_sea_level_pressure_hpa"],
            label="ensemble members" if member == forecast_tracks["member"].min() else None,
            **MEMBER_STYLE,
        )

    best = best_track.sort_values("time")
    axes.plot(
        best["time"], best["pressure_hpa"], label="JMA reference", **OBSERVED_STYLE
    )
    axes.invert_yaxis()
    axes.set_ylabel("central pressure [hPa]")
    axes.set_xlabel("valid time [UTC]")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8, loc="lower left")
    fig.autofmt_xdate()
    if title:
        axes.set_title(title)
    return _finish(fig, out_path, credit)
