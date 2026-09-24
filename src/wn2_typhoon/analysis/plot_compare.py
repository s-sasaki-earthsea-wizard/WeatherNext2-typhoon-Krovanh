"""Figures that put the cases side by side.

Every figure is drawn from the tables in :mod:`compare`; none reads the
forecast fields. The cases are ordered by initialization time and take one
hue, light to dark, in that order (``plot.CASE_RAMP``), with a marker shape
per case as the second encoding. The observed storm is always black.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wn2_typhoon.analysis.compare import CaseResult, within_column
from wn2_typhoon.analysis.plot import (
    CASE_MARKERS,
    CASE_RAMP,
    INK,
    MEMBER_STYLE,
    MEMBER_STYLE_ON_TILES,
    OBSERVED_STYLE,
    _finish,
    _grid_figure,
    _panel_title,
    map_axes,
    track_extent,
)
from wn2_typhoon.analysis.track_error import TRACK_TIME

LINE_WIDTH = 1.8
MARKER_SIZE = 4.5


def _style(index: int) -> dict:
    """Line style for the case at ``index`` in initialization order."""
    return {
        "color": CASE_RAMP[index % len(CASE_RAMP)],
        "marker": CASE_MARKERS[index % len(CASE_MARKERS)],
        "markersize": MARKER_SIZE,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
        "linewidth": LINE_WIDTH,
    }


def _label_line_ends(axes, ends: list[tuple], min_gap: float) -> None:
    """Put each case label just past the last point of its curve.

    Curves that end at nearly the same height would print their labels on
    top of each other, so the labels are nudged apart from the bottom up,
    keeping at least ``min_gap`` (in data units) between them.

    Args:
        axes: Axes holding the curves.
        ends: ``(x, y, text)`` for each curve's last point.
        min_gap: Smallest vertical distance allowed between two labels.
    """
    ordered = sorted(ends, key=lambda end: end[1])
    heights = []
    for _, y, _ in ordered:
        if heights and y - heights[-1] < min_gap:
            y = heights[-1] + min_gap
        heights.append(y)
    for (x, _, text), y in zip(ordered, heights, strict=True):
        axes.annotate(
            text, (x, y), xytext=(5, 0), textcoords="offset points",
            fontsize=7.5, color=INK["secondary"], va="center", ha="left",
        )


def _event_lines(axes, events: dict[str, pd.Timestamp], label: bool = True) -> None:
    """Draw the observed milestones as hairlines."""
    for index, (name, when) in enumerate(events.items()):
        axes.axvline(when, color=INK["muted"], linewidth=0.8, linestyle=":", zorder=1)
        if label:
            axes.annotate(
                name, (when, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -3 - 10 * (index % 2)), textcoords="offset points",
                fontsize=7, color=INK["muted"], va="top", ha="left",
            )


def plot_error_vs_lead(
    skill: pd.DataFrame,
    cases: list[CaseResult],
    labels: dict[str, str],
    out_path: Path,
    within_km: float,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Mean member error, spread and near-member count against lead time.

    Three panels rather than one because they answer different questions:
    how wrong a typical member was, how far apart the members were, and how
    many were still with the observed storm at all. A case whose mean error
    is small while its near-member count is low is being flattered by
    cancellation.

    Args:
        skill: Output of ``compare.skill_by_lead``.
        cases: The cases, in initialization order.
        labels: ``{case_id: label}`` from ``compare.case_labels``.
        out_path: PNG destination.
        within_km: Radius the near-member count was computed with.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(8.5, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.1},
    )
    near = within_column(within_km)
    ends = []
    for index, case in enumerate(cases):
        rows = skill.loc[skill["case"] == case.case_id].sort_values("lead_hours")
        style = _style(index)
        label = f"{labels[case.case_id]}  ({case.init_time:%m-%d %HZ})"
        top.plot(rows["lead_hours"], rows["mean_error_km"], label=label,
                 markevery=(0, 4), **style)
        middle.plot(rows["lead_hours"], rows["spread_km"], markevery=(0, 4), **style)
        bottom.plot(rows["lead_hours"], rows[near], markevery=(0, 4), **style)
        if len(rows):
            ends.append((rows["lead_hours"].iloc[-1], rows["mean_error_km"].iloc[-1],
                         labels[case.case_id]))
    _label_line_ends(top, ends, min_gap=0.045 * float(skill["mean_error_km"].max()))

    top.set_ylabel("mean member error [km]")
    top.set_ylim(bottom=0)
    top.legend(fontsize=8, loc="upper left", title="init relative to formation",
               title_fontsize=8)
    middle.set_ylabel("spread [km]")
    middle.set_ylim(bottom=0)
    bottom.set_ylabel(f"members within {within_km:.0f} km")
    bottom.set_ylim(-0.3, skill["n_members"].max() + 0.5)
    bottom.set_xlabel("lead time [h]")
    bottom.set_xlim(left=0)
    for axes in (top, middle, bottom):
        axes.grid(alpha=0.25)
    if title:
        top.set_title(title)
    return _finish(fig, out_path, credit)


def plot_error_vs_valid_time(
    skill: pd.DataFrame,
    cases: list[CaseResult],
    labels: dict[str, str],
    events: dict[str, pd.Timestamp],
    out_path: Path,
    within_km: float,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Mean member error and near-member count against valid time.

    This is the forecaster's view: for a given moment of the storm, did the
    run that started later know more? The observed milestones are marked so
    the loop and the weakening can be found on the time axis.

    Args:
        skill: Output of ``compare.skill_by_lead``.
        cases: The cases, in initialization order.
        labels: ``{case_id: label}`` from ``compare.case_labels``.
        events: ``{name: time}`` milestones to mark.
        out_path: PNG destination.
        within_km: Radius the near-member count was computed with.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.5, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.1},
    )
    near = within_column(within_km)
    ends = []
    for index, case in enumerate(cases):
        rows = skill.loc[skill["case"] == case.case_id].sort_values("valid_time")
        style = _style(index)
        label = f"{labels[case.case_id]}  ({case.init_time:%m-%d %HZ})"
        top.plot(rows["valid_time"], rows["mean_error_km"], label=label,
                 markevery=(0, 4), **style)
        bottom.plot(rows["valid_time"], rows[near], markevery=(0, 4), **style)
        if len(rows):
            ends.append((rows["valid_time"].iloc[-1], rows["mean_error_km"].iloc[-1],
                         labels[case.case_id]))
    _label_line_ends(top, ends, min_gap=0.045 * float(skill["mean_error_km"].max()))
    _event_lines(top, events)
    _event_lines(bottom, events, label=False)

    top.set_ylabel("mean member error [km]")
    top.set_ylim(bottom=0)
    # Anchored a little below the top edge so the milestone labels stay visible.
    top.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92),
               title="init relative to formation", title_fontsize=8)
    bottom.set_ylabel(f"members within {within_km:.0f} km")
    bottom.set_ylim(-0.3, skill["n_members"].max() + 0.5)
    bottom.set_xlabel("valid time [UTC]")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for axes in (top, bottom):
        axes.grid(alpha=0.25)
    if title:
        top.set_title(title)
    return _finish(fig, out_path, credit)


def plot_spread_vs_error(
    skill: pd.DataFrame,
    cases: list[CaseResult],
    labels: dict[str, str],
    out_path: Path,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Spread against the error of the ensemble mean, one point per lead.

    Points on the diagonal are what a well-calibrated ensemble gives; the
    cloud here sits above it, because every member starts from the same ERA5
    state and the spread has to grow from nothing. Only leads with the full
    ensemble are drawn, as in ``compare.spread_skill``.

    Args:
        skill: Output of ``compare.skill_by_lead``.
        cases: The cases, in initialization order.
        labels: ``{case_id: label}`` from ``compare.case_labels``.
        out_path: PNG destination.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, axes = plt.subplots(figsize=(6.5, 6.0))
    full = skill.loc[skill["n_members"] == skill.groupby("case")["n_members"].transform("max")]
    limit = float(np.nanmax(full[["spread_km", "ensemble_mean_error_km"]].to_numpy())) * 1.05
    axes.plot([0, limit], [0, limit], color=INK["muted"], linewidth=0.8, linestyle="--",
              label="spread = error", zorder=1)
    for index, case in enumerate(cases):
        rows = full.loc[full["case"] == case.case_id]
        style = _style(index)
        axes.scatter(
            rows["spread_km"], rows["ensemble_mean_error_km"],
            s=28, color=style["color"], marker=style["marker"],
            edgecolors="white", linewidths=0.6, alpha=0.9,
            label=f"{labels[case.case_id]}  ({case.init_time:%m-%d %HZ})", zorder=3,
        )
    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_aspect("equal")
    axes.set_xlabel("spread about the ensemble mean [km]")
    axes.set_ylabel("error of the ensemble mean [km]")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8, loc="upper left", title="init relative to formation",
                title_fontsize=8)
    if title:
        axes.set_title(title)
    return _finish(fig, out_path, credit)


def plot_case_tracks(
    cases: list[CaseResult],
    best_track: pd.DataFrame,
    labels: dict[str, str],
    events: dict[str, pd.Timestamp],
    out_path: Path,
    basemap: str = "natural-earth",
    extent=None,
    margin_deg: float = 2.0,
    reference_margin_deg: float = 1.5,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """One map per case, members over the observed track, on a shared window.

    Small multiples rather than one map: forty member tracks in five colours
    on one panel cannot be read, and the question is how the spread of each
    case sits against the same observed loop. The last panel carries the
    observed track alone, zoomed in, with its dates and milestones, so the
    other five need no annotation.

    The window is shared and fixed rather than fitted to the tracks, so that
    one member wandering off does not shrink the storm in every panel; a
    track that leaves the window is simply cut at its edge. Pass the stored
    region (``output.region`` in the config) for a window the reader already
    knows from the data.

    No ensemble-mean track is drawn. See the ``compare`` module docstring.

    Args:
        cases: The cases, in initialization order; at most five.
        best_track: Reference track.
        labels: ``{case_id: label}`` from ``compare.case_labels``.
        events: ``{name: time}`` milestones, marked on the reference panel.
        out_path: PNG destination.
        basemap: One of ``plot.BASEMAPS``.
        extent: ``[lon_min, lon_max, lat_min, lat_max]`` window shared by the
            case panels; fitted to the tracks plus ``margin_deg`` when None.
        margin_deg: Padding used when ``extent`` is None.
        reference_margin_deg: Padding around the observed track on its own
            panel.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    import cartopy.crs as ccrs

    columns, rows = 3, 2
    if len(cases) > columns * rows - 1:
        raise ValueError(f"at most {columns * rows - 1} cases fit on the grid")

    if extent is None:
        all_tracks = pd.concat([case["track"] for case in cases], ignore_index=True)
        extent = track_extent(all_tracks, best_track, margin_deg)
    best = best_track.sort_values("time")
    reference_extent = track_extent(best.iloc[0:0], best, reference_margin_deg)

    # Size the figure from the window's shape, so the fixed-aspect maps fill
    # their boxes instead of floating in whitespace.
    width, dpi = 11.5, 160
    margins = {"left": 0.05, "right": 0.98, "top": 0.9, "bottom": 0.07,
               "wspace": 0.1, "hspace": 0.2}
    fig, panel_width = _grid_figure(extent, basemap, rows, columns, width, margins)
    member_style = MEMBER_STYLE_ON_TILES if basemap == "osm" else MEMBER_STYLE
    panel_width_px = panel_width * dpi

    for index, case in enumerate(cases):
        row, column = divmod(index, columns)
        sides = tuple(
            side for side, on in (("left", column == 0), ("bottom", row == rows - 1)) if on
        )
        axes = map_axes(fig, (rows, columns, index + 1), extent, basemap,
                        width_px=panel_width_px, label_sides=sides)
        for _, group in case["track"].groupby("member", sort=True):
            group = group.sort_values(TRACK_TIME)
            axes.plot(group["lon"], group["lat"], transform=ccrs.PlateCarree(),
                      **member_style)
        axes.plot(best["lon"], best["lat"], transform=ccrs.PlateCarree(), **OBSERVED_STYLE)
        _panel_title(axes, f"{labels[case.case_id]}  (init {case.init_time:%m-%d %HZ})")

    # Reference panel: the observed track alone, zoomed, with dates and milestones.
    index = columns * rows - 1
    axes = map_axes(fig, (rows, columns, index + 1), reference_extent, basemap,
                    width_px=panel_width_px, label_sides=("left", "bottom"))
    axes.plot(best["lon"], best["lat"], transform=ccrs.PlateCarree(), **OBSERVED_STYLE)
    # One label per 00Z point. A milestone joins its date in one label and
    # goes below-left, plain dates go above-right, so neighbouring days on the
    # slow-moving parts of the track do not print over each other.
    milestones = {pd.Timestamp(when): name for name, when in events.items()}
    daily = best.loc[(best["time"].dt.hour == 0) | best["time"].isin(milestones)]
    axes.plot(daily["lon"], daily["lat"], transform=ccrs.PlateCarree(),
              marker="o", markersize=4, color=OBSERVED_STYLE["color"], linestyle="none",
              zorder=6)
    for _, point in daily.iterrows():
        milestone = milestones.get(pd.Timestamp(point["time"]))
        if milestone:
            axes.plot(point["lon"], point["lat"], transform=ccrs.PlateCarree(),
                      marker="o", markersize=10, markerfacecolor="none",
                      markeredgecolor=OBSERVED_STYLE["color"], markeredgewidth=1.6,
                      linestyle="none", zorder=7)
        axes.annotate(
            f"{point['time']:%m-%d} {milestone}" if milestone else f"{point['time']:%m-%d}",
            (point["lon"], point["lat"]),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(axes),
            xytext=(-7, -9) if milestone else (4, 3), textcoords="offset points",
            fontsize=7 if milestone else 6.5,
            color=INK["primary"] if milestone else INK["secondary"],
            ha="right" if milestone else "left",
        )
    _panel_title(axes, "JMA reference (zoomed), 00Z dates and milestones")

    if title:
        fig.suptitle(title, y=0.97)
    return _finish(fig, out_path, credit)


def plot_lifetime(
    fate: pd.DataFrame,
    cases: list[CaseResult],
    labels: dict[str, str],
    observed_min_pressure_hpa: float,
    out_path: Path,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """When each member's track ended and how deep it got, per case.

    One dot per member, so the reader sees the population and not a summary
    of it: the question is whether the model keeps the storm going and deepens
    it, and a median would hide the members that did not.

    Args:
        fate: Output of ``compare.member_fate`` merged with the per-member
            ``end_error_hours`` (see ``scripts/compare_cases.py``).
        cases: The cases, in initialization order.
        labels: ``{case_id: label}`` from ``compare.case_labels``.
        observed_min_pressure_hpa: The storm's observed minimum pressure.
        out_path: PNG destination.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.8))
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    positions = np.arange(len(cases))
    fate_markers = {"stayed": "o", "north": "^", "west": "<", "east": ">", "south": "v"}

    for index, case in enumerate(cases):
        rows = fate.loc[fate["case"] == case.case_id].sort_values("member")
        jitter = np.linspace(-0.22, 0.22, len(rows)) if len(rows) > 1 else np.zeros(len(rows))
        colour = _style(index)["color"]
        for offset, (_, member) in zip(jitter, rows.iterrows(), strict=True):
            marker = fate_markers.get(str(member["fate"]), "o")
            left.plot(index + offset, member["end_error_hours"], marker=marker,
                      markersize=7, color=colour, markeredgecolor="white",
                      markeredgewidth=0.8, linestyle="none", zorder=3)
            right.plot(index + offset, member["min_pressure_hpa"], marker=marker,
                       markersize=7, color=colour, markeredgecolor="white",
                       markeredgewidth=0.8, linestyle="none", zorder=3)

    left.axhline(0.0, color=INK["primary"], linewidth=0.9)
    left.annotate("observed weakening", (0.01, 0.0), xycoords=("axes fraction", "data"),
                  xytext=(0, 3), textcoords="offset points", fontsize=7,
                  color=INK["secondary"], ha="left", va="bottom")
    left.set_ylabel("track end relative to observed weakening [h]")
    right.axhline(observed_min_pressure_hpa, color=INK["primary"], linewidth=0.9)
    right.annotate("observed minimum", (0.99, observed_min_pressure_hpa),
                   xycoords=("axes fraction", "data"), xytext=(0, -2),
                   textcoords="offset points", fontsize=7, color=INK["secondary"],
                   ha="right", va="top")
    right.set_ylabel("minimum central pressure [hPa]")
    right.invert_yaxis()

    for axes in (left, right):
        axes.set_xticks(positions)
        axes.set_xticklabels(
            [f"{labels[case.case_id]}\n{case.init_time:%m-%d %HZ}" for case in cases],
            fontsize=8,
        )
        axes.set_xlim(-0.6, len(cases) - 0.4)
        axes.grid(alpha=0.25, axis="y")
    handles = [
        plt.Line2D([], [], marker=marker, color=INK["muted"], linestyle="none",
                   markersize=7, label=f"ended {name}" if name != "stayed" else "stayed in box")
        for name, marker in fate_markers.items()
        if name in set(fate["fate"].astype(str))
    ]
    left.legend(handles=handles, fontsize=7.5, loc="lower right", title="track end",
                title_fontsize=7.5)
    if title:
        fig.suptitle(title)
    return _finish(fig, out_path, credit)
