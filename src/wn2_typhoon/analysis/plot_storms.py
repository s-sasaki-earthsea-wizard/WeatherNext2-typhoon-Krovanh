"""Figures that put the storms side by side.

Every figure is drawn from the tables in :mod:`storms`. A storm keeps one
hue across every figure, in the order the storms are given; the hues are
the first three slots of a categorical palette validated for colour-vision
deficiency against the white background. The lightest of them is under 3:1
against white, so every curve is also labelled at its end and every figure
has a legend.

Curves are drawn only over the leads that every case of the storm reaches
(``n_cases`` in the table), so a curve never changes its sample halfway.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from wn2_typhoon.analysis.plot import INK, _finish
from wn2_typhoon.analysis.plot_compare import _label_line_ends
from wn2_typhoon.analysis.storms import StormResult

# Categorical slots 1-3 (blue, orange, aqua). Worst adjacent CVD separation
# Delta E 9.2 (deutan), normal-vision 27.6; aqua is 2.7:1 against white,
# hence the end labels.
STORM_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
LINE_WIDTH = 2.0
# Diverging blue-red with a grey midpoint (the dataviz reference pair). Red is
# "members less directed than the storm", the stall side.
DIRECTEDNESS_CMAP = LinearSegmentedColormap.from_list(
    "directedness", ["#a8302f", "#ee8c8b", "#f0efec", "#6da7ec", "#184f95"]
)
# log10 ratio at which the colour saturates: a factor of ten either way.
DIRECTEDNESS_LIMIT = 1.0
# How each kind of tropical end reads under a storm's name.
END_WORDING = {"depression": "weakened to a depression", "extratropical": "became extratropical"}


def _colour(index: int) -> str:
    if index >= len(STORM_COLOURS):
        raise ValueError(
            f"at most {len(STORM_COLOURS)} storms per figure; validate a longer "
            "palette before adding more"
        )
    return STORM_COLOURS[index]


def _full(table: pd.DataFrame, storm: StormResult) -> pd.DataFrame:
    """The storm's rows at the leads every one of its cases reaches."""
    rows = table.loc[table["storm"] == storm.label]
    return rows.loc[rows["n_cases"] == len(storm.cases)].sort_values("lead_hours")


def _lead_axis(axes, top: float) -> None:
    axes.set_xlabel("lead time [h]")
    axes.set_xticks(np.arange(0, top + 1, 24))
    axes.set_xlim(0, top + 12)


def _share_axis(axes, label: str) -> None:
    axes.set_ylabel(label)
    axes.set_ylim(-0.03, 1.05)
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def _tidy(*panels) -> None:
    for axes in panels:
        axes.grid(alpha=0.25)
        axes.spines[["top", "right"]].set_visible(False)


def plot_error_by_storm(
    table: pd.DataFrame,
    storms: list[StormResult],
    out_path: Path,
    within_km: float,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Mean member error and the share of members at the observed position.

    The plain view, in km, which the road-and-clock figure then takes apart.

    Args:
        table: Output of ``storms.road_and_clock``.
        storms: The storms, in colour order.
        out_path: PNG destination.
        within_km: Radius the counts were taken with.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.1},
    )
    top_ends, bottom_ends, last = [], [], 0
    for index, storm in enumerate(storms):
        rows = _full(table, storm)
        if rows.empty:
            continue
        style = {"color": _colour(index), "linewidth": LINE_WIDTH}
        share = rows["n_at_position"] / rows["n_present"]
        top.plot(rows["lead_hours"], rows["mean_error_km"], label=storm.label, **style)
        bottom.plot(rows["lead_hours"], share, **style)
        x = rows["lead_hours"].iloc[-1]
        top_ends.append((x, rows["mean_error_km"].iloc[-1], storm.name.title()))
        bottom_ends.append((x, share.iloc[-1], storm.name.title()))
        last = max(last, x)
    _label_line_ends(top, top_ends, min_gap=0.045 * top.get_ylim()[1])
    _label_line_ends(bottom, bottom_ends, min_gap=0.07)

    top.set_ylabel("mean member error [km]")
    top.set_ylim(bottom=0)
    top.legend(fontsize=8, loc="upper left", frameon=False)
    _share_axis(bottom, f"members within {within_km:.0f} km\nof the observed position")
    _lead_axis(bottom, last)
    _tidy(top, bottom)
    if title:
        top.set_title(title, fontsize=10, loc="left")
    return _finish(fig, out_path, credit)


def plot_road_and_clock(
    table: pd.DataFrame,
    storms: list[StormResult],
    out_path: Path,
    within_km: float,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Share of members on the observed path, and their clock along it.

    Top: members within ``within_km`` of the observed path at any time, over
    the members still tracking the storm and not beyond either end of the
    path. Bottom: the clock of those members, median and interquartile
    range, positive ahead of the storm. A storm's clock is drawn only over
    the leads at which every case's clock is readable (see
    ``path_offset``).

    Args:
        table: Output of ``storms.road_and_clock``.
        storms: The storms, in colour order.
        out_path: PNG destination.
        within_km: Radius the counts were taken with.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1},
    )
    top_ends, bottom_ends, last = [], [], 0
    for index, storm in enumerate(storms):
        rows = _full(table, storm)
        if rows.empty:
            continue
        colour = _colour(index)
        share = rows["n_on_path"] / (rows["n_present"] - rows["n_beyond"])
        top.plot(rows["lead_hours"], share, color=colour, linewidth=LINE_WIDTH,
                 label=storm.label)
        top_ends.append((rows["lead_hours"].iloc[-1], share.iloc[-1], storm.name.title()))
        last = max(last, rows["lead_hours"].iloc[-1])

        clocked = rows.loc[rows["n_clock_cases"] == len(storm.cases)]
        clocked = clocked.loc[clocked["n_clock"] > 0]
        if clocked.empty:
            continue
        bottom.fill_between(clocked["lead_hours"], clocked["clock_q25_hours"],
                            clocked["clock_q75_hours"], color=colour, alpha=0.15,
                            linewidth=0)
        bottom.plot(clocked["lead_hours"], clocked["clock_median_hours"], color=colour,
                    linewidth=LINE_WIDTH)
        bottom_ends.append((clocked["lead_hours"].iloc[-1],
                            clocked["clock_median_hours"].iloc[-1], storm.name.title()))
    _label_line_ends(top, top_ends, min_gap=0.07)
    _label_line_ends(bottom, bottom_ends, min_gap=0.06 * np.diff(bottom.get_ylim())[0])

    _share_axis(top, f"members within {within_km:.0f} km\nof the observed path")
    top.legend(fontsize=8, loc="lower left", frameon=False)
    bottom.axhline(0.0, color=INK["muted"], linewidth=0.8)
    bottom.set_ylabel("clock of the members on the path [h]\nmedian and IQR, + = ahead")
    _lead_axis(bottom, last)
    _tidy(top, bottom)
    if title:
        top.set_title(title, fontsize=10, loc="left")
    return _finish(fig, out_path, credit)


def plot_track_ends(
    ends: pd.DataFrame,
    storms: list[StormResult],
    out_path: Path,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """When each member's track ended, relative to the end of the tropical phase.

    One dot per member. Where the storm has a dissipation time, it is marked
    on the same scale, which is what the tracker's end should be read
    against: it follows the vortex, not the tropical storm.

    Args:
        ends: Output of ``storms.track_ends``.
        storms: The storms, in colour order.
        out_path: PNG destination.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, axes = plt.subplots(figsize=(8.5, 5.5))
    fig.subplots_adjust(bottom=0.16)
    rng = np.random.default_rng(0)
    for index, storm in enumerate(storms):
        rows = ends.loc[ends["storm"] == storm.label]
        jitter = rng.uniform(-0.25, 0.25, len(rows))
        axes.scatter(index + jitter, rows["after_tropical_end_hours"], s=22,
                     color=_colour(index), edgecolors="white", linewidths=0.6, zorder=3)
        if storm.dissipation_time is not None:
            offset = (storm.dissipation_time - pd.Timestamp(storm.end.time)) / pd.Timedelta(
                hours=1
            )
            axes.hlines(offset, index - 0.38, index + 0.38, color=INK["primary"],
                        linewidth=1.2, linestyle="--", zorder=4)
            axes.annotate("dissipation", (index + 0.38, offset), xytext=(3, 0),
                          textcoords="offset points", fontsize=7.5,
                          color=INK["secondary"], va="center")
    axes.axhline(0.0, color=INK["primary"], linewidth=0.9)
    axes.annotate("end of the tropical phase (JMA)", (0.01, 0.0),
                  xycoords=("axes fraction", "data"), xytext=(0, 3),
                  textcoords="offset points", fontsize=7.5, color=INK["secondary"],
                  va="bottom")
    axes.set_xticks(range(len(storms)))
    axes.set_xticklabels(
        [f"{storm.label}\n{END_WORDING.get(storm.end.kind, storm.end.kind)}"
         for storm in storms],
        fontsize=8.5,
    )
    axes.set_xlim(-0.6, len(storms) - 0.4)
    axes.set_ylabel("track end after the end of the tropical phase [h]")
    _tidy(axes)
    axes.grid(axis="x", visible=False)
    if title:
        axes.set_title(title, fontsize=10, loc="left")
    return _finish(fig, out_path, credit)


def plot_shape_and_clock(
    members: pd.DataFrame,
    storms: list[StormResult],
    out_path: Path,
    clock_lead: int = 72,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Each member on two axes: shape against the road, and clock along it.

    The same horizontal axis, the Chamfer distance to the observed path, in
    three panels that differ in how the clock is summarised: its median over
    the on-road positions, its value at one lead, and its trend with lead
    time. Which summary to keep is open; a member without a value in a panel
    is counted in that panel's title.

    Args:
        members: Output of ``storms.member_shape_and_clock``.
        storms: The storms, in colour order.
        out_path: PNG destination.
        clock_lead: Lead of the middle panel; ``clock_at_<lead>h`` must exist.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    panels = [
        ("clock_median_hours", "clock, median over on-road positions [h]"),
        (f"clock_at_{clock_lead}h", f"clock at lead {clock_lead} h [h]"),
        ("clock_trend", "clock trend [h per h]\n(along-road speed ratio - 1)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0), sharex=True)
    fig.subplots_adjust(wspace=0.32, bottom=0.17, top=0.82)
    for axis, (column, label) in zip(axes, panels, strict=True):
        missing = []
        for index, storm in enumerate(storms):
            rows = members.loc[members["storm"] == storm.label]
            shown = rows.dropna(subset=[column, "chamfer_km"])
            axis.scatter(shown["chamfer_km"], shown[column], s=26, color=_colour(index),
                         edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3,
                         label=storm.label)
            missing.append(f"{storm.name.title()} {len(rows) - len(shown)}")
        axis.axhline(0.0, color=INK["muted"], linewidth=0.8)
        axis.set_ylabel(label)
        axis.set_xlabel("shape: Chamfer distance to the observed path [km]")
        axis.set_title("no value: " + ", ".join(missing), fontsize=8, loc="left",
                       color=INK["secondary"])
        _tidy(axis)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False,
               loc="upper right", ncol=len(storms), bbox_to_anchor=(0.99, 0.95))
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return _finish(fig, out_path, credit)


def plot_directedness(
    table: pd.DataFrame,
    lags: pd.DataFrame,
    storms: list[StormResult],
    out_path: Path,
    title: str | None = None,
    credit: str | None = None,
) -> Path:
    """Scalogram of the members' directedness against the storm's.

    One panel per storm: valid time across, window length up, colour the
    median over members of log10(Pe_member / Pe_observed). Red is members
    less directed than the storm at that moment, blue more. Each window
    length is drawn at its own window centres, which fall between the
    six-hourly rows when the window has an odd number of steps.

    Args:
        table: Output of ``directedness.directedness_by_time``.
        lags: Output of ``directedness.directedness_lag``.
        storms: The storms, one panel each.
        out_path: PNG destination.
        title: Figure title.
        credit: Attribution line, or empty string to suppress it.

    Returns:
        ``out_path``.
    """
    fig, axes = plt.subplots(len(storms), 1, figsize=(10.0, 2.3 * len(storms) + 0.8))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(hspace=0.75, right=0.86, top=0.9, bottom=0.08)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-DIRECTEDNESS_LIMIT, vmax=DIRECTEDNESS_LIMIT)
    image = None
    for axis, storm in zip(axes, storms, strict=True):
        rows = table.loc[table["storm"] == storm.label]
        windows = sorted(rows["window_hours"].unique())
        for hours in windows:
            band = rows.loc[rows["window_hours"] == hours].sort_values("centre")
            centres = mdates.date2num(pd.to_datetime(band["centre"]).dt.to_pydatetime())
            half = 3.0 / 24.0
            edges = np.append(centres - half, centres[-1] + half)
            values = band["log_ratio_median"].to_numpy()[None, :]
            image = axis.pcolormesh(edges, [hours - 3, hours + 3], values,
                                    cmap=DIRECTEDNESS_CMAP, norm=norm, shading="flat")
        best = lags.loc[(lags["storm"] == storm.label) & lags["best"]]
        best = best.drop_duplicates("window_hours").sort_values("window_hours")
        summary = (
            f"best lag of the members [h] at {'/'.join(str(h) for h in best.window_hours)} h "
            f"windows: {' / '.join(f'{int(lag):+d}' for lag in best.lag_hours)}"
            if len(best) else "too few windows for a lag"
        )
        axis.set_title(f"{storm.label}    {summary}", fontsize=8.5, loc="left")
        axis.set_yticks(windows)
        axis.set_ylabel("window [h]")
        axis.xaxis_date()
        axis.xaxis.set_major_locator(mdates.DayLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if image is not None:
        bar = fig.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02)
        bar.set_label("median log10(Pe member / Pe observed)\nred: members less directed")
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return _finish(fig, out_path, credit)
