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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wn2_typhoon.analysis.plot import INK, _finish
from wn2_typhoon.analysis.plot_compare import _label_line_ends
from wn2_typhoon.analysis.storms import StormResult

# Categorical slots 1-3 (blue, orange, aqua). Worst adjacent CVD separation
# Delta E 9.2 (deutan), normal-vision 27.6; aqua is 2.7:1 against white,
# hence the end labels.
STORM_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
LINE_WIDTH = 2.0
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
