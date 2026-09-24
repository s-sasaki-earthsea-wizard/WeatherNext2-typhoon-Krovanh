"""Tests for the map layout helpers behind the member track figures.

The maps themselves are not rendered: their Natural Earth or OpenStreetMap
background needs the network on a cold cache. The axes here are bare cartopy
GeoAxes with gridlines only, which draw offline, set up the way
``plot.map_axes`` sets up its own.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.analysis.plot import (
    _daily,
    _grid_figure,
    _label_dates,
    _member_caption,
    _panel_title,
)

ccrs = pytest.importorskip("cartopy.crs")

EXTENT = [113.6, 142.7, 18.6, 43.4]
MARGINS = {"left": 0.07, "right": 0.98, "top": 0.93, "bottom": 0.04,
           "wspace": 0.08, "hspace": 0.16}


def _geo_axes(fig, subplot=(1, 1, 1), extent=(125.0, 135.0, 20.0, 30.0),
              projection=None):
    """A map axes with left and bottom gridline labels, and no background."""
    axes = fig.add_subplot(*subplot, projection=projection or ccrs.PlateCarree())
    axes.set_extent(list(extent), crs=ccrs.PlateCarree())
    gridlines = axes.gridlines(draw_labels=True)
    gridlines.top_labels = gridlines.right_labels = False
    return axes


def test_panel_title_is_drawn_when_the_top_labels_are_off() -> None:
    """Left to itself, matplotlib puts this title at y=inf and drops it."""
    fig = plt.figure(figsize=(6, 6))
    axes = _geo_axes(fig)
    _panel_title(axes, "member 0")
    fig.canvas.draw()
    title = axes.title.get_window_extent()
    assert np.isfinite(title.y0)
    assert title.y0 >= axes.get_window_extent().y1
    plt.close(fig)


@pytest.mark.parametrize(
    "basemap, projection",
    [("natural-earth", ccrs.PlateCarree()), ("osm", ccrs.GOOGLE_MERCATOR)],
)
def test_grid_figure_lets_each_map_fill_its_box(basemap, projection) -> None:
    fig, _ = _grid_figure(EXTENT, basemap, rows=4, columns=2, width=9.0, margins=MARGINS)
    axes = _geo_axes(fig, (4, 2, 1), EXTENT, projection)
    box = axes.get_position(original=True)
    axes.apply_aspect()
    active = axes.get_position()
    assert active.width == pytest.approx(box.width, rel=0.01)
    assert active.height == pytest.approx(box.height, rel=0.01)
    plt.close(fig)


def test_date_labels_keep_off_each_other_and_the_dots() -> None:
    """Two tracks whose daily dots crowd together, as over the observed loop.

    Labelled each on its preferred side, three pairs of these labels overlap;
    much denser and no set of spots around the dots would be free.
    """
    times = pd.date_range("2026-09-01", periods=6, freq="1D")
    steps = np.arange(len(times))
    member = pd.DataFrame(
        {"valid_time": times, "lat": 25.0 + 0.6 * steps, "lon": 130.0 + 0.3 * steps}
    )
    best = pd.DataFrame(
        {"time": times, "lat": 25.1 + 0.6 * steps, "lon": 130.5 + 0.2 * steps}
    )
    fig = plt.figure(figsize=(6, 6))
    axes = _geo_axes(fig)
    _label_dates(
        axes,
        [(member, "valid_time", "right", "#1f5fbf"), (best, "time", "left", "#111111")],
        size=6.5,
    )
    fig.canvas.draw()

    labels = [text for text in axes.texts if text.get_text()]
    assert sorted(label.get_text() for label in labels) == sorted(
        [f"{when:%m-%d}" for when in times] * 2
    )
    boxes = [label.get_window_extent() for label in labels]
    assert not any(
        boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i)
    )
    to_display = ccrs.PlateCarree()._as_mpl_transform(axes)
    for frame in (member, best):
        for x, y in to_display.transform(frame[["lon", "lat"]].to_numpy()):
            assert not any(box.contains(x, y) for box in boxes)
    plt.close(fig)


def test_daily_keeps_the_00z_rows_only() -> None:
    track = pd.DataFrame(
        {"valid_time": pd.date_range("2026-09-01 12:00", periods=6, freq="6h")}
    )
    assert list(_daily(track, "valid_time")["valid_time"]) == [
        pd.Timestamp("2026-09-02 00:00")
    ]


def test_member_caption_names_the_end_or_the_miss() -> None:
    tracks = pd.DataFrame(
        {
            "member": [0, 0],
            "valid_time": pd.to_datetime(["2026-09-05 18:00", "2026-09-06 00:00"]),
        }
    )
    assert _member_caption(tracks, 0) == "member 0, track ends 09-06 00Z"
    assert _member_caption(tracks, 3) == "member 3, no matching storm"
