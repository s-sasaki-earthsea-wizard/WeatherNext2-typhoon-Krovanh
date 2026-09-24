"""Figures: track maps (the ensemble, and member by member), error against lead time,
and central pressure.

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

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wn2_typhoon.analysis.track_error import TRACK_TIME
from wn2_typhoon.utils.geo import mean_position

MEMBER_STYLE = {"color": "#7f7f7f", "linewidth": 0.9, "alpha": 0.75, "zorder": 2}
ENSEMBLE_STYLE = {"color": "#d62728", "linewidth": 2.2, "zorder": 4}
OBSERVED_STYLE = {"color": "#111111", "linewidth": 2.2, "zorder": 5}
# The grey members fade into a tile map's own colours; darken them there.
MEMBER_STYLE_ON_TILES = {**MEMBER_STYLE, "color": "#4d4d4d", "alpha": 0.85}
# One member drawn on its own is the subject of its map, so it gets a hue.
# Validated against the ensemble-mean red and against the Natural Earth and
# OpenStreetMap sea and land colours; it sits above the mean, below the
# reference.
MEMBER_FOCUS_STYLE = {"color": "#1f5fbf", "linewidth": 1.8, "zorder": 4.5}
DAILY_MARKER_SIZE = 3.5
FORMATION_MARKER_SIZE = 11
DAILY_LABEL = "00Z positions"

# Map backgrounds. Natural Earth needs no network once cartopy has cached its
# shapefiles and is the cleaner rendering for a paper figure. The OpenStreetMap
# tiles put named islands and coastlines under the tracks, which helps a reader
# who does not know the Nansei chain by its shape; they need the network on the
# first draw and are cached under data/ after that. CARTO's light tiles were
# tried first and are key-gated now: without an API key every tile comes back
# watermarked "API KEY REQUIRED".
BASEMAPS = ("natural-earth", "osm")
BASEMAP_CREDIT = {
    "natural-earth": "",
    "osm": "basemap (c) OpenStreetMap contributors",
}
# OSM's tile usage policy asks for a user agent that identifies the
# application; cartopy's default, "CartoPy/<version>", is not one.
TILE_USER_AGENT = (
    "wn2-typhoon-krovanh/0.1 "
    "(+https://github.com/s-sasaki-earthsea-wizard/WeatherNext2-typhoon-Krovanh)"
)
# cartopy's own default is a directory under the system temp dir, so left
# alone every run would download the tiles again.
TILE_CACHE_DIR = Path("data/cache/tiles")
TILE_ZOOM_RANGE = (3, 10)

# One hue, light to dark in initialization order, for figures that put the
# cases side by side. Validated as an ordinal ramp: lightness monotone, every
# step at least 0.06 apart, light end 2.4:1 against white. The markers are the
# second encoding, so identity never rests on colour alone.
CASE_RAMP = ("#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
CASE_MARKERS = ("o", "s", "^", "D", "v")
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}


def _check_basemap(basemap: str) -> str:
    """Return ``basemap`` if it is a known background, else raise.

    Raises:
        ValueError: If ``basemap`` is not in :data:`BASEMAPS`.
    """
    if basemap not in BASEMAPS:
        raise ValueError(f"basemap must be one of {BASEMAPS}, got {basemap!r}")
    return basemap


def credit_line(source: str, basemap: str = "natural-earth") -> str:
    """Return the attribution required on a published figure.

    Args:
        source: Value of the reference track's ``source`` column, either
            "preliminary" or "post-analysis".
        basemap: Map background of the figure, one of :data:`BASEMAPS`. The
            OpenStreetMap tiles carry their own attribution requirement.

    Returns:
        A one-line credit naming the model, the initial conditions, the
        reference track and, when tiles were drawn, their provider. The
        reference's release is stated so a preliminary figure is never
        mistaken for a final one.
    """
    release = {
        "preliminary": "JMA preliminary position table",
        "post-analysis": "JMA post-analysis position table",
    }.get(source, f"JMA position table ({source})")
    line = (
        "WeatherNext 2 (Google DeepMind, CC BY 4.0) "
        "| initial conditions ERA5 (Copernicus) "
        f"| reference {release}"
    )
    extra = BASEMAP_CREDIT[_check_basemap(basemap)]
    return f"{line} | {extra}" if extra else line


def tile_zoom(extent, width_px: float) -> int:
    """Pick the web-map zoom that gives about one tile pixel per figure pixel.

    A tile is 256 px wide and at zoom ``z`` the world is ``2**z`` tiles across,
    so the zoom that matches the figure is the smallest one whose tiles are at
    least as fine as the pixels they are drawn on. One level coarser leaves the
    map soft; one finer downloads four times the tiles for nothing.

    Args:
        extent: ``[lon_min, lon_max, lat_min, lat_max]`` in degrees.
        width_px: Width of the map axes in pixels.

    Returns:
        The zoom level, clamped to :data:`TILE_ZOOM_RANGE`.
    """
    span = float(extent[1]) - float(extent[0])
    zoom = math.ceil(math.log2(360.0 / span * width_px / 256.0))
    low, high = TILE_ZOOM_RANGE
    return int(min(max(zoom, low), high))


def track_extent(
    forecast_tracks: pd.DataFrame, best_track: pd.DataFrame, margin_deg: float
) -> list[float]:
    """Bounding box of the drawn tracks, padded.

    Args:
        forecast_tracks: Forecast tracks with ``lat`` and ``lon`` columns.
        best_track: Reference track with ``lat`` and ``lon`` columns.
        margin_deg: Padding on every side, in degrees.

    Returns:
        ``[lon_min, lon_max, lat_min, lat_max]``.
    """
    lats = np.concatenate([forecast_tracks["lat"].to_numpy(), best_track["lat"].to_numpy()])
    lons = np.concatenate([forecast_tracks["lon"].to_numpy(), best_track["lon"].to_numpy()])
    return [
        float(lons.min()) - margin_deg,
        float(lons.max()) + margin_deg,
        float(lats.min()) - margin_deg,
        float(lats.max()) + margin_deg,
    ]


def map_axes(
    fig,
    subplot: tuple[int, int, int],
    extent,
    basemap: str = "natural-earth",
    width_px: float = 1360.0,
    label_sides: tuple[str, ...] = ("left", "bottom"),
):
    """Add a map axes to ``fig`` with the chosen background drawn.

    Everything plotted onto the returned axes must pass
    ``transform=ccrs.PlateCarree()``: the tile background lives in Web
    Mercator and the Natural Earth one in plate carree, and the transform is
    what lets the same plotting code serve both.

    Args:
        fig: Figure to add the axes to.
        subplot: ``(rows, columns, index)`` as for ``fig.add_subplot``.
        extent: ``[lon_min, lon_max, lat_min, lat_max]`` in degrees.
        basemap: One of :data:`BASEMAPS`.
        width_px: Width the axes will be rendered at, used to choose the tile
            zoom. Ignored for Natural Earth.
        label_sides: Which of "left" and "bottom" get gridline labels, so a
            grid of small maps can label only its outer edge.

    Returns:
        A cartopy ``GeoAxes``.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if _check_basemap(basemap) == "osm":
        from cartopy.io.img_tiles import OSM

        tiler = OSM(cache=str(TILE_CACHE_DIR), user_agent=TILE_USER_AGENT)
        axes = fig.add_subplot(*subplot, projection=tiler.crs)
        axes.set_extent(extent, crs=ccrs.PlateCarree())
        axes.add_image(tiler, tile_zoom(extent, width_px), interpolation="spline36")
        grid_colour = "#666666"
    else:
        axes = fig.add_subplot(*subplot, projection=ccrs.PlateCarree())
        axes.set_extent(extent, crs=ccrs.PlateCarree())
        axes.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f0eb", zorder=0)
        axes.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eaf1f7", zorder=0)
        axes.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            linewidth=0.6, edgecolor="#888888", zorder=1,
        )
        grid_colour = "#cccccc"

    gridlines = axes.gridlines(
        draw_labels=True, linewidth=0.3, color=grid_colour, alpha=0.6,
        xlabel_style={"size": 8}, ylabel_style={"size": 8},
    )
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.left_labels = "left" in label_sides
    gridlines.bottom_labels = "bottom" in label_sides
    return axes


def _panel_title(axes, text: str, fontsize: float = 9) -> None:
    """Title a map panel just above its frame.

    Matplotlib lifts an axes title clear of the axes' top decorations. With
    cartopy's top gridline labels switched off, as :func:`map_axes` does,
    that height comes back infinite and the title is put at y=inf, which
    means it is silently not drawn (matplotlib 3.11, cartopy 0.25). Giving
    ``y`` turns the automatic placement off; nothing sits above the frame
    to clear anyway.

    Args:
        axes: Map axes from :func:`map_axes`.
        text: Title text.
        fontsize: Font size.
    """
    axes.set_title(text, fontsize=fontsize, y=1.0)


def _extent_aspect(extent, basemap: str) -> float:
    """Height over width of a map drawn on ``extent`` in the basemap's projection."""
    lon_span = math.radians(float(extent[1]) - float(extent[0]))
    if basemap == "osm":
        # Web Mercator stretches latitude; the tiles are drawn in it.
        def northing(lat):
            return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        return (northing(extent[3]) - northing(extent[2])) / lon_span
    return math.radians(float(extent[3]) - float(extent[2])) / lon_span


def _grid_figure(
    extent, basemap: str, rows: int, columns: int, width: float, margins: dict
):
    """A figure whose height fits a grid of maps that all show ``extent``.

    A GeoAxes keeps the aspect of its window, so on a figure of arbitrary
    height the maps shrink inside their subplot boxes and float in
    whitespace. Deriving the height from the window's shape avoids that.

    Args:
        extent: ``[lon_min, lon_max, lat_min, lat_max]`` shared by the panels.
        basemap: One of :data:`BASEMAPS`; the tiles are in Web Mercator.
        rows: Number of panel rows.
        columns: Number of panel columns.
        width: Figure width in inches.
        margins: Keyword arguments for ``fig.subplots_adjust``; must include
            ``left``, ``right``, ``top``, ``bottom``, ``wspace`` and ``hspace``.

    Returns:
        The figure, with ``margins`` applied, and one panel's width in inches.
    """
    panel_width = width * (margins["right"] - margins["left"]) / (
        columns + margins["wspace"] * (columns - 1)
    )
    panel_height = panel_width * _extent_aspect(extent, basemap)
    height = panel_height * (rows + margins["hspace"] * (rows - 1)) / (
        margins["top"] - margins["bottom"]
    )
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(**margins)
    return fig, panel_width


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
    basemap: str = "natural-earth",
) -> Path:
    """Draw the ensemble tracks over the reference track on a map.

    Args:
        forecast_tracks: One track per member, from ``track_error.select_storm``.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        out_path: PNG destination.
        title: Figure title.
        margin_deg: Padding around the drawn tracks, in degrees.
        credit: Attribution line; built from the reference's ``source`` column
            and ``basemap`` when omitted. Pass an empty string to suppress it.
        basemap: Map background, one of :data:`BASEMAPS`.

    Returns:
        ``out_path``.
    """
    import cartopy.crs as ccrs

    if credit is None:
        credit = credit_line(str(best_track["source"].iloc[0]), basemap)

    extent = track_extent(forecast_tracks, best_track, margin_deg)
    figsize, dpi = (8.5, 7.5), 160
    fig = plt.figure(figsize=figsize)
    axes = map_axes(fig, (1, 1, 1), extent, basemap, width_px=figsize[0] * dpi)
    member_style = MEMBER_STYLE_ON_TILES if basemap == "osm" else MEMBER_STYLE

    for member, group in forecast_tracks.groupby("member", sort=True):
        group = group.sort_values("valid_time")
        axes.plot(
            group["lon"], group["lat"],
            transform=ccrs.PlateCarree(),
            label="ensemble members" if member == forecast_tracks["member"].min() else None,
            **member_style,
        )
    _draw_ensemble_mean(axes, forecast_tracks)
    _draw_reference(axes, best_track)

    # A GeoAxes keeps a fixed aspect, so it shrinks inside its subplot box and
    # an axes title floats far above the map. Put it on the figure instead.
    if title:
        fig.suptitle(title, y=0.92)
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return _finish(fig, out_path, credit)


def plot_member_track(
    forecast_tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    member: int,
    out_path: Path,
    title: str | None = None,
    extent=None,
    margin_deg: float = 3.0,
    credit: str | None = None,
    basemap: str = "natural-earth",
) -> Path:
    """Draw one member's track over the ensemble mean and the reference.

    The window is fitted to every member, not to the one drawn, so the
    figures of one case share it with each other and with
    :func:`plot_tracks` and can be flipped through without the map moving.

    Args:
        forecast_tracks: Every member's track, from
            ``track_error.select_storm``. The ensemble mean and the window
            come from all of them.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        member: The member to draw. One with no selected track gets a map
            with the reference and the mean only, and says so.
        out_path: PNG destination.
        title: Figure title.
        extent: ``[lon_min, lon_max, lat_min, lat_max]``; fitted to all
            tracks plus ``margin_deg`` when None.
        margin_deg: Padding used when ``extent`` is None.
        credit: Attribution line; built from the reference's ``source`` column
            and ``basemap`` when omitted. Pass an empty string to suppress it.
        basemap: Map background, one of :data:`BASEMAPS`.

    Returns:
        ``out_path``.
    """
    if credit is None:
        credit = credit_line(str(best_track["source"].iloc[0]), basemap)
    if extent is None:
        extent = track_extent(forecast_tracks, best_track, margin_deg)

    figsize, dpi = (8.5, 7.5), 160
    fig = plt.figure(figsize=figsize)
    axes = map_axes(fig, (1, 1, 1), extent, basemap, width_px=figsize[0] * dpi)
    _draw_member(axes, forecast_tracks, best_track, member,
                 label=_member_caption(forecast_tracks, member), dated=True)
    if title:
        fig.suptitle(title, y=0.92)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles + [_daily_legend_handle()], labels + [f"{DAILY_LABEL}, dated MM-DD"],
                loc="upper left", fontsize=8, framealpha=0.9)
    return _finish(fig, out_path, credit)


def plot_member_grid(
    forecast_tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    out_path: Path,
    members=None,
    columns: int = 4,
    title: str | None = None,
    extent=None,
    margin_deg: float = 3.0,
    credit: str | None = None,
    basemap: str = "natural-earth",
) -> Path:
    """One map per member, laid out in a grid on one shared window.

    The small-multiple counterpart of :func:`plot_tracks`: each panel is
    what :func:`plot_member_track` draws, so a member can be read on its own
    and still be compared with the others at a glance. The 00Z dots are not
    dated here; at this size the labels bury the tracks, and the member's
    own map has them.

    Args:
        forecast_tracks: Every member's track, from ``track_error.select_storm``.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
        out_path: PNG destination.
        members: Members in panel order, filled row by row. Pass every member
            of the ensemble, including those with no selected track, so the
            panels keep their positions; defaults to the members that have one.
        columns: Panels per row; the rows follow from the member count.
        title: Figure title.
        extent: ``[lon_min, lon_max, lat_min, lat_max]``; fitted to all
            tracks plus ``margin_deg`` when None.
        margin_deg: Padding used when ``extent`` is None.
        credit: Attribution line; built from the reference's ``source`` column
            and ``basemap`` when omitted. Pass an empty string to suppress it.
        basemap: Map background, one of :data:`BASEMAPS`.

    Returns:
        ``out_path``.
    """
    if credit is None:
        credit = credit_line(str(best_track["source"].iloc[0]), basemap)
    if extent is None:
        extent = track_extent(forecast_tracks, best_track, margin_deg)
    if members is None:
        members = sorted(forecast_tracks["member"].unique())
    rows = math.ceil(len(members) / columns)

    width, dpi = 16.0, 160
    margins = {"left": 0.04, "right": 0.99, "top": 0.87, "bottom": 0.07,
               "wspace": 0.06, "hspace": 0.14}
    fig, panel_width = _grid_figure(extent, basemap, rows, columns, width, margins)

    legend = {}
    for index, member in enumerate(members):
        row, column = divmod(index, columns)
        sides = tuple(
            side for side, on in (("left", column == 0), ("bottom", row == rows - 1)) if on
        )
        axes = map_axes(fig, (rows, columns, index + 1), extent, basemap,
                        width_px=panel_width * dpi, label_sides=sides)
        _draw_member(axes, forecast_tracks, best_track, member,
                     label="member track", dated=False)
        _panel_title(axes, _member_caption(forecast_tracks, member))
        # A panel without a track has no member handle, so gather from all.
        for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
            legend.setdefault(label, handle)
    legend[DAILY_LABEL] = _daily_legend_handle()

    fig.legend(legend.values(), legend.keys(), loc="upper center", ncol=len(legend),
               bbox_to_anchor=(0.5, 0.945), fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, y=0.99)
    return _finish(fig, out_path, credit)


def _draw_ensemble_mean(axes, forecast_tracks: pd.DataFrame) -> None:
    """Draw the ensemble-mean track while every member is still present.

    Past that point the mean is taken over a shrinking, diverging sample and
    wanders off along a path no member took: on the formation case it ran
    northwest to 30N 123E after the members began to end, which is not a
    forecast of anything. A tick marks where the first member ends.

    Args:
        axes: Map axes from :func:`map_axes`.
        forecast_tracks: Every member's track, not only the ones on show.
    """
    import cartopy.crs as ccrs

    mean_track = _ensemble_mean_track(forecast_tracks)
    complete = mean_track.loc[mean_track["n_members"] == mean_track["n_members"].max()]
    if complete.empty:
        return
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


def _draw_reference(axes, best_track: pd.DataFrame) -> None:
    """Draw the reference track and circle its first position, the formation.

    Args:
        axes: Map axes from :func:`map_axes`.
        best_track: Reference track from ``data.jma_besttrack.load_track``.
    """
    import cartopy.crs as ccrs

    best = best_track.sort_values("time")
    axes.plot(
        best["lon"], best["lat"],
        transform=ccrs.PlateCarree(), label="JMA reference", **OBSERVED_STYLE,
    )
    axes.plot(
        best["lon"].iloc[0], best["lat"].iloc[0],
        marker="o", markersize=FORMATION_MARKER_SIZE, markerfacecolor="none",
        markeredgecolor="#111111", markeredgewidth=2.0,
        transform=ccrs.PlateCarree(), zorder=6, linestyle="none",
        label="observed formation",
    )


def _daily(track: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """The rows of ``track`` at 00Z."""
    times = pd.to_datetime(track[time_column])
    return track.loc[(times.dt.hour == 0) & (times.dt.minute == 0)]


def _draw_daily_dots(axes, daily: pd.DataFrame, style: dict) -> None:
    """Dot a track's 00Z positions in its own colour, just above its line.

    With one member on a map the question is when as well as where: a member
    on the observed path but a day late is an error the line alone hides.
    Dots at the same hours on both tracks show it.

    Args:
        axes: Map axes from :func:`map_axes`.
        daily: The track's 00Z rows, from :func:`_daily`.
        style: The track's line style.
    """
    import cartopy.crs as ccrs

    if daily.empty:
        return
    axes.plot(
        daily["lon"], daily["lat"], transform=ccrs.PlateCarree(),
        marker="o", markersize=DAILY_MARKER_SIZE, color=style["color"],
        markeredgecolor="white", markeredgewidth=0.5,
        linestyle="none", zorder=style["zorder"] + 0.1,
    )


def _label_dates(
    axes,
    dated: list[tuple[pd.DataFrame, str, str, str]],
    size: float,
    rings: list[tuple[float, float, float]] = (),
) -> None:
    """Put an ``MM-DD`` label by every 00Z dot, keeping labels apart.

    The observed storm crawled for its first two days and looped later, so
    its daily positions sit close together, and a member that follows it puts
    its own beside them. Each label therefore tries eight spots around its
    dot, its preferred side first and the opposite side last, and takes the
    first that covers neither a marker nor a label already placed. When none
    is free it takes the one that overlaps least. Track lines are not
    avoided; the white outline keeps a label legible where one runs under it.

    A label takes its track's colour rather than an ink colour. Where the
    member runs along the observed path, a date sits between a blue dot and
    a black one, and its colour is what says which it belongs to. The
    member blue is 6:1 against the outline.

    Args:
        axes: Map axes from :func:`map_axes`, with the dots already drawn.
        dated: ``(daily rows, time column, preferred side, colour)`` per
            track, in the order their labels get first pick; side is
            ``"left"`` or ``"right"``.
        size: Font size.
        rings: ``(lon, lat, radius in points)`` of markers larger than a dot,
            such as the formation circle. Labels keep off them, and a dot
            inside one is labelled from outside its edge.
    """
    import cartopy.crs as ccrs
    from matplotlib import patheffects
    from matplotlib.transforms import Bbox

    renderer = axes.figure.canvas.get_renderer()
    # A GeoAxes settles its fixed-aspect position at draw time; the label
    # boxes are measured in display space, so settle it now.
    axes.apply_aspect()
    to_axes = ccrs.PlateCarree()._as_mpl_transform(axes)

    def display(lon, lat):
        return to_axes.transform([[float(lon), float(lat)]])[0]

    def square(xy, radius_pt):
        r = renderer.points_to_pixels(radius_pt)
        return Bbox([[xy[0] - r, xy[1] - r], [xy[0] + r, xy[1] + r]])

    dot_radius = DAILY_MARKER_SIZE / 2 + 0.5
    rings = [(display(lon, lat), radius) for lon, lat, radius in rings]
    occupied = [square(xy, radius) for xy, radius in rings]
    for daily, *_ in dated:
        occupied += [square(display(lon, lat), dot_radius)
                     for lon, lat in zip(daily["lon"], daily["lat"], strict=True)]

    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    gap = renderer.points_to_pixels(1.0)

    def place(point, time_column: str, colour: str, spot: str, clearance: float):
        across, up = clearance + 2, clearance + 1
        offset, ha, va = {
            "right": ((across, 0), "left", "center"),
            "left": ((-across, 0), "right", "center"),
            "above": ((0, up), "center", "bottom"),
            "below": ((0, -up), "center", "top"),
            "above right": ((across, up), "left", "bottom"),
            "below right": ((across, -up), "left", "top"),
            "above left": ((-across, up), "right", "bottom"),
            "below left": ((-across, -up), "right", "top"),
        }[spot]
        label = axes.annotate(
            f"{pd.Timestamp(point[time_column]):%m-%d}",
            (point["lon"], point["lat"]), xycoords=to_axes,
            xytext=offset, textcoords="offset points",
            fontsize=size, color=colour, ha=ha, va=va,
            path_effects=halo, zorder=7,
        )
        # Padded by the outline's width, so labels that only touch count.
        return label, label.get_window_extent(renderer).padded(gap)

    def overlap(box) -> float:
        total = 0.0
        for other in occupied:
            both = Bbox.intersection(box, other)
            if both is not None:
                total += both.width * both.height
        return total

    for daily, time_column, side, colour in dated:
        other = "left" if side == "right" else "right"
        order = (side, f"above {side}", f"below {side}", "above", "below",
                 f"above {other}", f"below {other}", other)
        for _, point in daily.iterrows():
            xy = display(point["lon"], point["lat"])
            clearance = max(
                [dot_radius] + [radius for centre, radius in rings
                                if square(centre, radius).contains(*xy)]
            )
            tried = []
            for spot in order:
                label, box = place(point, time_column, colour, spot, clearance)
                if not any(box.overlaps(other) for other in occupied):
                    break
                label.remove()
                tried.append((overlap(box), spot))
            else:
                _, box = place(point, time_column, colour, min(tried)[1], clearance)
            occupied.append(box)


def _draw_member(
    axes,
    forecast_tracks: pd.DataFrame,
    best_track: pd.DataFrame,
    member: int,
    label: str,
    dated: bool,
) -> None:
    """Draw one member over the ensemble mean and the reference, dotted at 00Z.

    Args:
        axes: Map axes from :func:`map_axes`.
        forecast_tracks: Every member's track; the mean is taken over them all.
        best_track: Reference track.
        member: The member to draw.
        label: Legend label of the member's line.
        dated: Whether to label the 00Z dots with their dates.
    """
    import cartopy.crs as ccrs

    track = forecast_tracks.loc[forecast_tracks["member"] == member]
    track = track.sort_values(TRACK_TIME)
    best = best_track.sort_values("time")
    # Drawn first so its legend entry leads.
    if not track.empty:
        axes.plot(track["lon"], track["lat"], transform=ccrs.PlateCarree(),
                  label=label, **MEMBER_FOCUS_STYLE)
    else:
        axes.text(0.5, 0.5, "no matching storm", transform=axes.transAxes,
                  ha="center", va="center", fontsize=9, color=INK["secondary"])
    _draw_ensemble_mean(axes, forecast_tracks)
    _draw_reference(axes, best)

    member_daily, best_daily = _daily(track, TRACK_TIME), _daily(best, "time")
    _draw_daily_dots(axes, member_daily, MEMBER_FOCUS_STYLE)
    _draw_daily_dots(axes, best_daily, OBSERVED_STYLE)
    if dated:
        formation = (best["lon"].iloc[0], best["lat"].iloc[0], FORMATION_MARKER_SIZE / 2 + 1)
        _label_dates(axes, [(member_daily, TRACK_TIME, "right", MEMBER_FOCUS_STYLE["color"]),
                            (best_daily, "time", "left", OBSERVED_STYLE["color"])],
                     size=6.5, rings=[formation])


def _member_caption(forecast_tracks: pd.DataFrame, member: int) -> str:
    """Name a member and say when its track ends, or that it has none."""
    times = forecast_tracks.loc[forecast_tracks["member"] == member, TRACK_TIME]
    if times.empty:
        return f"member {member}, no matching storm"
    return f"member {member}, track ends {pd.Timestamp(times.max()):%m-%d %HZ}"


def _daily_legend_handle():
    """Legend entry for the 00Z dots, which are drawn in each track's colour."""
    from matplotlib.lines import Line2D

    return Line2D([], [], marker="o", markersize=DAILY_MARKER_SIZE, linestyle="none",
                  color=INK["secondary"], markeredgecolor="white", markeredgewidth=0.5)


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
