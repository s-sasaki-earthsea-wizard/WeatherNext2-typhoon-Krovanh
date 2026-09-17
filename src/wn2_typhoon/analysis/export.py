"""GeoJSON export of the selected tracks, for QGIS or any other GIS.

The figures are fixed views. A GIS lets the reader pan, switch the basemap and
query a point, and ``analysis/track.csv`` already holds the centres as rows,
but loading that means Points-to-Path by member on every visit. The lines are
written ready-made instead, with the reference track beside them.

Lines and points go to separate files. GeoJSON allows mixed geometry in one
collection, but QGIS then splits it into sub-layers and asks which to load,
which is one prompt too many for a file meant to be dropped onto a map.

Coordinates follow the GeoJSON convention: ``[longitude, latitude]`` in WGS 84
with longitude in [-180, 180]. The tracker's longitudes run [0, 360), so they
are wrapped on the way out. Timestamps are UTC ISO 8601 strings, which QGIS
reads as date-time fields. Missing values are written as ``null``: a bare
``NaN`` is not JSON, and QGIS refuses the whole file over one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wn2_typhoon.analysis.track_error import TRACK_PRESSURE, TRACK_TIME, TRACK_WIND


def wrap_longitude(lon: float) -> float:
    """Map a longitude in degrees east onto [-180, 180).

    Args:
        lon: Longitude in degrees, any range.

    Returns:
        The same meridian expressed in [-180, 180).
    """
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _stamp(value) -> str | None:
    """Format a timestamp as UTC ISO 8601, or None when missing."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(value) -> float | None:
    """Return ``value`` as a float, or None when missing."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _feature(geometry_type: str, coordinates, properties: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": geometry_type, "coordinates": coordinates},
        "properties": properties,
    }


def forecast_features(
    forecast_tracks: pd.DataFrame, case_id: str, init_time
) -> tuple[list[dict], list[dict]]:
    """Turn one case's selected tracks into GeoJSON features.

    Args:
        forecast_tracks: One track per member, from
            ``track_error.select_storm``.
        case_id: Case identifier, written into every feature so that several
            cases can share one file.
        init_time: Initialization time of the case, for the lead hours.

    Returns:
        ``(lines, points)``: one LineString per member and one Point per row.
    """
    start = pd.Timestamp(str(init_time))
    tracks = forecast_tracks.copy()
    tracks[TRACK_TIME] = pd.to_datetime(tracks[TRACK_TIME])
    lines, points = [], []
    for member, group in tracks.groupby("member", sort=True):
        group = group.sort_values(TRACK_TIME)
        coordinates = [
            [wrap_longitude(lon), float(lat)]
            for lat, lon in zip(group["lat"], group["lon"], strict=True)
        ]
        common = {
            "kind": "forecast",
            "case": case_id,
            "init_time": _stamp(start),
            "member": int(member),
            "track_id": str(group["track_id"].iloc[0]),
        }
        lines.append(
            _feature(
                "LineString",
                coordinates,
                {
                    **common,
                    "start": _stamp(group[TRACK_TIME].iloc[0]),
                    "end": _stamp(group[TRACK_TIME].iloc[-1]),
                    "n_points": len(group),
                    "min_pressure_hpa": _number(group[TRACK_PRESSURE].min()),
                },
            )
        )
        for row, coordinate in zip(group.itertuples(index=False), coordinates, strict=True):
            valid_time = getattr(row, TRACK_TIME)
            points.append(
                _feature(
                    "Point",
                    coordinate,
                    {
                        **common,
                        "valid_time": _stamp(valid_time),
                        "lead_hours": int((valid_time - start).total_seconds() // 3600),
                        "pressure_hpa": _number(getattr(row, TRACK_PRESSURE)),
                        "wind_kt": _number(getattr(row, TRACK_WIND)),
                    },
                )
            )
    return lines, points


def reference_features(best_track: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Turn the reference track into GeoJSON features.

    Args:
        best_track: Reference track from ``data.jma_besttrack.load_track``.

    Returns:
        ``(lines, points)``: one LineString and one Point per row.
    """
    best = best_track.sort_values("time")
    coordinates = [
        [wrap_longitude(lon), float(lat)]
        for lat, lon in zip(best["lat"], best["lon"], strict=True)
    ]
    source = str(best["source"].iloc[0]) if len(best) else None
    common = {"kind": "reference", "source": source}
    lines = [
        _feature(
            "LineString",
            coordinates,
            {
                **common,
                "start": _stamp(best["time"].iloc[0]) if len(best) else None,
                "end": _stamp(best["time"].iloc[-1]) if len(best) else None,
                "n_points": len(best),
            },
        )
    ]
    points = [
        _feature(
            "Point",
            coordinate,
            {
                **common,
                "time": _stamp(row.time),
                "pressure_hpa": _number(row.pressure_hpa),
                "wind_kt": _number(row.wind_kt),
                "grade": _number(row.grade),
            },
        )
        for row, coordinate in zip(best.itertuples(index=False), coordinates, strict=True)
    ]
    return lines, points


def write_geojson(features: list[dict], path: Path) -> Path:
    """Write features as one FeatureCollection.

    Args:
        features: GeoJSON feature dicts.
        path: Destination; parent directories are created.

    Returns:
        ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {"type": "FeatureCollection", "features": features}
    with path.open("w", encoding="utf-8") as handle:
        # allow_nan=False turns a stray NaN into an error here rather than an
        # unreadable file later; every value passes through _number first.
        json.dump(collection, handle, allow_nan=False)
    return path


def write_tracks(
    cases: list[tuple[str, str, pd.DataFrame]],
    best_track: pd.DataFrame,
    out_dir: Path,
    stem: str = "tracks",
) -> tuple[Path, Path]:
    """Write the forecast tracks of one or more cases plus the reference.

    Args:
        cases: ``(case_id, init_time, forecast_tracks)`` per case.
        best_track: Reference track, written once.
        out_dir: Directory for ``<stem>-lines.geojson`` and
            ``<stem>-points.geojson``.
        stem: File name prefix.

    Returns:
        ``(lines_path, points_path)``.
    """
    lines, points = [], []
    for case_id, init_time, forecast_tracks in cases:
        case_lines, case_points = forecast_features(forecast_tracks, case_id, init_time)
        lines.extend(case_lines)
        points.extend(case_points)
    reference_lines, reference_points = reference_features(best_track)
    lines.extend(reference_lines)
    points.extend(reference_points)
    out_dir = Path(out_dir)
    return (
        write_geojson(lines, out_dir / f"{stem}-lines.geojson"),
        write_geojson(points, out_dir / f"{stem}-points.geojson"),
    )
