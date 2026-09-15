"""Geodesic helpers."""

from __future__ import annotations

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres.

    Args:
        lat1: Latitude of point 1 [deg].
        lon1: Longitude of point 1 [deg].
        lat2: Latitude of point 2 [deg].
        lon2: Longitude of point 2 [deg].

    Returns:
        Distance in km.
    """
    raise NotImplementedError("TODO")
