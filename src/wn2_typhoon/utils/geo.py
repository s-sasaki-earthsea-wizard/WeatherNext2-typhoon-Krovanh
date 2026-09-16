"""Geodesic helpers.

Every function here takes scalars or numpy arrays interchangeably and returns
whatever numpy returns, so they can be applied to a whole track at once.

Longitudes are degrees east. The Krovanh domain is far from the dateline, but
the differences are wrapped anyway so a track that crosses it stays finite.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def _wrap_degrees(delta):
    """Fold a longitude difference into [-180, 180)."""
    return (np.asarray(delta, dtype=float) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in kilometres.

    Args:
        lat1: Latitude of point 1 [deg].
        lon1: Longitude of point 1 [deg].
        lat2: Latitude of point 2 [deg].
        lon2: Longitude of point 2 [deg].

    Returns:
        Distance in km, as a float or an array matching the broadcast inputs.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(_wrap_degrees(np.asarray(lon2, dtype=float)
                                    - np.asarray(lon1, dtype=float)))
    inner = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Forward azimuth from point 1 to point 2.

    Args:
        lat1: Latitude of point 1 [deg].
        lon1: Longitude of point 1 [deg].
        lat2: Latitude of point 2 [deg].
        lon2: Longitude of point 2 [deg].

    Returns:
        Bearing in degrees clockwise from north, in [0, 360).
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(_wrap_degrees(np.asarray(lon2, dtype=float)
                                    - np.asarray(lon1, dtype=float)))
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return np.degrees(np.arctan2(y, x)) % 360.0


def track_bearing_deg(lat, lon):
    """Direction of travel at each point of a track.

    Uses the centred difference where both neighbours exist and a one-sided
    difference at the ends, which is what tropical-cyclone verification does
    when it resolves an error into along- and cross-track parts.

    Args:
        lat: Latitudes along the track [deg], in time order.
        lon: Longitudes along the track [deg], in time order.

    Returns:
        Bearing at each point, degrees clockwise from north. A track of one
        point has no direction and gives NaN.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size < 2:
        return np.full(lat.shape, np.nan)
    previous = np.roll(np.arange(lat.size), 1)
    following = np.roll(np.arange(lat.size), -1)
    previous[0] = 0
    following[-1] = lat.size - 1
    return initial_bearing_deg(
        lat[previous], lon[previous], lat[following], lon[following]
    )


def along_cross_track_km(obs_lat, obs_lon, fcst_lat, fcst_lon, motion_bearing_deg):
    """Split a position error into along-track and cross-track parts.

    The total error is the hypotenuse of the two, so a large along-track error
    means the forecast is fast or slow while a large cross-track error means it
    is off to one side.

    Args:
        obs_lat: Observed latitude [deg].
        obs_lon: Observed longitude [deg].
        fcst_lat: Forecast latitude [deg].
        fcst_lon: Forecast longitude [deg].
        motion_bearing_deg: Direction the observed storm is travelling,
            from :func:`track_bearing_deg`.

    Returns:
        ``(along_km, cross_km)``. Along-track is positive when the forecast is
        ahead of the observed centre along its direction of travel, so positive
        means too fast. Cross-track is positive to the right of that direction.
    """
    distance = haversine_km(obs_lat, obs_lon, fcst_lat, fcst_lon)
    bearing = initial_bearing_deg(obs_lat, obs_lon, fcst_lat, fcst_lon)
    offset = np.radians(bearing - np.asarray(motion_bearing_deg, dtype=float))
    return distance * np.cos(offset), distance * np.sin(offset)


def mean_position(lat, lon):
    """Centroid of a set of positions, taken on the sphere.

    Averaging degrees directly is wrong near the dateline and is slightly wrong
    everywhere; this averages the unit vectors instead.

    Args:
        lat: Latitudes [deg].
        lon: Longitudes [deg].

    Returns:
        ``(lat, lon)`` of the centroid, longitude east in [0, 360).
    """
    phi, lam = np.radians(np.asarray(lat, dtype=float)), np.radians(
        np.asarray(lon, dtype=float)
    )
    x = np.mean(np.cos(phi) * np.cos(lam))
    y = np.mean(np.cos(phi) * np.sin(lam))
    z = np.mean(np.sin(phi))
    return (
        float(np.degrees(np.arctan2(z, np.hypot(x, y)))),
        float(np.degrees(np.arctan2(y, x)) % 360.0),
    )
