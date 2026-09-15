"""Figures: track map and error-vs-lead-time curves."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_tracks(forecast_tracks: pd.DataFrame, best_track: pd.DataFrame, out_path: Path) -> None:
    """Draw ensemble tracks over the JMA best track on a map.

    Args:
        forecast_tracks: Tracker output with a ``member`` column.
        best_track: JMA best track.
        out_path: PNG destination.
    """
    raise NotImplementedError("TODO: cartopy map")


def plot_error_vs_lead(errors: pd.DataFrame, out_path: Path) -> None:
    """Plot position error against lead time.

    Args:
        errors: Output of ``analysis.track_error.position_errors``.
        out_path: PNG destination.
    """
    raise NotImplementedError("TODO")
