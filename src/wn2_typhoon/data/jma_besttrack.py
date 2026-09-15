"""Load the JMA typhoon position table (best track).

Source: https://www.data.jma.go.jp/typhoon/position_table/table2026.csv
Columns (all UTC): year, month, day, hour, storm number, name, grade,
lat, lon, central pressure [hPa], max wind [kt; 0 when < 34 kt],
50 kt / 30 kt wind-radius ellipse fields, landfall flag.

Only the official post-analysis CSV is supported. As of 2026-09-16 storm
2624 is not yet in the CSV (preliminary values exist only as PDF); the
comparison step waits for the CSV by decision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POSITION_TABLE_URL = "https://www.data.jma.go.jp/typhoon/position_table/table{year}.csv"


def fetch_position_table(year: int, out_path: Path) -> Path:
    """Download the yearly position table CSV (Shift_JIS) to ``out_path``.

    Args:
        year: Four-digit year.
        out_path: Destination file.

    Returns:
        ``out_path``.
    """
    raise NotImplementedError("TODO: download and re-encode to UTF-8")


def load_track(csv_path: Path, storm_number: str) -> pd.DataFrame:
    """Return the best track for one storm as a tidy DataFrame.

    Args:
        csv_path: Position table CSV.
        storm_number: Four-digit JMA number, e.g. "2624".

    Returns:
        DataFrame indexed by UTC time with lat, lon, pressure_hpa, wind_kt, grade.
    """
    raise NotImplementedError("TODO: parse, filter by storm number, build datetime index")
