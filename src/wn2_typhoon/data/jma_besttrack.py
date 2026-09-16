"""Load the JMA typhoon position table, from either of its two releases.

The same storm is published twice, on very different schedules:

* **Preliminary** (``T<number>.pdf``): available while the season is running.
  3-hourly in JST, pressure in hPa, maximum wind in m/s, and it begins at
  formation, so it carries nothing from the depression stage before that.
* **Post-analysis** (``table<year>.csv``, Shift_JIS): the archival record,
  6-hourly in UTC, wind in knots, including the depression stage as grade 2.
  Measured on 2026-09-16 it lagged about 3.7 months, so storm 2624 is not
  expected before the turn of the year.

Both are parsed into one schema so the comparison can start on the preliminary
values and be re-run against the CSV without changing anything downstream:

    time           UTC, timezone-naive like everything else in this project
    lat, lon       degrees, longitude east
    pressure_hpa   central pressure
    wind_kt        maximum sustained wind, 10-minute mean
    grade          JMA grade code, only the CSV has it
    remark         table annotations, only the PDF has them
    source         "preliminary" or "post-analysis"

Wind radii are deliberately not parsed. The two releases do not describe them
the same way -- the PDF gives storm-force and gale-force radii by compass
direction, the CSV gives 50 kt and 30 kt ellipse axes -- so they would not
survive the swap, and this project compares storm centres.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRELIMINARY_PDF_URL = "https://www.data.jma.go.jp/typhoon/data/T{number}.pdf"
POSITION_TABLE_URL = (
    "https://www.data.jma.go.jp/typhoon/position_table/table{year}.csv"
)

# The preliminary table is stamped in Japan Standard Time. Everything this
# project stores is UTC and timezone-naive, so JST is attached only long enough
# to convert.
JST = timezone(timedelta(hours=9))
KNOTS_PER_METRE_PER_SECOND = 1.0 / 0.514444

COLUMNS = [
    "time", "lat", "lon", "pressure_hpa", "wind_kt", "grade", "remark", "source"
]

# "2026年台風第24号  KROVANH (2624)"
_HEADER = re.compile(r"(?P<year>\d{4})年台風第\d{1,2}号\s+(?P<name>\S+)\s+\((?P<number>\d{4})\)")

# One data row. Month and day are only printed when they change, so the two
# leading integers are optional and carried forward by the caller.
_ROW = re.compile(
    r"^\s*(?P<lead>(?:\d{1,2}\s+){0,2})(?P<hour>\d{2})\s+"
    r"(?P<lat>\d{1,2}\.\d)\s*N?\s+"
    r"(?P<lon>\d{2,3}\.\d)\s*E?\s+"
    r"(?P<pressure>\d{3,4})\s+"
    r"(?P<wind>\d{1,3}|--)\s+"
    r"(?P<rest>.*)$"
)
_JAPANESE = re.compile(r"[぀-ヿ一-鿿]+")


@dataclass(frozen=True)
class StormHeader:
    """Identity of the storm a preliminary table describes.

    Attributes:
        year: Four-digit year the storm was numbered in.
        number: Four-digit JMA number, e.g. "2624".
        name: International name, e.g. "KROVANH".
    """

    year: int
    number: str
    name: str


def extract_pdf_text(pdf_path: Path) -> str:
    """Read a preliminary table as layout-preserving text.

    Args:
        pdf_path: The downloaded PDF.

    Returns:
        The page text with columns kept aligned.
    """
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(
        page.extract_text(extraction_mode="layout") for page in reader.pages
    )


def parse_header(text: str) -> StormHeader:
    """Read the storm identity from the top of a preliminary table.

    Args:
        text: Output of :func:`extract_pdf_text`.

    Returns:
        The storm's year, number and name.

    Raises:
        ValueError: If the heading is not where it is expected.
    """
    match = _HEADER.search(text)
    if match is None:
        raise ValueError("no '<year>年台風第<n>号 <NAME> (<number>)' heading found")
    return StormHeader(
        year=int(match.group("year")),
        number=match.group("number"),
        name=match.group("name"),
    )


def _remark(rest: str) -> str:
    """Return any Japanese annotation trailing a row."""
    found = _JAPANESE.findall(rest)
    return "".join(found) if found else ""


def parse_preliminary_text(text: str) -> pd.DataFrame:
    """Parse the text of a preliminary position table.

    Month and day appear only when they change and are carried forward; a month
    that goes backwards means the storm crossed into the next year. Times are
    converted from JST to UTC.

    Args:
        text: Output of :func:`extract_pdf_text`.

    Returns:
        A track table in the shared schema, sorted by time.

    Raises:
        ValueError: If the heading is missing or no rows parse.
    """
    header = parse_header(text)

    rows: list[dict] = []
    year, month, day = header.year, None, None
    pending_remark = ""

    for line in text.splitlines():
        match = _ROW.match(line)
        if match is None:
            # A standalone annotation such as 台風発生 belongs to the next row.
            stripped = line.strip()
            if stripped and _JAPANESE.fullmatch(stripped):
                pending_remark = stripped
            continue

        lead = match.group("lead").split()
        if len(lead) == 2:
            new_month, day = int(lead[0]), int(lead[1])
            if month is not None and new_month < month:
                year += 1
            month = new_month
        elif len(lead) == 1:
            day = int(lead[0])
        if month is None or day is None:
            raise ValueError(f"row before any month/day was given: {line!r}")

        jst = datetime(year, month, day, int(match.group("hour")), tzinfo=JST)
        utc = jst.astimezone(UTC).replace(tzinfo=None)
        wind = match.group("wind")
        remark = pending_remark or _remark(match.group("rest"))
        pending_remark = ""

        rows.append(
            {
                "time": np.datetime64(utc, "ns"),
                "lat": float(match.group("lat")),
                "lon": float(match.group("lon")),
                "pressure_hpa": float(match.group("pressure")),
                "wind_kt": (
                    np.nan
                    if wind == "--"
                    else float(wind) * KNOTS_PER_METRE_PER_SECOND
                ),
                "grade": pd.NA,
                "remark": remark,
                "source": "preliminary",
            }
        )

    if not rows:
        raise ValueError("no data rows parsed; has the table layout changed?")

    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["time"] = frame["time"].astype("datetime64[ns]")
    logger.info(
        "%s (%s): %d preliminary rows, %s to %s",
        header.name, header.number, len(frame),
        frame.time.min(), frame.time.max(),
    )
    return frame.sort_values("time", ignore_index=True)


def load_preliminary(pdf_path: Path) -> pd.DataFrame:
    """Parse a downloaded preliminary table.

    Args:
        pdf_path: The PDF from :func:`fetch_preliminary`.

    Returns:
        A track table in the shared schema.
    """
    return parse_preliminary_text(extract_pdf_text(pdf_path))


def load_post_analysis(csv_path: Path, storm_number: str) -> pd.DataFrame:
    """Return one storm's post-analysis track from a yearly position table.

    Args:
        csv_path: The Shift_JIS CSV from :func:`fetch_position_table`.
        storm_number: Four-digit JMA number, e.g. "2624".

    Returns:
        A track table in the shared schema.

    Raises:
        KeyError: If the storm is not in the file yet, naming what is.
    """
    raw = pd.read_csv(csv_path, encoding="shift_jis")
    columns = list(raw.columns)
    # Positional rather than by name: the headers are Japanese and have moved
    # between years, but the leading columns have not.
    raw = raw.rename(
        columns={
            columns[0]: "year", columns[1]: "month", columns[2]: "day",
            columns[3]: "hour", columns[4]: "number", columns[5]: "name",
            columns[6]: "grade", columns[7]: "lat", columns[8]: "lon",
            columns[9]: "pressure_hpa", columns[10]: "wind_kt",
        }
    )
    raw["number"] = raw["number"].astype(str).str.zfill(4)

    storm = raw[raw["number"] == storm_number]
    if storm.empty:
        available = sorted(raw["number"].unique())
        raise KeyError(
            f"storm {storm_number} is not in {csv_path.name} yet; it holds "
            f"{available[0]}..{available[-1]}. The post-analysis lags the "
            "season by months; use the preliminary table until then."
        )

    frame = pd.DataFrame(
        {
            # Both loaders must agree on resolution or a merge on time silently
            # matches nothing; the CSV would otherwise come back in seconds.
            "time": pd.to_datetime(storm[["year", "month", "day", "hour"]])
            .astype("datetime64[ns]")
            .to_numpy(),
            "lat": storm["lat"].astype(float).to_numpy(),
            "lon": storm["lon"].astype(float).to_numpy(),
            "pressure_hpa": storm["pressure_hpa"].astype(float).to_numpy(),
            # The CSV reports 0 below 34 kt, which means "under the threshold",
            # not "calm"; it is not a measurement and must not be compared.
            "wind_kt": storm["wind_kt"]
            .astype(float)
            .replace(0.0, np.nan)
            .to_numpy(),
            "grade": storm["grade"].astype("Int64").to_numpy(),
            "remark": "",
            "source": "post-analysis",
        },
        columns=COLUMNS,
    )
    logger.info(
        "storm %s: %d post-analysis rows, %s to %s",
        storm_number, len(frame), frame.time.min(), frame.time.max(),
    )
    return frame.sort_values("time", ignore_index=True)


def _download(url: str, out_path: Path) -> Path:
    """Fetch a URL to a file, creating the parent directory."""
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:
        out_path.write_bytes(response.read())
    return out_path


def fetch_preliminary(storm_number: str, out_path: Path) -> Path:
    """Download the preliminary position table for one storm.

    Args:
        storm_number: Four-digit JMA number, e.g. "2624".
        out_path: Destination PDF.

    Returns:
        ``out_path``.
    """
    return _download(PRELIMINARY_PDF_URL.format(number=storm_number), out_path)


def fetch_position_table(year: int, out_path: Path) -> Path:
    """Download the yearly post-analysis position table.

    Args:
        year: Four-digit year.
        out_path: Destination CSV, left in its original Shift_JIS encoding.

    Returns:
        ``out_path``.
    """
    return _download(POSITION_TABLE_URL.format(year=year), out_path)


def load_track(csv_path: Path) -> pd.DataFrame:
    """Read the reference track written by ``scripts/fetch_besttrack.py``.

    Both releases arrive here in the same schema, so nothing downstream has to
    know which one it got beyond reading the ``source`` column.

    Args:
        csv_path: Path to the stored track, normally
            ``data/interim/besttrack.csv``.

    Returns:
        The track in :data:`COLUMNS` order, sorted by time, with ``time`` as
        timezone-naive UTC.

    Raises:
        FileNotFoundError: If the file does not exist; ``make fetch-besttrack``
            creates it.
        ValueError: If the file does not carry the expected schema.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} is missing; run 'make fetch-besttrack' to create it"
        )
    frame = pd.read_csv(csv_path, parse_dates=["time"])
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")
    return frame[COLUMNS].sort_values("time").reset_index(drop=True)
