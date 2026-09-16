"""CLI entry point: fetch_besttrack.

Downloads the JMA reference track for the configured storm and writes it as a
normalised CSV. By default it prefers the post-analysis table and falls back to
the preliminary one, so the same command keeps working when storm 2624 finally
appears in the yearly CSV around the turn of the year and the comparison can
simply be re-run.

Usage:
    uv run python scripts/fetch_besttrack.py --config configs/krovanh.yaml
    uv run python scripts/fetch_besttrack.py --source preliminary --refresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wn2_typhoon.config import load_raw
from wn2_typhoon.data.jma_besttrack import (
    fetch_position_table,
    fetch_preliminary,
    load_post_analysis,
    load_preliminary,
)
from wn2_typhoon.utils.logs import configure

logger = configure("fetch_besttrack")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    parser.add_argument(
        "--source",
        choices=["auto", "post-analysis", "preliminary"],
        default="auto",
        help="auto prefers the post-analysis table and falls back",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/jma"))
    parser.add_argument("--out", type=Path, help="default data/interim/besttrack.csv")
    parser.add_argument(
        "--refresh", action="store_true", help="re-download even if cached"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="keep third-party INFO logs"
    )
    return parser.parse_args()


def main() -> None:
    """Fetch and normalise the reference track."""
    args = parse_args()
    configure(logger.name, verbose=args.verbose)
    cfg = load_raw(args.config)
    number = cfg["storm"]["jma_number"]
    year = int(number[:2]) + 2000

    csv_path = args.raw_dir / f"table{year}.csv"
    pdf_path = args.raw_dir / f"T{number}.pdf"

    track = None
    if args.source in ("auto", "post-analysis"):
        if args.refresh or not csv_path.exists():
            fetch_position_table(year, csv_path)
        try:
            track = load_post_analysis(csv_path, number)
        except KeyError as missing:
            if args.source == "post-analysis":
                raise
            logger.info("%s", missing.args[0])
            logger.info("Falling back to the preliminary table")

    if track is None:
        if args.refresh or not pdf_path.exists():
            fetch_preliminary(number, pdf_path)
        track = load_preliminary(pdf_path)

    out = args.out or Path("data/interim/besttrack.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(out, index=False)

    source = track["source"].iloc[0]
    logger.info(
        "%s track for %s: %d rows, %s to %s -> %s",
        source, number, len(track), track.time.min(), track.time.max(), out,
    )
    if source == "preliminary":
        logger.info(
            "Preliminary values. They start at formation, so the pre-formation "
            "case has no observed position to compare its genesis against until "
            "the post-analysis CSV arrives."
        )


if __name__ == "__main__":
    main()
