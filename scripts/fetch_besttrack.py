"""CLI entry point: fetch_besttrack.

Usage:
    uv run python scripts/fetch_besttrack.py --config configs/krovanh.yaml --case init-2026-08-31T18
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wn2_typhoon.config import get_case, load_raw


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/krovanh.yaml"))
    parser.add_argument("--case", required=True, help="case id from the config")
    return parser.parse_args()


def main() -> None:
    """Run the step for one case."""
    args = parse_args()
    cfg = load_raw(args.config)
    case = get_case(cfg, args.case)
    raise NotImplementedError(f"TODO: fetch_besttrack for case {case.id} (init {case.init_time})")


if __name__ == "__main__":
    main()
