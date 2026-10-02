"""Experiment configuration loading.

There is one YAML file per storm under ``configs/`` (``configs/krovanh.yaml``
is the default everywhere). This module turns it into typed objects, resolves
one *case* (initialization time) by id, and derives the paths that are kept
per storm, so that two storms never write over each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Where the normalised reference tracks are written, one file per storm.
BESTTRACK_DIR = Path("data/interim")
#: Root of the per-case results; the across-case comparison sits under it.
OUTPUTS_DIR = Path("outputs")


@dataclass(frozen=True)
class Case:
    """One forecast run.

    Attributes:
        id: Case identifier used in file names (e.g. "init-2026-08-31T18").
        init_time: Initialization time, UTC ISO 8601.
        note: Free-text description.
        observed_position: Analysed storm centre at ``init_time`` as
            ``(lat, lon_east)``, or None when no published position exists
            yet. This is reference data, not an input: tracking always runs
            in cyclogenesis mode. See :func:`get_case`.
    """

    id: str
    init_time: str
    note: str = ""
    observed_position: tuple[float, float] | None = None


def load_raw(path: Path) -> dict[str, Any]:
    """Load the YAML config as a plain dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration.
    """
    return yaml.safe_load(Path(path).read_text())


def get_case(cfg: dict[str, Any], case_id: str) -> Case:
    """Return the case with the given id.

    ``observed_position`` was called ``seed_position`` while it was fed to the
    tracker. Measured on 2026-09-17, seeding changes nothing past lead 0, so
    it is no longer used that way and the name would now mislead; the old key
    is rejected rather than silently ignored.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        case_id: Value of the ``id`` field under ``cases``.

    Raises:
        KeyError: If no case matches.
        ValueError: If a case still carries the retired ``seed_position`` key.
    """
    for entry in cfg["cases"]:
        if entry["id"] == case_id:
            if "seed_position" in entry:
                raise ValueError(
                    f"Case {case_id} uses seed_position, which was retired: "
                    "tracking no longer seeds. Rename it to observed_position."
                )
            position = entry.get("observed_position")
            return Case(
                id=entry["id"],
                init_time=entry["init_time"],
                note=entry.get("note", ""),
                observed_position=(
                    None if position is None
                    else (float(position[0]), float(position[1]))
                ),
            )
    raise KeyError(f"Unknown case id: {case_id}")



def besttrack_path(cfg: dict[str, Any], interim_dir: Path = BESTTRACK_DIR) -> Path:
    """Path of the normalised reference track for the configured storm.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        interim_dir: Directory holding the reference tracks.

    Returns:
        ``<interim_dir>/besttrack-<jma_number>.csv``.
    """
    return Path(interim_dir) / f"besttrack-{cfg['storm']['jma_number']}.csv"


def comparison_dir(cfg: dict[str, Any], outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Directory of the across-case comparison for the configured storm.

    The per-case directories need no such split, because case ids carry the
    initialization date and so never collide between storms.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        outputs_dir: Root of the results.

    Returns:
        ``<outputs_dir>/comparison/<jma_number>``.
    """
    return Path(outputs_dir) / "comparison" / str(cfg["storm"]["jma_number"])
