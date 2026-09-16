"""Experiment configuration loading.

The single source of truth is ``configs/krovanh.yaml``. This module turns it
into typed objects and resolves one *case* (initialization time) by id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
