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
        seed_position: Observed storm centre at ``init_time`` as
            ``(lat, lon_east)``, used to seed the cyclone tracker. None leaves
            the tracker in cyclogenesis mode.
    """

    id: str
    init_time: str
    note: str = ""
    seed_position: tuple[float, float] | None = None


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

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        case_id: Value of the ``id`` field under ``cases``.

    Raises:
        KeyError: If no case matches.
    """
    for entry in cfg["cases"]:
        if entry["id"] == case_id:
            seed = entry.get("seed_position")
            return Case(
                id=entry["id"],
                init_time=entry["init_time"],
                note=entry.get("note", ""),
                seed_position=None if seed is None else (float(seed[0]), float(seed[1])),
            )
    raise KeyError(f"Unknown case id: {case_id}")
