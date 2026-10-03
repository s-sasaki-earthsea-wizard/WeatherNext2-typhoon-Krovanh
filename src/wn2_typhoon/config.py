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
#: Root of the results. Each storm has a directory under it, named by its
#: JMA number and name, holding its cases and their comparison.
OUTPUTS_DIR = Path("outputs")

#: How a storm stops being a tropical storm in the JMA tables, and the key in
#: the ``storm`` block that records when. Exactly one must be present.
TROPICAL_END_KEYS = {
    "depression": "depression_time",
    "extratropical": "extratropical_time",
}


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


@dataclass(frozen=True)
class TropicalEnd:
    """When the storm stopped being a tropical storm, which ends the comparison.

    Attributes:
        time: Time of JMA's transition row, UTC ISO 8601.
        kind: ``"depression"`` if it weakened to a tropical depression,
            ``"extratropical"`` if it became an extratropical low.
    """

    time: str
    kind: str


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


def tropical_end(cfg: dict[str, Any]) -> TropicalEnd:
    """Return when and how the configured storm stopped being a tropical storm.

    Krovanh weakened to a depression and records ``depression_time``; storms
    that recurve past Japan become extratropical and record
    ``extratropical_time`` instead.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).

    Raises:
        ValueError: If the ``storm`` block has neither key or both.
    """
    storm = cfg["storm"]
    present = [
        (kind, storm[key]) for kind, key in TROPICAL_END_KEYS.items() if storm.get(key)
    ]
    if len(present) != 1:
        keys = " or ".join(TROPICAL_END_KEYS.values())
        raise ValueError(
            f"storm {storm.get('jma_number')} must set exactly one of {keys}, "
            f"found {len(present)}"
        )
    kind, time = present[0]
    return TropicalEnd(time=str(time), kind=kind)


def besttrack_path(cfg: dict[str, Any], interim_dir: Path = BESTTRACK_DIR) -> Path:
    """Path of the normalised reference track for the configured storm.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        interim_dir: Directory holding the reference tracks.

    Returns:
        ``<interim_dir>/besttrack-<jma_number>.csv``.
    """
    return Path(interim_dir) / f"besttrack-{cfg['storm']['jma_number']}.csv"


def storm_dir(cfg: dict[str, Any], outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Directory holding everything of the configured storm.

    Named by number and name, ``2624-krovanh``: the number because names are
    reused across years, the name so a listing reads without a lookup. On the
    Mac it is a symlink to the same directory on the NAS.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        outputs_dir: Root of the results.

    Returns:
        ``<outputs_dir>/<jma_number>-<name>``.
    """
    storm = cfg["storm"]
    return Path(outputs_dir) / f"{storm['jma_number']}-{str(storm['name']).lower()}"


def case_dir(cfg: dict[str, Any], case_id: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Directory of one case's forecast output and analysis.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        case_id: Value of the ``id`` field under ``cases``.
        outputs_dir: Root of the results.

    Returns:
        ``<storm_dir>/<case_id>``.
    """
    return storm_dir(cfg, outputs_dir) / case_id


def comparison_dir(cfg: dict[str, Any], outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Directory of the across-case comparison for the configured storm.

    Args:
        cfg: Parsed configuration (see :func:`load_raw`).
        outputs_dir: Root of the results.

    Returns:
        ``<storm_dir>/comparison``.
    """
    return storm_dir(cfg, outputs_dir) / "comparison"
