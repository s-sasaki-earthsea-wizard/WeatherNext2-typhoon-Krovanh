"""Tests for the per-storm parts of the configuration.

The last two tests read every config under configs/, so a new storm that is
missing its tropical end, reuses a case id or mistypes an init time fails here
rather than on the pod.
"""

from pathlib import Path

import pytest

from wn2_typhoon.config import (
    besttrack_path,
    case_dir,
    comparison_dir,
    get_case,
    load_raw,
    storm_dir,
    tropical_end,
)

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


def _cfg(**storm) -> dict:
    return {"storm": {"jma_number": "2699", "name": "TESTSTORM", **storm}}


def test_paths_are_derived_from_the_storm_number_and_name() -> None:
    assert besttrack_path(_cfg()) == Path("data/interim/besttrack-2699.csv")
    assert storm_dir(_cfg()) == Path("outputs/2699-teststorm")
    assert case_dir(_cfg(), "init-2026-09-01T00") == Path("outputs/2699-teststorm/init-2026-09-01T00")
    assert comparison_dir(_cfg()) == Path("outputs/2699-teststorm/comparison")


def test_a_storm_that_weakened_ends_at_its_depression_row() -> None:
    end = tropical_end(_cfg(depression_time="2026-09-07T00:00"))
    assert (end.time, end.kind) == ("2026-09-07T00:00", "depression")


def test_a_storm_that_went_extratropical_ends_at_its_transition_row() -> None:
    end = tropical_end(_cfg(extratropical_time="2026-09-22T12:00"))
    assert (end.time, end.kind) == ("2026-09-22T12:00", "extratropical")


@pytest.mark.parametrize(
    "storm",
    [{}, {"depression_time": "2026-09-07T00:00", "extratropical_time": "2026-09-07T00:00"}],
    ids=["neither", "both"],
)
def test_the_end_must_be_unambiguous(storm) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        tropical_end(_cfg(**storm))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_every_storm_config_is_complete(path) -> None:
    """A tropical end, and cases whose ids spell their init times."""
    cfg = load_raw(path)
    tropical_end(cfg)
    for entry in cfg["cases"]:
        case = get_case(cfg, entry["id"])
        assert case.id == "init-" + case.init_time[:13]


def test_storm_configs_share_no_number_and_no_case() -> None:
    cfgs = [load_raw(path) for path in CONFIGS]
    numbers = [cfg["storm"]["jma_number"] for cfg in cfgs]
    assert len(set(numbers)) == len(numbers)
    ids = [entry["id"] for cfg in cfgs for entry in cfg["cases"]]
    assert len(set(ids)) == len(ids)
