"""Tests for the per-storm parts of the configuration.

The last two tests read every config under configs/, so a new storm that
reuses a case id or mistypes an init time fails here rather than on the pod.
"""

from pathlib import Path

import pytest

from wn2_typhoon.config import (
    besttrack_path,
    comparison_dir,
    get_case,
    load_raw,
)

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


def _cfg(**storm) -> dict:
    return {"storm": {"jma_number": "2699", **storm}}


def test_paths_are_derived_from_the_storm_number() -> None:
    assert besttrack_path(_cfg()) == Path("data/interim/besttrack-2699.csv")
    assert comparison_dir(_cfg()) == Path("outputs/comparison/2699")



@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_every_storm_config_is_complete(path) -> None:
    """Cases whose ids spell their init times."""
    cfg = load_raw(path)
    for entry in cfg["cases"]:
        case = get_case(cfg, entry["id"])
        assert case.id == "init-" + case.init_time[:13]


def test_storm_configs_share_no_number_and_no_case() -> None:
    cfgs = [load_raw(path) for path in CONFIGS]
    numbers = [cfg["storm"]["jma_number"] for cfg in cfgs]
    assert len(set(numbers)) == len(numbers)
    ids = [entry["id"] for cfg in cfgs for entry in cfg["cases"]]
    assert len(set(ids)) == len(ids)
