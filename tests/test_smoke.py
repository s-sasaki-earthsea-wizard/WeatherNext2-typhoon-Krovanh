"""Smoke tests: the package imports and the experiment config resolves."""

from pathlib import Path

import pytest

import wn2_typhoon
from wn2_typhoon.config import get_case, load_raw

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "krovanh.yaml"


def test_version() -> None:
    assert wn2_typhoon.__version__


def test_config_spans_twelve_hours_either_side_of_formation() -> None:
    """Five init times, 6 h apart, straddling the observed formation."""
    cfg = load_raw(CONFIG)
    assert cfg["storm"]["jma_number"] == "2624"
    assert [c["id"] for c in cfg["cases"]] == [
        "init-2026-08-31T12",
        "init-2026-08-31T18",
        "init-2026-09-01T00",
        "init-2026-09-01T06",
        "init-2026-09-01T12",
    ]


def test_get_case() -> None:
    cfg = load_raw(CONFIG)
    case = get_case(cfg, "init-2026-09-01T00")
    assert case.init_time == "2026-09-01T00:00"


def test_cases_carry_the_observed_position_where_one_exists() -> None:
    """Reference positions only, and only from formation onwards.

    Nothing feeds these to the tracker: every case is tracked in cyclogenesis
    mode. They are here so the analysis can state the initial-state error.
    """
    cfg = load_raw(CONFIG)
    assert get_case(cfg, "init-2026-08-31T12").observed_position is None
    assert get_case(cfg, "init-2026-08-31T18").observed_position is None
    assert get_case(cfg, "init-2026-09-01T00").observed_position == (22.6, 131.9)
    assert get_case(cfg, "init-2026-09-01T12").observed_position == (22.4, 131.7)


def test_retired_seed_position_key_is_rejected() -> None:
    """A stale config must fail loudly rather than silently stop seeding."""
    cfg = {"cases": [{"id": "c", "init_time": "2026-09-01T00:00",
                      "seed_position": [22.6, 131.9]}]}
    with pytest.raises(ValueError, match="seed_position"):
        get_case(cfg, "c")


def test_every_case_has_a_distinct_init_time() -> None:
    """Cases are keyed by init time; a duplicate would overwrite outputs."""
    cfg = load_raw(CONFIG)
    times = [entry["init_time"] for entry in cfg["cases"]]
    assert len(times) == len(set(times))
