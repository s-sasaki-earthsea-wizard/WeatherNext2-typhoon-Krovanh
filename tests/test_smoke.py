"""Smoke tests: the package imports and the experiment config resolves."""

from pathlib import Path

import wn2_typhoon
from wn2_typhoon.config import get_case, load_raw

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "krovanh.yaml"


def test_version() -> None:
    assert wn2_typhoon.__version__


def test_config_has_two_cases() -> None:
    cfg = load_raw(CONFIG)
    assert cfg["storm"]["jma_number"] == "2624"
    assert [c["id"] for c in cfg["cases"]] == ["init-2026-08-31T18", "init-2026-09-01T00"]


def test_get_case() -> None:
    cfg = load_raw(CONFIG)
    case = get_case(cfg, "init-2026-09-01T00")
    assert case.init_time == "2026-09-01T00:00"


def test_cases_carry_their_tracker_seed() -> None:
    """The pre-formation case has no observed position to seed from."""
    cfg = load_raw(CONFIG)
    assert get_case(cfg, "init-2026-08-31T18").seed_position is None
    assert get_case(cfg, "init-2026-09-01T00").seed_position == (22.6, 131.9)
