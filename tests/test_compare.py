"""Tests for the across-case comparison and the GeoJSON export.

The per-case tables are built here through the real evaluation functions
from constructed tracks, so the comparison is tested on the schema it will
meet, not on a hand-written imitation of it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.analysis.compare import (
    CASE_TABLES,
    CaseResult,
    case_labels,
    fate_counts,
    genesis_by_case,
    lifetime_by_case,
    load_case,
    member_fate,
    offset_label,
    pivot,
    skill_by_lead,
    spread_skill,
)
from wn2_typhoon.analysis.export import (
    forecast_features,
    reference_features,
    wrap_longitude,
    write_tracks,
)
from wn2_typhoon.analysis.plot import credit_line, tile_zoom
from wn2_typhoon.analysis.track_error import (
    ensemble_summary,
    genesis_report,
    lifetime_report,
    position_errors,
    select_storm,
)

FORMATION = "2026-09-01T00:00"
WEAKENING = "2026-09-02T00:00"


def _best_track():
    """A reference track moving north by one degree every six hours."""
    times = pd.date_range("2026-09-01 00:00", "2026-09-02 00:00", freq="6h")
    return pd.DataFrame(
        {
            "time": times,
            "lat": 20.0 + np.arange(len(times)),
            "lon": [130.0] * len(times),
            "pressure_hpa": [990.0] * len(times),
            "wind_kt": [40.0] * len(times),
            "grade": [np.nan] * len(times),
            "remark": [""] * len(times),
            "source": ["preliminary"] * len(times),
        }
    )


def _tracks(rows):
    frame = pd.DataFrame(
        rows,
        columns=[
            "member", "track_id", "valid_time", "lat", "lon",
            "minimum_sea_level_pressure_hpa",
        ],
    )
    frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    frame["maximum_sustained_wind_speed_knots"] = 45.0
    return frame


def _case(case_id: str, init: str, rows) -> CaseResult:
    """Build a case's tables the way scripts/evaluate.py does."""
    best = _best_track()
    tracks = _tracks(rows)
    selected, selections = select_storm(tracks, best, init)
    errors = position_errors(selected, best, init)
    tables = {
        "summary": ensemble_summary(errors),
        "errors": errors,
        "genesis": genesis_report(selected, selections, FORMATION, init, 6),
        "lifetime": lifetime_report(selected, selections, WEAKENING),
        "track": selected,
        "selection": pd.DataFrame(
            [{"member": s.member, "found": s.found, "track_id": s.track_id,
              "anchor_time": s.anchor_time, "distance_km": s.distance_km,
              "runner_up_km": s.runner_up_km, "separation_ratio": np.nan}
             for s in selections]
        ),
    }
    return CaseResult(case_id, pd.Timestamp(init), tables)


def _two_cases():
    """An early case with one member wandering off, and a later exact one.

    The early case starts 12 h before formation, so its first rollout step
    (08-31 18Z) precedes the observed formation and genesis timing is
    resolvable; the reference track only starts at formation, so errors for
    it begin at lead 12 h.
    """
    early = _case(
        "early", "2026-08-31T12:00",
        [
            # Member 0 exists from the first rollout step and follows the truth.
            (0, 1, "2026-08-31 18:00", 19.0, 130.0, 998.0),
            (0, 1, "2026-09-01 00:00", 20.0, 130.0, 995.0),
            (0, 1, "2026-09-01 06:00", 21.0, 130.0, 990.0),
            (0, 1, "2026-09-01 12:00", 22.0, 130.0, 985.0),
            (0, 1, "2026-09-01 18:00", 23.0, 130.0, 985.0),
            (0, 1, "2026-09-02 00:00", 24.0, 130.0, 995.0),
            (0, 1, "2026-09-02 06:00", 25.0, 130.0, 1000.0),
            # Member 1 starts later, drifts far west, and ends early.
            (1, 4, "2026-09-01 06:00", 21.5, 130.0, 992.0),
            (1, 4, "2026-09-01 12:00", 22.0, 127.0, 992.0),
            (1, 4, "2026-09-01 18:00", 22.0, 122.0, 998.0),
        ],
    )
    late = _case(
        "late", "2026-09-01T06:00",
        [
            (0, 2, "2026-09-01 12:00", 22.0, 130.0, 990.0),
            (0, 2, "2026-09-01 18:00", 23.0, 130.0, 990.0),
            (0, 2, "2026-09-02 00:00", 24.0, 130.0, 990.0),
            (1, 5, "2026-09-01 12:00", 22.0, 130.5, 990.0),
            (1, 5, "2026-09-01 18:00", 23.0, 130.5, 990.0),
            (1, 5, "2026-09-02 00:00", 24.0, 130.5, 990.0),
        ],
    )
    return [early, late]


def test_case_labels_are_relative_to_formation() -> None:
    cases = _two_cases()
    assert case_labels(cases, FORMATION) == {"early": "-12 h", "late": "+6 h"}
    same = CaseResult("at", pd.Timestamp(FORMATION), {})
    assert case_labels([same], FORMATION) == {"at": "0 h"}


def test_offset_label_reads_as_prose() -> None:
    assert offset_label("2026-08-31T12:00", FORMATION) == "12 h before formation"
    assert offset_label("2026-09-01T00:00", FORMATION) == "at formation"
    assert offset_label("2026-09-01T06:00", FORMATION) == "6 h after formation"


def test_skill_by_lead_carries_valid_time_and_the_near_count() -> None:
    cases = _two_cases()
    skill = skill_by_lead(cases, within_km=200.0)
    early = skill.loc[skill["case"] == "early"].set_index("lead_hours")
    # The reference starts at formation, which is lead 12 h of the early case.
    assert early.index.min() == 12
    assert early.loc[12, "valid_time"] == pd.Timestamp("2026-09-01 00:00")
    assert early.loc[12, "n_within_200km"] == 1  # only member 0 exists yet
    # At lead 24 h member 1 sits 3 degrees west (~310 km): one member near.
    assert early.loc[24, "n_members"] == 2
    assert early.loc[24, "n_within_200km"] == 1
    wide = pivot(skill, "mean_error_km", cases)
    assert list(wide.columns) == ["early", "late"]
    assert np.isnan(wide.loc[30, "late"])  # the late case ends at lead 18 h
    assert not np.isnan(wide.loc[30, "early"])


def test_pivot_by_valid_time_aligns_the_cases_on_the_clock() -> None:
    cases = _two_cases()
    skill = skill_by_lead(cases)
    wide = pivot(skill, "mean_error_km", cases, index="valid_time")
    row = wide.loc[pd.Timestamp("2026-09-01 12:00")]
    # Early case, lead 24 h: one member on the truth, one 3 degrees west.
    assert row["early"] == pytest.approx(np.mean([0.0, 309.0]), abs=3)
    # Late case, lead 6 h: one member on the truth, one half a degree east.
    assert row["late"] == pytest.approx(np.mean([0.0, 51.5]), abs=2)


def test_genesis_by_case_counts_members_at_the_floor() -> None:
    cases = _two_cases()
    table = genesis_by_case(cases, step_hours=6).set_index("case")
    assert bool(table.loc["early", "resolvable"]) is True
    assert table.loc["early", "n_at_earliest"] == 1  # member 0 at lead 6 h
    assert table.loc["early", "n_found"] == 2
    assert table.loc["early", "min_lead_error_hours"] == pytest.approx(-6.0)
    assert table.loc["early", "max_lead_error_hours"] == pytest.approx(6.0)
    assert bool(table.loc["late", "resolvable"]) is False


def test_lifetime_by_case_counts_members_outliving_the_storm() -> None:
    cases = _two_cases()
    table = lifetime_by_case(cases, observed_min_pressure_hpa=990.0).set_index("case")
    # Early member 0 ends 6 h after the observed weakening, member 1 before.
    assert table.loc["early", "n_outliving_observed"] == 1
    assert table.loc["early", "max_end_error_hours"] == pytest.approx(6.0)
    assert table.loc["early", "n_deeper_than_observed"] == 1  # 985 hPa
    assert table.loc["late", "n_deeper_than_observed"] == 0


def test_spread_skill_uses_only_full_ensemble_leads() -> None:
    cases = _two_cases()
    table = spread_skill(cases).set_index("case")
    # The late case has two members at every lead and nonzero spread.
    assert table.loc["late", "n_leads"] == 3
    assert table.loc["late", "mean_error_to_spread"] > 0


def test_member_fate_is_read_against_the_observed_box() -> None:
    cases = _two_cases()
    fate = member_fate(cases, _best_track(), margin_deg=1.5).set_index(["case", "member"])
    assert fate.loc[("early", 0), "fate"] == "stayed"   # ends 25N, box top 25.5N
    assert fate.loc[("early", 1), "fate"] == "west"     # ends 122E, box west 128.5E
    counts = fate_counts(fate.reset_index(), cases).set_index("case")
    assert counts.loc["early", "stayed"] == 1
    assert counts.loc["early", "west"] == 1
    assert counts.loc["early", "n_members"] == 2
    assert counts.loc["late", "stayed"] == 2


def test_load_case_reads_what_evaluate_writes(tmp_path) -> None:
    case = _two_cases()[0]
    for name, table in case.tables.items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    loaded = load_case("early", "2026-08-31T18:00", tmp_path)
    assert set(loaded.tables) == set(CASE_TABLES)
    assert pd.api.types.is_datetime64_any_dtype(loaded["errors"]["valid_time"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["track"]["valid_time"])
    with pytest.raises(FileNotFoundError):
        load_case("missing", "2026-08-31T18:00", tmp_path / "nowhere")


def test_geojson_wraps_longitude_and_orders_coordinates() -> None:
    assert wrap_longitude(251.0) == pytest.approx(-109.0)
    assert wrap_longitude(130.0) == pytest.approx(130.0)
    assert wrap_longitude(180.0) == pytest.approx(-180.0)
    tracks = _tracks(
        [
            (0, 3, "2026-09-01 06:00", 21.0, 250.0, np.nan),
            (0, 3, "2026-09-01 12:00", 22.0, 251.0, 990.0),
        ]
    )
    lines, points = forecast_features(tracks, "case-a", "2026-09-01T00:00")
    assert len(lines) == 1 and len(points) == 2
    assert lines[0]["geometry"]["coordinates"][0] == [pytest.approx(-110.0), 21.0]
    assert lines[0]["properties"]["member"] == 0
    assert lines[0]["properties"]["case"] == "case-a"
    assert points[0]["properties"]["pressure_hpa"] is None
    assert points[0]["properties"]["lead_hours"] == 6
    assert points[1]["properties"]["valid_time"] == "2026-09-01T12:00:00Z"


def test_geojson_files_are_valid_json_with_the_reference(tmp_path) -> None:
    tracks = _tracks([(0, 3, "2026-09-01 06:00", 21.0, 130.0, np.nan)])
    lines_path, points_path = write_tracks(
        [("case-a", "2026-09-01T00:00", tracks)], _best_track(), tmp_path
    )
    lines = json.loads(lines_path.read_text())
    points = json.loads(points_path.read_text())
    assert lines["type"] == "FeatureCollection"
    kinds = [f["properties"]["kind"] for f in lines["features"]]
    assert kinds == ["forecast", "reference"]
    assert all(f["geometry"]["type"] == "Point" for f in points["features"])
    reference_lines, reference_points = reference_features(_best_track())
    assert reference_lines[0]["properties"]["n_points"] == len(reference_points)


def test_tile_zoom_follows_extent_and_pixel_width() -> None:
    extent = [118.0, 145.0, 19.0, 40.0]
    assert tile_zoom(extent, 1360) == 7
    assert tile_zoom(extent, 450) == 5
    assert tile_zoom([0.0, 360.0, -80.0, 80.0], 100) == 3      # clamped low
    assert tile_zoom([130.0, 130.1, 25.0, 25.1], 4000) == 10   # clamped high


def test_credit_line_names_the_tiles_only_when_drawn() -> None:
    plain = credit_line("preliminary")
    assert "OpenStreetMap" not in plain
    assert "OpenStreetMap" in credit_line("preliminary", "osm")
    with pytest.raises(ValueError):
        credit_line("preliminary", "google")


def test_comparison_figures_render(tmp_path) -> None:
    """The non-map figures draw from constructed tables without a network."""
    from wn2_typhoon.analysis.plot_compare import (
        plot_error_vs_lead,
        plot_error_vs_valid_time,
        plot_lifetime,
        plot_spread_vs_error,
    )

    cases = _two_cases()
    labels = case_labels(cases, FORMATION)
    skill = skill_by_lead(cases)
    fate = member_fate(cases, _best_track())
    ends = pd.concat(
        [c["lifetime"].assign(case=c.case_id)[["case", "member", "end_error_hours"]]
         for c in cases]
    )
    fate = fate.merge(ends, on=["case", "member"])
    events = {"formation": pd.Timestamp(FORMATION), "depression": pd.Timestamp(WEAKENING)}
    written = [
        plot_error_vs_lead(skill, cases, labels, tmp_path / "a.png", 200.0, credit=""),
        plot_error_vs_valid_time(skill, cases, labels, events, tmp_path / "b.png", 200.0,
                                 credit=""),
        plot_spread_vs_error(skill, cases, labels, tmp_path / "c.png", credit=""),
        plot_lifetime(fate, cases, labels, 990.0, tmp_path / "d.png", credit=""),
    ]
    assert all(path.exists() and path.stat().st_size > 5000 for path in written)
