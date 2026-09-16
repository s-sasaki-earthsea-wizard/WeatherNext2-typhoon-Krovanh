"""Tests for the JMA reference-track loaders.

The preliminary parser takes text, not a PDF, so these run without poppler,
without pypdf and without network access. Layout below is copied from the real
T2624.pdf as pypdf renders it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wn2_typhoon.data.jma_besttrack import (
    COLUMNS,
    load_post_analysis,
    parse_header,
    parse_preliminary_text,
)

PRELIMINARY = """2026年台風第24号  KROVANH (2624)
                             位   置   表   （速報値）
（日本時）           中心位置           中心    最大      暴風域半径
月   日  時     緯度        経度      気圧    風速
                                hPa  m/s          km
                                                                台風発生
  9  1 09  22.6  N   131.9  E   996    18      ---     S:  560    N:  390
       12  22.5      131.8      996    18      ---     S:  560    N:  390
    2 00  22.4      131.8      996    18      ---     S:  560    N:  330
       03  22.5      131.8      996    18      ---     S:  560    N:  330
       09  26.0      132.0      992    --      ---         ---  熱帯低気圧に変わる
"""

# A storm that runs from December into January, to exercise the year rollover.
NEW_YEAR = """2026年台風第30号  TESTSTORM (2630)
月   日  時     緯度        経度      気圧    風速
 12 31 21  10.0  N   140.0  E   1000    18      ---
  1  1 00  10.5      140.5      1000    18      ---
"""


def test_header_is_read() -> None:
    header = parse_header(PRELIMINARY)
    assert (header.year, header.number, header.name) == (2026, "2624", "KROVANH")


def test_missing_header_is_fatal() -> None:
    with pytest.raises(ValueError, match="heading"):
        parse_preliminary_text("月   日  時\n  9  1 09  22.6  N  131.9  E  996  18\n")


def test_no_rows_is_fatal() -> None:
    with pytest.raises(ValueError, match="no data rows"):
        parse_preliminary_text("2026年台風第24号  KROVANH (2624)\n位 置 表\n")


def test_schema() -> None:
    frame = parse_preliminary_text(PRELIMINARY)
    assert list(frame.columns) == COLUMNS
    assert frame["time"].dtype == np.dtype("datetime64[ns]")
    assert (frame["source"] == "preliminary").all()


def test_jst_is_converted_to_utc() -> None:
    """09 JST on 1 September is 00 UTC the same day."""
    frame = parse_preliminary_text(PRELIMINARY)
    assert frame["time"].iloc[0] == pd.Timestamp("2026-09-01T00:00")


def test_month_and_day_are_carried_forward() -> None:
    """They are only printed when they change."""
    frame = parse_preliminary_text(PRELIMINARY)
    assert list(frame["time"]) == [
        pd.Timestamp("2026-09-01T00:00"),  # 9/1 09 JST
        pd.Timestamp("2026-09-01T03:00"),  # 9/1 12 JST, day carried
        pd.Timestamp("2026-09-01T15:00"),  # 9/2 00 JST, month carried
        pd.Timestamp("2026-09-01T18:00"),  # 9/2 03 JST, both carried
        pd.Timestamp("2026-09-02T00:00"),  # 9/2 09 JST
    ]


def test_year_rolls_over_when_the_month_goes_backwards() -> None:
    frame = parse_preliminary_text(NEW_YEAR)
    assert frame["time"].iloc[0] == pd.Timestamp("2026-12-31T12:00")
    assert frame["time"].iloc[1] == pd.Timestamp("2026-12-31T15:00")


def test_wind_is_converted_from_metres_per_second() -> None:
    frame = parse_preliminary_text(PRELIMINARY)
    assert frame["wind_kt"].iloc[0] == pytest.approx(18 / 0.514444, rel=1e-6)


def test_absent_wind_is_not_zero() -> None:
    frame = parse_preliminary_text(PRELIMINARY)
    assert np.isnan(frame["wind_kt"].iloc[-1])


def test_remarks_are_attached_to_the_right_rows() -> None:
    frame = parse_preliminary_text(PRELIMINARY)
    # 台風発生 sits on its own line above the first row.
    assert frame["remark"].iloc[0] == "台風発生"
    assert frame["remark"].iloc[1] == ""
    assert frame["remark"].iloc[-1] == "熱帯低気圧に変わる"


def test_positions_survive_the_n_and_e_suffixes() -> None:
    frame = parse_preliminary_text(PRELIMINARY)
    assert frame["lat"].iloc[0] == 22.6
    assert frame["lon"].iloc[0] == 131.9
    assert frame["lat"].iloc[1] == 22.5


CSV_HEADER = (
    "年,月,日,時（UTC）,台風番号,台風名,階級,緯度,経度,中心気圧,最大風速,"
    "50KT長径方向,50KT長径,50KT短径,30KT長径方向,30KT長径,30KT短径,上陸\n"
)
CSV_ROWS = (
    "2026,5,4,18,2605,HAGUPIT,2,8.1 ,150.0 ,1006,0,0,0,0,0,0,0,0\n"
    "2026,5,5,0,2605,HAGUPIT,3,8.1 ,149.6 ,1000,35,0,0,0,7,220,170,0\n"
    "2026,5,5,6,2601,NOKAEN,2,9.0 ,148.0 ,1004,0,0,0,0,0,0,0,0\n"
)


@pytest.fixture
def position_table(tmp_path: Path) -> Path:
    path = tmp_path / "table2026.csv"
    path.write_bytes((CSV_HEADER + CSV_ROWS).encode("shift_jis"))
    return path


def test_post_analysis_schema_matches_the_preliminary_one(position_table) -> None:
    csv = load_post_analysis(position_table, "2605")
    pdf = parse_preliminary_text(PRELIMINARY)
    assert list(csv.columns) == list(pdf.columns)
    for column in ("time", "lat", "lon", "pressure_hpa", "wind_kt"):
        assert csv[column].dtype == pdf[column].dtype, column


def test_only_the_requested_storm_is_returned(position_table) -> None:
    csv = load_post_analysis(position_table, "2605")
    assert len(csv) == 2
    assert csv["time"].iloc[0] == pd.Timestamp("2026-05-04T18:00")


def test_grade_is_kept(position_table) -> None:
    assert list(load_post_analysis(position_table, "2605")["grade"]) == [2, 3]


def test_sub_threshold_wind_is_not_reported_as_calm(position_table) -> None:
    """The CSV writes 0 kt below 34 kt, which is a threshold, not a measurement."""
    csv = load_post_analysis(position_table, "2605")
    assert np.isnan(csv["wind_kt"].iloc[0])
    assert csv["wind_kt"].iloc[1] == 35.0


def test_a_storm_not_yet_analysed_says_what_is_there(position_table) -> None:
    with pytest.raises(KeyError, match="2601..2605"):
        load_post_analysis(position_table, "2624")


# The real table, if `make fetch-besttrack` has cached it. This is the only
# guard against the published layout changing under us; data/ is git-ignored,
# so it skips on a fresh clone.
REAL_PDF = Path(__file__).resolve().parents[1] / "data/raw/jma/T2624.pdf"


@pytest.fixture(scope="module")
def real_track() -> pd.DataFrame:
    if not REAL_PDF.exists():
        pytest.skip(f"run `make fetch-besttrack` to cache {REAL_PDF.name}")
    pytest.importorskip("pypdf")
    from wn2_typhoon.data.jma_besttrack import load_preliminary

    return load_preliminary(REAL_PDF)


def test_real_table_is_three_hourly_over_the_storm(real_track) -> None:
    steps = real_track["time"].diff().dropna().dt.total_seconds() / 3600
    assert set(steps) == {3.0}
    assert real_track["time"].iloc[0] == pd.Timestamp("2026-09-01T00:00")
    assert real_track["time"].iloc[-1] == pd.Timestamp("2026-09-07T00:00")


def test_real_table_agrees_with_the_experiment_config(real_track) -> None:
    """Formation and peak in configs/krovanh.yaml came from this table."""
    first = real_track.iloc[0]
    assert (first["lat"], first["lon"], first["pressure_hpa"]) == (22.6, 131.9, 996.0)
    assert first["remark"] == "台風発生"

    peak = real_track.loc[real_track["pressure_hpa"].idxmin()]
    assert peak["time"] == pd.Timestamp("2026-09-03T00:00")
    assert (peak["lat"], peak["lon"], peak["pressure_hpa"]) == (26.3, 130.5, 985.0)


def test_the_storm_weakened_rather_than_going_extratropical(real_track) -> None:
    assert real_track["remark"].iloc[-1] == "熱帯低気圧に変わる"
