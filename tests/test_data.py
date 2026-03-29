"""Tests for distances/data.py — loading and extractor functions."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from distances.data import (
    ACTIVITY_TYPES,
    activity_sequences,
    load_activities,
    load_attributes,
    participation_matrix,
    time_use_matrix,
)

FIXTURES = Path(__file__).parent / "fixtures"
ACTIVITIES_CSV = FIXTURES / "activities.csv"
ATTRIBUTES_CSV = FIXTURES / "attributes.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_activities():
    df = load_activities(ACTIVITIES_CSV)
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"pid", "seq", "act", "zone", "start", "end"}
    assert df["pid"].dtype == pl.Utf8
    assert df["seq"].dtype == pl.Int32
    assert df["start"].dtype == pl.Int32
    assert df["end"].dtype == pl.Int32
    assert len(df) == 14  # 3 + 10 + 1 rows


def test_load_attributes():
    df = load_attributes(ATTRIBUTES_CSV)
    assert isinstance(df, pl.DataFrame)
    assert "pid" in df.columns
    assert "source" in df.columns
    assert df["pid"].dtype == pl.Utf8
    assert df["age"].dtype == pl.Int32
    assert df["weight"].dtype == pl.Float64
    assert len(df) == 3


# ---------------------------------------------------------------------------
# participation_matrix
# ---------------------------------------------------------------------------


def test_participation_matrix_shape():
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = participation_matrix(df)
    assert len(pids) == 3
    assert mat.shape == (3, len(ACTIVITY_TYPES))
    assert mat.dtype == np.float64


def test_participation_matrix_rows_sum_to_one():
    df = load_activities(ACTIVITIES_CSV)
    _, mat = participation_matrix(df)
    np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-6)


def test_participation_matrix_values():
    # cmap_pid_a: home=480+480=960min, work=480min out of 1440
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = participation_matrix(df)
    idx = pids.index("cmap_pid_a")
    home_col = ACTIVITY_TYPES.index("home")
    work_col = ACTIVITY_TYPES.index("work")
    np.testing.assert_almost_equal(mat[idx, home_col], 960 / 1440, decimal=6)
    np.testing.assert_almost_equal(mat[idx, work_col], 480 / 1440, decimal=6)


def test_participation_matrix_home_only():
    # ktdb_pid_c: home all day -> home fraction = 1.0, rest = 0
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = participation_matrix(df)
    idx = pids.index("ktdb_pid_c")
    home_col = ACTIVITY_TYPES.index("home")
    np.testing.assert_almost_equal(mat[idx, home_col], 1.0, decimal=6)
    np.testing.assert_almost_equal(mat[idx, :home_col].sum() + mat[idx, home_col + 1:].sum(), 0.0, decimal=6)


# ---------------------------------------------------------------------------
# time_use_matrix
# ---------------------------------------------------------------------------


def test_time_use_matrix_shape_default():
    # Default resolution=10 -> 1440/10 = 144 bins
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = time_use_matrix(df)
    assert len(pids) == 3
    assert mat.shape == (3, 144)
    assert mat.dtype == np.int32


def test_time_use_matrix_shape_resolution_1():
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = time_use_matrix(df, resolution=1)
    assert mat.shape == (3, 1440)


def test_time_use_matrix_shape_resolution_5():
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = time_use_matrix(df, resolution=5)
    assert mat.shape == (3, 288)


def test_time_use_matrix_invalid_resolution():
    df = load_activities(ACTIVITIES_CSV)
    with pytest.raises(ValueError, match="does not divide 1440"):
        time_use_matrix(df, resolution=7)


def test_time_use_matrix_values_resolution_1():
    # At 1-minute resolution check exact minute boundaries
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = time_use_matrix(df, resolution=1)
    idx = pids.index("cmap_pid_a")
    home_idx = ACTIVITY_TYPES.index("home")
    work_idx = ACTIVITY_TYPES.index("work")
    # home 0-480, work 480-960, home 960-1440
    assert mat[idx, 0] == home_idx
    assert mat[idx, 479] == home_idx
    assert mat[idx, 480] == work_idx
    assert mat[idx, 959] == work_idx
    assert mat[idx, 960] == home_idx
    assert mat[idx, 1439] == home_idx


def test_time_use_matrix_values_default_resolution():
    # At resolution=10, cmap_pid_a: home bins 0-47, work bins 48-95, home bins 96-143
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = time_use_matrix(df)
    idx = pids.index("cmap_pid_a")
    home_idx = ACTIVITY_TYPES.index("home")
    work_idx = ACTIVITY_TYPES.index("work")
    assert mat[idx, 0] == home_idx     # bin 0: minutes 0-9 -> home
    assert mat[idx, 47] == home_idx    # bin 47: minutes 470-479 -> home
    assert mat[idx, 48] == work_idx    # bin 48: minutes 480-489 -> work
    assert mat[idx, 95] == work_idx    # bin 95: minutes 950-959 -> work
    assert mat[idx, 96] == home_idx    # bin 96: minutes 960-969 -> home
    assert mat[idx, 143] == home_idx   # bin 143: minutes 1430-1439 -> home


def test_time_use_matrix_home_only():
    df = load_activities(ACTIVITIES_CSV)
    pids, mat = time_use_matrix(df)
    idx = pids.index("ktdb_pid_c")
    home_idx = ACTIVITY_TYPES.index("home")
    assert (mat[idx] == home_idx).all()


# ---------------------------------------------------------------------------
# activity_sequences
# ---------------------------------------------------------------------------


def test_activity_sequences_shape():
    df = load_activities(ACTIVITIES_CSV)
    pids, seqs = activity_sequences(df)
    assert len(pids) == 3
    assert len(seqs) == 3


def test_activity_sequences_values():
    df = load_activities(ACTIVITIES_CSV)
    pids, seqs = activity_sequences(df)
    idx_a = pids.index("cmap_pid_a")
    assert seqs[idx_a] == ["home", "work", "home"]
    idx_c = pids.index("ktdb_pid_c")
    assert seqs[idx_c] == ["home"]


def test_activity_sequences_all_types_present():
    # cmap_pid_b has all 9 activity types
    df = load_activities(ACTIVITIES_CSV)
    pids, seqs = activity_sequences(df)
    idx_b = pids.index("cmap_pid_b")
    assert set(seqs[idx_b]) == set(ACTIVITY_TYPES)


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------


def test_no_nts_in_fixture():
    df = load_attributes(ATTRIBUTES_CSV)
    assert df.filter(pl.col("source") == "nts").is_empty()
