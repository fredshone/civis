"""Tests for distance functions in the distances package."""

import numpy as np
import pytest

from distances.participation import (
    participation_distance,
    pairwise_participation_distance,
)
from distances.sequence import (
    DEFAULT_COST_MATRIX,
    edit_distance,
    pairwise_sequence_distance,
)
from distances.timing import (
    activity_timing_distance,
    pairwise_activity_timing_distance,
    pairwise_timing_distance,
    timing_distance,
)
from distances.composite import composite_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec(*fractions: float) -> np.ndarray:
    """Build a participation vector with the given values (length 9, padded with 0)."""
    v = np.zeros(9, dtype=np.float64)
    for i, f in enumerate(fractions):
        v[i] = f
    return v


# ---------------------------------------------------------------------------
# participation_distance
# ---------------------------------------------------------------------------


class TestParticipationDistance:
    def test_identical_schedules_distance_zero(self):
        v = _vec(0.5, 0.3, 0.2)
        assert participation_distance(v, v) == pytest.approx(0.0)

    def test_identical_copy_distance_zero(self):
        v = _vec(0.5, 0.3, 0.2)
        assert participation_distance(v, v.copy()) == pytest.approx(0.0)

    def test_no_shared_activity_types_distance_one(self):
        # v1 spends all time in activity 0 (home), v2 in activity 1 (work)
        v1 = _vec(1.0)
        v2 = np.zeros(9, dtype=np.float64)
        v2[1] = 1.0
        assert participation_distance(v1, v2) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # Both spend time in home (index 0) but differ in remainder
        v1 = _vec(0.5, 0.5)  # 50% home, 50% work
        v2 = np.zeros(9, dtype=np.float64)
        v2[0] = 0.5  # 50% home
        v2[2] = 0.5  # 50% education
        d = participation_distance(v1, v2)
        assert 0.0 < d < 1.0
        assert d == pytest.approx(0.5)

    def test_differ_only_in_duration_not_type(self):
        # Same activity types present, just different fractions
        v1 = _vec(0.6, 0.4)
        v2 = _vec(0.4, 0.6)
        d = participation_distance(v1, v2)
        assert 0.0 < d < 1.0
        assert d == pytest.approx(0.2)

    def test_returns_float(self):
        v = _vec(0.5, 0.5)
        assert isinstance(participation_distance(v, v), float)

    def test_symmetry(self):
        v1 = _vec(0.7, 0.3)
        v2 = _vec(0.2, 0.5, 0.3)
        assert participation_distance(v1, v2) == pytest.approx(
            participation_distance(v2, v1)
        )

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            raw = rng.random(9)
            v1 = raw / raw.sum()
            raw2 = rng.random(9)
            v2 = raw2 / raw2.sum()
            d = participation_distance(v1, v2)
            assert 0.0 <= d <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# pairwise_participation_distance
# ---------------------------------------------------------------------------


class TestPairwiseParticipationDistance:
    def _sample_matrix(self, n: int = 5) -> np.ndarray:
        rng = np.random.default_rng(0)
        raw = rng.random((n, 9))
        return raw / raw.sum(axis=1, keepdims=True)

    def test_diagonal_is_zero(self):
        m = self._sample_matrix()
        D = pairwise_participation_distance(m)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetric(self):
        m = self._sample_matrix()
        D = pairwise_participation_distance(m)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_shape(self):
        m = self._sample_matrix(7)
        D = pairwise_participation_distance(m)
        assert D.shape == (7, 7)

    def test_matches_scalar_function(self):
        m = self._sample_matrix(4)
        D = pairwise_participation_distance(m)
        for i in range(len(m)):
            for j in range(len(m)):
                expected = participation_distance(m[i], m[j])
                assert D[i, j] == pytest.approx(expected, abs=1e-12)

    def test_all_identical_rows_distance_zero(self):
        v = _vec(0.5, 0.3, 0.2)
        m = np.tile(v, (5, 1))
        D = pairwise_participation_distance(m)
        np.testing.assert_allclose(D, 0.0, atol=1e-12)

    def test_values_in_unit_interval(self):
        m = self._sample_matrix(10)
        D = pairwise_participation_distance(m)
        assert D.min() >= -1e-12
        assert D.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# edit_distance
# ---------------------------------------------------------------------------


class TestEditDistance:
    def test_identical_sequences_distance_zero(self):
        seq = ["home", "work", "home"]
        assert edit_distance(seq, seq) == pytest.approx(0.0)

    def test_identical_copy_distance_zero(self):
        seq = ["home", "work", "leisure", "home"]
        assert edit_distance(seq, list(seq)) == pytest.approx(0.0)

    def test_both_empty_distance_zero(self):
        assert edit_distance([], []) == pytest.approx(0.0)

    def test_one_empty_distance_one(self):
        # Deleting all elements from a length-3 sequence costs 3, denom = 3
        assert edit_distance(["home", "work", "home"], []) == pytest.approx(1.0)

    def test_completely_different_unit_costs(self):
        # All substitutions, same length → all substitutions cost 1 each → 1.0
        s1 = ["home", "work", "education"]
        s2 = ["medical", "escort", "shop"]
        assert edit_distance(s1, s2) == pytest.approx(1.0)

    def test_different_lengths(self):
        s1 = ["home", "work", "home"]  # len 3
        s2 = ["home", "work", "leisure", "home"]  # len 4
        d = edit_distance(s1, s2)
        assert 0.0 < d < 1.0
        # One insertion needed, normalised by 4
        assert d == pytest.approx(1 / 4)

    def test_returns_float(self):
        assert isinstance(edit_distance(["home"], ["work"]), float)

    def test_symmetry(self):
        s1 = ["home", "work", "home"]
        s2 = ["home", "education", "leisure", "home"]
        assert edit_distance(s1, s2) == pytest.approx(edit_distance(s2, s1))

    def test_result_in_unit_interval(self):
        seqs = [
            ["home", "work", "home"],
            ["home", "leisure", "visit", "home"],
            ["home", "education", "home"],
            [],
        ]
        for s1 in seqs:
            for s2 in seqs:
                d = edit_distance(s1, s2)
                assert 0.0 <= d <= 1.0 + 1e-9

    def test_semantic_cost_lower_than_unit(self):
        # work ↔ education has cost 0.5 < 1.0 under DEFAULT_COST_MATRIX
        s1 = ["home", "work", "home"]
        s2 = ["home", "education", "home"]
        d_unit = edit_distance(s1, s2, cost_matrix=None)
        d_semantic = edit_distance(s1, s2, cost_matrix=DEFAULT_COST_MATRIX)
        assert d_semantic < d_unit

    def test_semantic_cost_identical_gives_zero(self):
        seq = ["home", "work", "leisure", "home"]
        assert edit_distance(
            seq, seq, cost_matrix=DEFAULT_COST_MATRIX
        ) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# pairwise_sequence_distance
# ---------------------------------------------------------------------------


class TestPairwiseSequenceDistance:
    def _sample_seqs(self) -> list[list[str]]:
        return [
            ["home", "work", "home"],
            ["home", "education", "home"],
            ["home", "leisure", "visit", "home"],
            ["home", "home"],
        ]

    def test_diagonal_is_zero(self):
        seqs = self._sample_seqs()
        D = pairwise_sequence_distance(seqs)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetric(self):
        seqs = self._sample_seqs()
        D = pairwise_sequence_distance(seqs)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_shape(self):
        seqs = self._sample_seqs()
        D = pairwise_sequence_distance(seqs)
        assert D.shape == (4, 4)

    def test_matches_scalar(self):
        seqs = self._sample_seqs()
        D = pairwise_sequence_distance(seqs)
        for i in range(len(seqs)):
            for j in range(len(seqs)):
                assert D[i, j] == pytest.approx(
                    edit_distance(seqs[i], seqs[j]), abs=1e-12
                )

    def test_all_identical_distance_zero(self):
        seq = ["home", "work", "home"]
        D = pairwise_sequence_distance([seq, seq, seq])
        np.testing.assert_allclose(D, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# timing_distance
# ---------------------------------------------------------------------------


class TestTimingDistance:
    def _row(self, pattern: list[int], t: int = 12) -> np.ndarray:
        """Build a time-use row of length t from a repeating pattern."""
        row = np.zeros(t, dtype=np.int32)
        for i in range(t):
            row[i] = pattern[i % len(pattern)]
        return row

    def test_identical_rows_distance_zero(self):
        r = self._row([0, 1, 0])
        assert timing_distance(r, r) == pytest.approx(0.0)

    def test_all_different_distance_one(self):
        # Every bin disagrees
        r1 = np.zeros(10, dtype=np.int32)
        r2 = np.ones(10, dtype=np.int32)
        assert timing_distance(r1, r2) == pytest.approx(1.0)

    def test_half_disagree(self):
        r1 = np.array([0, 0, 1, 1], dtype=np.int32)
        r2 = np.array([0, 0, 0, 0], dtype=np.int32)
        assert timing_distance(r1, r2) == pytest.approx(0.5)

    def test_symmetry(self):
        r1 = self._row([0, 1, 2])
        r2 = self._row([1, 0, 2])
        assert timing_distance(r1, r2) == pytest.approx(timing_distance(r2, r1))

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            r1 = rng.integers(0, 9, size=1440).astype(np.int32)
            r2 = rng.integers(0, 9, size=1440).astype(np.int32)
            assert 0.0 <= timing_distance(r1, r2) <= 1.0


class TestActivityTimingDistance:
    def test_neither_has_activity_distance_zero(self):
        r1 = np.zeros(10, dtype=np.int32)
        r2 = np.zeros(10, dtype=np.int32)
        assert activity_timing_distance(r1, r2, activity_type_idx=1) == pytest.approx(
            0.0
        )

    def test_one_has_activity_distance_one(self):
        r1 = np.array([1] * 5 + [0] * 5, dtype=np.int32)
        r2 = np.zeros(10, dtype=np.int32)
        assert activity_timing_distance(r1, r2, activity_type_idx=1) == pytest.approx(
            1.0
        )

    def test_same_timing_distance_zero(self):
        r = np.array([0, 1, 1, 0], dtype=np.int32)
        assert activity_timing_distance(
            r, r.copy(), activity_type_idx=1
        ) == pytest.approx(0.0)

    def test_shifted_timing_positive(self):
        # Activity 1 at start vs. end
        r1 = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        r2 = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int32)
        d = activity_timing_distance(r1, r2, activity_type_idx=1)
        assert d > 0.0
        assert d <= 1.0

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            r1 = rng.integers(0, 4, size=100).astype(np.int32)
            r2 = rng.integers(0, 4, size=100).astype(np.int32)
            d = activity_timing_distance(r1, r2, activity_type_idx=1)
            assert 0.0 <= d <= 1.0 + 1e-9


class TestPairwiseTimingDistance:
    def _sample_matrix(self, n: int = 5, t: int = 48) -> np.ndarray:
        rng = np.random.default_rng(11)
        return rng.integers(0, 9, size=(n, t)).astype(np.int32)

    def test_diagonal_is_zero(self):
        m = self._sample_matrix()
        D = pairwise_timing_distance(m)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetric(self):
        m = self._sample_matrix()
        D = pairwise_timing_distance(m)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_shape(self):
        m = self._sample_matrix(6)
        D = pairwise_timing_distance(m)
        assert D.shape == (6, 6)

    def test_matches_scalar(self):
        m = self._sample_matrix(4)
        D = pairwise_timing_distance(m)
        for i in range(len(m)):
            for j in range(len(m)):
                assert D[i, j] == pytest.approx(timing_distance(m[i], m[j]), abs=1e-12)

    def test_all_identical_distance_zero(self):
        row = np.zeros(20, dtype=np.int32)
        m = np.tile(row, (5, 1))
        D = pairwise_timing_distance(m)
        np.testing.assert_allclose(D, 0.0, atol=1e-12)


class TestPairwiseActivityTimingDistance:
    def test_diagonal_is_zero(self):
        rng = np.random.default_rng(5)
        m = rng.integers(0, 4, size=(4, 20)).astype(np.int32)
        D = pairwise_activity_timing_distance(m, activity_type_idx=1)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetric(self):
        rng = np.random.default_rng(6)
        m = rng.integers(0, 4, size=(4, 20)).astype(np.int32)
        D = pairwise_activity_timing_distance(m, activity_type_idx=1)
        np.testing.assert_allclose(D, D.T, atol=1e-12)


# ---------------------------------------------------------------------------
# composite_distance
# ---------------------------------------------------------------------------


class TestCompositeDistance:
    def _make_inputs(self):
        rng = np.random.default_rng(99)
        raw1 = rng.random(9)
        part1 = raw1 / raw1.sum()
        raw2 = rng.random(9)
        part2 = raw2 / raw2.sum()
        seq1 = np.zeros(81, dtype=np.float64)
        seq2 = np.zeros(81, dtype=np.float64)
        seq1[[0, 10, 20]] = 1 / 3
        seq2[[5, 22, 40]] = 1 / 3
        t1 = np.zeros(144, dtype=np.int32)
        t2 = np.ones(144, dtype=np.int32)
        return part1, part2, seq1, seq2, t1, t2

    def test_result_in_unit_interval(self):
        part1, part2, seq1, seq2, t1, t2 = self._make_inputs()
        d = composite_distance(part1, part2, seq1, seq2, t1, t2)
        assert 0.0 <= d <= 1.0 + 1e-9

    def test_identical_inputs_distance_zero(self):
        rng = np.random.default_rng(42)
        raw = rng.random(9)
        part = raw / raw.sum()
        seq = np.zeros(81, dtype=np.float64)
        seq[0] = 1.0
        t = np.zeros(144, dtype=np.int32)
        d = composite_distance(part, part, seq, seq, t, t)
        assert d == pytest.approx(0.0)

    def test_custom_weights_sum_to_one_internally(self):
        part1, part2, seq1, seq2, t1, t2 = self._make_inputs()
        # Equal weights (1,1,1) should give same result as (1/3, 1/3, 1/3)
        d1 = composite_distance(part1, part2, seq1, seq2, t1, t2, weights=(1, 1, 1))
        d2 = composite_distance(
            part1, part2, seq1, seq2, t1, t2, weights=(1 / 3, 1 / 3, 1 / 3)
        )
        assert d1 == pytest.approx(d2)

    def test_zero_weight_ignores_component(self):
        part1, part2, seq1, seq2, t1, t2 = self._make_inputs()
        # With only timing weight, result should equal timing_distance
        from distances.timing import timing_distance as td

        d_comp = composite_distance(part1, part2, seq1, seq2, t1, t2, weights=(0, 0, 1))
        d_time = td(t1, t2)
        assert d_comp == pytest.approx(d_time)
