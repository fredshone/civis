"""Tests for evaluation/geometry.py.

Uses the shared fixture CSV files (3 persons) and a tiny AdditionEmbedder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datasets.encoding import AttributeEncoder, default_attribute_configs
from distances.data import load_activities, load_attributes
from evaluation.geometry import GeometryAnalyser, GeometryAnalyserConfig, _linear_cka
from models import AdditionEmbedder, AttributeEmbedderConfig
from datasets.masking import AttributeMasker

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def activities_df():
    return load_activities(_FIXTURES / "activities.csv")


@pytest.fixture
def attributes_df():
    return load_attributes(_FIXTURES / "attributes.csv")


@pytest.fixture
def encoder(attributes_df):
    enc = AttributeEncoder(default_attribute_configs())
    enc.fit(attributes_df)
    return enc


@pytest.fixture
def all_pids(attributes_df):
    return attributes_df["pid"].to_list()


@pytest.fixture
def tiny_config(encoder):
    return AttributeEmbedderConfig.from_encoder(encoder, d_embed=8, d_model=8, dropout=0.0)


@pytest.fixture
def tiny_embedder(tiny_config):
    model = AdditionEmbedder(tiny_config)
    model.eval()
    return model


@pytest.fixture
def dummy_distance_fn():
    """Returns a fixed distance of 0.5 for any pair."""
    return lambda a, b: 0.5


@pytest.fixture
def analyser(tiny_embedder, attributes_df, all_pids, encoder, dummy_distance_fn):
    return GeometryAnalyser(
        embedder=tiny_embedder,
        distance_fn=dummy_distance_fn,
        test_attributes=attributes_df,
        test_pids=all_pids,
        encoder=encoder,
        config=GeometryAnalyserConfig(seed=0),
    )


# ---------------------------------------------------------------------------
# alignment_uniformity
# ---------------------------------------------------------------------------


class TestAlignmentUniformity:
    def test_returns_expected_keys(self, analyser):
        result = analyser.alignment_uniformity()
        assert "alignment" in result
        assert "uniformity" in result

    def test_uniformity_is_float(self, analyser):
        result = analyser.alignment_uniformity()
        assert isinstance(result["uniformity"], float)
        assert np.isfinite(result["uniformity"])

    def test_alignment_none_without_masker(self, analyser):
        # No masker attached → alignment should be None
        result = analyser.alignment_uniformity()
        assert result["alignment"] is None

    def test_alignment_float_with_masker(
        self, tiny_embedder, attributes_df, all_pids, encoder, dummy_distance_fn
    ):
        masker = AttributeMasker(
            mask_probs={name: 0.3 for name in [cfg.name for cfg in encoder.configs]},
        )
        analyser = GeometryAnalyser(
            embedder=tiny_embedder,
            distance_fn=dummy_distance_fn,
            test_attributes=attributes_df,
            test_pids=all_pids,
            encoder=encoder,
            masker=masker,
            config=GeometryAnalyserConfig(seed=0),
        )
        result = analyser.alignment_uniformity()
        assert result["alignment"] is not None
        assert isinstance(result["alignment"], float)
        assert result["alignment"] >= 0.0


# ---------------------------------------------------------------------------
# rank_correlation
# ---------------------------------------------------------------------------


class TestRankCorrelation:
    def test_returns_float(self, analyser):
        n = len(analyser.test_pids)
        # Provide precomputed schedule distances (random) so distance_fn is not called
        n_pairs = min(6, n * (n - 1))
        rng = np.random.default_rng(0)
        sched_dists = rng.uniform(0, 1, size=n_pairs)
        corr = analyser.rank_correlation(n_pairs=n_pairs, schedule_distances=sched_dists)
        assert isinstance(corr, float)

    def test_perfect_correlation(self, analyser):
        """When embedding distances equal schedule distances, ρ ≈ 1."""
        n = len(analyser.test_pids)
        n_pairs = min(6, max(3, n))
        rng = np.random.default_rng(analyser.config.seed)
        i_idx = rng.integers(0, n, size=n_pairs)
        j_idx = rng.integers(0, n, size=n_pairs)
        same = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % n

        # Set schedule distances equal to embedding distances
        diff = analyser._embeddings[i_idx] - analyser._embeddings[j_idx]
        emb_dists = np.linalg.norm(diff, axis=1)

        corr = analyser.rank_correlation(n_pairs=n_pairs, schedule_distances=emb_dists)
        # With identical rank orderings ρ = 1 (or nan if all distances equal)
        if np.isnan(corr):
            pytest.skip("All embedding distances are identical — cannot check correlation")
        assert corr > 0.9


# ---------------------------------------------------------------------------
# neighbourhood_overlap
# ---------------------------------------------------------------------------


class TestNeighbourhoodOverlap:
    def test_returns_none_without_matrix(self, analyser):
        result = analyser.neighbourhood_overlap(k_values=[2])
        assert result == {2: 0.0}

    def test_overlap_in_range(self, analyser):
        n = len(analyser.test_pids)
        dist_mat = np.random.RandomState(0).rand(n, n)
        np.fill_diagonal(dist_mat, 0.0)
        dist_mat = (dist_mat + dist_mat.T) / 2

        result = analyser.neighbourhood_overlap(
            k_values=[1, 2], schedule_distance_matrix=dist_mat
        )
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"k={k}: overlap {v} out of [0,1]"

    def test_perfect_overlap(self, analyser):
        """When schedule and embedding spaces have the same kNN ordering, overlap = 1."""
        n = len(analyser.test_pids)
        emb = analyser._embeddings

        # Build distance matrix from embedding distances directly
        dist_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                dist_mat[i, j] = float(np.linalg.norm(emb[i] - emb[j]))

        result = analyser.neighbourhood_overlap(
            k_values=[1], schedule_distance_matrix=dist_mat
        )
        # With identical orderings overlap should be 1
        assert result[1] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# source_separation
# ---------------------------------------------------------------------------


class TestSourceSeparation:
    def test_returns_required_keys(self, analyser):
        result = analyser.source_separation(source_column="source")
        assert "mean_wasserstein" in result
        assert "source_accuracy" in result
        assert "per_source_pair" in result

    def test_values_are_floats(self, analyser):
        result = analyser.source_separation(source_column="source")
        assert isinstance(result["mean_wasserstein"], float)
        assert isinstance(result["source_accuracy"], float)

    def test_missing_column_returns_zeros(self, analyser):
        result = analyser.source_separation(source_column="nonexistent_col")
        assert result["mean_wasserstein"] == 0.0
        assert result["source_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------


class TestCKA:
    def test_cka_perfect(self):
        """K = L → CKA = 1."""
        rng = np.random.default_rng(0)
        n = 20
        K = rng.random((n, n))
        K = K @ K.T  # make it PSD
        result = _linear_cka(K, K)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_cka_returns_float(self, analyser):
        n = len(analyser.test_pids)
        dist_mat = np.random.RandomState(0).rand(n, n)
        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0.0)
        result = analyser.cka_with_schedule_kernel(
            n_samples=n, schedule_distance_matrix=dist_mat
        )
        assert isinstance(result, float)

    def test_cka_no_matrix_returns_zero(self, analyser):
        result = analyser.cka_with_schedule_kernel(schedule_distance_matrix=None)
        assert result == 0.0


# ---------------------------------------------------------------------------
# full_report
# ---------------------------------------------------------------------------


class TestFullReport:
    def test_runs_and_returns_dict(self, analyser, tmp_path):
        result = analyser.full_report(output_dir=str(tmp_path))
        assert isinstance(result, dict)
        assert "alignment_uniformity" in result
        assert "rank_correlation" in result
        assert "neighbourhood_overlap" in result
        assert "source_separation" in result
        assert "cka" in result

    def test_creates_markdown(self, analyser, tmp_path):
        analyser.full_report(output_dir=str(tmp_path))
        assert (tmp_path / "geometry_report.md").exists()
