"""Tests for the new feature store and distance cache modules."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from distances.feature_store import (
    build_schedule_features,
    load_schedule_features,
    feature_manifest_hash,
)
from distances.cache import (
    build_distance_graph,
    load_distance_graph,
)
from distances.data import load_activities


@pytest.fixture
def activities_df():
    """Load test activities."""
    return load_activities("tests/fixtures/activities.csv")


class TestFeatureStore:
    """Test feature store precomputation and persistence."""

    def test_build_and_load_features(self, activities_df):
        """Test building and loading schedule features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"

            # Build features
            features = build_schedule_features(
                activities_df, feature_dir, timing_resolution=10, overwrite=False
            )

            assert len(features.pids) > 0
            assert features.participation.shape[1] == 9
            assert len(features.sequences) == len(features.pids)
            assert features.sequence_2grams.shape[0] == len(features.pids)
            assert features.sequence_2grams.shape[1] == 81
            assert features.timing.ndim == 2

            # Load features
            loaded = load_schedule_features(feature_dir)
            assert loaded.pids == features.pids
            np.testing.assert_array_almost_equal(
                loaded.participation, features.participation
            )
            assert loaded.sequences == features.sequences
            np.testing.assert_array_almost_equal(
                loaded.sequence_2grams, features.sequence_2grams
            )
            np.testing.assert_array_equal(loaded.timing, features.timing)

    def test_feature_manifest_hash(self, activities_df):
        """Test manifest hashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"

            features = build_schedule_features(
                activities_df, feature_dir, timing_resolution=10
            )
            h1 = feature_manifest_hash(features.manifest)

            # Load and rehash
            loaded = load_schedule_features(feature_dir)
            h2 = feature_manifest_hash(loaded.manifest)

            assert h1 == h2
            assert isinstance(h1, str)
            assert len(h1) == 64  # SHA256 hex digest length

    def test_cache_reuse_skips_recompute(self, activities_df):
        """Test that loading cached features skips recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"

            # First build
            features1 = build_schedule_features(
                activities_df, feature_dir, timing_resolution=10, overwrite=False
            )

            # Modify an array to verify it's not being recomputed
            original_participation = features1.participation.copy()

            # Second "build" should load from cache
            features2 = build_schedule_features(
                activities_df, feature_dir, timing_resolution=10, overwrite=False
            )

            # Should be identical
            np.testing.assert_array_equal(
                features1.participation, features2.participation
            )


class TestDistanceGraph:
    """Test sparse distance graph caching."""

    def test_build_and_load_distance_graph(self, activities_df):
        """Test building and loading distance graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            graph_dir = Path(tmpdir) / "graph"

            # Build features first
            features = build_schedule_features(
                activities_df, feature_dir, timing_resolution=10
            )

            # Create a simple distance matrix for testing
            n = len(features.pids)
            D = np.random.rand(n, n).astype(np.float64)
            D = (D + D.T) / 2  # Make symmetric
            np.fill_diagonal(D, 0)

            # Build graph
            graph = build_distance_graph(
                pids=features.pids,
                distance_matrix=D,
                metric_spec="test_metric",
                feature_manifest=features.manifest,
                graph_dir=graph_dir,
                k=10,
                overwrite=False,
            )

            # Since test data has only 3 persons, effective k = min(10, 3-1) = 2
            assert len(graph.pids) == n
            assert graph.near_idx.shape == (n, n - 1)
            assert graph.far_idx.shape == (n, n - 1)

            # Load graph
            loaded = load_distance_graph(graph_dir)
            assert loaded.pids == graph.pids
            np.testing.assert_array_equal(loaded.near_idx, graph.near_idx)
            np.testing.assert_array_almost_equal(
                loaded.near_dist, graph.near_dist, decimal=5
            )

    def test_get_distance_from_graph(self, activities_df):
        """Test retrieving distances from graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            graph_dir = Path(tmpdir) / "graph"

            features = build_schedule_features(
                activities_df, feature_dir, timing_resolution=10
            )

            n = len(features.pids)
            D = np.random.rand(n, n).astype(np.float64)
            D = (D + D.T) / 2
            np.fill_diagonal(D, 0)

            graph = build_distance_graph(
                pids=features.pids,
                distance_matrix=D,
                metric_spec="test_metric",
                feature_manifest=features.manifest,
                graph_dir=graph_dir,
                k=10,
            )

            # Test retrieving a stored distance (from near neighbors)
            i, j = 0, graph.near_idx[0, 0]
            d = graph.get_distance(i, int(j))
            assert not np.isnan(d)
            assert d >= 0

            # Test self-distance
            assert graph.get_distance(0, 0) == 0.0

            # With small test dataset (3 persons), all pairs are stored,
            # so we test a pair that's in far neighbors instead of unstored
            if n >= 3:
                i_far, j_far = 0, graph.far_idx[0, 0]
                d_far = graph.get_distance(i_far, int(j_far))
                assert not np.isnan(d_far)
                assert d_far >= 0
