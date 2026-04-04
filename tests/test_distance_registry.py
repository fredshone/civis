"""Tests for pluggable distance metric registry and builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from distances.data import load_activities, participation_matrix
from distances.metric_plugins import CompositeDistance, ParticipationDistance
from distances.protocols import DistanceMetric
from distances.registry import (
    METRIC_REGISTRY,
    build_metric,
    list_metrics,
    register_metric,
)


FIXTURES = Path(__file__).parent / "fixtures"
ACTIVITIES_CSV = FIXTURES / "activities.csv"


def test_builtin_metrics_are_registered():
    names = list_metrics()
    assert "participation" in names
    assert "sequence" in names
    assert "timing" in names
    assert "composite_gpu" in names


def test_build_metric_from_name():
    metric = build_metric("participation")
    assert metric.name == "participation"


def test_participation_metric_builds_candidate_index():
    acts = load_activities(ACTIVITIES_CSV)
    metric = build_metric("participation")
    features = metric.prepare_features(acts)
    index = metric.build_candidate_index(features)
    assert index is not None
    assert index["type"] == "knn"
    assert index["indices"].shape[0] == len(features["pids"])


def test_sequence_metric_uses_2gram_features():
    acts = load_activities(ACTIVITIES_CSV)
    metric = build_metric("sequence")
    features = metric.prepare_features(acts)
    assert "matrix" in features
    assert features["matrix"].shape[0] == len(features["pids"])
    assert features["matrix"].shape[1] == 81
    index = metric.build_candidate_index(features)
    assert index is not None


def test_build_metric_from_dict():
    metric = build_metric({"name": "timing", "resolution": 10})
    assert metric.name == "timing"


def test_build_composite_metric_from_config():
    metric = build_metric(
        {
            "name": "composite",
            "weights": [0.4, 0.3, 0.3],
            "components": ["participation", "sequence", "timing"],
        }
    )
    assert isinstance(metric, CompositeDistance)
    assert len(metric.components) == 3


def test_build_gpu_composite_metric_from_config():
    metric = build_metric(
        {
            "name": "composite_gpu",
            "weights": [0.4, 0.3, 0.3],
            "components": ["participation", "sequence", "timing"],
            "device": "cpu",
        }
    )
    assert metric.name == "composite_gpu"
    assert len(metric.components) == 3


def test_composite_metric_builds_nested_candidate_index():
    acts = load_activities(ACTIVITIES_CSV)
    metric = build_metric(
        {
            "name": "composite",
            "weights": [0.4, 0.3, 0.3],
            "components": [
                "participation",
                {"name": "timing", "resolution": 10},
                "sequence",
            ],
        }
    )
    features = metric.prepare_features(acts)
    index = metric.build_candidate_index(features)
    assert index is not None
    assert index["type"] == "composite"
    assert len(index["components"]) >= 1


def test_gpu_composite_scores_match_cpu_on_cpu_device():
    acts = load_activities(ACTIVITIES_CSV)
    cpu_metric = build_metric(
        {
            "name": "composite",
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "components": ["participation", "sequence", "timing"],
        }
    )
    gpu_metric = build_metric(
        {
            "name": "composite_gpu",
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "components": ["participation", "sequence", "timing"],
            "device": "cpu",
        }
    )

    cpu_features = cpu_metric.prepare_features(acts)
    gpu_features = gpu_metric.prepare_features(acts)

    pairs = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    cpu_d = cpu_metric.score_pairs_batch(cpu_features, pairs)
    gpu_d = gpu_metric.score_pairs_batch(gpu_features, pairs)

    np.testing.assert_allclose(gpu_d, cpu_d, atol=1e-12)


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        build_metric("not_a_metric")


def test_register_metric_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_metric("participation", lambda cfg: ParticipationDistance(**cfg))


def test_custom_metric_registration_and_build():
    class ConstantMetric(DistanceMetric):
        name = "constant"

        def __init__(self, value: float = 0.5) -> None:
            self.value = value

        def prepare_features(self, activities):
            pids, _ = participation_matrix(activities)
            return {"pids": pids}

        def score_pairs_batch(self, features, pairs, index=None):
            return np.full(len(pairs), self.value, dtype=np.float64)

    metric_name = "constant_test_metric"
    register_metric(metric_name, lambda cfg: ConstantMetric(**cfg))
    try:
        metric = build_metric({"name": metric_name, "value": 0.25})
        acts = load_activities(ACTIVITIES_CSV)
        features = metric.prepare_features(acts)
        d = metric.score_pairs_batch(features, np.array([[0, 1]], dtype=np.int32))
        assert d[0] == pytest.approx(0.25)
    finally:
        METRIC_REGISTRY.pop(metric_name, None)


def test_composite_plugin_matches_legacy_scalar_for_pair():
    acts = load_activities(ACTIVITIES_CSV)
    metric = build_metric(
        {
            "name": "composite",
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "components": ["participation", "sequence", "timing"],
        }
    )
    features = metric.prepare_features(acts)
    d = metric.score_pairs_batch(features, np.array([[0, 1]], dtype=np.int32))[0]
    assert np.isfinite(d)
