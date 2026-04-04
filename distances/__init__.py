"""Distance utilities for schedule similarity modelling."""

from distances.protocols import DistanceMetric
from distances.registry import (
    METRIC_REGISTRY,
    build_metric,
    list_metrics,
    register_metric,
)
from distances.feature_store import (
    FeatureManifest,
    ScheduleFeatures,
    build_schedule_features,
    load_schedule_features,
    feature_manifest_hash,
)
from distances.cache import (
    DistanceGraphManifest,
    DistanceGraph,
    build_distance_graph,
    load_distance_graph,
)

__all__ = [
    "DistanceMetric",
    "METRIC_REGISTRY",
    "register_metric",
    "list_metrics",
    "build_metric",
    "FeatureManifest",
    "ScheduleFeatures",
    "build_schedule_features",
    "load_schedule_features",
    "feature_manifest_hash",
    "DistanceGraphManifest",
    "DistanceGraph",
    "build_distance_graph",
    "load_distance_graph",
]
