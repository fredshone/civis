"""Registry and builders for pluggable distance metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from distances.metric_plugins import (
    CompositeDistance,
    GPUCompositeDistance,
    ParticipationDistance,
    TwoGramDistance,
    TimingDistance,
    default_sequence_metric,
)
from distances.protocols import DistanceMetric

MetricFactory = Callable[[dict[str, Any]], DistanceMetric]

METRIC_REGISTRY: dict[str, MetricFactory] = {}


def register_metric(name: str, factory: MetricFactory) -> None:
    """Register a metric factory under a unique name."""
    if name in METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' is already registered")
    METRIC_REGISTRY[name] = factory


def list_metrics() -> list[str]:
    """Return registered metric names in sorted order."""
    return sorted(METRIC_REGISTRY)


def build_metric(spec: str | dict[str, Any]) -> DistanceMetric:
    """Build a metric from a name or config dict.

    Examples
    --------
    ``build_metric("participation")``

    ``build_metric({"name": "timing", "resolution": 10})``

    ``build_metric({
        "name": "composite",
        "weights": [0.4, 0.3, 0.3],
        "components": ["participation", "sequence", "timing"],
    })``
    """
    if isinstance(spec, str):
        cfg: dict[str, Any] = {"name": spec}
    else:
        cfg = dict(spec)

    name = cfg.pop("name", None)
    if name is None:
        raise ValueError("Metric spec must include a 'name' field")

    if name == "composite":
        component_specs = cfg.pop("components", ["participation", "sequence", "timing"])
        components = [build_metric(s) for s in component_specs]
        return CompositeDistance(components=components, **cfg)

    if name == "composite_gpu":
        component_specs = cfg.pop("components", ["participation", "sequence", "timing"])
        components = [build_metric(s) for s in component_specs]
        return GPUCompositeDistance(components=components, **cfg)

    factory = METRIC_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown metric '{name}'. Available metrics: {list_metrics()} and 'composite'"
        )
    return factory(cfg)


def _register_builtin_metrics() -> None:
    if METRIC_REGISTRY:
        return

    register_metric("participation", lambda cfg: ParticipationDistance(**cfg))
    register_metric(
        "sequence",
        lambda cfg: (
            default_sequence_metric() if len(cfg) == 0 else TwoGramDistance(**cfg)
        ),
    )
    register_metric("timing", lambda cfg: TimingDistance(**cfg))
    register_metric("composite_gpu", lambda cfg: GPUCompositeDistance(**cfg))


_register_builtin_metrics()
