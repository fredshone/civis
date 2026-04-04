"""Built-in pluggable distance metrics.

Includes CPU and GPU composite variants for lazy pairwise scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.neighbors import NearestNeighbors

from distances.data import (
    activity_sequences,
    participation_matrix,
    sequence_2gram_matrix_from_sequences,
    time_use_matrix,
)
from distances.protocols import DistanceMetric
from distances.timing import timing_distance


@dataclass
class ParticipationFeatures:
    pids: list[str]
    matrix: np.ndarray  # (N, 9) float64


@dataclass
class TwoGramFeatures:
    pids: list[str]
    matrix: np.ndarray  # (N, 81) float64


@dataclass
class TimingFeatures:
    pids: list[str]
    matrix: np.ndarray  # (N, T) int32


@dataclass
class CompositeFeatures:
    pids: list[str]
    components: list[dict[str, Any]]  # features from each component metric


def _build_nn_index(
    matrix: np.ndarray, pids: list[str], k: int = 50
) -> dict[str, Any] | None:
    """Build a simple k-NN candidate index for a feature matrix.

    The returned structure is intentionally lightweight and serialisable:
    it stores the nearest-neighbour indices and distances for reuse by
    downstream candidate generation code.
    """
    n = matrix.shape[0]
    if n < 2:
        return None

    effective_k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)

    # Drop self-neighbour at position 0.
    return {
        "type": "knn",
        "k": effective_k,
        "metric": "euclidean",
        "pids": list(pids),
        "indices": indices[:, 1:].astype(np.int32),
        "distances": distances[:, 1:].astype(np.float32),
    }


def _matrix_to_tensor(matrix: np.ndarray, device: str) -> torch.Tensor:
    """Convert a NumPy feature matrix to a dense torch tensor on *device*."""
    return torch.from_numpy(np.asarray(matrix)).to(device=device)


def _tensor_features_to_numpy(features: dict[str, Any]) -> dict[str, Any]:
    """Convert tensor-valued feature dict entries back to NumPy arrays."""
    converted: dict[str, Any] = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            converted[key] = value.detach().cpu().numpy()
        else:
            converted[key] = value
    return converted


class ParticipationDistance(DistanceMetric):
    name = "participation"

    def prepare_features(self, activities: pl.DataFrame) -> dict[str, Any]:
        pids, matrix = participation_matrix(activities)
        return {
            "pids": pids,
            "matrix": matrix,
        }

    def build_candidate_index(self, features: dict[str, Any]) -> dict[str, Any] | None:
        return _build_nn_index(features["matrix"], features["pids"])

    def score_pairs_batch(
        self,
        features: dict[str, Any],
        pairs: np.ndarray,
        index: dict[str, Any] | None = None,
    ) -> np.ndarray:
        matrix = features["matrix"]
        pairs = np.asarray(pairs, dtype=np.intp)
        left = matrix[pairs[:, 0]]
        right = matrix[pairs[:, 1]]
        return np.sum(np.abs(left - right), axis=1) / 2.0


class TwoGramDistance(DistanceMetric):
    name = "sequence"

    def prepare_features(self, activities: pl.DataFrame) -> dict[str, Any]:
        pids, sequences = activity_sequences(activities)
        matrix = sequence_2gram_matrix_from_sequences(sequences)
        return {
            "pids": pids,
            "matrix": matrix,
        }

    def build_candidate_index(self, features: dict[str, Any]) -> dict[str, Any] | None:
        return _build_nn_index(features["matrix"], features["pids"])

    def score_pairs_batch(
        self,
        features: dict[str, Any],
        pairs: np.ndarray,
        index: dict[str, Any] | None = None,
    ) -> np.ndarray:
        matrix = features["matrix"]
        pairs = np.asarray(pairs, dtype=np.intp)
        left = matrix[pairs[:, 0]]
        right = matrix[pairs[:, 1]]
        return np.sum(np.abs(left - right), axis=1) / 2.0


class TimingDistance(DistanceMetric):
    name = "timing"

    def __init__(self, resolution: int = 1) -> None:
        self.resolution = resolution

    def prepare_features(self, activities: pl.DataFrame) -> dict[str, Any]:
        pids, matrix = time_use_matrix(activities, resolution=self.resolution)
        return {
            "pids": pids,
            "matrix": matrix,
        }

    def build_candidate_index(self, features: dict[str, Any]) -> dict[str, Any] | None:
        return _build_nn_index(features["matrix"], features["pids"])

    def score_pairs_batch(
        self,
        features: dict[str, Any],
        pairs: np.ndarray,
        index: dict[str, Any] | None = None,
    ) -> np.ndarray:
        matrix = features["matrix"]
        distances = np.zeros(len(pairs), dtype=np.float64)
        for idx, (i, j) in enumerate(pairs):
            distances[idx] = timing_distance(matrix[i], matrix[j])
        return distances


class CompositeDistance(DistanceMetric):
    name = "composite"

    def __init__(
        self,
        components: list[DistanceMetric],
        weights: tuple[float, ...] | list[float] | None = None,
        normalize_weights: bool = True,
    ) -> None:
        if not components:
            raise ValueError("CompositeDistance requires at least one component metric")
        self.components = components
        if weights is None:
            w = np.ones(len(components), dtype=np.float64)
        else:
            if len(weights) != len(components):
                raise ValueError(
                    f"weights length ({len(weights)}) must match components ({len(components)})"
                )
            w = np.array(weights, dtype=np.float64)
        if normalize_weights:
            total = float(w.sum())
            if total <= 0.0:
                raise ValueError("weights must sum to a positive value")
            w = w / total
        self.weights = w

    def prepare_features(self, activities: pl.DataFrame) -> dict[str, Any]:
        comp_features = []
        for metric in self.components:
            comp_features.append(metric.prepare_features(activities))

        # Validate pid ordering consistency
        pids = comp_features[0]["pids"]
        for cf in comp_features[1:]:
            if cf["pids"] != pids:
                raise ValueError("Component metrics produced mismatched pid ordering")

        return {
            "pids": pids,
            "components": comp_features,
        }

    def build_candidate_index(self, features: dict[str, Any]) -> dict[str, Any] | None:
        component_indices: list[dict[str, Any]] = []
        for metric, component_features in zip(self.components, features["components"]):
            idx = metric.build_candidate_index(component_features)
            if idx is not None:
                component_indices.append(idx)

        if not component_indices:
            return None

        return {
            "type": "composite",
            "pids": features["pids"],
            "components": component_indices,
            "component_names": [metric.name for metric in self.components],
        }

    def score_pairs_batch(
        self,
        features: dict[str, Any],
        pairs: np.ndarray,
        index: dict[str, Any] | None = None,
    ) -> np.ndarray:
        comp_features = features["components"]
        distances = np.zeros(len(pairs), dtype=np.float64)

        for w, metric, cf in zip(self.weights, self.components, comp_features):
            d = metric.score_pairs_batch(cf, pairs, index)
            distances += float(w) * d

        return distances


class GPUCompositeDistance(CompositeDistance):
    """Composite distance scorer that evaluates pair batches on a torch device.

    The feature extraction path is shared with :class:`CompositeDistance`, but
    the per-pair scoring step is vectorised with torch so it can run on CUDA.
    Candidate indexes are still built on CPU when requested.
    """

    name = "composite_gpu"

    def __init__(
        self,
        components: list[DistanceMetric],
        weights: tuple[float, ...] | list[float] | None = None,
        normalize_weights: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            components=components,
            weights=weights,
            normalize_weights=normalize_weights,
        )
        self.device = torch.device(device)

    def to(self, device: str) -> "GPUCompositeDistance":
        """Move future pair scoring to *device*."""
        self.device = torch.device(device)
        return self

    def prepare_features(self, activities: pl.DataFrame) -> dict[str, Any]:
        features = super().prepare_features(activities)
        return {
            "pids": features["pids"],
            "components": [
                {
                    **component_features,
                    "matrix": _matrix_to_tensor(
                        component_features["matrix"], str(self.device)
                    ),
                }
                for component_features in features["components"]
            ],
        }

    def build_candidate_index(self, features: dict[str, Any]) -> dict[str, Any] | None:
        cpu_features = {
            "pids": features["pids"],
            "components": [
                _tensor_features_to_numpy(component_features)
                for component_features in features["components"]
            ],
        }
        return super().build_candidate_index(cpu_features)

    def _score_component_torch(
        self,
        metric: DistanceMetric,
        features: dict[str, Any],
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        matrix = features["matrix"]
        if not torch.is_tensor(matrix):
            matrix = _matrix_to_tensor(np.asarray(matrix), str(self.device))
        else:
            matrix = matrix.to(self.device)

        left = matrix[pairs[:, 0]]
        right = matrix[pairs[:, 1]]

        if metric.name in {"participation", "sequence"}:
            return torch.sum(torch.abs(left - right), dim=1) / 2.0
        if metric.name == "timing":
            return torch.mean((left != right).to(torch.float64), dim=1)

        # Fallback for unsupported custom component metrics.
        component_np = _tensor_features_to_numpy(features)
        pair_np = pairs.detach().cpu().numpy()
        return torch.as_tensor(
            metric.score_pairs_batch(component_np, pair_np, index=None),
            dtype=torch.float64,
            device=self.device,
        )

    def score_pairs_batch(
        self,
        features: dict[str, Any],
        pairs: np.ndarray,
        index: dict[str, Any] | None = None,
    ) -> np.ndarray:
        comp_features = features["components"]
        pair_tensor = torch.as_tensor(
            np.asarray(pairs, dtype=np.int64), device=self.device
        )
        distances = torch.zeros(
            pair_tensor.shape[0], dtype=torch.float64, device=self.device
        )

        for weight, metric, component_features in zip(
            self.weights, self.components, comp_features
        ):
            component_distance = self._score_component_torch(
                metric=metric,
                features=component_features,
                pairs=pair_tensor,
            )
            distances.add_(float(weight) * component_distance)

        return distances.detach().cpu().numpy()


def default_sequence_metric() -> TwoGramDistance:
    return TwoGramDistance()
