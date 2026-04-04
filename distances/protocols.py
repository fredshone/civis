"""Protocols and base types for pluggable schedule distance metrics.

All metrics use the scalable V2 API for efficient feature extraction and batch scoring.

Metrics implement:
- prepare_features(activities) → precomputed features dict
- build_candidate_index(features) → optional ANN index
- score_pairs_batch(features, pairs, index) → distance array

This enables O(N·k) storage and computation instead of O(N²).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl


class DistanceMetric(ABC):
    """Abstract base class for scalable schedule distance metrics.

    All metrics extract precomputed features once, optionally build a
    candidate index, then score pairs in batches.
    """

    name: str

    @abstractmethod
    def prepare_features(self, activities: pl.DataFrame) -> dict[str, Any]:
        """Extract precomputed features for scalable distance computation.

        Parameters
        ----------
        activities : pl.DataFrame
            Activities data.

        Returns
        -------
        dict[str, Any]
            Feature dict (schema is metric-specific). Example:
            ``{"pids": list[str], "participation": ndarray, ...}``
        """

    @abstractmethod
    def score_pairs_batch(
        self,
        features: dict[str, Any],
        pairs: np.ndarray,
        index: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Score a batch of schedule index pairs using precomputed features.

        Parameters
        ----------
        features : dict[str, Any]
            Output from prepare_features().
        pairs : np.ndarray, shape (M, 2), int32
            Schedule index pairs to score.
        index : dict[str, Any] | None
            Optional index from build_candidate_index(), or None.

        Returns
        -------
        np.ndarray, shape (M,), float64
            Distance for each pair.
        """

    def build_candidate_index(self, features: dict[str, Any]) -> dict[str, Any] | None:
        """Build an optional ANN or brute-force index for candidate generation.

        Override to enable efficient approximate nearest neighbour queries.
        Return None to use brute-force at scoring time.

        Parameters
        ----------
        features : dict[str, Any]
            Output from prepare_features().

        Returns
        -------
        dict[str, Any] | None
            Index object (schema is metric-specific), or None for brute-force.
        """
        return None
