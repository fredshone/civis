"""Distance cache based on sparse edge lists and neighbor graphs.

Replaces the dense N×N matrix representation with sharded edge lists and
k-neighbor indices, enabling storage of distance information for datasets
where a full dense matrix is infeasible.

Public API
----------
DistanceGraphManifest
build_distance_graph
load_distance_graph
DistanceGraph
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
from typing import Any

import numpy as np

from distances.feature_store import FeatureManifest, feature_manifest_hash


@dataclass
class DistanceGraphManifest:
    """Metadata for a cached distance graph.

    Parameters
    ----------
    feature_manifest_hash : str
        Hash of the FeatureManifest used to compute distances.
    metric_spec : str
        Description of the distance metric (e.g. "composite", weights).
    metric_hash : str
        Content hash of the metric specification.
    extraction_timestamp : str
        ISO 8601 timestamp when distances were computed.
    n_persons : int
        Number of unique persons in the graph.
    k_neighbors : int
        Number of nearest and furthest neighbors stored per person.
    graph_dir : str
        Directory where graph artifacts are stored.
    """

    feature_manifest_hash: str
    metric_spec: str
    metric_hash: str
    extraction_timestamp: str
    n_persons: int
    k_neighbors: int
    graph_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DistanceGraphManifest":
        return DistanceGraphManifest(**d)


@dataclass
class DistanceGraph:
    """Sparse distance graph with k-nearest and k-furthest neighbors.

    Parameters
    ----------
    pids : list[str]
        Person identifiers in sorted order.
    near_idx : np.ndarray, shape (N, k), int32
        Indices of k nearest neighbors for each person.
    near_dist : np.ndarray, shape (N, k), float32
        Distances to k nearest neighbors.
    far_idx : np.ndarray, shape (N, k), int32
        Indices of k furthest neighbors for each person.
    far_dist : np.ndarray, shape (N, k), float32
        Distances to k furthest neighbors.
    manifest : DistanceGraphManifest
        Metadata and versioning info.
    """

    pids: list[str]
    near_idx: np.ndarray
    near_dist: np.ndarray
    far_idx: np.ndarray
    far_dist: np.ndarray
    manifest: DistanceGraphManifest

    def get_distance(self, i: int, j: int) -> float:
        """Return the distance between persons i and j, or NaN if not stored."""
        if i == j:
            return 0.0

        # Check near neighbors of i
        near_matches = np.where(self.near_idx[i] == j)[0]
        if len(near_matches) > 0:
            return float(self.near_dist[i, near_matches[0]])

        # Check far neighbors of i
        far_matches = np.where(self.far_idx[i] == j)[0]
        if len(far_matches) > 0:
            return float(self.far_dist[i, far_matches[0]])

        return float("nan")

    def get_neighbors(self, i: int, kind: str = "near") -> np.ndarray:
        """Return indices of k neighbors for person i.

        Parameters
        ----------
        i : int
            Person index.
        kind : str
            ``"near"`` or ``"far"``.

        Returns
        -------
        np.ndarray, dtype int32
            Neighbor indices (including -1 for unfilled slots).
        """
        if kind == "near":
            return self.near_idx[i]
        elif kind == "far":
            return self.far_idx[i]
        else:
            raise ValueError(f"kind must be 'near' or 'far', got {kind!r}")


def build_distance_graph(
    pids: list[str],
    distance_matrix: np.ndarray,
    metric_spec: str,
    feature_manifest: FeatureManifest,
    graph_dir: str | Path,
    k: int = 500,
    overwrite: bool = False,
) -> DistanceGraph:
    """Build and cache a sparse distance graph from a dense matrix.

    Parameters
    ----------
    pids : list[str]
        Person identifiers (must match rows/cols of distance_matrix).
    distance_matrix : np.ndarray, shape (N, N), float64
        Symmetric distance matrix.
    metric_spec : str
        Human-readable metric specification (e.g., "composite: w=(0.33, 0.33, 0.34)").
    feature_manifest : FeatureManifest
        Feature manifest used to compute these distances.
    graph_dir : str | Path
        Directory where graph artifacts will be saved.
    k : int
        Number of nearest and furthest neighbors to retain per person.
    overwrite : bool
        If False and graph exists, load cached version.
        If True, recompute and overwrite.

    Returns
    -------
    DistanceGraph
        Sparse graph with manifest.
    """
    from datetime import datetime, timezone

    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = graph_dir / "manifest.json"

    # Check if cached graph exists
    if manifest_path.exists() and not overwrite:
        return load_distance_graph(graph_dir)

    n = len(pids)
    effective_k = min(k, n - 1)

    near_idx = np.zeros((n, effective_k), dtype=np.int32)
    near_dist = np.zeros((n, effective_k), dtype=np.float32)
    far_idx = np.zeros((n, effective_k), dtype=np.int32)
    far_dist = np.zeros((n, effective_k), dtype=np.float32)

    for i in range(n):
        row = distance_matrix[i].copy()
        row[i] = np.inf  # exclude self
        order = np.argsort(row)

        # Nearest neighbors
        near_idx[i] = order[:effective_k].astype(np.int32)
        near_dist[i] = row[order[:effective_k]].astype(np.float32)

        # Furthest neighbors (reverse order, excluding inf)
        finite_order = order[np.isfinite(row[order])]
        far_slice = finite_order[-effective_k:][::-1]
        actual_k = len(far_slice)

        if actual_k < effective_k:
            far_idx[i, :actual_k] = far_slice.astype(np.int32)
            far_idx[i, actual_k:] = -1
            far_dist[i, :actual_k] = row[far_slice].astype(np.float32)
            far_dist[i, actual_k:] = np.inf
        else:
            far_idx[i] = far_slice.astype(np.int32)
            far_dist[i] = row[far_slice].astype(np.float32)

    # Compute metric hash
    metric_hash = hashlib.sha256(metric_spec.encode("utf-8")).hexdigest()

    manifest = DistanceGraphManifest(
        feature_manifest_hash=feature_manifest_hash(feature_manifest),
        metric_spec=metric_spec,
        metric_hash=metric_hash,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        n_persons=n,
        k_neighbors=effective_k,
        graph_dir=str(graph_dir),
    )

    # Save graph
    graph_path = graph_dir / "graph.npz"
    np.savez_compressed(
        graph_path,
        pids=np.array(pids, dtype=object),
        near_idx=near_idx,
        near_dist=near_dist,
        far_idx=far_idx,
        far_dist=far_dist,
    )

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return DistanceGraph(
        pids=pids,
        near_idx=near_idx,
        near_dist=near_dist,
        far_idx=far_idx,
        far_dist=far_dist,
        manifest=manifest,
    )


def load_distance_graph(graph_dir: str | Path) -> DistanceGraph:
    """Load a cached distance graph.

    Parameters
    ----------
    graph_dir : str | Path
        Directory containing manifest.json and graph.npz.

    Returns
    -------
    DistanceGraph
        Loaded graph with manifest.

    Raises
    ------
    FileNotFoundError
        If manifest or graph files are missing.
    """
    graph_dir = Path(graph_dir)

    # Load manifest
    manifest_path = graph_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest_dict = json.load(f)
    manifest = DistanceGraphManifest.from_dict(manifest_dict)

    # Load graph
    graph_path = graph_dir / "graph.npz"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found at {graph_path}")

    data = np.load(graph_path, allow_pickle=True)
    pids = list(data["pids"])
    near_idx = data["near_idx"].astype(np.int32)
    near_dist = data["near_dist"].astype(np.float32)
    far_idx = data["far_idx"].astype(np.int32)
    far_dist = data["far_dist"].astype(np.float32)

    return DistanceGraph(
        pids=pids,
        near_idx=near_idx,
        near_dist=near_dist,
        far_idx=far_idx,
        far_dist=far_dist,
        manifest=manifest,
    )
