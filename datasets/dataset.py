"""PyTorch Dataset for contrastive schedule embedding training.

Serves (attributes, schedule_distance) pairs from a precomputed distance
matrix and a set of encoded attribute dictionaries.

Supported training modes
------------------------
pairwise
    Each sample is a pair ``(attrs_i, attrs_j, distance_ij)`` where *j* is
    sampled randomly from all other persons.  Suitable for distance-regression
    and NTXent losses.

triplet
    Each sample is a tuple
    ``(anchor, positive, negative, d_anchor_positive, d_anchor_negative)``
    where the positive has distance < *positive_threshold* and the negative
    has distance > *negative_threshold*.

single
    Each sample is ``(attrs, distances_row)`` where *distances_row* is shape
    ``(N,)``.  Used with :class:`~training.losses.SoftNearestNeighbourLoss`
    which needs an ``(B, B)`` distance matrix for the full batch.

Distance storage
----------------
For large datasets the N×N dense float64 matrix is impractical.
:class:`SparseDistanceMatrix` stores only the *k* nearest and *k* furthest
neighbours per person, reducing memory from O(N²) to O(N·k).

Hard negative mining
--------------------
:class:`HardNegativeSampler` maintains a periodically-refreshed k-NN index
in current embedding space.  Hard negatives are persons that are *close in
embedding space* (similar according to the model) but *far in schedule space*
(actually dissimilar).

Public API
----------
ScheduleEmbeddingDataset
LazyPairwiseDataset
SparseDistanceMatrix
HardNegativeSampler
collate_fn
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.masking import AttributeMasker
from distances.protocols import DistanceMetric


class SparseDistanceMatrix:
    """Compressed distance matrix storing only k nearest and k furthest neighbours.

    Parameters
    ----------
    k:
        Number of nearest (and furthest) neighbours to retain per person.
    """

    def __init__(self, k: int = 50) -> None:
        self.k = k
        self._near_idx: np.ndarray | None = None  # (N, k) int32
        self._near_dist: np.ndarray | None = None  # (N, k) float32
        self._far_idx: np.ndarray | None = None  # (N, k) int32
        self._far_dist: np.ndarray | None = None  # (N, k) float32
        self._n: int = 0

    @classmethod
    def from_dense(cls, D: np.ndarray, k: int = 50) -> "SparseDistanceMatrix":
        """Build a sparse matrix from a dense N×N distance array.

        Parameters
        ----------
        D:
            Symmetric (N, N) float distance matrix.  Diagonal should be 0.
        k:
            Number of neighbours to retain at each end.
        """
        n = D.shape[0]
        effective_k = min(k, n - 1)
        obj = cls(k=effective_k)
        obj._n = n

        near_idx = np.zeros((n, effective_k), dtype=np.int32)
        near_dist = np.zeros((n, effective_k), dtype=np.float32)
        far_idx = np.zeros((n, effective_k), dtype=np.int32)
        far_dist = np.zeros((n, effective_k), dtype=np.float32)

        for i in range(n):
            row = D[i].copy()
            row[i] = np.inf  # exclude self
            order = np.argsort(row)
            near_idx[i] = order[:effective_k].astype(np.int32)
            near_dist[i] = row[order[:effective_k]].astype(np.float32)
            # furthest: reverse the sorted order, skip inf-filled self
            finite_order = order[np.isfinite(row[order])]
            far_slice = finite_order[-effective_k:][::-1]
            actual_k = len(far_slice)
            if actual_k < effective_k:
                # Pad with -1 / inf if fewer than k valid neighbours
                far_idx[i, :actual_k] = far_slice.astype(np.int32)
                far_idx[i, actual_k:] = -1
                far_dist[i, :actual_k] = row[far_slice].astype(np.float32)
                far_dist[i, actual_k:] = np.inf
            else:
                far_idx[i] = far_slice.astype(np.int32)
                far_dist[i] = row[far_slice].astype(np.float32)

        obj._near_idx = near_idx
        obj._near_dist = near_dist
        obj._far_idx = far_idx
        obj._far_dist = far_dist
        return obj

    def get_distance(self, i: int, j: int) -> float:
        """Return the stored distance between persons *i* and *j*.

        Returns ``float('nan')`` if the pair is not stored.
        """
        if i == j:
            return 0.0
        # Check near neighbours of i
        if self._near_idx is not None:
            near_matches = np.where(self._near_idx[i] == j)[0]
            if len(near_matches) > 0:
                return float(self._near_dist[i, near_matches[0]])
        # Check far neighbours of i
        if self._far_idx is not None:
            far_matches = np.where(self._far_idx[i] == j)[0]
            if len(far_matches) > 0:
                return float(self._far_dist[i, far_matches[0]])
        return float("nan")

    def get_neighbours(self, i: int) -> np.ndarray:
        """Return indices of the k nearest neighbours of person *i*."""
        if self._near_idx is None:
            return np.array([], dtype=np.int32)
        return self._near_idx[i]

    def get_furthest(self, i: int) -> np.ndarray:
        """Return indices of the k furthest neighbours of person *i*."""
        if self._far_idx is None:
            return np.array([], dtype=np.int32)
        valid = self._far_idx[i]
        return valid[valid >= 0]

    def __len__(self) -> int:
        return self._n


class HardNegativeSampler:
    """Periodically-updated index for mining hard negatives.

    A hard negative for anchor *i* is a person that is *close in embedding
    space* (the model currently thinks they're similar) but *far in schedule
    space* (they're actually dissimilar according to the distance matrix).

    Call :meth:`refresh` after each training epoch (or every N steps) to
    rebuild the embedding-space index.

    Parameters
    ----------
    k:
        Number of nearest embedding-space neighbours to consider as
        hard-negative candidates.
    """

    def __init__(self, k: int = 50) -> None:
        self.k = k
        self._emb_near_idx: np.ndarray | None = None  # (N, k)

    def refresh(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> None:
        """Recompute embeddings for all persons and rebuild the k-NN index.

        Parameters
        ----------
        model:
            A callable ``model(attributes) -> Tensor(batch, d)`` that produces
            embeddings from attribute dictionaries.
        dataloader:
            DataLoader yielding ``(attributes, ...)`` tuples where the first
            element is a dict of attribute tensors (one row per person).
        device:
            Torch device string.
        """
        model.eval()
        all_embs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                attrs = batch[0] if isinstance(batch, (list, tuple)) else batch
                attrs = {k: v.to(device) for k, v in attrs.items()}
                embs = model(attrs)
                all_embs.append(embs.cpu())
        embeddings = torch.cat(all_embs, dim=0).numpy()  # (N, d)

        n = embeddings.shape[0]
        effective_k = min(self.k, n - 1)
        near_idx = np.zeros((n, effective_k), dtype=np.int32)

        # Simple brute-force; for large N use faiss
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normed = embeddings / norms
        sim = normed @ normed.T  # (N, N) cosine similarity
        np.fill_diagonal(sim, -np.inf)
        for i in range(n):
            near_idx[i] = np.argsort(sim[i])[::-1][:effective_k].astype(np.int32)

        self._emb_near_idx = near_idx

    def sample_hard_negative(
        self,
        anchor_idx: int,
        distance_matrix: np.ndarray | SparseDistanceMatrix,
        schedule_threshold: float = 0.5,
    ) -> int:
        """Sample a hard negative for *anchor_idx*.

        Looks for embedding-space neighbours that exceed *schedule_threshold*
        in schedule distance.  Falls back to a random far neighbour if the
        index hasn't been refreshed.

        Parameters
        ----------
        anchor_idx:
            Index of the anchor person.
        distance_matrix:
            Dense ndarray or :class:`SparseDistanceMatrix`.
        schedule_threshold:
            Minimum schedule distance for a valid hard negative.
        """
        if self._emb_near_idx is None:
            raise RuntimeError("Call refresh() before sample_hard_negative().")

        candidates = self._emb_near_idx[anchor_idx]
        for cand in candidates:
            if isinstance(distance_matrix, np.ndarray):
                d = float(distance_matrix[anchor_idx, cand])
            else:
                d = distance_matrix.get_distance(anchor_idx, int(cand))
            if d > schedule_threshold:
                return int(cand)

        # Fallback: return last candidate
        return int(candidates[-1])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ScheduleEmbeddingDataset(Dataset):
    """PyTorch Dataset yielding contrastive training samples.

    Parameters
    ----------
    attributes:
        ``dict[str, Tensor]`` where each tensor has shape ``(N,)`` and
        corresponds to one attribute across all *N* persons.
    distance_matrix:
        Dense ``(N, N)`` float64 ndarray **or** a
        :class:`SparseDistanceMatrix`.
    masker:
        Optional :class:`~datasets.masking.AttributeMasker` applied
        independently to each returned attribute set.
    mode:
        ``"pairwise"`` or ``"triplet"``.
    positive_threshold:
        (triplet mode) Maximum distance for a positive pair.
    negative_threshold:
        (triplet mode) Minimum distance for a negative pair.
    sampling_strategy:
        ``"random"`` (always) or ``"hard_negative"`` (requires a refreshed
        :class:`HardNegativeSampler` passed via *hard_negative_sampler*).
    hard_negative_sampler:
        Required when *sampling_strategy* is ``"hard_negative"``.
    """

    def __init__(
        self,
        attributes: dict[str, torch.Tensor],
        distance_matrix: np.ndarray | SparseDistanceMatrix,
        masker: AttributeMasker | None = None,
        mode: Literal["pairwise", "triplet", "single"] = "pairwise",
        positive_threshold: float = 0.2,
        negative_threshold: float = 0.5,
        sampling_strategy: Literal["random", "hard_negative"] = "random",
        hard_negative_sampler: HardNegativeSampler | None = None,
    ) -> None:
        self.attributes = attributes
        self.distance_matrix = distance_matrix
        self.masker = masker
        self.mode = mode
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.sampling_strategy = sampling_strategy
        self.hard_negative_sampler = hard_negative_sampler

        # Determine N from the first attribute tensor
        self._n = next(iter(attributes.values())).shape[0]

        # Precompute positive/negative candidate pools for triplet mode.
        # Each pool[i] is a sorted array of valid candidate indices for anchor i.
        # O(N²) at init but O(1) per sample — only built for dense matrices.
        self._positive_pool: list[np.ndarray] | None = None
        self._negative_pool: list[np.ndarray] | None = None
        if self.mode == "triplet" and isinstance(self.distance_matrix, np.ndarray):
            idx = np.arange(self._n)
            self._positive_pool = []
            self._negative_pool = []
            for i in range(self._n):
                row = self.distance_matrix[i]
                self._positive_pool.append(
                    np.where((row < self.positive_threshold) & (idx != i))[0]
                )
                self._negative_pool.append(
                    np.where((row > self.negative_threshold) & (idx != i))[0]
                )

    def __len__(self) -> int:
        return self._n

    def _get_attrs(self, idx: int) -> dict[str, torch.Tensor]:
        """Return the attribute dict for person *idx*, applying masker if set."""
        attrs = {k: v[idx] for k, v in self.attributes.items()}
        if self.masker is not None:
            attrs = self.masker(attrs)
        return attrs

    def _get_distance(self, i: int, j: int) -> float:
        if isinstance(self.distance_matrix, np.ndarray):
            return float(self.distance_matrix[i, j])
        return self.distance_matrix.get_distance(i, j)

    def _sample_positive(self, anchor: int) -> int:
        """Sample a person within *positive_threshold* of *anchor*."""
        if self._positive_pool is not None:
            pool = self._positive_pool[anchor]
            if len(pool) > 0:
                return int(np.random.choice(pool))
        # Fallback: O(N) scan (pool empty or not precomputed)
        n = self._n
        indices = list(range(n))
        indices.remove(anchor)
        np.random.shuffle(indices)
        for j in indices:
            if self._get_distance(anchor, j) < self.positive_threshold:
                return j
        return min(indices, key=lambda j: self._get_distance(anchor, j))

    def _sample_negative(self, anchor: int) -> int:
        """Sample a person beyond *negative_threshold* from *anchor*."""
        if (
            self.sampling_strategy == "hard_negative"
            and self.hard_negative_sampler is not None
        ):
            return self.hard_negative_sampler.sample_hard_negative(
                anchor, self.distance_matrix, self.negative_threshold
            )
        if self._negative_pool is not None:
            pool = self._negative_pool[anchor]
            if len(pool) > 0:
                return int(np.random.choice(pool))
        # Fallback: O(N) scan (pool empty or not precomputed)
        n = self._n
        indices = list(range(n))
        indices.remove(anchor)
        np.random.shuffle(indices)
        for j in indices:
            if self._get_distance(anchor, j) > self.negative_threshold:
                return j
        return max(indices, key=lambda j: self._get_distance(anchor, j))

    def __getitem__(self, idx: int):
        if self.mode == "pairwise":
            return self._getitem_pairwise(idx)
        elif self.mode == "triplet":
            return self._getitem_triplet(idx)
        elif self.mode == "single":
            return self._getitem_single(idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    def _getitem_pairwise(self, idx: int):
        j = idx
        while j == idx:
            j = int(np.random.randint(0, self._n))
        attrs_i = self._get_attrs(idx)
        attrs_j = self._get_attrs(j)
        dist = torch.tensor(self._get_distance(idx, j), dtype=torch.float32)
        return attrs_i, attrs_j, dist

    def _getitem_single(self, idx: int):
        """Return anchor attributes, their full distance row, and their dataset index.

        Yields ``(attrs, distances_row, idx)`` where *distances_row* is a float32
        tensor of shape ``(N,)`` containing distances from *idx* to all other
        persons.  The dataset index *idx* is included so the trainer can
        construct an exact (B, B) schedule-distance sub-matrix from the
        collated (B, N) block.  Suitable for
        :class:`~training.losses.SoftNearestNeighbourLoss`.
        """
        attrs = self._get_attrs(idx)
        n = self._n
        if isinstance(self.distance_matrix, np.ndarray):
            row = torch.tensor(self.distance_matrix[idx].astype(np.float32))
        else:
            row = torch.full((n,), float("nan"), dtype=torch.float32)
            for j in range(n):
                row[j] = self.distance_matrix.get_distance(idx, j)
        return attrs, row, idx

    def _getitem_triplet(self, idx: int):
        pos = self._sample_positive(idx)
        neg = self._sample_negative(idx)
        anchor = self._get_attrs(idx)
        positive = self._get_attrs(pos)
        negative = self._get_attrs(neg)
        d_ap = torch.tensor(self._get_distance(idx, pos), dtype=torch.float32)
        d_an = torch.tensor(self._get_distance(idx, neg), dtype=torch.float32)
        return anchor, positive, negative, d_ap, d_an


class LazyPairwiseDataset(Dataset):
    """Pairwise dataset with distances computed lazily via a metric plugin.

    This backend avoids precomputing and storing an explicit N×N matrix.
    For each sampled pair, distance is computed from V2 metric features using
    ``score_pairs_batch`` on a single pair.

    Parameters
    ----------
    distance_device:
        Device used for metric feature tensors.  ``"cpu"`` keeps the current
        NumPy-based path; ``"cuda"`` moves supported feature arrays to torch
        tensors once during initialisation.
    """

    def __init__(
        self,
        attributes: dict[str, torch.Tensor],
        global_indices: np.ndarray,
        metric: DistanceMetric,
        metric_features: dict[str, Any],
        candidate_index: dict[str, Any] | None = None,
        masker: AttributeMasker | None = None,
        distance_cache: dict[tuple[int, int], float] | None = None,
        max_cached_pairs: int = 500_000,
        distance_device: str = "cpu",
    ) -> None:
        self.attributes = attributes
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.metric = metric
        self.distance_device = distance_device
        self.metric_features = self._prepare_metric_features(metric_features)
        self.candidate_index = candidate_index
        self.masker = masker
        self.distance_cache = distance_cache if distance_cache is not None else {}
        self.max_cached_pairs = max_cached_pairs
        self._n = next(iter(attributes.values())).shape[0]
        if self.global_indices.shape[0] != self._n:
            raise ValueError(
                f"global_indices length ({self.global_indices.shape[0]}) must match "
                f"number of local samples ({self._n})"
            )

    def _prepare_metric_features(
        self, metric_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Move metric feature arrays to the requested device when supported."""
        if self.distance_device == "cpu":
            return metric_features

        def _move(value: Any) -> Any:
            if isinstance(value, np.ndarray) and value.dtype != object:
                return torch.as_tensor(value, device=self.distance_device)
            if isinstance(value, dict):
                return {k: _move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_move(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_move(v) for v in value)
            return value

        moved = _move(metric_features)
        if hasattr(self.metric, "to"):
            try:
                self.metric = self.metric.to(self.distance_device)
            except Exception:
                # Keep the metric usable even if it does not support device moves.
                pass
        return moved

    def __len__(self) -> int:
        return self._n

    def _get_attrs(self, idx: int) -> dict[str, torch.Tensor]:
        attrs = {k: v[idx] for k, v in self.attributes.items()}
        if self.masker is not None:
            attrs = self.masker(attrs)
        return attrs

    def __getitem__(self, idx: int):
        j = idx
        while j == idx:
            j = int(np.random.randint(0, self._n))

        gi = int(self.global_indices[idx])
        gj = int(self.global_indices[j])
        cache_key = (gi, gj) if gi < gj else (gj, gi)
        cached = self.distance_cache.get(cache_key)
        if cached is None:
            pair = np.array([[gi, gj]], dtype=np.int32)
            d = float(
                self.metric.score_pairs_batch(
                    self.metric_features,
                    pair,
                    self.candidate_index,
                )[0]
            )
            if len(self.distance_cache) >= self.max_cached_pairs:
                # Keep implementation simple: evict one arbitrary item.
                self.distance_cache.pop(next(iter(self.distance_cache)))
            self.distance_cache[cache_key] = d
        else:
            d = float(cached)

        attrs_i = self._get_attrs(idx)
        attrs_j = self._get_attrs(j)
        dist = torch.tensor(d, dtype=torch.float32)
        return attrs_i, attrs_j, dist


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_fn(batch):
    """Collate a list of dataset samples into batched tensors.

    Supports both pairwise samples
    ``(attrs_i, attrs_j, dist)``
    and triplet samples
    ``(anchor, positive, negative, d_ap, d_an)``.

    Each ``attrs_*`` is a ``dict[str, Tensor]``; the collated form stacks
    each tensor along a new batch dimension (dim 0).
    """
    if len(batch[0]) == 3:
        if isinstance(batch[0][1], dict):
            # Pairwise: (attrs_i, attrs_j, dist)
            attrs_i_list, attrs_j_list, dists = zip(*batch)
            return (
                _collate_attr_dicts(attrs_i_list),
                _collate_attr_dicts(attrs_j_list),
                torch.stack(dists),
            )
        else:
            # Single-with-index: (attrs, dist_row, idx)
            attrs_list, rows, idxs = zip(*batch)
            return (
                _collate_attr_dicts(attrs_list),
                torch.stack(rows),  # (B, N)
                torch.tensor(idxs, dtype=torch.long),  # (B,)
            )
    elif len(batch[0]) == 5:
        # Triplet: (anchor, positive, negative, d_ap, d_an)
        anchors, positives, negatives, d_aps, d_ans = zip(*batch)
        return (
            _collate_attr_dicts(anchors),
            _collate_attr_dicts(positives),
            _collate_attr_dicts(negatives),
            torch.stack(d_aps),
            torch.stack(d_ans),
        )
    else:
        raise ValueError(f"Unexpected batch element length: {len(batch[0])}")


def _collate_attr_dicts(
    dicts: tuple[dict[str, torch.Tensor], ...],
) -> dict[str, torch.Tensor]:
    """Stack a sequence of scalar-valued attribute dicts into batched dicts."""
    keys = dicts[0].keys()
    return {k: torch.stack([d[k] for d in dicts], dim=0) for k in keys}
