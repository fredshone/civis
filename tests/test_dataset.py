"""Tests for datasets/dataset.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from datasets.dataset import (
    HardNegativeSampler,
    LazyPairwiseDataset,
    ScheduleEmbeddingDataset,
    SparseDistanceMatrix,
    collate_fn,
)
from datasets.masking import AttributeMasker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_distance_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n, n)).astype(np.float64)
    D = (raw + raw.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def _make_attributes(n: int) -> dict[str, torch.Tensor]:
    return {
        "sex": torch.randint(0, 3, (n,)),
        "age": torch.rand(n),
        "country": torch.randint(0, 5, (n,)),
    }


# ---------------------------------------------------------------------------
# SparseDistanceMatrix
# ---------------------------------------------------------------------------


class TestSparseDistanceMatrix:
    def test_from_dense_shape(self):
        D = _make_distance_matrix(10)
        sparse = SparseDistanceMatrix.from_dense(D, k=3)
        assert sparse._near_idx.shape == (10, 3)
        assert sparse._far_idx.shape == (10, 3)

    def test_self_distance_is_zero(self):
        D = _make_distance_matrix(5)
        sparse = SparseDistanceMatrix.from_dense(D, k=2)
        for i in range(5):
            assert sparse.get_distance(i, i) == 0.0

    def test_near_distances_match_dense(self):
        D = _make_distance_matrix(8)
        sparse = SparseDistanceMatrix.from_dense(D, k=4)
        for i in range(8):
            for idx, j in enumerate(sparse._near_idx[i]):
                stored = sparse.get_distance(i, int(j))
                assert stored == pytest.approx(D[i, j], abs=1e-5)

    def test_get_neighbours_returns_array(self):
        D = _make_distance_matrix(6)
        sparse = SparseDistanceMatrix.from_dense(D, k=2)
        nb = sparse.get_neighbours(0)
        assert isinstance(nb, np.ndarray)
        assert len(nb) == 2

    def test_get_furthest_returns_array(self):
        D = _make_distance_matrix(6)
        sparse = SparseDistanceMatrix.from_dense(D, k=2)
        fb = sparse.get_furthest(0)
        assert isinstance(fb, np.ndarray)
        assert len(fb) >= 1

    def test_furthest_farther_than_nearest(self):
        D = _make_distance_matrix(10)
        sparse = SparseDistanceMatrix.from_dense(D, k=3)
        for i in range(10):
            near = sparse._near_dist[i]
            far = sparse._far_dist[i, sparse._far_idx[i] >= 0]
            if len(far) > 0 and len(near) > 0:
                assert near.min() <= far.max() + 1e-6

    def test_len(self):
        D = _make_distance_matrix(7)
        sparse = SparseDistanceMatrix.from_dense(D, k=3)
        assert len(sparse) == 7

    def test_k_larger_than_n_clamped(self):
        D = _make_distance_matrix(3)
        sparse = SparseDistanceMatrix.from_dense(D, k=100)
        assert sparse.k <= 2


# ---------------------------------------------------------------------------
# ScheduleEmbeddingDataset — pairwise mode
# ---------------------------------------------------------------------------


class TestPairwiseDataset:
    @pytest.fixture()
    def dataset(self):
        n = 8
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)
        return ScheduleEmbeddingDataset(attrs, D, mode="pairwise")

    def test_len(self, dataset):
        assert len(dataset) == 8

    def test_getitem_returns_triple(self, dataset):
        sample = dataset[0]
        assert len(sample) == 3

    def test_attrs_are_dicts(self, dataset):
        attrs_i, attrs_j, dist = dataset[0]
        assert isinstance(attrs_i, dict)
        assert isinstance(attrs_j, dict)

    def test_distance_is_scalar_tensor(self, dataset):
        _, _, dist = dataset[0]
        assert isinstance(dist, torch.Tensor)
        assert dist.shape == ()

    def test_distance_in_unit_interval(self, dataset):
        for idx in range(len(dataset)):
            _, _, dist = dataset[idx]
            assert 0.0 <= dist.item() <= 1.0 + 1e-6

    def test_distance_matches_matrix(self):
        n = 6
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)
        ds = ScheduleEmbeddingDataset(attrs, D, mode="pairwise")
        # We can't control j directly, but we can verify the distance value is valid
        for idx in range(n):
            _, _, dist = ds[idx]
            assert any(
                abs(dist.item() - D[idx, j]) < 1e-5 for j in range(n) if j != idx
            )

    def test_masker_applied(self):
        n = 6
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)
        # prob=1.0 means sex always zeroed out
        masker = AttributeMasker({"sex": 1.0})
        ds = ScheduleEmbeddingDataset(attrs, D, masker=masker, mode="pairwise")
        attrs_i, attrs_j, _ = ds[0]
        assert attrs_i["sex"].item() == 0
        assert attrs_j["sex"].item() == 0


# ---------------------------------------------------------------------------
# ScheduleEmbeddingDataset — triplet mode
# ---------------------------------------------------------------------------


class TestTripletDataset:
    @pytest.fixture()
    def dataset(self):
        n = 12
        attrs = _make_attributes(n)
        # Structured D: first 6 persons close to each other, last 6 far
        D = np.ones((n, n)) * 0.8
        D[:6, :6] = 0.1
        D[6:, 6:] = 0.1
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2
        return ScheduleEmbeddingDataset(
            attrs,
            D,
            mode="triplet",
            positive_threshold=0.2,
            negative_threshold=0.5,
        )

    def test_getitem_returns_5tuple(self, dataset):
        sample = dataset[0]
        assert len(sample) == 5

    def test_positive_distance_below_threshold(self, dataset):
        for idx in range(6):
            _, _, _, d_ap, _ = dataset[idx]
            assert (
                d_ap.item() < 0.2 + 1e-5
            ), f"Positive distance {d_ap.item()} >= threshold 0.2"

    def test_negative_distance_above_threshold(self, dataset):
        for idx in range(6):
            _, _, _, _, d_an = dataset[idx]
            assert (
                d_an.item() > 0.5 - 1e-5
            ), f"Negative distance {d_an.item()} <= threshold 0.5"

    def test_distance_tensors_are_scalar(self, dataset):
        _, _, _, d_ap, d_an = dataset[0]
        assert d_ap.shape == ()
        assert d_an.shape == ()

    def test_masker_applied_to_all_three(self):
        n = 12
        attrs = _make_attributes(n)
        D = np.ones((n, n)) * 0.8
        D[:6, :6] = 0.1
        D[6:, 6:] = 0.1
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2
        masker = AttributeMasker({"sex": 1.0})
        ds = ScheduleEmbeddingDataset(
            attrs,
            D,
            masker=masker,
            mode="triplet",
            positive_threshold=0.2,
            negative_threshold=0.5,
        )
        anchor, pos, neg, _, _ = ds[0]
        assert anchor["sex"].item() == 0
        assert pos["sex"].item() == 0
        assert neg["sex"].item() == 0


# ---------------------------------------------------------------------------
# SparseDistanceMatrix with dataset
# ---------------------------------------------------------------------------


class TestSparseWithDataset:
    def test_pairwise_with_sparse_matrix(self):
        n = 8
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)
        sparse = SparseDistanceMatrix.from_dense(D, k=4)
        ds = ScheduleEmbeddingDataset(attrs, sparse, mode="pairwise")
        # Should not raise; distances may be nan for unstored pairs
        for idx in range(n):
            attrs_i, attrs_j, dist = ds[idx]
            assert isinstance(dist, torch.Tensor)


# ---------------------------------------------------------------------------
# LazyPairwiseDataset
# ---------------------------------------------------------------------------


class TestLazyPairwiseDataset:
    def test_getitem_returns_pairwise_sample(self):
        class L1Metric:
            def score_pairs_batch(self, features, pairs, index=None):
                matrix = features["matrix"]
                left = matrix[pairs[:, 0]]
                right = matrix[pairs[:, 1]]
                return np.sum(np.abs(left - right), axis=1)

        n = 10
        attrs = _make_attributes(n)
        global_idx = np.arange(n, dtype=np.int64)
        matrix = np.random.default_rng(0).random((n, 4)).astype(np.float64)
        metric_features = {"matrix": matrix}

        ds = LazyPairwiseDataset(
            attributes=attrs,
            global_indices=global_idx,
            metric=L1Metric(),
            metric_features=metric_features,
            candidate_index=None,
            masker=None,
        )

        a_i, a_j, d = ds[0]
        assert isinstance(a_i, dict)
        assert isinstance(a_j, dict)
        assert isinstance(d, torch.Tensor)
        assert d.shape == ()

    def test_distance_device_cpu_keeps_numpy_features(self):
        class DeviceAwareMetric:
            def __init__(self):
                self.to_calls: list[str] = []

            def to(self, device: str):
                self.to_calls.append(device)
                return self

            def score_pairs_batch(self, features, pairs, index=None):
                assert isinstance(features["matrix"], np.ndarray)
                matrix = features["matrix"]
                left = matrix[pairs[:, 0]]
                right = matrix[pairs[:, 1]]
                return np.sum(np.abs(left - right), axis=1)

        n = 6
        attrs = _make_attributes(n)
        global_idx = np.arange(n, dtype=np.int64)
        matrix = np.random.default_rng(3).random((n, 4)).astype(np.float64)
        metric = DeviceAwareMetric()

        ds = LazyPairwiseDataset(
            attributes=attrs,
            global_indices=global_idx,
            metric=metric,
            metric_features={"matrix": matrix},
            distance_device="cpu",
        )

        _ = ds[0]
        assert metric.to_calls == []

    def test_masker_is_applied(self):
        class L1Metric:
            def score_pairs_batch(self, features, pairs, index=None):
                matrix = features["matrix"]
                left = matrix[pairs[:, 0]]
                right = matrix[pairs[:, 1]]
                return np.sum(np.abs(left - right), axis=1)

        n = 8
        attrs = _make_attributes(n)
        global_idx = np.arange(n, dtype=np.int64)
        matrix = np.random.default_rng(1).random((n, 3)).astype(np.float64)
        masker = AttributeMasker({"sex": 1.0})
        ds = LazyPairwiseDataset(
            attributes=attrs,
            global_indices=global_idx,
            metric=L1Metric(),
            metric_features={"matrix": matrix},
            masker=masker,
        )

        a_i, a_j, _ = ds[0]
        assert a_i["sex"].item() == 0
        assert a_j["sex"].item() == 0

    def test_global_index_length_must_match_attributes(self):
        class DummyMetric:
            def score_pairs_batch(self, features, pairs, index=None):
                return np.zeros(len(pairs), dtype=np.float64)

        n = 5
        attrs = _make_attributes(n)
        with pytest.raises(ValueError, match="global_indices length"):
            LazyPairwiseDataset(
                attributes=attrs,
                global_indices=np.arange(n - 1),
                metric=DummyMetric(),
                metric_features={},
            )

    def test_distance_cache_reuses_computed_pairs(self, monkeypatch):
        class CountingMetric:
            def __init__(self):
                self.calls = 0

            def score_pairs_batch(self, features, pairs, index=None):
                self.calls += 1
                matrix = features["matrix"]
                left = matrix[pairs[:, 0]]
                right = matrix[pairs[:, 1]]
                return np.sum(np.abs(left - right), axis=1)

        n = 2
        attrs = _make_attributes(n)
        global_idx = np.arange(n, dtype=np.int64)
        matrix = np.random.default_rng(2).random((n, 3)).astype(np.float64)
        metric = CountingMetric()
        shared_cache: dict[tuple[int, int], float] = {}

        ds = LazyPairwiseDataset(
            attributes=attrs,
            global_indices=global_idx,
            metric=metric,
            metric_features={"matrix": matrix},
            distance_cache=shared_cache,
        )

        # Pre-fill cache for pair (0,1) and force sampled j=1.
        key = (0, 1)
        pair = np.array([[0, 1]], dtype=np.int32)
        d1 = metric.score_pairs_batch({"matrix": matrix}, pair)[0]
        shared_cache[key] = float(d1)
        calls_after_first = metric.calls

        monkeypatch.setattr(np.random, "randint", lambda low, high=None: 1)
        _ = ds[0]
        calls_after_second = metric.calls

        assert calls_after_second == calls_after_first


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


class TestCollateFn:
    def test_pairwise_collate(self):
        n = 6
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)
        ds = ScheduleEmbeddingDataset(attrs, D, mode="pairwise")
        dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(dl))
        assert len(batch) == 3
        attrs_i_batch, attrs_j_batch, dists = batch
        assert isinstance(attrs_i_batch, dict)
        assert isinstance(attrs_j_batch, dict)
        assert dists.shape == (4,)

    def test_pairwise_batch_attrs_have_batch_dim(self):
        n = 6
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)
        ds = ScheduleEmbeddingDataset(attrs, D, mode="pairwise")
        dl = DataLoader(ds, batch_size=3, collate_fn=collate_fn)
        batch = next(iter(dl))
        attrs_i_batch = batch[0]
        for key, tensor in attrs_i_batch.items():
            assert (
                tensor.shape[0] == 3
            ), f"{key}: expected batch_size=3, got {tensor.shape}"

    def test_triplet_collate(self):
        n = 12
        attrs = _make_attributes(n)
        D = np.ones((n, n)) * 0.8
        D[:6, :6] = 0.1
        D[6:, 6:] = 0.1
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2
        ds = ScheduleEmbeddingDataset(
            attrs,
            D,
            mode="triplet",
            positive_threshold=0.2,
            negative_threshold=0.5,
        )
        dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(dl))
        assert len(batch) == 5
        anchors, positives, negatives, d_aps, d_ans = batch
        assert d_aps.shape == (4,)
        assert d_ans.shape == (4,)

    def test_dataloader_iterates_fully(self):
        n = 8
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)
        ds = ScheduleEmbeddingDataset(attrs, D, mode="pairwise")
        dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        count = sum(1 for _ in dl)
        assert count == 4  # 8 / 2


# ---------------------------------------------------------------------------
# HardNegativeSampler (basic)
# ---------------------------------------------------------------------------


class TestHardNegativeSampler:
    def test_refresh_raises_before_call(self):
        sampler = HardNegativeSampler(k=3)
        D = _make_distance_matrix(5)
        with pytest.raises(RuntimeError, match="refresh"):
            sampler.sample_hard_negative(0, D)

    def test_sample_returns_valid_index(self):
        n = 8
        attrs = _make_attributes(n)
        D = _make_distance_matrix(n)

        # Build a trivial model that returns random embeddings
        class IdentityModel(torch.nn.Module):
            def forward(self, attributes):
                # Return the age feature as a 1-d "embedding"
                return attributes["age"].unsqueeze(1).float()

        model = IdentityModel()
        ds = ScheduleEmbeddingDataset(attrs, D, mode="pairwise")
        dl = DataLoader(ds, batch_size=n, collate_fn=collate_fn)
        sampler = HardNegativeSampler(k=3)
        sampler.refresh(model, dl)

        idx = sampler.sample_hard_negative(0, D, schedule_threshold=0.0)
        assert 0 <= idx < n
