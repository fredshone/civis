"""Tests for datasets/dataset.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from datasets.dataset import (
    HardNegativeSampler,
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
        "sex":     torch.randint(0, 3, (n,)),
        "age":     torch.rand(n),
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
                abs(dist.item() - D[idx, j]) < 1e-5
                for j in range(n) if j != idx
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
            attrs, D,
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
            assert d_ap.item() < 0.2 + 1e-5, (
                f"Positive distance {d_ap.item()} >= threshold 0.2"
            )

    def test_negative_distance_above_threshold(self, dataset):
        for idx in range(6):
            _, _, _, _, d_an = dataset[idx]
            assert d_an.item() > 0.5 - 1e-5, (
                f"Negative distance {d_an.item()} <= threshold 0.5"
            )

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
            attrs, D, masker=masker,
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
            assert tensor.shape[0] == 3, f"{key}: expected batch_size=3, got {tensor.shape}"

    def test_triplet_collate(self):
        n = 12
        attrs = _make_attributes(n)
        D = np.ones((n, n)) * 0.8
        D[:6, :6] = 0.1
        D[6:, 6:] = 0.1
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2
        ds = ScheduleEmbeddingDataset(
            attrs, D,
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
