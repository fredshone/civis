"""Unit tests for training/losses.py."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from training.losses import (
    LOSS_REGISTRY,
    DistanceRegressionLoss,
    NTXentLoss,
    RankCorrelationLoss,
    SoftNearestNeighbourLoss,
    build_loss,
    _pairwise_euclidean,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def B() -> int:
    return 16


@pytest.fixture
def d() -> int:
    return 32


@pytest.fixture
def perfect_pair(B, d):
    """emb_i, emb_j, distances where embedding distance ≈ schedule distance."""
    torch.manual_seed(0)
    emb_i = torch.randn(B, d)
    # emb_j offset so that ||emb_i - emb_j|| is correlated with distances
    directions = torch.randn(B, d)
    distances = torch.rand(B)  # schedule distances in [0, 1]
    emb_j = emb_i + directions / directions.norm(dim=-1, keepdim=True) * distances.unsqueeze(-1)
    return emb_i.detach(), emb_j.detach(), distances.detach()


@pytest.fixture
def random_pair(B, d):
    """emb_i, emb_j, distances with no correlation."""
    torch.manual_seed(1)
    emb_i = torch.randn(B, d)
    emb_j = torch.randn(B, d)
    distances = torch.rand(B)
    return emb_i.detach(), emb_j.detach(), distances.detach()


@pytest.fixture
def identical_pair(B, d):
    """emb_i == emb_j, distances all 0."""
    torch.manual_seed(2)
    emb = torch.randn(B, d)
    return emb.detach(), emb.detach(), torch.zeros(B)


# ---------------------------------------------------------------------------
# _pairwise_euclidean
# ---------------------------------------------------------------------------

def test_pairwise_euclidean_identical(B, d):
    emb = torch.randn(B, d)
    dists = _pairwise_euclidean(emb, emb)
    assert dists.shape == (B,)
    # All distances should be very small (only +1e-8 offset)
    assert (dists < 1e-3).all()


def test_pairwise_euclidean_known():
    a = torch.tensor([[3.0, 0.0]])
    b = torch.tensor([[0.0, 4.0]])
    dist = _pairwise_euclidean(a, b)
    assert abs(dist.item() - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# DistanceRegressionLoss
# ---------------------------------------------------------------------------

class TestDistanceRegressionLoss:
    def test_loss_low_when_aligned(self, perfect_pair):
        emb_i, emb_j, distances = perfect_pair
        loss_fn = DistanceRegressionLoss(normalize_emb_dist=False)
        loss, diag = loss_fn(emb_i, emb_j, distances)
        assert loss.item() >= 0
        assert "loss" in diag
        assert "mean_emb_dist" in diag
        assert "mean_schedule_dist" in diag
        assert "dist_correlation" in diag

    def test_loss_decreases_with_training(self, d):
        torch.manual_seed(42)
        B = 32
        distances = torch.rand(B) * 0.8
        emb_i = nn.Parameter(torch.randn(B, d))
        emb_j = nn.Parameter(torch.randn(B, d))
        loss_fn = DistanceRegressionLoss(normalize_emb_dist=False)
        opt = torch.optim.Adam([emb_i, emb_j], lr=0.05)

        initial_loss, _ = loss_fn(emb_i, emb_j, distances)
        for _ in range(50):
            opt.zero_grad()
            loss, _ = loss_fn(emb_i, emb_j, distances)
            loss.backward()
            opt.step()

        assert loss.item() < initial_loss.item()

    def test_nan_distances_skipped(self, B, d):
        torch.manual_seed(3)
        emb_i = torch.randn(B, d)
        emb_j = torch.randn(B, d)
        distances = torch.rand(B)
        distances[::2] = float("nan")
        loss_fn = DistanceRegressionLoss()
        loss, diag = loss_fn(emb_i, emb_j, distances)
        assert not math.isnan(loss.item())

    def test_all_nan_returns_zero(self, B, d):
        emb_i = torch.randn(B, d)
        emb_j = torch.randn(B, d)
        distances = torch.full((B,), float("nan"))
        loss_fn = DistanceRegressionLoss()
        loss, _ = loss_fn(emb_i, emb_j, distances)
        assert loss.item() == 0.0

    def test_huber_mode(self, perfect_pair):
        emb_i, emb_j, distances = perfect_pair
        loss_fn = DistanceRegressionLoss(use_huber=True, huber_delta=0.1)
        loss, _ = loss_fn(emb_i, emb_j, distances)
        assert loss.item() >= 0

    def test_diagnostics_keys(self, random_pair):
        emb_i, emb_j, distances = random_pair
        loss_fn = DistanceRegressionLoss()
        _, diag = loss_fn(emb_i, emb_j, distances)
        assert set(diag) == {"loss", "mean_emb_dist", "mean_schedule_dist", "dist_correlation"}


# ---------------------------------------------------------------------------
# SoftNearestNeighbourLoss
# ---------------------------------------------------------------------------

class TestSoftNearestNeighbourLoss:
    def _make_batch(self, B=8, d=16):
        torch.manual_seed(10)
        emb = torch.randn(B, d)
        dist_matrix = torch.rand(B, B)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        dist_matrix.fill_diagonal_(0.0)
        return emb, dist_matrix

    def test_loss_positive(self):
        loss_fn = SoftNearestNeighbourLoss()
        emb, dist_matrix = self._make_batch()
        loss, diag = loss_fn(emb, dist_matrix)
        assert loss.item() >= 0
        assert "loss" in diag
        assert "tau_embed" in diag
        assert "tau_schedule" in diag
        assert "mean_entropy_target" in diag
        assert "mean_entropy_pred" in diag

    def test_loss_decreases_with_training(self):
        B, d = 16, 32
        torch.manual_seed(20)
        dist_matrix = torch.rand(B, B)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        dist_matrix.fill_diagonal_(0.0)

        emb = nn.Parameter(torch.randn(B, d))
        loss_fn = SoftNearestNeighbourLoss(tau_schedule=0.5, tau_embed=0.5)
        opt = torch.optim.Adam([emb], lr=0.05)

        initial_loss, _ = loss_fn(emb, dist_matrix)
        for _ in range(50):
            opt.zero_grad()
            loss, _ = loss_fn(emb, dist_matrix)
            loss.backward()
            opt.step()

        assert loss.item() < initial_loss.item()

    def test_learnable_tau(self):
        loss_fn = SoftNearestNeighbourLoss(tau_embed=1.0, learnable_tau=True)
        assert isinstance(loss_fn.log_tau_embed, torch.nn.Parameter)
        assert abs(loss_fn.tau_embed - 1.0) < 0.01

    def test_set_step_annealing(self):
        loss_fn = SoftNearestNeighbourLoss(
            tau_embed=1.0, tau_anneal_steps=100, tau_anneal_final=0.1
        )
        loss_fn.set_step(50)
        # Should be halfway between 1.0 and 0.1
        assert 0.5 < loss_fn.tau_embed < 0.6

    def test_diagnostics_keys(self):
        loss_fn = SoftNearestNeighbourLoss()
        emb, dist_matrix = self._make_batch()
        _, diag = loss_fn(emb, dist_matrix)
        assert set(diag) == {
            "loss", "tau_embed", "tau_schedule",
            "mean_entropy_target", "mean_entropy_pred",
        }


# ---------------------------------------------------------------------------
# RankCorrelationLoss
# ---------------------------------------------------------------------------

class TestRankCorrelationLoss:
    def test_low_loss_when_aligned(self, perfect_pair):
        emb_i, emb_j, distances = perfect_pair
        loss_fn = RankCorrelationLoss()
        loss, diag = loss_fn(emb_i, emb_j, distances)
        # Correlation should be positive → loss < 1
        assert loss.item() < 1.0

    def test_loss_decreases_with_training(self, d):
        torch.manual_seed(30)
        B = 32
        distances = torch.arange(B, dtype=torch.float32) / B
        emb_i = nn.Parameter(torch.randn(B, d))
        emb_j = nn.Parameter(torch.randn(B, d))
        loss_fn = RankCorrelationLoss()
        opt = torch.optim.Adam([emb_i, emb_j], lr=0.05)

        initial_loss, _ = loss_fn(emb_i, emb_j, distances)
        for _ in range(50):
            opt.zero_grad()
            loss, _ = loss_fn(emb_i, emb_j, distances)
            loss.backward()
            opt.step()

        assert loss.item() < initial_loss.item()

    def test_nan_handling(self, B, d):
        torch.manual_seed(4)
        emb_i = torch.randn(B, d)
        emb_j = torch.randn(B, d)
        distances = torch.rand(B)
        distances[:B // 2] = float("nan")
        loss_fn = RankCorrelationLoss()
        loss, _ = loss_fn(emb_i, emb_j, distances)
        assert not math.isnan(loss.item())

    def test_diagnostics_keys(self, random_pair):
        emb_i, emb_j, distances = random_pair
        loss_fn = RankCorrelationLoss()
        _, diag = loss_fn(emb_i, emb_j, distances)
        assert set(diag) == {
            "loss", "spearman_approx", "mean_emb_dist", "mean_schedule_dist"
        }


# ---------------------------------------------------------------------------
# NTXentLoss
# ---------------------------------------------------------------------------

class TestNTXentLoss:
    def test_loss_positive(self, B, d):
        torch.manual_seed(5)
        emb_i = torch.randn(B, d)
        emb_j = torch.randn(B, d)
        distances = torch.rand(B)
        loss_fn = NTXentLoss(tau=0.1, positive_threshold=0.3)
        loss, diag = loss_fn(emb_i, emb_j, distances)
        if diag["fraction_anchors_with_positives"] > 0:
            assert loss.item() >= 0

    def test_loss_decreases_with_training(self, d):
        torch.manual_seed(40)
        B = 32
        # Make some clear positives (distance = 0.1) and negatives (distance = 0.8)
        distances = torch.cat([
            torch.full((B // 2,), 0.1),
            torch.full((B // 2,), 0.8),
        ])
        emb_i = nn.Parameter(torch.randn(B, d))
        emb_j = nn.Parameter(torch.randn(B, d))
        loss_fn = NTXentLoss(tau=0.1, positive_threshold=0.3, negative_threshold=0.6)
        opt = torch.optim.Adam([emb_i, emb_j], lr=0.05)

        initial_loss, initial_diag = loss_fn(emb_i, emb_j, distances)
        if initial_diag["fraction_anchors_with_positives"] == 0:
            pytest.skip("No positives in batch, skipping convergence test")

        for _ in range(50):
            opt.zero_grad()
            loss, _ = loss_fn(emb_i, emb_j, distances)
            if loss.item() == 0:
                break
            loss.backward()
            opt.step()

        assert loss.item() <= initial_loss.item()

    def test_nan_distances_handled(self, B, d):
        torch.manual_seed(6)
        emb_i = torch.randn(B, d)
        emb_j = torch.randn(B, d)
        distances = torch.rand(B)
        distances[0] = float("nan")
        loss_fn = NTXentLoss(positive_threshold=0.4)
        # Should not raise
        loss, _ = loss_fn(emb_i, emb_j, distances)

    def test_diagnostics_keys(self, B, d):
        torch.manual_seed(7)
        emb_i = torch.randn(B, d)
        emb_j = torch.randn(B, d)
        distances = torch.rand(B)
        loss_fn = NTXentLoss()
        _, diag = loss_fn(emb_i, emb_j, distances)
        assert set(diag) == {
            "loss", "mean_positive_sim", "mean_negative_sim",
            "fraction_anchors_with_positives",
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestLossRegistry:
    def test_all_registered(self):
        assert "distance_regression" in LOSS_REGISTRY
        assert "soft_nearest_neighbour" in LOSS_REGISTRY
        assert "rank_correlation" in LOSS_REGISTRY
        assert "ntxent" in LOSS_REGISTRY

    def test_build_loss_distance_regression(self):
        loss = build_loss({"name": "distance_regression", "use_huber": True})
        assert isinstance(loss, DistanceRegressionLoss)
        assert loss.use_huber is True

    def test_build_loss_soft_nearest_neighbour(self):
        loss = build_loss({"name": "soft_nearest_neighbour", "tau_schedule": 0.5})
        assert isinstance(loss, SoftNearestNeighbourLoss)
        assert loss.tau_schedule == 0.5

    def test_build_loss_rank_correlation(self):
        loss = build_loss({"name": "rank_correlation"})
        assert isinstance(loss, RankCorrelationLoss)

    def test_build_loss_ntxent(self):
        loss = build_loss({"name": "ntxent", "tau": 0.07})
        assert isinstance(loss, NTXentLoss)

    def test_build_loss_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss({"name": "nonexistent_loss"})
