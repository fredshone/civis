"""Loss functions for contrastive schedule embedding training.

All standard losses accept ``(emb_i, emb_j, distances)`` and return
``(loss, diagnostics)`` where *diagnostics* is a ``dict[str, float]``
with per-batch metrics.  :class:`SoftNearestNeighbourLoss` has a distinct
signature ``(emb, dist_matrix)`` suitable for the ``"single"`` dataset mode.

Public API
----------
DistanceRegressionLoss
SoftNearestNeighbourLoss
RankCorrelationLoss
NTXentLoss
LOSS_REGISTRY
build_loss
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _pairwise_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between paired rows.

    Parameters
    ----------
    a, b:
        Tensors of shape ``(B, d)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``.  ``+1e-8`` under the square-root avoids NaN gradients
        when ``a == b``.
    """
    return (a - b).pow(2).sum(-1).add(1e-8).sqrt()


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between two 1-D tensors."""
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


# ---------------------------------------------------------------------------
# Distance regression
# ---------------------------------------------------------------------------

class DistanceRegressionLoss(nn.Module):
    """MSE (or Huber) between pairwise embedding distances and schedule distances.

    Parameters
    ----------
    use_huber:
        If ``True`` use Huber loss instead of MSE.
    huber_delta:
        Transition point for Huber loss.
    normalize_emb_dist:
        Divide embedding distances by ``sqrt(d_model)`` before comparing to
        schedule distances, which live in ``[0, 1]``.
    """

    def __init__(
        self,
        use_huber: bool = False,
        huber_delta: float = 0.1,
        normalize_emb_dist: bool = True,
    ) -> None:
        super().__init__()
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.normalize_emb_dist = normalize_emb_dist

    def forward(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss.

        Parameters
        ----------
        emb_i, emb_j:
            Embeddings of shape ``(B, d)``.
        distances:
            Schedule distances of shape ``(B,)``.  May contain ``nan`` for
            pairs not stored in a sparse matrix; those pairs are skipped.

        Returns
        -------
        loss : torch.Tensor
            Scalar.
        diagnostics : dict[str, float]
        """
        valid = ~distances.isnan()
        if valid.sum() == 0:
            zero = emb_i.sum() * 0.0
            return zero, {"loss": 0.0, "mean_emb_dist": 0.0,
                          "mean_schedule_dist": 0.0, "dist_correlation": 0.0}

        emb_i, emb_j, distances = emb_i[valid], emb_j[valid], distances[valid]
        emb_dist = _pairwise_euclidean(emb_i, emb_j)

        if self.normalize_emb_dist:
            d = emb_i.shape[-1]
            emb_dist = emb_dist / math.sqrt(d)

        if self.use_huber:
            loss = F.huber_loss(emb_dist, distances, delta=self.huber_delta)
        else:
            loss = F.mse_loss(emb_dist, distances)

        with torch.no_grad():
            corr = _pearson(emb_dist.detach(), distances).item()

        diag = {
            "loss": loss.item(),
            "mean_emb_dist": emb_dist.mean().item(),
            "mean_schedule_dist": distances.mean().item(),
            "dist_correlation": corr,
        }
        return loss, diag


# ---------------------------------------------------------------------------
# Soft nearest-neighbour
# ---------------------------------------------------------------------------

class SoftNearestNeighbourLoss(nn.Module):
    """Cross-entropy between soft neighbourhood distributions.

    For each anchor the schedule-distance-weighted softmax defines a target
    neighbourhood distribution; the embedding-distance-weighted softmax is
    the predicted distribution.  Loss = mean KL(target ‖ predicted).

    This loss uses a **different signature** from the other losses:
    ``forward(emb, dist_matrix)`` where *dist_matrix* is the ``(B, B)``
    pairwise schedule distance block for the batch.  Use with
    ``ScheduleEmbeddingDataset(mode="single")``.

    Parameters
    ----------
    tau_schedule:
        Temperature for the target (schedule-distance) distribution.
        Lower = sharper neighbourhood.
    tau_embed:
        Initial temperature for the predicted (embedding-distance) distribution.
    learnable_tau:
        If ``True``, ``tau_embed`` becomes a learned parameter.
    tau_anneal_steps:
        If > 0, linearly decay effective ``tau_embed`` from its initial value
        to ``tau_anneal_final`` over this many training steps.
    tau_anneal_final:
        Target ``tau_embed`` after annealing.  Required when
        ``tau_anneal_steps > 0``.
    """

    def __init__(
        self,
        tau_schedule: float = 1.0,
        tau_embed: float = 1.0,
        learnable_tau: bool = False,
        tau_anneal_steps: int = 0,
        tau_anneal_final: float | None = None,
    ) -> None:
        super().__init__()
        self.tau_schedule = tau_schedule
        self.tau_anneal_steps = tau_anneal_steps
        self.tau_anneal_final = tau_anneal_final
        self._step: int = 0

        if learnable_tau:
            self.log_tau_embed = nn.Parameter(
                torch.tensor(math.log(max(tau_embed, 1e-6)))
            )
        else:
            self.log_tau_embed: nn.Parameter | None = None  # type: ignore[assignment]
            self._tau_embed_fixed = tau_embed

    @property
    def tau_embed(self) -> float:
        """Effective tau_embed at the current training step."""
        if self.log_tau_embed is not None:
            return float(self.log_tau_embed.exp().item())
        if self.tau_anneal_steps > 0 and self.tau_anneal_final is not None:
            progress = min(self._step / self.tau_anneal_steps, 1.0)
            return self._tau_embed_fixed + progress * (
                self.tau_anneal_final - self._tau_embed_fixed
            )
        return self._tau_embed_fixed

    def set_step(self, step: int) -> None:
        """Update the training step counter (for tau annealing)."""
        self._step = step

    def forward(
        self,
        emb: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute soft nearest-neighbour loss.

        Parameters
        ----------
        emb:
            Embeddings of shape ``(B, d)``.
        dist_matrix:
            Pairwise schedule distances of shape ``(B, B)``.

        Returns
        -------
        loss : torch.Tensor
            Scalar.
        diagnostics : dict[str, float]
        """
        B = emb.shape[0]
        # Embedding pairwise distances
        emb_cdist = torch.cdist(emb, emb)  # (B, B)

        # Build log-softmax distributions with diagonal masked to -inf
        mask = torch.eye(B, dtype=torch.bool, device=emb.device)

        # Target: based on schedule distances (no grad needed)
        sched = dist_matrix.detach().clone()
        sched[mask] = float("inf")
        log_p = F.log_softmax(-sched / self.tau_schedule, dim=-1)  # (B, B)

        # Prediction: based on embedding distances
        if self.log_tau_embed is not None:
            tau_emb = self.log_tau_embed.exp()
        else:
            tau_emb = self.tau_embed
        emb_cdist_masked = emb_cdist.clone()
        emb_cdist_masked[mask] = float("inf")
        log_q = F.log_softmax(-emb_cdist_masked / tau_emb, dim=-1)  # (B, B)

        # KL(p || q) = sum_j p_j * (log_p_j - log_q_j)
        # Use p * log_p convention: 0 * log(0) = 0
        p = log_p.exp()
        kl = p * (log_p - log_q)
        kl = torch.where(p > 0, kl, torch.zeros_like(kl))
        loss = kl.sum(dim=-1).mean()

        with torch.no_grad():
            ent_target = -(p * log_p).sum(dim=-1).mean().item()
            q = log_q.exp()
            ent_pred = -(q * log_q).sum(dim=-1).mean().item()

        diag = {
            "loss": loss.item(),
            "tau_embed": self.tau_embed,
            "tau_schedule": self.tau_schedule,
            "mean_entropy_target": ent_target,
            "mean_entropy_pred": ent_pred,
        }
        return loss, diag


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

class RankCorrelationLoss(nn.Module):
    """Differentiable approximation to Spearman rank correlation.

    Uses soft-ranks (differentiable via sigmoid) to approximate rank order,
    then computes 1 − Pearson(soft_rank(emb_dist), soft_rank(schedule_dist)).

    Parameters
    ----------
    soft_rank_eps:
        Temperature for the sigmoid in soft-rank.  Smaller = harder (closer
        to true ranks but less smooth gradient).
    """

    def __init__(self, soft_rank_eps: float = 1.0) -> None:
        super().__init__()
        self.soft_rank_eps = soft_rank_eps

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable rank approximation.

        Parameters
        ----------
        x:
            1-D tensor of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)``, ranks approximately in ``[0.5, B - 0.5]``.
        """
        # P(x_j < x_i) ≈ sigmoid((x_i - x_j) / eps), sum over j gives rank
        diffs = x.unsqueeze(0) - x.unsqueeze(1)  # (B, B): diffs[i,j] = x_i - x_j
        above = torch.sigmoid(diffs / self.soft_rank_eps).sum(dim=1)  # (B,)
        return above + 0.5

    def forward(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss.

        Parameters
        ----------
        emb_i, emb_j:
            Embeddings of shape ``(B, d)``.
        distances:
            Schedule distances of shape ``(B,)``.

        Returns
        -------
        loss : torch.Tensor
            Scalar.
        diagnostics : dict[str, float]
        """
        valid = ~distances.isnan()
        if valid.sum() < 2:
            zero = emb_i.sum() * 0.0
            return zero, {"loss": 0.0, "spearman_approx": 0.0,
                          "mean_emb_dist": 0.0, "mean_schedule_dist": 0.0}

        emb_i, emb_j, distances = emb_i[valid], emb_j[valid], distances[valid]
        emb_dist = _pairwise_euclidean(emb_i, emb_j)

        rank_emb = self._soft_rank(emb_dist)
        rank_sched = self._soft_rank(distances.detach())

        corr = _pearson(rank_emb, rank_sched)
        loss = 1.0 - corr

        diag = {
            "loss": loss.item(),
            "spearman_approx": corr.item(),
            "mean_emb_dist": emb_dist.mean().item(),
            "mean_schedule_dist": distances.mean().item(),
        }
        return loss, diag


# ---------------------------------------------------------------------------
# NTXent (multi-positive variant)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """NT-Xent loss with threshold-based positive definition.

    Unlike SimCLR (which uses augmented views as the single positive),
    positives here are all pairs whose schedule distance is below
    ``positive_threshold``.  Supports zero or multiple positives per anchor.

    Parameters
    ----------
    tau:
        Temperature for cosine similarity scaling.
    positive_threshold:
        Schedule distance below which a pair is treated as positive.
    negative_threshold:
        Not used directly in the loss; kept for API symmetry.
    """

    def __init__(
        self,
        tau: float = 0.07,
        positive_threshold: float = 0.2,
        negative_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def forward(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-positive NT-Xent loss.

        Parameters
        ----------
        emb_i, emb_j:
            Embeddings of shape ``(B, d)``.
        distances:
            Paired schedule distances of shape ``(B,)``.

        Returns
        -------
        loss : torch.Tensor
            Scalar.
        diagnostics : dict[str, float]
        """
        B = emb_i.shape[0]
        # Concatenate both sides: first B rows from emb_i, next B from emb_j
        all_emb = torch.cat([emb_i, emb_j], dim=0)              # (2B, d)
        all_emb = F.normalize(all_emb, dim=-1)
        sim = (all_emb @ all_emb.T) / self.tau                   # (2B, 2B)

        # Build (2B, 2B) schedule distance block from paired distances
        # Block structure: [A_i x A_i = 0, A_i x A_j = diag(distances);
        #                   A_j x A_i = diag(distances), A_j x A_j = 0]
        dist_block = torch.zeros(2 * B, 2 * B, device=distances.device)
        # Fill with nan for unknown pairs (only paired distances are known)
        dist_block.fill_(float("nan"))
        # Diagonal of each block = same person, distance = 0
        for k in range(B):
            dist_block[k, k] = 0.0
            dist_block[B + k, B + k] = 0.0
        # Off-diagonal pairs: i-th emb_i paired with i-th emb_j
        for k in range(B):
            d = distances[k].item()
            dist_block[k, B + k] = d
            dist_block[B + k, k] = d

        # Positive mask: distance below threshold, excluding self (diagonal)
        diag_mask = torch.eye(2 * B, dtype=torch.bool, device=emb_i.device)
        positive_mask = (dist_block < self.positive_threshold) & ~diag_mask

        # Mask diagonal in sim to -inf
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # Compute multi-positive cross-entropy per anchor
        loss_sum = torch.tensor(0.0, device=emb_i.device)
        n_valid = 0
        log_softmax_sim = F.log_softmax(sim, dim=-1)  # (2B, 2B)

        n_pos_per_anchor = positive_mask.sum(dim=-1).float()
        has_positive = n_pos_per_anchor > 0

        if has_positive.sum() == 0:
            zero = emb_i.sum() * 0.0
            return zero, {
                "loss": 0.0,
                "mean_positive_sim": 0.0,
                "mean_negative_sim": 0.0,
                "fraction_anchors_with_positives": 0.0,
            }

        for i in range(2 * B):
            if not has_positive[i]:
                continue
            n_pos = int(n_pos_per_anchor[i].item())
            pos_log_probs = log_softmax_sim[i][positive_mask[i]]
            loss_sum = loss_sum - pos_log_probs.mean()
            n_valid += 1

        loss = loss_sum / n_valid

        with torch.no_grad():
            known = ~dist_block.isnan()
            pos_sims = sim[positive_mask & known]
            neg_sims = sim[(dist_block >= self.positive_threshold) & known & ~diag_mask]
            mean_pos = pos_sims.mean().item() if pos_sims.numel() > 0 else float("nan")
            mean_neg = neg_sims.mean().item() if neg_sims.numel() > 0 else float("nan")

        diag = {
            "loss": loss.item(),
            "mean_positive_sim": mean_pos,
            "mean_negative_sim": mean_neg,
            "fraction_anchors_with_positives": has_positive.float().mean().item(),
        }
        return loss, diag


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSS_REGISTRY: dict[str, type] = {
    "distance_regression": DistanceRegressionLoss,
    "soft_nearest_neighbour": SoftNearestNeighbourLoss,
    "rank_correlation": RankCorrelationLoss,
    "ntxent": NTXentLoss,
}


def build_loss(config: dict) -> nn.Module:
    """Instantiate a loss function from a config dict.

    Parameters
    ----------
    config:
        Must contain a ``"name"`` key matching a key in :data:`LOSS_REGISTRY`.
        All other keys are passed as keyword arguments to the loss constructor.
    """
    config = dict(config)
    name = config.pop("name")
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss {name!r}. Available: {list(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name](**config)
