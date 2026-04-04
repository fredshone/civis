"""PyTorch Lightning training module for contrastive schedule embedding.

Public API
----------
TrainerConfig
EmbeddingTrainer
EmbeddingCheckpoint
CollapseMonitor
AttentionLogger
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, SubsetRandomSampler

from training.losses import SoftNearestNeighbourLoss, _pairwise_euclidean


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyper-parameters for :class:`EmbeddingTrainer`.

    Parameters
    ----------
    lr:
        Peak learning rate for AdamW.
    weight_decay:
        AdamW weight decay.
    max_epochs:
        Number of training epochs.
    warmup_steps:
        Linear warmup steps before cosine decay begins.
    hard_negative_refresh_steps:
        Refresh the hard-negative k-NN index every this many global steps.
    hard_negative_subset_size:
        Maximum number of training examples to use when refreshing the index.
    log_every_n_steps:
        How often to log training metrics.
    collapse_monitor_threshold:
        Within/between-source variance ratio above which to log a warning.
    attention_log_every_n_epochs:
        How often to log attention heatmaps (attention models only).
    """
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    warmup_steps: int = 1000
    hard_negative_refresh_steps: int = 500
    hard_negative_subset_size: int = 5000
    log_every_n_steps: int = 50
    collapse_monitor_threshold: float = 5.0
    attention_log_every_n_epochs: int = 5


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class EmbeddingTrainer(pl.LightningModule):
    """Lightning module for contrastive attribute embedding training.

    Parameters
    ----------
    model:
        A :class:`~models.base.BaseAttributeEmbedder` subclass.
    loss_fn:
        A loss from :mod:`training.losses`.
    config:
        Training hyperparameters.
    val_pairs:
        Pre-collated tuple ``(attrs_i, attrs_j, distances)`` of 1 000 CPU
        tensors used for the fixed held-out rank-correlation metric.
        Pass ``None`` to skip that metric.
    hard_negative_sampler:
        Optional :class:`~datasets.dataset.HardNegativeSampler`.
    masker:
        Optional :class:`~datasets.masking.AttributeMasker` whose curriculum
        step is updated each training step.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainerConfig,
        val_pairs: tuple | None = None,
        hard_negative_sampler: Any | None = None,
        masker: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self._val_pairs_cpu = val_pairs
        self.val_pairs: tuple | None = None  # moved to device in on_fit_start
        self.hard_negative_sampler = hard_negative_sampler
        self.masker = masker

        self.validation_step_outputs: list[dict] = []
        self._train_dataset = None

        self.save_hyperparameters(ignore=["model", "loss_fn", "masker",
                                          "hard_negative_sampler", "val_pairs"])

    # ------------------------------------------------------------------
    # Setup hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        if self._val_pairs_cpu is not None:
            dev = self.device
            attrs_i, attrs_j, dists = self._val_pairs_cpu
            self.val_pairs = (
                {k: v.to(dev) for k, v in attrs_i.items()},
                {k: v.to(dev) for k, v in attrs_j.items()},
                dists.to(dev),
            )

    def on_train_start(self) -> None:
        try:
            self._train_dataset = self.trainer.train_dataloader.dataset
        except Exception:
            self._train_dataset = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if isinstance(self.loss_fn, SoftNearestNeighbourLoss):
            attrs, dist_matrix, batch_indices = batch
            dist_matrix = dist_matrix.to(self.device)
            batch_indices = batch_indices.to(self.device)
            attrs = {k: v.to(self.device) for k, v in attrs.items()}
            emb = self.model(attrs)
            # dist_matrix is (B, N_total); batch_indices are the dataset indices
            # of the B batch members.  Selecting those columns gives the exact
            # (B, B) schedule-distance sub-matrix for this batch.
            dist_bxb = dist_matrix[:, batch_indices]
            loss, diag = self.loss_fn(emb, dist_bxb)
        else:
            attrs_i, attrs_j, distances = batch
            attrs_i = {k: v.to(self.device) for k, v in attrs_i.items()}
            attrs_j = {k: v.to(self.device) for k, v in attrs_j.items()}
            distances = distances.to(self.device)
            emb_i = self.model(attrs_i)
            emb_j = self.model(attrs_j)
            loss, diag = self.loss_fn(emb_i, emb_j, distances)

        for k, v in diag.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False,
                     prog_bar=(k == "loss"))

        # Update step counters on loss and masker
        if hasattr(self.loss_fn, "set_step"):
            self.loss_fn.set_step(self.global_step)
        if self.masker is not None and hasattr(self.masker, "set_step"):
            self.masker.set_step(self.global_step)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if (
            self.hard_negative_sampler is not None
            and self._train_dataset is not None
            and self.global_step > 0
            and self.global_step % self.config.hard_negative_refresh_steps == 0
        ):
            self._refresh_hard_negatives()

    def _refresh_hard_negatives(self) -> None:
        dataset = self._train_dataset
        n = len(dataset)
        subset_size = min(self.config.hard_negative_subset_size, n)
        indices = np.random.choice(n, size=subset_size, replace=False).tolist()
        sampler = SubsetRandomSampler(indices)
        # Build a minimal DataLoader for refresh (no masking needed)
        from datasets.dataset import collate_fn
        loader = DataLoader(
            dataset,
            batch_size=256,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )
        self.hard_negative_sampler.refresh(
            self.model, loader, device=str(self.device)
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        if isinstance(self.loss_fn, SoftNearestNeighbourLoss):
            attrs, dist_matrix, batch_indices = batch
            dist_matrix = dist_matrix.to(self.device)
            batch_indices = batch_indices.to(self.device)
            attrs = {k: v.to(self.device) for k, v in attrs.items()}
            emb = self.model(attrs)
            dist_bxb = dist_matrix[:, batch_indices]
            loss, _ = self.loss_fn(emb, dist_bxb)
            # For metrics accumulation, treat emb as both sides
            self.validation_step_outputs.append({
                "emb_i": emb.detach(),
                "emb_j": emb.detach(),
                "distances": dist_bxb.diagonal().detach(),
                "loss": loss.detach(),
            })
        else:
            attrs_i, attrs_j, distances = batch
            attrs_i = {k: v.to(self.device) for k, v in attrs_i.items()}
            attrs_j = {k: v.to(self.device) for k, v in attrs_j.items()}
            distances = distances.to(self.device)
            emb_i = self.model(attrs_i)
            emb_j = self.model(attrs_j)
            loss, _ = self.loss_fn(emb_i, emb_j, distances)
            self.validation_step_outputs.append({
                "emb_i": emb_i.detach(),
                "emb_j": emb_j.detach(),
                "distances": distances.detach(),
                "loss": loss.detach(),
            })

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        if not outputs:
            return

        emb_i = torch.cat([o["emb_i"] for o in outputs], dim=0)
        emb_j = torch.cat([o["emb_j"] for o in outputs], dim=0)
        distances = torch.cat([o["distances"] for o in outputs], dim=0)
        mean_loss = torch.stack([o["loss"] for o in outputs]).mean()
        self.validation_step_outputs.clear()

        self.log("val/loss", mean_loss, prog_bar=True)

        # Alignment: mean embedding distance between paired embeddings
        valid = ~distances.isnan()
        if valid.sum() > 0:
            align = _pairwise_euclidean(emb_i[valid], emb_j[valid]).mean()
            self.log("val/alignment", align)

        # Uniformity: log mean pairwise Gaussian kernel on unit-normalised embs
        all_emb = torch.cat([emb_i, emb_j], dim=0)
        max_unif = 2000
        if all_emb.shape[0] > max_unif:
            idx = torch.randperm(all_emb.shape[0], device=all_emb.device)[:max_unif]
            all_emb_u = all_emb[idx]
        else:
            all_emb_u = all_emb
        normed = F.normalize(all_emb_u, dim=-1)
        sq_dists = torch.pdist(normed).pow(2)
        uniformity = (-2.0 * sq_dists).exp().mean().log()
        self.log("val/uniformity", uniformity)

        # Rank correlation on fixed held-out val_pairs
        if self.val_pairs is not None:
            vi, vj, vd = self.val_pairs
            with torch.no_grad():
                hi = self.model(vi)
                hj = self.model(vj)
            ed = _pairwise_euclidean(hi, hj).cpu().numpy()
            sd = vd.cpu().numpy()
            if np.std(ed) >= 1e-8 and np.std(sd) >= 1e-8:
                rho = spearmanr(ed, sd).statistic
                if not np.isnan(rho):
                    self.log("val/rank_correlation", float(rho), prog_bar=True)

        # Neighbourhood overlap is only reported in post-training geometry analysis
        # via GeometryAnalyser.neighbourhood_overlap, which uses the full N×N matrix.

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        params = list(self.model.parameters())
        # Include learnable tau if present in loss
        if hasattr(self.loss_fn, "log_tau_embed") and isinstance(
            self.loss_fn.log_tau_embed, nn.Parameter
        ):
            params.append(self.loss_fn.log_tau_embed)

        optimizer = torch.optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return max(step / max(self.config.warmup_steps, 1), 1e-8)
            progress = (step - self.config.warmup_steps) / max(
                total_steps - self.config.warmup_steps, 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EmbeddingCheckpoint(pl.Callback):
    """Save the best checkpoint by validation rank correlation.

    Parameters
    ----------
    dirpath:
        Directory where checkpoints are saved.
    filename:
        Filename stem (without extension).
    """

    def __init__(self, dirpath: str, filename: str = "best_rank_corr") -> None:
        super().__init__()
        self._ckpt = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor="val/rank_correlation",
            mode="max",
            save_top_k=1,
        )

    def setup(self, trainer, pl_module, stage):
        self._ckpt.setup(trainer, pl_module, stage)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._ckpt.on_validation_epoch_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self._ckpt.on_save_checkpoint(trainer, pl_module, checkpoint)


class CollapseMonitor(pl.Callback):
    """Monitor within-source vs between-source embedding variance.

    A high ratio indicates the embedding has learned to separate data sources
    rather than scheduling behaviour.

    Parameters
    ----------
    source_labels:
        Integer tensor of shape ``(N,)`` mapping each sample to a source index.
    sample_attrs:
        Full attribute dict of shape ``(N, ...)``; the monitor uses up to
        ``max_samples`` examples.
    threshold:
        Variance ratio above which a warning is logged.
    check_every_n_epochs:
        How often to compute the metric.
    max_samples:
        Maximum number of samples to use for variance computation.
    """

    def __init__(
        self,
        source_labels: torch.Tensor,
        sample_attrs: dict[str, torch.Tensor],
        threshold: float = 5.0,
        check_every_n_epochs: int = 1,
        max_samples: int = 2000,
    ) -> None:
        super().__init__()
        n = min(len(source_labels), max_samples)
        self._source_labels = source_labels[:n]
        self._sample_attrs = {k: v[:n] for k, v in sample_attrs.items()}
        self.threshold = threshold
        self.check_every_n_epochs = check_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return

        device = pl_module.device
        attrs = {k: v.to(device) for k, v in self._sample_attrs.items()}
        labels = self._source_labels.to(device)

        with torch.no_grad():
            emb = pl_module.model(attrs)  # (N, d)

        unique_sources = labels.unique()
        within_vars = []
        source_means = []
        for src in unique_sources:
            mask = labels == src
            if mask.sum() < 2:
                continue
            src_emb = emb[mask]
            source_means.append(src_emb.mean(dim=0))
            # Trace of covariance ≈ mean squared deviation from mean
            within_vars.append(src_emb.var(dim=0).sum().item())

        if not within_vars or len(source_means) < 2:
            return

        within_var = float(np.mean(within_vars))
        # Between-source variance: variance of source mean embeddings
        means_stack = torch.stack(source_means)
        between_var = means_stack.var(dim=0).sum().item()
        ratio = between_var / (within_var + 1e-8)

        trainer.logger.log_metrics({
            "monitor/within_source_var": within_var,
            "monitor/between_source_var": between_var,
            "monitor/collapse_ratio": ratio,
        }, step=trainer.global_step)

        if ratio > self.threshold:
            try:
                from pytorch_lightning.loggers import TensorBoardLogger
                if isinstance(trainer.logger, TensorBoardLogger):
                    trainer.logger.experiment.add_text(
                        "warning/collapse",
                        f"Epoch {trainer.current_epoch}: collapse ratio "
                        f"{ratio:.2f} exceeds threshold {self.threshold}",
                        global_step=trainer.global_step,
                    )
            except Exception:
                pass


class AttentionLogger(pl.Callback):
    """Log mean per-layer attention weight heatmaps to TensorBoard.

    Only active for :class:`~models.attention.SelfAttentionEmbedder`.

    Parameters
    ----------
    sample_attrs:
        Attribute dict used for the forward pass; the first 16 samples are used.
    log_every_n_epochs:
        How often to log heatmaps.
    """

    def __init__(
        self,
        sample_attrs: dict[str, torch.Tensor],
        log_every_n_epochs: int = 5,
    ) -> None:
        super().__init__()
        self._sample_attrs = {k: v[:16] for k, v in sample_attrs.items()}
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        from models.attention import SelfAttentionEmbedder
        if not isinstance(pl_module.model, SelfAttentionEmbedder):
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            if not isinstance(trainer.logger, TensorBoardLogger):
                return
        except Exception:
            return

        device = pl_module.device
        attrs = {k: v.to(device) for k, v in self._sample_attrs.items()}

        with torch.no_grad():
            _, attn_weights = pl_module.model(attrs, return_attention=True)

        import matplotlib.pyplot as plt
        import io

        tb = trainer.logger.experiment
        for layer_idx, attn in enumerate(attn_weights):
            # attn: (batch, heads, seq, seq) — average over batch and heads
            mean_attn = attn.mean(dim=(0, 1)).cpu().numpy()  # (seq, seq)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(mean_attn, vmin=0, aspect="auto", cmap="viridis")
            ax.set_title(f"Attention layer {layer_idx}")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=72)
            plt.close(fig)
            buf.seek(0)

            import numpy as np
            from PIL import Image
            img = Image.open(buf)
            img_arr = np.array(img).astype(np.float32) / 255.0  # (H, W, C)
            # TensorBoard expects (C, H, W)
            img_tensor = torch.from_numpy(img_arr[:, :, :3]).permute(2, 0, 1)
            tb.add_image(
                f"attention/layer_{layer_idx}",
                img_tensor,
                global_step=trainer.global_step,
            )
