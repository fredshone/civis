"""End-to-end experiment runner.

Usage
-----
::

    uv run python experiments/run.py experiments/configs/baseline_addition.yaml

The script:
1. Loads the experiment config.
2. Loads and/or computes the pairwise schedule distance matrix (cached to disk).
3. Splits data into train / val / test by person index.
4. Fits an :class:`~datasets.encoding.AttributeEncoder` and encodes attributes.
5. Builds a :class:`~datasets.masking.AttributeMasker`.
6. Constructs :class:`~datasets.dataset.ScheduleEmbeddingDataset` instances.
7. Builds the embedding model and loss function.
8. Runs PyTorch Lightning training with callbacks.
9. (Optional) runs downstream evaluation.

Public API
----------
run_experiment
    CLI entry point — takes a YAML config path and returns None.

run_experiment_returning_metrics
    Programmatic entry point — takes an :class:`~experiments.configs.ExperimentConfig`
    and returns a flat ``dict[str, float]`` of training and evaluation metrics.
    Used by :class:`~experiments.ablations.AblationRunner`.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import lightning as _pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.dataset import (
    HardNegativeSampler,
    ScheduleEmbeddingDataset,
    SparseDistanceMatrix,
    collate_fn,
)
from datasets.encoding import AttributeEncoder, default_attribute_configs
from datasets.masking import AttributeMasker
from distances.composite import pairwise_composite_distance
from distances.data import load_activities, load_attributes
from experiments.configs import ExperimentConfig, load_config
from models.base import AttributeEmbedderConfig, BaseAttributeEmbedder
from models.registry import build_model
from training.losses import build_loss
from training.trainer import (
    AttentionLogger,
    CollapseMonitor,
    EmbeddingCheckpoint,
    EmbeddingTrainer,
    TrainerConfig,
)


# ---------------------------------------------------------------------------
# Internal result type
# ---------------------------------------------------------------------------

@dataclass
class _TrainingResult:
    """All state produced by a completed training run."""
    model: BaseAttributeEmbedder
    encoder: AttributeEncoder
    masker: AttributeMasker
    activities_df: pl.DataFrame
    attributes_df: pl.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    D: np.ndarray          # full pairwise N×N distance matrix
    output_path: Path
    n_params: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_val_pairs(
    attrs: dict[str, torch.Tensor],
    D_dense: np.ndarray,
    n_pairs: int,
    seed: int,
) -> tuple[dict, dict, torch.Tensor]:
    """Sample a fixed set of val pairs for held-out rank-correlation evaluation."""
    rng = np.random.default_rng(seed)
    N = D_dense.shape[0]
    idx_i = rng.integers(0, N, size=n_pairs)
    idx_j = rng.integers(0, N, size=n_pairs)
    # Ensure no self-pairs
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % N

    distances = torch.tensor(
        D_dense[idx_i, idx_j].astype(np.float32), dtype=torch.float32
    )
    attrs_i = {k: v[idx_i] for k, v in attrs.items()}
    attrs_j = {k: v[idx_j] for k, v in attrs.items()}
    return attrs_i, attrs_j, distances


# ---------------------------------------------------------------------------
# Core training pipeline (shared by both public entry points)
# ---------------------------------------------------------------------------

def _run_training(config: ExperimentConfig) -> _TrainingResult:
    """Run the full training pipeline and return all artefacts.

    This is the single source of truth for the training loop.  Both
    :func:`run_experiment` and :func:`run_experiment_returning_metrics` call
    this function.
    """
    _pl.seed_everything(config.seed, workers=True)

    output_path = Path(config.output_dir) / config.name
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print(f"Loading data from {config.data.data_path} ...")
    activities_df = load_activities(config.data.data_path)
    attributes_df = load_attributes(config.data.attributes_path)
    N = len(attributes_df)
    print(f"  {N} persons, {len(activities_df)} activity records")

    # ------------------------------------------------------------------
    # 2. Distance matrix (compute or load from cache)
    # ------------------------------------------------------------------
    dist_cache = output_path / "distance_matrix.npy"
    if dist_cache.exists():
        print(f"Loading cached distance matrix from {dist_cache}")
        D = np.load(dist_cache)
    else:
        print("Computing pairwise composite distance matrix ...")
        D = pairwise_composite_distance(
            activities_df, weights=config.data.distance_weights
        )
        np.save(dist_cache, D)
        print(f"  Saved to {dist_cache}")

    # ------------------------------------------------------------------
    # 3. Train / val / test split
    # ------------------------------------------------------------------
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(N)
    n_train = int(N * config.data.train_fraction)
    n_val = int(N * config.data.val_fraction)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    # ------------------------------------------------------------------
    # 4. Encode attributes
    # ------------------------------------------------------------------
    encoder = AttributeEncoder(default_attribute_configs())
    encoder.fit(attributes_df)
    all_attrs = encoder.transform(attributes_df)
    train_attrs = {k: v[train_idx] for k, v in all_attrs.items()}
    val_attrs = {k: v[val_idx] for k, v in all_attrs.items()}

    # ------------------------------------------------------------------
    # 5. Attribute masker
    # ------------------------------------------------------------------
    masker = AttributeMasker.from_data(
        attributes_df,
        base_rate=config.data.masking_base_rate,
        missingness_weighted=config.data.masking_missingness_weighted,
        strategy=config.data.masking_strategy,
    )

    # ------------------------------------------------------------------
    # 6. Distance matrices
    # ------------------------------------------------------------------
    train_D_sparse = SparseDistanceMatrix.from_dense(
        D[np.ix_(train_idx, train_idx)], k=config.data.sparse_k
    )
    val_D_dense = D[np.ix_(val_idx, val_idx)]

    # ------------------------------------------------------------------
    # 7. Fixed validation pairs
    # ------------------------------------------------------------------
    val_pairs = _sample_val_pairs(
        val_attrs, val_D_dense, config.data.n_val_pairs, config.data.val_pairs_seed
    )

    # ------------------------------------------------------------------
    # 8. Datasets and DataLoaders
    # ------------------------------------------------------------------
    train_dataset = ScheduleEmbeddingDataset(
        attributes=train_attrs,
        distance_matrix=train_D_sparse,
        masker=masker,
        mode=config.data.mode,
        positive_threshold=config.data.positive_threshold,
        negative_threshold=config.data.negative_threshold,
    )
    val_dataset = ScheduleEmbeddingDataset(
        attributes=val_attrs,
        distance_matrix=val_D_dense,
        mode=config.data.mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
    )

    # ------------------------------------------------------------------
    # 9. Embedding model
    # ------------------------------------------------------------------
    embedder_config = AttributeEmbedderConfig.from_encoder(
        encoder,
        d_embed=config.model.d_embed,
        d_model=config.model.d_model,
        dropout=config.model.dropout,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        use_cls_token=config.model.use_cls_token,
        pooling=config.model.pooling,
        context_attributes=config.model.context_attributes,
        attribute_groups=config.model.attribute_groups,
    )
    model = build_model(
        {"architecture": config.model.architecture, **asdict(embedder_config)}
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {type(model).__name__}, {n_params:,} parameters")

    # ------------------------------------------------------------------
    # 10. Loss function
    # ------------------------------------------------------------------
    loss_fn = build_loss(
        {"name": config.training.loss_name, **config.training.loss_kwargs}
    )

    # ------------------------------------------------------------------
    # 11. Hard negative sampler
    # ------------------------------------------------------------------
    hard_negative_sampler: HardNegativeSampler | None = None
    if config.data.mode in ("pairwise", "triplet"):
        hard_negative_sampler = HardNegativeSampler(k=50)

    # ------------------------------------------------------------------
    # 12. Collapse monitor — source labels from training split
    # ------------------------------------------------------------------
    try:
        source_col = attributes_df["source"].to_list()
        source_vocab = {s: i for i, s in enumerate(sorted(set(source_col)))}
        all_source_labels = torch.tensor(
            [source_vocab.get(str(s), 0) for s in source_col], dtype=torch.long
        )
        monitor_n = min(2000, len(train_idx))
        monitor_labels = all_source_labels[train_idx[:monitor_n]]
        monitor_attrs = {k: v[:monitor_n] for k, v in train_attrs.items()}
    except Exception:
        monitor_labels = torch.zeros(min(2000, n_train), dtype=torch.long)
        monitor_attrs = {k: v[:2000] for k, v in train_attrs.items()}

    # ------------------------------------------------------------------
    # 13. Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        EmbeddingCheckpoint(dirpath=str(output_path)),
        CollapseMonitor(
            source_labels=monitor_labels,
            sample_attrs=monitor_attrs,
            threshold=config.training.collapse_monitor_threshold,
        ),
    ]
    if config.model.architecture == "attention":
        attn_sample = {k: v[:16] for k, v in train_attrs.items()}
        callbacks.append(
            AttentionLogger(
                sample_attrs=attn_sample,
                log_every_n_epochs=config.training.attention_log_every_n_epochs,
            )
        )

    # ------------------------------------------------------------------
    # 14. EmbeddingTrainer
    # ------------------------------------------------------------------
    trainer_cfg = TrainerConfig(
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        max_epochs=config.training.max_epochs,
        warmup_steps=config.training.warmup_steps,
        hard_negative_refresh_steps=config.training.hard_negative_refresh_steps,
        hard_negative_subset_size=config.training.hard_negative_subset_size,
        log_every_n_steps=config.training.log_every_n_steps,
        collapse_monitor_threshold=config.training.collapse_monitor_threshold,
        attention_log_every_n_epochs=config.training.attention_log_every_n_epochs,
    )
    embedding_trainer = EmbeddingTrainer(
        model=model,
        loss_fn=loss_fn,
        config=trainer_cfg,
        val_pairs=val_pairs,
        hard_negative_sampler=hard_negative_sampler,
        masker=masker,
    )

    # ------------------------------------------------------------------
    # 15. Lightning Trainer
    # ------------------------------------------------------------------
    logger = TensorBoardLogger(save_dir=str(output_path), name="logs")
    pl_trainer = _pl.Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        deterministic=True,
    )
    pl_trainer.fit(embedding_trainer, train_loader, val_loader)
    print(f"\nTraining complete. Outputs in {output_path}")

    # ------------------------------------------------------------------
    # 16. Persist artefacts for report generation
    # ------------------------------------------------------------------
    import json as _json
    from dataclasses import asdict as _asdict

    encoder.save(output_path / "encoder.pkl")
    torch.save(embedding_trainer.model.state_dict(), output_path / "model.pt")
    with open(output_path / "config.json", "w") as _fh:
        _json.dump(_asdict(config), _fh, indent=2)

    return _TrainingResult(
        model=embedding_trainer.model,
        encoder=encoder,
        masker=masker,
        activities_df=activities_df,
        attributes_df=attributes_df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        D=D,
        output_path=output_path,
        n_params=n_params,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_experiment(config_path: str | Path) -> None:
    """Run a full training experiment from a YAML config file.

    Parameters
    ----------
    config_path:
        Path to a YAML config file (see :mod:`experiments.configs`).
    """
    config = load_config(config_path)
    _run_training(config)


def run_experiment_returning_metrics(
    config: ExperimentConfig,
) -> dict[str, float]:
    """Run training and evaluation, returning a flat metrics dict.

    Designed for programmatic use by :class:`~experiments.ablations.AblationRunner`.
    Accepts an already-constructed :class:`~experiments.configs.ExperimentConfig`
    rather than a YAML path, so callers can override individual fields.

    Parameters
    ----------
    config:
        Fully populated experiment configuration.

    Returns
    -------
    dict[str, float]
        Flat dictionary with keys prefixed by domain:

        - ``meta/n_params``, ``meta/train_time_s``
        - ``intrinsic/alignment``, ``intrinsic/uniformity``,
          ``intrinsic/rank_correlation``, ``intrinsic/neighbourhood_overlap_k{k}``,
          ``intrinsic/source_mean_wasserstein``, ``intrinsic/source_accuracy``,
          ``intrinsic/cka``
        - ``discrete/work_participation_{head}_{metric}``
        - ``continuous/work_duration_{head}_{metric}``
        - ``continuous/trip_count_{head}_{metric}``
    """
    import time

    from evaluation import (
        DownstreamEvaluatorConfig,
        GeometryAnalyser,
        GeometryAnalyserConfig,
        TripCountConfig,
        TripCountEvaluator,
        WorkDurationConfig,
        WorkDurationEvaluator,
        WorkParticipationConfig,
        WorkParticipationEvaluator,
    )

    t0 = time.monotonic()
    result = _run_training(config)
    train_time = time.monotonic() - t0

    metrics: dict[str, float] = {
        "meta/n_params": float(result.n_params),
        "meta/train_time_s": train_time,
    }

    # ------------------------------------------------------------------
    # Split DataFrames by pid
    # ------------------------------------------------------------------
    all_pids: list[str] = result.attributes_df["pid"].to_list()
    train_pids = [all_pids[i] for i in result.train_idx]
    test_pids = [all_pids[i] for i in result.test_idx]

    train_pid_set = set(train_pids)
    test_pid_set = set(test_pids)

    train_attributes = result.attributes_df[result.train_idx.tolist()]
    test_attributes = result.attributes_df[result.test_idx.tolist()]
    train_activities = result.activities_df.filter(
        pl.col("pid").is_in(train_pid_set)
    )
    test_activities = result.activities_df.filter(
        pl.col("pid").is_in(test_pid_set)
    )

    # ------------------------------------------------------------------
    # Intrinsic geometry analysis
    # ------------------------------------------------------------------
    try:
        D_test = result.D[np.ix_(result.test_idx, result.test_idx)]
        analyser_cfg = GeometryAnalyserConfig(
            seed=config.seed,
            report_dir=str(result.output_path / "geometry"),
        )

        # Pre-sample rank pairs aligned to analyser's internal RNG
        n_test = len(test_pids)
        n_rank_pairs = 5000
        rng_rank = np.random.default_rng(analyser_cfg.seed)
        ri = rng_rank.integers(0, n_test, size=n_rank_pairs)
        rj = rng_rank.integers(0, n_test, size=n_rank_pairs)
        same = ri == rj
        rj[same] = (rj[same] + 1) % n_test
        rank_sched_dists = D_test[ri, rj]

        analyser = GeometryAnalyser(
            embedder=result.model,
            distance_fn=lambda x, y: 0.0,  # unused; precomputed distances provided
            test_attributes=test_attributes,
            test_pids=test_pids,
            encoder=result.encoder,
            masker=result.masker,
            config=analyser_cfg,
        )
        geo_report = analyser.full_report(
            output_dir=str(result.output_path / "geometry"),
            schedule_distance_matrix=D_test,
            rank_schedule_distances=rank_sched_dists,
        )

        au = geo_report.get("alignment_uniformity", {})
        if au.get("alignment") is not None:
            metrics["intrinsic/alignment"] = float(au["alignment"])
        if au.get("uniformity") is not None:
            metrics["intrinsic/uniformity"] = float(au["uniformity"])

        rc = geo_report.get("rank_correlation")
        if rc is not None:
            metrics["intrinsic/rank_correlation"] = float(rc)

        no = geo_report.get("neighbourhood_overlap", {})
        for k in [5, 10, 20, 50]:
            if k in no:
                metrics[f"intrinsic/neighbourhood_overlap_k{k}"] = float(no[k])

        ss = geo_report.get("source_separation", {})
        if "mean_wasserstein" in ss:
            metrics["intrinsic/source_mean_wasserstein"] = float(
                ss["mean_wasserstein"]
            )
        if "source_accuracy" in ss:
            metrics["intrinsic/source_accuracy"] = float(ss["source_accuracy"])

        cka = geo_report.get("cka")
        if cka is not None:
            metrics["intrinsic/cka"] = float(cka)

    except Exception as exc:
        print(f"  [warning] geometry analysis failed: {exc}")

    # ------------------------------------------------------------------
    # Downstream: work participation (classification)
    # ------------------------------------------------------------------
    try:
        eval_cfg = DownstreamEvaluatorConfig(seed=config.seed)
        wp_cfg = WorkParticipationConfig(seed=config.seed)
        wp_eval = WorkParticipationEvaluator(result.model, wp_cfg)
        wp_metrics = wp_eval.run(
            train_activities=train_activities,
            train_attributes=train_attributes,
            test_activities=test_activities,
            test_attributes=test_attributes,
            encoder=result.encoder,
            train_pids=train_pids,
            test_pids=test_pids,
        )
        for k, v in wp_metrics.items():
            metrics[f"discrete/work_participation_{k.replace('/', '_')}"] = float(v)
    except Exception as exc:
        print(f"  [warning] work participation evaluation failed: {exc}")

    # ------------------------------------------------------------------
    # Downstream: work duration (regression)
    # ------------------------------------------------------------------
    try:
        wd_cfg = WorkDurationConfig(seed=config.seed)
        wd_eval = WorkDurationEvaluator(result.model, wd_cfg)
        wd_metrics = wd_eval.run(
            train_activities=train_activities,
            train_attributes=train_attributes,
            test_activities=test_activities,
            test_attributes=test_attributes,
            encoder=result.encoder,
            train_pids=train_pids,
            test_pids=test_pids,
        )
        for k, v in wd_metrics.items():
            metrics[f"continuous/work_duration_{k.replace('/', '_')}"] = float(v)
    except Exception as exc:
        print(f"  [warning] work duration evaluation failed: {exc}")

    # ------------------------------------------------------------------
    # Downstream: trip count (regression)
    # ------------------------------------------------------------------
    try:
        tc_cfg = TripCountConfig(seed=config.seed)
        tc_eval = TripCountEvaluator(result.model, tc_cfg)
        tc_metrics = tc_eval.run(
            train_activities=train_activities,
            train_attributes=train_attributes,
            test_activities=test_activities,
            test_attributes=test_attributes,
            encoder=result.encoder,
            train_pids=train_pids,
            test_pids=test_pids,
        )
        for k, v in tc_metrics.items():
            metrics[f"continuous/trip_count_{k.replace('/', '_')}"] = float(v)
    except Exception as exc:
        print(f"  [warning] trip count evaluation failed: {exc}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python experiments/run.py <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])
