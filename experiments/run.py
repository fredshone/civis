"""End-to-end experiment runner.

Usage
-----
::

    uv run python experiments/run.py experiments/configs/baseline_addition.yaml

The script:
1. Loads the experiment config.
2. Builds or reuses feature caches and the pairwise schedule distance matrix.
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
import pytorch_lightning as _pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.dataset import (
    LazyPairwiseDataset,
    collate_fn,
)
from datasets.encoding import AttributeEncoder, default_attribute_configs
from datasets.masking import AttributeMasker
from distances import build_schedule_features
from distances.data import load_activities, load_attributes
from distances.metric_plugins import (
    CompositeDistance,
    GPUCompositeDistance,
    ParticipationDistance,
    TimingDistance,
    TwoGramDistance,
)
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
    pids: list[str]  # person IDs in global activities order
    D: np.ndarray | None  # optional full pairwise N×N distance matrix
    output_path: Path
    n_params: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_distance_cache(path: Path) -> dict[tuple[int, int], float]:
    """Load a persisted lazy pair-distance cache.

    Cache format is a compressed ``npz`` with:
    - ``pairs``: int32 array shape (M, 2)
    - ``distances``: float32 array shape (M,)
    """
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    if "pairs" not in data or "distances" not in data:
        return {}
    pairs = data["pairs"]
    distances = data["distances"]
    cache: dict[tuple[int, int], float] = {}
    for (i, j), d in zip(pairs, distances):
        ii = int(i)
        jj = int(j)
        key = (ii, jj) if ii < jj else (jj, ii)
        cache[key] = float(d)
    return cache


def _save_distance_cache(
    cache: dict[tuple[int, int], float],
    path: Path,
) -> None:
    """Persist lazy pair-distance cache as compressed ``npz``."""
    if not cache:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = np.array(list(cache.keys()), dtype=np.int32)
    distances = np.array(list(cache.values()), dtype=np.float32)
    np.savez_compressed(path, pairs=pairs, distances=distances)


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
    # 2. Distance setup (lazy composite only)
    # ------------------------------------------------------------------
    feature_store_dir = output_path / "feature_store"
    lazy_cache_path = output_path / config.data.lazy_distance_cache_file
    D: np.ndarray | None = None
    pids: list[str] = sorted(activities_df["pid"].unique().to_list())
    metric = None
    metric_features = None
    metric_index = None
    lazy_distance_cache: dict[tuple[int, int], float] | None = None

    if config.data.mode != "pairwise":
        raise ValueError("Only mode='pairwise' is supported")

    print("Building/loading feature store for lazy composite pairwise scoring ...")
    requested_distance_device = config.data.distance_device
    distance_device = (
        requested_distance_device
        if requested_distance_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    if requested_distance_device != "cpu" and distance_device != "cuda":
        print("  CUDA unavailable; using CPU distance scoring for this run")

    if distance_device == "cuda":
        metric = GPUCompositeDistance(
            components=[
                ParticipationDistance(),
                TwoGramDistance(),
                TimingDistance(resolution=config.data.timing_resolution),
            ],
            weights=config.data.distance_weights,
            normalize_weights=True,
            device=distance_device,
        )
    else:
        metric = CompositeDistance(
            components=[
                ParticipationDistance(),
                TwoGramDistance(),
                TimingDistance(resolution=config.data.timing_resolution),
            ],
            weights=config.data.distance_weights,
            normalize_weights=True,
        )
    schedule_features = build_schedule_features(
        activities_df,
        feature_dir=feature_store_dir,
        timing_resolution=config.data.timing_resolution,
        overwrite=False,
    )
    metric_features = {
        "pids": schedule_features.pids,
        "components": [
            {
                "pids": schedule_features.pids,
                "matrix": schedule_features.participation,
            },
            {
                "pids": schedule_features.pids,
                "matrix": schedule_features.sequence_2grams,
            },
            {"pids": schedule_features.pids, "matrix": schedule_features.timing},
        ],
    }
    pids = metric_features["pids"]
    metric_index = metric.build_candidate_index(metric_features)
    lazy_distance_cache = _load_distance_cache(lazy_cache_path)
    if lazy_distance_cache:
        print(f"  Loaded lazy pair-distance cache: {len(lazy_distance_cache):,} pairs")

    # ------------------------------------------------------------------
    # 3. Align attribute rows to metric pid order, then split
    # ------------------------------------------------------------------
    attr_pids: list[str] = attributes_df["pid"].to_list()
    metric_pos = {pid: i for i, pid in enumerate(pids)}
    try:
        attr_to_metric_idx = np.array(
            [metric_pos[pid] for pid in attr_pids], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(
            f"pid {exc.args[0]!r} from attributes not found in activities-derived metric space"
        )

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(N)
    n_train = int(N * config.data.train_fraction)
    n_val = int(N * config.data.val_fraction)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    print(
        f"  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test"
    )

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
    # 6. Distance supervision and fixed validation pairs
    # ------------------------------------------------------------------
    rng_pairs = np.random.default_rng(config.data.val_pairs_seed)
    n_val = len(val_idx)
    vi = rng_pairs.integers(0, n_val, size=config.data.n_val_pairs)
    vj = rng_pairs.integers(0, n_val, size=config.data.n_val_pairs)
    same = vi == vj
    vj[same] = (vj[same] + 1) % n_val
    global_i = attr_to_metric_idx[val_idx[vi]]
    global_j = attr_to_metric_idx[val_idx[vj]]
    pair_arr = np.stack([global_i, global_j], axis=1).astype(np.int32)
    d_val = metric.score_pairs_batch(metric_features, pair_arr, metric_index)
    val_pairs = (
        {k: v[vi] for k, v in val_attrs.items()},
        {k: v[vj] for k, v in val_attrs.items()},
        torch.tensor(d_val.astype(np.float32), dtype=torch.float32),
    )

    # ------------------------------------------------------------------
    # 8. Datasets and DataLoaders
    # ------------------------------------------------------------------
    train_dataset = LazyPairwiseDataset(
        attributes=train_attrs,
        global_indices=attr_to_metric_idx[train_idx],
        metric=metric,
        metric_features=metric_features,
        candidate_index=metric_index,
        masker=masker,
        distance_cache=lazy_distance_cache,
        max_cached_pairs=config.data.lazy_max_cached_pairs,
        distance_device=distance_device,
    )
    val_dataset = LazyPairwiseDataset(
        attributes=val_attrs,
        global_indices=attr_to_metric_idx[val_idx],
        metric=metric,
        metric_features=metric_features,
        candidate_index=metric_index,
        masker=None,
        distance_cache=lazy_distance_cache,
        max_cached_pairs=config.data.lazy_max_cached_pairs,
        distance_device=distance_device,
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
    hard_negative_sampler = None

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

    if lazy_distance_cache is not None:
        _save_distance_cache(lazy_distance_cache, lazy_cache_path)
        print(
            f"Saved lazy pair-distance cache: {len(lazy_distance_cache):,} pairs "
            f"-> {lazy_cache_path}"
        )

    return _TrainingResult(
        model=embedding_trainer.model,
        encoder=encoder,
        masker=masker,
        activities_df=activities_df,
        attributes_df=attributes_df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        pids=pids,
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
    train_activities = result.activities_df.filter(pl.col("pid").is_in(train_pid_set))
    test_activities = result.activities_df.filter(pl.col("pid").is_in(test_pid_set))

    # ------------------------------------------------------------------
    # Intrinsic geometry analysis
    # ------------------------------------------------------------------
    try:
        if result.D is None:
            raise RuntimeError(
                "Dense distance matrix unavailable (lazy composite-only backend); "
                "skipping intrinsic geometry metrics"
            )
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
        if rc is not None and not (isinstance(rc, float) and np.isnan(rc)):
            metrics["intrinsic/rank_correlation"] = float(rc)

        no = geo_report.get("neighbourhood_overlap", {})
        for k in [5, 10, 20, 50]:
            if k in no:
                metrics[f"intrinsic/neighbourhood_overlap_k{k}"] = float(no[k])

        ss = geo_report.get("source_separation", {})
        if "mean_wasserstein" in ss:
            metrics["intrinsic/source_mean_wasserstein"] = float(ss["mean_wasserstein"])
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
