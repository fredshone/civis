"""YAML-based experiment configuration for schedule embedding training.

Public API
----------
DataConfig
ModelConfig
TrainingConfig
EvaluationConfig
ExperimentConfig
load_config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _normalise_distance_weights(
    w: dict[str, float] | tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert dict form to canonical (participation, sequence, timing) tuple."""
    if isinstance(w, tuple):
        return w
    required = ("participation", "sequence", "timing")
    missing = [k for k in required if k not in w]
    if missing:
        raise ValueError(f"distance_weights missing keys: {missing}")
    return (w["participation"], w["sequence"], w["timing"])


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Dataset and dataloader configuration.

    Parameters
    ----------
    data_path:
        Path to the activities parquet/csv file.
    attributes_path:
        Path to the attributes parquet/csv file.
    distance_weights:
        Weights for participation, sequence and timing components of the
        composite distance.
    train_fraction, val_fraction:
        Fractions of data for training and validation (test = remainder).
    batch_size:
        Samples per mini-batch.
    num_workers:
        DataLoader worker processes.
    mode:
        Dataset mode: ``"pairwise"``, ``"triplet"``, or ``"single"``.
    masking_strategy:
        ``"independent"``, ``"grouped"``, or ``"curriculum"``.
    masking_base_rate:
        Base attribute masking probability.
    masking_missingness_weighted:
        Scale per-attribute masking rate by empirical missingness.
    timing_resolution:
        Time-use bin size in minutes for timing distance components.
    distance_device:
        Requested device for lazy distance scoring.  CUDA is used when
        available; otherwise the runner falls back to CPU.
    lazy_max_cached_pairs:
        Maximum number of lazily-computed schedule distances kept in the
        in-memory on-the-fly cache before eviction.
    lazy_distance_cache_file:
        File name under the run output directory used to persist lazy
        pair-distance cache across runs.
    val_pairs_seed:
        RNG seed for sampling fixed validation pairs.
    n_val_pairs:
        Number of fixed validation pairs for rank-correlation computation.
    """

    data_path: str = "data/activities.parquet"
    attributes_path: str = "data/attributes.parquet"
    distance_weights: dict[str, float] | tuple[float, float, float] = field(
        default_factory=lambda: {
            "participation": 1 / 3,
            "sequence": 1 / 3,
            "timing": 1 / 3,
        }
    )
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    batch_size: int = 128
    num_workers: int = 4
    mode: str = "pairwise"
    masking_strategy: str = "independent"
    masking_base_rate: float = 0.15
    masking_missingness_weighted: bool = True
    timing_resolution: int = 10
    distance_device: str = "cuda"
    lazy_max_cached_pairs: int = 500000
    lazy_distance_cache_file: str = "lazy_distance_cache.npz"
    val_pairs_seed: int = 42
    n_val_pairs: int = 1000

    def __post_init__(self):
        self.distance_weights = _normalise_distance_weights(self.distance_weights)
        if self.mode != "pairwise":
            raise ValueError(f"Only mode='pairwise' is supported, got {self.mode!r}")


@dataclass
class ModelConfig:
    """Embedding model architecture configuration.

    Parameters
    ----------
    architecture:
        Model type: ``"addition"``, ``"attention"``, or ``"film"``.
    d_embed:
        Per-attribute embedding dimension.
    d_model:
        Final output embedding dimension.
    dropout:
        Dropout rate.
    n_heads:
        Number of attention heads (attention model only).
    n_layers:
        Number of transformer encoder layers (attention model only).
    use_cls_token:
        Use a learned ``[CLS]`` token (attention model only).
    pooling:
        Pooling strategy: ``"cls"``, ``"mean"``, or ``"sum"`` (attention only).
    context_attributes:
        Attribute names used as conditioning context (FiLM model only).
    attribute_groups:
        Optional mapping of attribute name → group label for positional
        encodings (attention model only).
    """

    architecture: str = "addition"
    d_embed: int = 64
    d_model: int = 128
    dropout: float = 0.1
    n_heads: int = 4
    n_layers: int = 2
    use_cls_token: bool = True
    pooling: str = "cls"
    context_attributes: list[str] = field(default_factory=list)
    attribute_groups: dict[str, str] | None = None


@dataclass
class TrainingConfig:
    """Optimiser and training loop configuration.

    Parameters
    ----------
    loss_name:
        Key in :data:`~training.losses.LOSS_REGISTRY`.
    loss_kwargs:
        Extra keyword arguments forwarded to the loss constructor.
    lr:
        Peak AdamW learning rate.
    weight_decay:
        AdamW weight decay.
    max_epochs:
        Training epochs.
    warmup_steps:
        Linear LR warmup steps.
    hard_negative_refresh_steps:
        Steps between hard-negative k-NN index refreshes.
    hard_negative_subset_size:
        Samples used for each hard-negative refresh.
    log_every_n_steps:
        Training metric logging frequency.
    collapse_monitor_threshold:
        Variance ratio above which to log a collapse warning.
    attention_log_every_n_epochs:
        Epochs between attention heatmap logs.
    """

    loss_name: str = "distance_regression"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    warmup_steps: int = 1000
    hard_negative_refresh_steps: int = 500
    hard_negative_subset_size: int = 5000
    log_every_n_steps: int = 50
    collapse_monitor_threshold: float = 5.0
    attention_log_every_n_epochs: int = 5


@dataclass
class EvaluationConfig:
    """Downstream evaluation configuration.

    Parameters
    ----------
    run_downstream:
        Whether to run downstream evaluation after training.
    downstream_tasks:
        Task identifiers to evaluate.
    n_linear_probe_epochs:
        Epochs for training the linear probe head.
    """

    run_downstream: bool = True
    downstream_tasks: list[str] = field(
        default_factory=lambda: ["mode_share", "departure_time", "chain_length"]
    )
    n_linear_probe_epochs: int = 50


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Parameters
    ----------
    name:
        Unique experiment name (used for output directories).
    seed:
        Global random seed.
    output_dir:
        Root directory for checkpoints and logs.
    data, model, training, evaluation:
        Sub-configurations.
    """

    name: str = "experiment"
    seed: int = 42
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML file containing experiment configuration.

    Returns
    -------
    ExperimentConfig

    Raises
    ------
    ValueError
        If splits don't sum correctly or architecture is unknown.
    """
    from models.registry import MODEL_REGISTRY

    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = raw or {}
    data_raw = dict(raw.get("data", {}))

    data = DataConfig(
        **{k: v for k, v in data_raw.items() if k in DataConfig.__dataclass_fields__}
    )
    model = ModelConfig(
        **{
            k: v
            for k, v in raw.get("model", {}).items()
            if k in ModelConfig.__dataclass_fields__
        }
    )
    training = TrainingConfig(
        **{
            k: v
            for k, v in raw.get("training", {}).items()
            if k in TrainingConfig.__dataclass_fields__
        }
    )
    evaluation = EvaluationConfig(
        **{
            k: v
            for k, v in raw.get("evaluation", {}).items()
            if k in EvaluationConfig.__dataclass_fields__
        }
    )

    cfg = ExperimentConfig(
        name=raw.get("name", "experiment"),
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "outputs"),
        data=data,
        model=model,
        training=training,
        evaluation=evaluation,
    )

    # Validation
    if cfg.data.train_fraction + cfg.data.val_fraction > 1.0:
        raise ValueError(
            f"train_fraction ({cfg.data.train_fraction}) + "
            f"val_fraction ({cfg.data.val_fraction}) must be <= 1.0"
        )
    if cfg.model.architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture {cfg.model.architecture!r}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    return cfg
