"""Shared evaluation protocol and utilities for downstream tasks.

All downstream evaluators follow the same protocol:
1. Freeze the pre-trained embedder.
2. Compute embeddings for all samples in train/val/test splits.
3. Train a lightweight task head on train embeddings only.
4. Evaluate on test embeddings.
5. Report metrics.

Public API
----------
DownstreamEvaluatorConfig
    Configuration dataclass shared across all evaluators.

DownstreamEvaluator
    Abstract base class implementing the embedding step and the full
    ``run`` pipeline.  Subclasses implement ``extract_labels``, ``fit``,
    and ``evaluate``.

LinearHead
    Single linear layer — tests what is *linearly decodable* from the
    embedding (standard linear probe).

MLPHead
    Two-layer MLP — tests what is decodable at all, as an upper bound on
    embedding utility.

compare_embeddings(embedders, task, ...) -> pl.DataFrame
    Run the same evaluation for multiple embedders and return a long-format
    comparison table.

random_baseline(task, ...) -> dict[str, float]
    Lower bound: evaluate with Gaussian-noise embeddings.

frozen_attribute_baseline(task, ...) -> dict[str, float]
    Upper-bound reference: evaluate with raw one-hot + continuous features
    (no learned embedding).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

from datasets.encoding import AttributeEncoder
from models.base import BaseAttributeEmbedder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DownstreamEvaluatorConfig:
    """Configuration for downstream evaluators.

    Parameters
    ----------
    head_type:
        Which head to use as the primary reported head (``'linear'`` or
        ``'mlp'``).  Both are always trained; this controls which is used
        when a single-head result is requested.
    cache_dir:
        Directory for caching computed embeddings to disk.  ``None``
        disables caching.  Cache files are stored as
        ``{cache_dir}/{cache_tag}.npz``.
    embed_batch_size:
        Batch size for the embedding forward pass.
    device:
        PyTorch device string for embedding inference.
    seed:
        Random seed for head training.
    mlp_hidden_dim:
        Hidden dimension for the MLP head.
    mlp_max_iter:
        Maximum training iterations for the MLP head.
    """

    head_type: Literal["linear", "mlp"] = "linear"
    cache_dir: str | None = None
    embed_batch_size: int = 512
    device: str = "cpu"
    seed: int = 42
    mlp_hidden_dim: int = 256
    mlp_max_iter: int = 500


# ---------------------------------------------------------------------------
# Head classes
# ---------------------------------------------------------------------------


class LinearHead:
    """Single linear layer task head (standard linear probe).

    Uses ``sklearn.linear_model.LogisticRegression`` for classification and
    ``sklearn.linear_model.Ridge`` for regression.

    Parameters
    ----------
    task_type:
        ``'classification'`` or ``'regression'``.
    seed:
        Random seed for ``LogisticRegression``.
    """

    def __init__(
        self,
        task_type: Literal["classification", "regression"],
        seed: int = 42,
    ) -> None:
        self.task_type = task_type
        if task_type == "classification":
            self._model = LogisticRegression(max_iter=1000, random_state=seed)
        else:
            self._model = Ridge()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the head to training data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification heads.")
        return self._model.predict_proba(X)


class MLPHead:
    """Two-layer MLP task head.

    Uses ``sklearn.neural_network.MLPClassifier`` / ``MLPRegressor``.

    Note: sklearn's MLP does not support dropout.  The ``mlp_max_iter``
    config parameter controls the maximum number of training iterations.

    Parameters
    ----------
    task_type:
        ``'classification'`` or ``'regression'``.
    hidden_dim:
        Width of each hidden layer.
    max_iter:
        Maximum training iterations.
    seed:
        Random seed.
    """

    def __init__(
        self,
        task_type: Literal["classification", "regression"],
        hidden_dim: int = 256,
        max_iter: int = 500,
        seed: int = 42,
    ) -> None:
        self.task_type = task_type
        hidden = (hidden_dim, hidden_dim)
        if task_type == "classification":
            self._model = MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=max_iter,
                random_state=seed,
            )
        else:
            self._model = MLPRegressor(
                hidden_layer_sizes=hidden,
                max_iter=max_iter,
                random_state=seed,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the head to training data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification heads.")
        return self._model.predict_proba(X)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DownstreamEvaluator(ABC):
    """Abstract base class for downstream evaluation tasks.

    Subclasses implement :meth:`extract_labels`, :meth:`fit`, and
    :meth:`evaluate`.  The shared :meth:`embed_dataset` and :meth:`run`
    are provided here.

    Parameters
    ----------
    embedder:
        A trained :class:`~models.base.BaseAttributeEmbedder`.  It is set to
        eval mode and run without gradient tracking during embedding.
    config:
        Evaluator configuration.
    """

    def __init__(
        self,
        embedder: BaseAttributeEmbedder,
        config: DownstreamEvaluatorConfig,
    ) -> None:
        self.embedder = embedder
        self.config = config

    def embed_dataset(
        self,
        attributes_df: pl.DataFrame,
        encoder: AttributeEncoder,
        pids: list[str],
        cache_tag: str | None = None,
    ) -> np.ndarray:
        """Compute frozen embeddings for a list of persons.

        Filters ``attributes_df`` to the requested ``pids`` in order,
        encodes them with ``encoder``, then runs the frozen embedder.

        Parameters
        ----------
        attributes_df:
            Full attributes DataFrame (must contain a ``pid`` column).
        encoder:
            Fitted :class:`~datasets.encoding.AttributeEncoder`.
        pids:
            Ordered list of person IDs.  The returned array has one row per
            pid, in the same order.
        cache_tag:
            If set and ``config.cache_dir`` is not ``None``, embeddings are
            cached to ``{cache_dir}/{cache_tag}.npz`` and reloaded on
            subsequent calls with the same tag.

        Returns
        -------
        np.ndarray
            Shape ``(len(pids), embedder.embed_dim)``, dtype float32.
        """
        if cache_tag is not None and self.config.cache_dir is not None:
            cache_path = Path(self.config.cache_dir) / f"{cache_tag}.npz"
            if cache_path.exists():
                return np.load(cache_path)["embeddings"]

        # Filter to requested pids in order
        pid_series = pl.Series("pid", pids)
        filtered = (
            pl.DataFrame({"pid": pid_series})
            .join(attributes_df, on="pid", how="left")
        )

        attributes = encoder.transform(filtered)
        device = torch.device(self.config.device)
        attributes = {k: v.to(device) for k, v in attributes.items()}

        self.embedder.eval()
        self.embedder.to(device)
        batch_size = self.config.embed_batch_size
        n = len(pids)
        chunks: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = {k: v[start: start + batch_size] for k, v in attributes.items()}
                emb = self.embedder(batch)
                chunks.append(emb.cpu().numpy())

        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)

        if cache_tag is not None and self.config.cache_dir is not None:
            cache_path = Path(self.config.cache_dir) / f"{cache_tag}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, embeddings=embeddings)

        return embeddings

    @abstractmethod
    def extract_labels(
        self,
        activities_df: pl.DataFrame,
        attributes_df: pl.DataFrame,
        pids: list[str],
    ) -> np.ndarray:
        """Extract task labels aligned to ``pids``.

        Parameters
        ----------
        activities_df:
            Activities DataFrame (pid, seq, act, zone, start, end).
        attributes_df:
            Attributes DataFrame.
        pids:
            Ordered list of person IDs.

        Returns
        -------
        np.ndarray
            Shape ``(len(pids),)``.
        """
        ...

    @abstractmethod
    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        """Train the task head(s) on training embeddings and labels.

        Parameters
        ----------
        train_embeddings:
            Shape ``(n_train, embed_dim)``.
        train_labels:
            Shape ``(n_train,)``.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate on test embeddings and labels.

        Parameters
        ----------
        test_embeddings:
            Shape ``(n_test, embed_dim)``.
        test_labels:
            Shape ``(n_test,)``.

        Returns
        -------
        dict[str, float]
            Metric name → value.
        """
        ...

    def run(
        self,
        train_activities: pl.DataFrame,
        train_attributes: pl.DataFrame,
        test_activities: pl.DataFrame,
        test_attributes: pl.DataFrame,
        encoder: AttributeEncoder,
        train_pids: list[str],
        test_pids: list[str],
    ) -> dict[str, float]:
        """Run the full evaluation pipeline.

        Embeds both splits, extracts labels, trains the head, evaluates.

        Parameters
        ----------
        train_activities / train_attributes:
            Training split DataFrames.
        test_activities / test_attributes:
            Test split DataFrames.
        encoder:
            Fitted :class:`~datasets.encoding.AttributeEncoder`.
        train_pids / test_pids:
            Ordered person ID lists for each split.

        Returns
        -------
        dict[str, float]
            Metrics from :meth:`evaluate`.
        """
        train_emb = self.embed_dataset(
            train_attributes, encoder, train_pids, cache_tag="train"
        )
        test_emb = self.embed_dataset(
            test_attributes, encoder, test_pids, cache_tag="test"
        )
        train_labels = self.extract_labels(train_activities, train_attributes, train_pids)
        test_labels = self.extract_labels(test_activities, test_attributes, test_pids)
        self.fit(train_emb, train_labels)
        return self.evaluate(test_emb, test_labels)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compare_embeddings(
    embedders: dict[str, BaseAttributeEmbedder],
    task: DownstreamEvaluator,
    train_activities: pl.DataFrame,
    train_attributes: pl.DataFrame,
    test_activities: pl.DataFrame,
    test_attributes: pl.DataFrame,
    encoder: AttributeEncoder,
    train_pids: list[str],
    test_pids: list[str],
) -> pl.DataFrame:
    """Run the same evaluation for multiple embedders and compare results.

    Parameters
    ----------
    embedders:
        Mapping from embedder name to embedder instance.
    task:
        A configured :class:`DownstreamEvaluator`.  Its ``embedder`` will be
        temporarily replaced for each entry in ``embedders``.

    Returns
    -------
    pl.DataFrame
        Long-format table with columns ``[embedder_name, metric, value]``.
    """
    rows: list[dict] = []
    for name, embedder in embedders.items():
        task.embedder = embedder
        metrics = task.run(
            train_activities,
            train_attributes,
            test_activities,
            test_attributes,
            encoder,
            train_pids,
            test_pids,
        )
        for metric, value in metrics.items():
            rows.append({"embedder_name": name, "metric": metric, "value": float(value)})
    return pl.DataFrame(rows)


def random_baseline(
    task: DownstreamEvaluator,
    test_activities: pl.DataFrame,
    test_attributes: pl.DataFrame,
    test_pids: list[str],
    embed_dim: int,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate with Gaussian-noise embeddings as a lower bound.

    Both train and test embeddings are drawn from N(0, 1).

    Parameters
    ----------
    embed_dim:
        Dimensionality of the noise vectors (should match the real embedder).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Metrics from :meth:`~DownstreamEvaluator.evaluate`.
    """
    rng = np.random.default_rng(seed)
    n_test = len(test_pids)
    n_train = n_test  # use same size for noise train set

    train_emb = rng.standard_normal((n_train, embed_dim)).astype(np.float32)
    test_emb = rng.standard_normal((n_test, embed_dim)).astype(np.float32)

    # Use test pids as proxy train pids (labels come from same evaluator)
    train_labels = task.extract_labels(test_activities, test_attributes, test_pids)
    test_labels = task.extract_labels(test_activities, test_attributes, test_pids)

    task.fit(train_emb, train_labels)
    return task.evaluate(test_emb, test_labels)


def frozen_attribute_baseline(
    task: DownstreamEvaluator,
    train_activities: pl.DataFrame,
    train_attributes: pl.DataFrame,
    test_activities: pl.DataFrame,
    test_attributes: pl.DataFrame,
    encoder: AttributeEncoder,
    train_pids: list[str],
    test_pids: list[str],
) -> dict[str, float]:
    """Evaluate using raw one-hot + continuous features (no learned embedding).

    Discrete attributes are one-hot encoded using the fitted encoder's
    vocabulary.  Continuous attributes are min-max normalised (already
    in [0, 1] from the encoder).  All features are concatenated into a
    single vector used as the "embedding".

    This serves as an upper-bound reference: any learned embedder should
    approach or exceed this performance.

    Returns
    -------
    dict[str, float]
        Metrics from :meth:`~DownstreamEvaluator.evaluate`.
    """

    def _build_features(attributes_df: pl.DataFrame, pids: list[str]) -> np.ndarray:
        pid_series = pl.Series("pid", pids)
        filtered = (
            pl.DataFrame({"pid": pid_series})
            .join(attributes_df, on="pid", how="left")
        )
        parts: list[np.ndarray] = []
        for cfg in encoder.configs:
            if cfg.name not in filtered.columns:
                continue
            col = filtered[cfg.name]
            if cfg.kind == "discrete":
                vocab = encoder._vocab.get(cfg.name)
                if vocab is None:
                    continue
                vocab_size = len(vocab)
                val_to_idx = {v: i for i, v in enumerate(vocab)}
                indices = np.array(
                    [val_to_idx.get(v, 0) if v is not None else 0 for v in col.to_list()],
                    dtype=np.int32,
                )
                onehot = np.zeros((len(pids), vocab_size), dtype=np.float32)
                onehot[np.arange(len(pids)), indices] = 1.0
                parts.append(onehot)
            else:
                vmin = encoder._min.get(cfg.name, 0.0)
                vmax = encoder._max.get(cfg.name, 1.0)
                values = col.cast(pl.Float64).to_list()
                normed = np.array(
                    [
                        float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))
                        if v is not None
                        else 0.0
                        for v in values
                    ],
                    dtype=np.float32,
                )
                parts.append(normed.reshape(-1, 1))
        if not parts:
            return np.zeros((len(pids), 1), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    train_emb = _build_features(train_attributes, train_pids)
    test_emb = _build_features(test_attributes, test_pids)
    train_labels = task.extract_labels(train_activities, train_attributes, train_pids)
    test_labels = task.extract_labels(test_activities, test_attributes, test_pids)
    task.fit(train_emb, train_labels)
    return task.evaluate(test_emb, test_labels)
