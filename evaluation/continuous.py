"""Continuous regression downstream evaluation tasks.

Two tasks:
- Work duration: total minutes spent on work activities (workers only).
- Trip count: number of activity transitions (excluding home→home).

Both use the same regression protocol: freeze embedder → embed → train
Ridge (linear) and MLP (nonlinear) heads → evaluate.

Public API
----------
WorkDurationConfig / WorkDurationEvaluator
    Regression of total work time in minutes.

TripCountConfig / TripCountEvaluator
    Regression of trip count (activity transitions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datasets.encoding import AttributeEncoder
from models.base import BaseAttributeEmbedder

from .base import DownstreamEvaluator, DownstreamEvaluatorConfig, LinearHead, MLPHead

if TYPE_CHECKING:
    import matplotlib.figure


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class WorkDurationConfig(DownstreamEvaluatorConfig):
    """Configuration for :class:`WorkDurationEvaluator`.

    Parameters
    ----------
    source_column:
        Attributes column for survey source.
    employment_column:
        Attributes column for employment status.
    zone_column:
        Attributes column for household zone type.
    """

    source_column: str = "source"
    employment_column: str = "employment"
    zone_column: str = "hh_zone"


@dataclass
class TripCountConfig(DownstreamEvaluatorConfig):
    """Configuration for :class:`TripCountEvaluator`.

    Parameters
    ----------
    source_column:
        Attributes column for survey source.
    employment_column:
        Attributes column for employment status.
    zone_column:
        Attributes column for household zone type.
    exclude_home_to_home:
        If ``True`` (default), consecutive home→home transitions are not
        counted as trips.
    """

    source_column: str = "source"
    employment_column: str = "employment"
    zone_column: str = "hh_zone"
    exclude_home_to_home: bool = True


# ---------------------------------------------------------------------------
# Shared regression helpers
# ---------------------------------------------------------------------------


def _regression_metrics(
    head: LinearHead | MLPHead,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    preds = head.predict(test_embeddings)
    mae = float(mean_absolute_error(test_labels, preds))
    rmse = float(np.sqrt(mean_squared_error(test_labels, preds)))
    r2 = float(r2_score(test_labels, preds))
    spearman_corr, _ = spearmanr(test_labels, preds)
    return {
        f"{prefix}/mae": mae,
        f"{prefix}/rmse": rmse,
        f"{prefix}/r2": r2,
        f"{prefix}/spearman": float(spearman_corr),
    }


def _evaluate_stratified(
    evaluator: DownstreamEvaluator,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_attributes: pl.DataFrame,
    test_pids: list[str],
    stratify_by: str,
) -> dict[str, dict[str, float]]:
    pid_to_idx = {pid: i for i, pid in enumerate(test_pids)}
    pid_to_group: dict[str, str] = {}
    if stratify_by in test_attributes.columns:
        for row in test_attributes.select(["pid", stratify_by]).iter_rows():
            pid, group = row
            pid_to_group[pid] = str(group) if group is not None else "unknown"

    groups: dict[str, list[int]] = {}
    for pid in test_pids:
        g = pid_to_group.get(pid, "unknown")
        groups.setdefault(g, []).append(pid_to_idx[pid])

    results: dict[str, dict[str, float]] = {}
    for group, indices in groups.items():
        idx = np.array(indices)
        sub_emb = test_embeddings[idx]
        sub_labels = test_labels[idx]
        if len(sub_labels) < 2:
            continue
        results[group] = evaluator.evaluate(sub_emb, sub_labels)
    return results


def _plot_residuals(
    head: LinearHead | MLPHead,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    title: str,
) -> "matplotlib.figure.Figure":
    preds = head.predict(test_embeddings)
    residuals = test_labels - preds

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(preds, residuals, alpha=0.4, s=15)
    axes[0].axhline(0, color="grey", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{title}: residuals vs predicted")

    axes[1].hist(residuals, bins=30, edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title}: residual distribution")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# WorkDurationEvaluator
# ---------------------------------------------------------------------------


class WorkDurationEvaluator(DownstreamEvaluator):
    """Regression of total work time in minutes.

    Labels are the sum of ``(end - start)`` for activities with
    ``act == 'work'``.  Persons with no work activity receive 0.0.

    Both heads (Ridge and MLP) are trained on the workers-only subset
    (``label > 0``) and evaluated on the workers-only test subset.

    Parameters
    ----------
    embedder:
        Pre-trained attribute embedder.
    config:
        Task configuration.
    """

    def __init__(
        self,
        embedder: BaseAttributeEmbedder,
        config: WorkDurationConfig,
    ) -> None:
        super().__init__(embedder, config)
        self.config: WorkDurationConfig
        self._linear: LinearHead | None = None
        self._mlp: MLPHead | None = None

    def extract_labels(
        self,
        activities_df: pl.DataFrame,
        attributes_df: pl.DataFrame,
        pids: list[str],
    ) -> np.ndarray:
        """Extract total work duration (minutes) aligned to ``pids``.

        Non-workers receive 0.0.

        Returns
        -------
        np.ndarray
            Float32 array, shape ``(len(pids),)``.
        """
        work_durations = (
            activities_df
            .filter(pl.col("act") == "work")
            .with_columns((pl.col("end") - pl.col("start")).alias("duration"))
            .group_by("pid")
            .agg(pl.col("duration").sum().alias("work_minutes"))
        )
        pid_to_duration: dict[str, float] = {
            row[0]: float(row[1])
            for row in work_durations.iter_rows()
        }
        return np.array(
            [pid_to_duration.get(p, 0.0) for p in pids],
            dtype=np.float32,
        )

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        """Train both heads on the workers-only subset (``label > 0``)."""
        mask = train_labels > 0
        if mask.sum() < 2:
            # Fallback: train on all data if too few workers
            X, y = train_embeddings, train_labels
        else:
            X, y = train_embeddings[mask], train_labels[mask]

        self._linear = LinearHead("regression", seed=self.config.seed)
        self._mlp = MLPHead(
            "regression",
            hidden_dim=self.config.mlp_hidden_dim,
            max_iter=self.config.mlp_max_iter,
            seed=self.config.seed,
        )
        self._linear.fit(X, y)
        self._mlp.fit(X, y)

    def evaluate(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute regression metrics on the workers-only test subset.

        Returns
        -------
        dict[str, float]
            Keys: ``{linear,mlp}/{mae,rmse,r2,spearman}``.
        """
        if self._linear is None or self._mlp is None:
            raise RuntimeError("Call fit() before evaluate().")

        mask = test_labels > 0
        if mask.sum() < 2:
            return {}

        emb = test_embeddings[mask]
        labels = test_labels[mask]

        metrics: dict[str, float] = {}
        metrics.update(_regression_metrics(self._linear, emb, labels, "linear"))
        metrics.update(_regression_metrics(self._mlp, emb, labels, "mlp"))
        return metrics

    def evaluate_stratified(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        test_attributes: pl.DataFrame,
        test_pids: list[str],
        stratify_by: str,
    ) -> dict[str, dict[str, float]]:
        """Evaluate separately for each stratum of a given attribute."""
        return _evaluate_stratified(
            self, test_embeddings, test_labels, test_attributes, test_pids, stratify_by
        )

    def cross_source_evaluate(
        self,
        all_activities: pl.DataFrame,
        all_attributes: pl.DataFrame,
        encoder: AttributeEncoder,
        source_a: str,
        source_b: str,
    ) -> dict[str, float]:
        """Train on source A, evaluate on source B."""
        source_col = self.config.source_column
        pids_a = (
            all_attributes.filter(pl.col(source_col) == source_a)["pid"].to_list()
        )
        pids_b = (
            all_attributes.filter(pl.col(source_col) == source_b)["pid"].to_list()
        )
        emb_a = self.embed_dataset(all_attributes, encoder, pids_a)
        emb_b = self.embed_dataset(all_attributes, encoder, pids_b)
        labels_a = self.extract_labels(all_activities, all_attributes, pids_a)
        labels_b = self.extract_labels(all_activities, all_attributes, pids_b)
        self.fit(emb_a, labels_a)
        return self.evaluate(emb_b, labels_b)

    def plot_residuals(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        head: Literal["linear", "mlp"] = "linear",
    ) -> "matplotlib.figure.Figure":
        """Plot residuals vs predicted and residual histogram.

        Parameters
        ----------
        head:
            Which head to use for predictions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._linear is None or self._mlp is None:
            raise RuntimeError("Call fit() before plot_residuals().")

        mask = test_labels > 0
        emb = test_embeddings[mask]
        labels = test_labels[mask]
        h = self._linear if head == "linear" else self._mlp
        return _plot_residuals(h, emb, labels, f"Work duration ({head})")


# ---------------------------------------------------------------------------
# TripCountEvaluator
# ---------------------------------------------------------------------------


class TripCountEvaluator(DownstreamEvaluator):
    """Regression of the number of trips (activity transitions) in a day.

    A trip is a consecutive pair of activities ``(act_i, act_{i+1})``.
    When ``config.exclude_home_to_home`` is ``True`` (default), transitions
    where both activities are ``'home'`` are not counted.

    Parameters
    ----------
    embedder:
        Pre-trained attribute embedder.
    config:
        Task configuration.
    """

    def __init__(
        self,
        embedder: BaseAttributeEmbedder,
        config: TripCountConfig,
    ) -> None:
        super().__init__(embedder, config)
        self.config: TripCountConfig
        self._linear: LinearHead | None = None
        self._mlp: MLPHead | None = None

    def extract_labels(
        self,
        activities_df: pl.DataFrame,
        attributes_df: pl.DataFrame,
        pids: list[str],
    ) -> np.ndarray:
        """Extract trip counts aligned to ``pids``.

        Persons with a single activity or absent from ``activities_df``
        receive count 0.

        Returns
        -------
        np.ndarray
            Int array, shape ``(len(pids),)``.
        """
        sorted_acts = activities_df.sort(["pid", "start"])

        # Build previous activity column within each pid group
        prev_act = sorted_acts.with_columns(
            pl.col("act").shift(1).over("pid").alias("prev_act")
        )

        # A transition exists where prev_act is not null (i.e., not the first activity)
        transitions = prev_act.filter(pl.col("prev_act").is_not_null())

        if self.config.exclude_home_to_home:
            transitions = transitions.filter(
                ~((pl.col("prev_act") == "home") & (pl.col("act") == "home"))
            )

        trip_counts = (
            transitions
            .group_by("pid")
            .agg(pl.len().alias("trip_count"))
        )

        pid_to_count: dict[str, int] = {
            row[0]: int(row[1]) for row in trip_counts.iter_rows()
        }
        return np.array([pid_to_count.get(p, 0) for p in pids], dtype=np.int64)

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        """Train both heads on all training data."""
        self._linear = LinearHead("regression", seed=self.config.seed)
        self._mlp = MLPHead(
            "regression",
            hidden_dim=self.config.mlp_hidden_dim,
            max_iter=self.config.mlp_max_iter,
            seed=self.config.seed,
        )
        self._linear.fit(train_embeddings, train_labels.astype(np.float32))
        self._mlp.fit(train_embeddings, train_labels.astype(np.float32))

    def evaluate(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute regression metrics.

        Returns
        -------
        dict[str, float]
            Keys: ``{linear,mlp}/{mae,rmse,r2,spearman}``.
        """
        if self._linear is None or self._mlp is None:
            raise RuntimeError("Call fit() before evaluate().")

        labels = test_labels.astype(np.float32)
        metrics: dict[str, float] = {}
        metrics.update(_regression_metrics(self._linear, test_embeddings, labels, "linear"))
        metrics.update(_regression_metrics(self._mlp, test_embeddings, labels, "mlp"))
        return metrics

    def evaluate_stratified(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        test_attributes: pl.DataFrame,
        test_pids: list[str],
        stratify_by: str,
    ) -> dict[str, dict[str, float]]:
        """Evaluate separately for each stratum of a given attribute."""
        return _evaluate_stratified(
            self, test_embeddings, test_labels, test_attributes, test_pids, stratify_by
        )

    def cross_source_evaluate(
        self,
        all_activities: pl.DataFrame,
        all_attributes: pl.DataFrame,
        encoder: AttributeEncoder,
        source_a: str,
        source_b: str,
    ) -> dict[str, float]:
        """Train on source A, evaluate on source B."""
        source_col = self.config.source_column
        pids_a = (
            all_attributes.filter(pl.col(source_col) == source_a)["pid"].to_list()
        )
        pids_b = (
            all_attributes.filter(pl.col(source_col) == source_b)["pid"].to_list()
        )
        emb_a = self.embed_dataset(all_attributes, encoder, pids_a)
        emb_b = self.embed_dataset(all_attributes, encoder, pids_b)
        labels_a = self.extract_labels(all_activities, all_attributes, pids_a)
        labels_b = self.extract_labels(all_activities, all_attributes, pids_b)
        self.fit(emb_a, labels_a)
        return self.evaluate(emb_b, labels_b)

    def plot_residuals(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        head: Literal["linear", "mlp"] = "linear",
    ) -> "matplotlib.figure.Figure":
        """Plot residuals vs predicted and residual histogram.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._linear is None or self._mlp is None:
            raise RuntimeError("Call fit() before plot_residuals().")

        h = self._linear if head == "linear" else self._mlp
        return _plot_residuals(h, test_embeddings, test_labels.astype(np.float32),
                               f"Trip count ({head})")
