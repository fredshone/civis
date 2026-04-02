"""Discrete choice downstream evaluation task.

Predicts whether a person goes to work on a given day — a binary
classification task derived directly from the activity schedule.

Public API
----------
WorkParticipationConfig
    Configuration for the work participation task.

WorkParticipationEvaluator
    Binary classification of work participation from attribute embeddings.
    Supports stratified evaluation, cross-source generalisation testing,
    calibration plotting, and error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from datasets.encoding import AttributeEncoder
from models.base import BaseAttributeEmbedder

from .base import DownstreamEvaluator, DownstreamEvaluatorConfig, LinearHead, MLPHead

if TYPE_CHECKING:
    import matplotlib.figure


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WorkParticipationConfig(DownstreamEvaluatorConfig):
    """Configuration for :class:`WorkParticipationEvaluator`.

    Parameters
    ----------
    source_column:
        Column in the attributes DataFrame identifying the survey source.
    employment_column:
        Column in the attributes DataFrame for employment status.
    calibration_n_bins:
        Number of bins for the calibration curve.
    """

    source_column: str = "source"
    employment_column: str = "employment"
    calibration_n_bins: int = 10


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class WorkParticipationEvaluator(DownstreamEvaluator):
    """Binary classification: does this person participate in work today?

    Labels are derived from the activities schedule — 1 if any activity
    has ``act == 'work'``, 0 otherwise.  Persons absent from
    ``activities_df`` receive label 0.

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
        config: WorkParticipationConfig,
    ) -> None:
        super().__init__(embedder, config)
        self.config: WorkParticipationConfig
        self._linear: LinearHead | None = None
        self._mlp: MLPHead | None = None

    def extract_labels(
        self,
        activities_df: pl.DataFrame,
        attributes_df: pl.DataFrame,
        pids: list[str],
    ) -> np.ndarray:
        """Extract binary work participation labels aligned to ``pids``.

        Parameters
        ----------
        activities_df:
            Activities DataFrame with columns ``pid``, ``act``.
        attributes_df:
            Unused; present for API consistency.
        pids:
            Ordered person IDs.

        Returns
        -------
        np.ndarray
            Binary int64 array, shape ``(len(pids),)``.  1 = works, 0 = does not.
        """
        work_pids = set(
            activities_df
            .filter(pl.col("act") == "work")
            ["pid"]
            .to_list()
        )
        return np.array([1 if p in work_pids else 0 for p in pids], dtype=np.int64)

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        """Train both linear and MLP heads."""
        self._linear = LinearHead("classification", seed=self.config.seed)
        self._mlp = MLPHead(
            "classification",
            hidden_dim=self.config.mlp_hidden_dim,
            max_iter=self.config.mlp_max_iter,
            seed=self.config.seed,
        )
        self._linear.fit(train_embeddings, train_labels)
        self._mlp.fit(train_embeddings, train_labels)

    def evaluate(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute classification metrics for both heads.

        Returns
        -------
        dict[str, float]
            Keys: ``{linear,mlp}/{accuracy,auc,f1,brier}``.
        """
        if self._linear is None or self._mlp is None:
            raise RuntimeError("Call fit() before evaluate().")

        metrics: dict[str, float] = {}
        for prefix, head in [("linear", self._linear), ("mlp", self._mlp)]:
            preds = head.predict(test_embeddings)
            proba = head.predict_proba(test_embeddings)[:, 1]
            metrics[f"{prefix}/accuracy"] = float(accuracy_score(test_labels, preds))
            metrics[f"{prefix}/auc"] = float(roc_auc_score(test_labels, proba))
            metrics[f"{prefix}/f1"] = float(
                f1_score(test_labels, preds, zero_division=0)
            )
            metrics[f"{prefix}/brier"] = float(brier_score_loss(test_labels, proba))
        return metrics

    def evaluate_stratified(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        test_attributes: pl.DataFrame,
        test_pids: list[str],
        stratify_by: str,
    ) -> dict[str, dict[str, float]]:
        """Evaluate separately for each stratum of a given attribute.

        Parameters
        ----------
        test_embeddings:
            Shape ``(n_test, embed_dim)``.
        test_labels:
            Shape ``(n_test,)``.
        test_attributes:
            Attributes DataFrame for the test split.
        test_pids:
            Ordered person IDs aligned to ``test_embeddings``.
        stratify_by:
            Column name in ``test_attributes`` to stratify on.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping from stratum value to metrics dict.
        """
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
            if len(np.unique(sub_labels)) < 2:
                continue  # skip degenerate strata
            results[group] = self.evaluate(sub_emb, sub_labels)
        return results

    def cross_source_evaluate(
        self,
        all_activities: pl.DataFrame,
        all_attributes: pl.DataFrame,
        encoder: AttributeEncoder,
        source_a: str,
        source_b: str,
    ) -> dict[str, float]:
        """Train on source A, evaluate on source B.

        Parameters
        ----------
        all_activities / all_attributes:
            Full DataFrames covering both sources.
        encoder:
            Fitted encoder.
        source_a:
            Training source name (value of ``config.source_column``).
        source_b:
            Test source name.

        Returns
        -------
        dict[str, float]
            Metrics from :meth:`evaluate`.
        """
        source_col = self.config.source_column
        pids_a = (
            all_attributes
            .filter(pl.col(source_col) == source_a)
            ["pid"]
            .to_list()
        )
        pids_b = (
            all_attributes
            .filter(pl.col(source_col) == source_b)
            ["pid"]
            .to_list()
        )
        emb_a = self.embed_dataset(all_attributes, encoder, pids_a)
        emb_b = self.embed_dataset(all_attributes, encoder, pids_b)
        labels_a = self.extract_labels(all_activities, all_attributes, pids_a)
        labels_b = self.extract_labels(all_activities, all_attributes, pids_b)
        self.fit(emb_a, labels_a)
        return self.evaluate(emb_b, labels_b)

    def plot_calibration(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        ax: "plt.Axes | None" = None,
    ) -> "matplotlib.figure.Figure":
        """Plot calibration curves for both heads.

        Parameters
        ----------
        ax:
            Optional existing axes.  If ``None``, a new figure is created.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._linear is None or self._mlp is None:
            raise RuntimeError("Call fit() before plot_calibration().")

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.get_figure()

        n_bins = self.config.calibration_n_bins
        for prefix, head, color in [
            ("linear", self._linear, "steelblue"),
            ("mlp", self._mlp, "darkorange"),
        ]:
            proba = head.predict_proba(test_embeddings)[:, 1]
            frac_pos, mean_pred = calibration_curve(
                test_labels, proba, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=prefix, color=color)

        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration: work participation")
        ax.legend()
        return fig

    def error_analysis(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        test_attributes: pl.DataFrame,
        test_pids: list[str],
    ) -> pl.DataFrame:
        """Return a DataFrame of misclassified cases with attribute context.

        Parameters
        ----------
        test_embeddings:
            Shape ``(n_test, embed_dim)``.
        test_labels:
            True binary labels.
        test_attributes:
            Attributes DataFrame for the test split.
        test_pids:
            Ordered person IDs.

        Returns
        -------
        pl.DataFrame
            Columns: ``pid``, ``true_label``, ``predicted_label``,
            ``predicted_proba``, plus all attribute columns present in
            ``test_attributes``.
        """
        if self._linear is None:
            raise RuntimeError("Call fit() before error_analysis().")

        preds = self._linear.predict(test_embeddings)
        proba = self._linear.predict_proba(test_embeddings)[:, 1]

        misclassified = np.where(preds != test_labels)[0]

        result = pl.DataFrame({
            "pid": pl.Series([test_pids[i] for i in misclassified], dtype=pl.Utf8),
            "true_label": test_labels[misclassified].tolist(),
            "predicted_label": preds[misclassified].tolist(),
            "predicted_proba": proba[misclassified].tolist(),
        })

        if len(result) == 0:
            return result

        pid_filter = set(result["pid"].to_list())
        attrs_sub = test_attributes.filter(pl.col("pid").is_in(pid_filter))
        result = result.join(attrs_sub, on="pid", how="left")
        return result
