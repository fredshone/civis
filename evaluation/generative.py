"""Generative evaluation task: schedule generation with ActVAE (Caveat).

This is the most demanding downstream test — it evaluates whether the
embedding is useful for *conditional generation*, not just discrimination or
regression.  The pre-trained attribute embedder replaces the label encoder in
ActVAE (Caveat) and the quality of generated schedules is measured using
density-estimation metrics from the existing distances module.

Integration status
------------------
:meth:`GenerativeEvaluator.fit` requires an external ActVAE implementation
and raises :class:`NotImplementedError` until that integration is in place.
All metric computation in :meth:`GenerativeEvaluator.evaluate` is fully
implemented using the existing ``distances`` module.

Public API
----------
GenerativeEvaluatorConfig
    Configuration for :class:`GenerativeEvaluator`.

GenerativeEvaluator
    Evaluates embedding quality via schedule generation.
    ``fit()`` → trains ActVAE (pending integration);
    ``evaluate()`` → computes density-estimation scores on (real, synthetic) schedule pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from datasets.encoding import AttributeEncoder
from distances.data import participation_matrix, time_use_matrix
from models.base import BaseAttributeEmbedder

from .base import DownstreamEvaluator, DownstreamEvaluatorConfig
from .caveat_adapter import CaveatAdapter, CaveatAdapterConfig


TransferMode = Literal["frozen", "fine_tuned", "random_init"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GenerativeEvaluatorConfig(DownstreamEvaluatorConfig):
    """Configuration for :class:`GenerativeEvaluator`.

    Parameters
    ----------
    transfer_mode:
        How to use the pre-trained embedder inside ActVAE.  See
        :class:`~evaluation.caveat_adapter.CaveatAdapterConfig` for semantics.
    caveat_config:
        Forwarded to the ActVAE constructor.  Contents depend on the
        ActVAE implementation.
    n_synthetic_samples:
        Number of synthetic schedules to generate during ``evaluate()``.
    source_column:
        Attributes column identifying the survey source.
    """

    transfer_mode: TransferMode = "frozen"
    caveat_config: dict = field(default_factory=dict)
    n_synthetic_samples: int = 1000
    source_column: str = "source"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class GenerativeEvaluator(DownstreamEvaluator):
    """Evaluates embedding quality via conditional schedule generation.

    Wraps the pre-trained embedder in a :class:`~evaluation.caveat_adapter.CaveatAdapter`
    and uses it as the label encoder for ActVAE.

    Metric computation in :meth:`evaluate` is fully implemented and uses the
    existing ``distances`` module:

    - Participation Wasserstein distance: marginal distribution over
      activity participation vectors (real vs. synthetic).
    - Timing Wasserstein distance: marginal distribution over time-use
      vectors (real vs. synthetic), computed per activity type.

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
        config: GenerativeEvaluatorConfig,
    ) -> None:
        super().__init__(embedder, config)
        self.config: GenerativeEvaluatorConfig
        self._adapter = CaveatAdapter(
            embedder,
            CaveatAdapterConfig(transfer_mode=config.transfer_mode),
        )
        self._synthetic_activities: pl.DataFrame | None = None

    def extract_labels(
        self,
        activities_df: pl.DataFrame,
        attributes_df: pl.DataFrame,
        pids: list[str],
    ) -> np.ndarray:
        """Return the activities DataFrame as an object array of rows per pid.

        For generative evaluation, labels are the full activity sequences
        rather than a scalar.  This returns a structured object array
        where each element is a list of activity rows for that pid.

        Returns
        -------
        np.ndarray
            Object array of shape ``(len(pids),)``.  Each element is a list
            of ``(act, start, end)`` tuples.
        """
        pid_to_rows: dict[str, list[tuple]] = {}
        for row in activities_df.select(["pid", "act", "start", "end"]).iter_rows():
            pid, act, start, end = row
            pid_to_rows.setdefault(pid, []).append((act, start, end))

        result = np.empty(len(pids), dtype=object)
        for i, pid in enumerate(pids):
            result[i] = pid_to_rows.get(pid, [])
        return result

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        """Train ActVAE with the wrapped label encoder.

        .. note::
            This method requires an external ActVAE implementation to be
            installed and configured.  Until that integration is available,
            calling this method raises :class:`NotImplementedError`.

            To integrate ActVAE:

            1. Install the ActVAE package.
            2. Subclass :class:`GenerativeEvaluator` and override ``fit()``.
            3. Use ``self._adapter`` as the label encoder — it satisfies
               :class:`~evaluation.caveat_adapter.LabelEncoderProtocol`.
            4. After training, store generated schedules in
               ``self._synthetic_activities`` as a Polars DataFrame with the
               same schema as the activities DataFrame.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "GenerativeEvaluator.fit() requires ActVAE integration.\n"
            "Use self._adapter (a CaveatAdapter) as ActVAE's label encoder.\n"
            "Store generated schedules in self._synthetic_activities after training."
        )

    def evaluate(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute density-estimation scores between real and synthetic schedules.

        Uses the existing ``distances`` module to compute Wasserstein distances
        between the marginal distributions of real and synthetic schedules.

        This method can be called independently of :meth:`fit` by setting
        ``self._synthetic_activities`` directly to a pre-generated activities
        DataFrame.

        Parameters
        ----------
        test_embeddings:
            Unused (kept for API consistency with base class).
        test_labels:
            Object array of real activity sequences (output of
            :meth:`extract_labels`).

        Returns
        -------
        dict[str, float]
            Keys: ``participation_wasserstein``, ``timing_wasserstein``.

        Raises
        ------
        RuntimeError
            If ``self._synthetic_activities`` has not been set.
        """
        if self._synthetic_activities is None:
            raise RuntimeError(
                "No synthetic activities available. "
                "Call fit() or set self._synthetic_activities directly."
            )

        real_activities = self._labels_to_activities(test_labels)
        synthetic_activities = self._synthetic_activities

        return self._compute_metrics(real_activities, synthetic_activities)

    def _labels_to_activities(self, labels: np.ndarray) -> pl.DataFrame:
        """Reconstruct an activities DataFrame from extract_labels output."""
        rows: list[dict] = []
        for i, seq in enumerate(labels):
            pid = f"pid_{i}"
            for j, (act, start, end) in enumerate(seq):
                rows.append({"pid": pid, "seq": j, "act": act, "start": start, "end": end})
        if not rows:
            return pl.DataFrame(schema={"pid": pl.Utf8, "seq": pl.Int32,
                                         "act": pl.Utf8, "start": pl.Int32, "end": pl.Int32})
        return pl.DataFrame(rows)

    def _compute_metrics(
        self,
        real_activities: pl.DataFrame,
        synthetic_activities: pl.DataFrame,
    ) -> dict[str, float]:
        """Compute Wasserstein distances between real and synthetic distributions."""
        from scipy.stats import wasserstein_distance

        metrics: dict[str, float] = {}

        # Participation Wasserstein: marginal over activity-type fractions
        _, real_part = participation_matrix(real_activities)
        _, syn_part = participation_matrix(synthetic_activities)
        # Compare marginal distributions per activity type
        part_distances = [
            wasserstein_distance(real_part[:, j], syn_part[:, j])
            for j in range(real_part.shape[1])
        ]
        metrics["participation_wasserstein"] = float(np.mean(part_distances))

        # Timing Wasserstein: marginal over per-minute time-use
        _, real_time = time_use_matrix(real_activities, resolution=10)
        _, syn_time = time_use_matrix(synthetic_activities, resolution=10)
        # Compare per-bin distributions
        n_bins = real_time.shape[1]
        timing_distances = [
            wasserstein_distance(real_time[:, t].astype(float),
                                 syn_time[:, t].astype(float))
            for t in range(n_bins)
        ]
        metrics["timing_wasserstein"] = float(np.mean(timing_distances))

        return metrics

    def cross_source_evaluate(
        self,
        all_activities: pl.DataFrame,
        all_attributes: pl.DataFrame,
        encoder: AttributeEncoder,
        source_a: str,
        source_b: str,
    ) -> dict[str, float]:
        """Train on source A, evaluate on source B.

        Raises
        ------
        NotImplementedError
            Until ActVAE integration is complete.
        """
        raise NotImplementedError(
            "cross_source_evaluate() requires ActVAE integration. "
            "See fit() for integration instructions."
        )

    def ablation_table(
        self,
        all_activities: pl.DataFrame,
        all_attributes: pl.DataFrame,
        encoder: AttributeEncoder,
        train_pids: list[str],
        test_pids: list[str],
        transfer_modes: list[TransferMode] | None = None,
    ) -> pl.DataFrame:
        """Run the evaluator for each transfer mode and return a comparison table.

        Parameters
        ----------
        transfer_modes:
            List of transfer modes to compare.  Defaults to
            ``['frozen', 'fine_tuned', 'random_init']``.

        Returns
        -------
        pl.DataFrame
            Long-format table with columns ``[transfer_mode, metric, value]``.

        Raises
        ------
        NotImplementedError
            Until ActVAE integration is complete.
        """
        raise NotImplementedError(
            "ablation_table() requires ActVAE integration. "
            "See fit() for integration instructions."
        )

    def conditional_collapse_profile(
        self,
        test_activities: pl.DataFrame,
        test_attributes: pl.DataFrame,
        test_pids: list[str],
        attributes_to_test: list[str],
    ) -> dict[str, float]:
        """Compute a conditionality profile across attributes.

        For each attribute in ``attributes_to_test``, splits ``test_pids``
        into two extreme groups (e.g., ``'employed'`` vs ``'retired'``),
        generates schedules for each group, and computes the Wasserstein
        distance between the generated distributions.  Higher distance
        indicates better attribute conditionality.

        Raises
        ------
        NotImplementedError
            Until ActVAE integration is complete.
        """
        raise NotImplementedError(
            "conditional_collapse_profile() requires ActVAE integration. "
            "See fit() for integration instructions."
        )
