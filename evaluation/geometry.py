"""Intrinsic embedding geometry analysis.

Evaluates embedding quality directly without downstream task labels.
All metrics compare the embedding space geometry against the schedule distance
function that was used as the training signal.

Public API
----------
GeometryAnalyserConfig
    Configuration dataclass.

GeometryAnalyser
    Computes alignment/uniformity, rank correlation, neighbourhood overlap,
    source separation, and CKA metrics.  Provides a ``full_report`` method
    that runs all analyses and saves a markdown report with plots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, wasserstein_distance

from datasets.encoding import AttributeEncoder
from datasets.masking import AttributeMasker
from models.base import BaseAttributeEmbedder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GeometryAnalyserConfig:
    """Configuration for :class:`GeometryAnalyser`.

    Parameters
    ----------
    device:
        PyTorch device for embedding inference.
    embed_batch_size:
        Batch size for the embedding forward pass.
    cache_dir:
        Directory to cache computed embeddings.
    report_dir:
        Default directory for :meth:`GeometryAnalyser.full_report` output.
    seed:
        Random seed used for all sampling operations.
    """

    device: str = "cpu"
    embed_batch_size: int = 512
    cache_dir: str | None = None
    report_dir: str | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# GeometryAnalyser
# ---------------------------------------------------------------------------


class GeometryAnalyser:
    """Intrinsic evaluation suite for embedding geometry.

    Computes five metrics that directly compare the embedding space against
    the schedule distance function:

    - **alignment / uniformity** (Wang & Isola 2020)
    - **rank correlation** — Spearman ρ between embedding and schedule distances
    - **neighbourhood overlap** — kNN recall at several k
    - **source separation** — cross-source Wasserstein and linear decodability
    - **CKA** — centred kernel alignment between the two kernel matrices

    Parameters
    ----------
    embedder:
        A trained :class:`~models.base.BaseAttributeEmbedder` (frozen during
        all inference calls).
    distance_fn:
        Callable ``(row_i, row_j) -> float`` that returns the schedule
        distance between two persons given their raw representation arrays.
        Used when schedule distances are not precomputed.
    test_attributes:
        Attributes DataFrame (must contain a ``pid`` column).
    test_pids:
        Ordered list of person IDs for the test set.
    encoder:
        Fitted :class:`~datasets.encoding.AttributeEncoder`.
    masker:
        Optional :class:`~datasets.masking.AttributeMasker` used to generate
        augmented positive pairs for the alignment metric.
    config:
        Analyser configuration.
    """

    def __init__(
        self,
        embedder: BaseAttributeEmbedder,
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
        test_attributes: pl.DataFrame,
        test_pids: list[str],
        encoder: AttributeEncoder,
        masker: AttributeMasker | None = None,
        config: GeometryAnalyserConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.distance_fn = distance_fn
        self.test_attributes = test_attributes
        self.test_pids = test_pids
        self.encoder = encoder
        self.masker = masker
        self.config = config or GeometryAnalyserConfig()

        self._embeddings: np.ndarray = self._compute_embeddings(test_attributes, test_pids)

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _compute_embeddings(
        self,
        attributes_df: pl.DataFrame,
        pids: list[str],
        cache_tag: str | None = None,
    ) -> np.ndarray:
        """Compute frozen embeddings for a list of persons.

        Mirrors the pattern from ``DownstreamEvaluator.embed_dataset``.
        """
        if cache_tag is not None and self.config.cache_dir is not None:
            cache_path = Path(self.config.cache_dir) / f"{cache_tag}.npz"
            if cache_path.exists():
                return np.load(cache_path)["embeddings"]

        pid_series = pl.Series("pid", pids)
        filtered = (
            pl.DataFrame({"pid": pid_series})
            .join(attributes_df, on="pid", how="left")
        )
        attributes = self.encoder.transform(filtered)
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
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, embeddings=embeddings)

        return embeddings

    # ------------------------------------------------------------------
    # Metric: alignment / uniformity
    # ------------------------------------------------------------------

    def alignment_uniformity(self) -> dict[str, float]:
        """Compute Wang & Isola (2020) alignment and uniformity metrics.

        Alignment
            Mean squared L2 distance between pairs of embeddings produced from
            the same attributes under two independent applications of the masker
            (augmented positive pairs).  Lower = better.  Requires
            ``self.masker`` to be set; ``"alignment"`` is ``None`` if not.

        Uniformity
            ``log mean exp(-2 ||z_i - z_j||²)`` over random pairs on the
            unit-normalised embedding sphere.  Lower = more uniform.

        Returns
        -------
        dict with keys ``"alignment"`` (float or None) and ``"uniformity"``
        (float).
        """
        result: dict[str, Any] = {}

        # --- alignment ---
        if self.masker is not None:
            pid_series = pl.Series("pid", self.test_pids)
            filtered = (
                pl.DataFrame({"pid": pid_series})
                .join(self.test_attributes, on="pid", how="left")
            )
            attributes = self.encoder.transform(filtered)
            device = torch.device(self.config.device)
            attributes = {k: v.to(device) for k, v in attributes.items()}

            self.embedder.eval()
            self.embedder.to(device)
            batch_size = self.config.embed_batch_size
            n = len(self.test_pids)
            chunks_a: list[np.ndarray] = []
            chunks_b: list[np.ndarray] = []

            with torch.no_grad():
                for start in range(0, n, batch_size):
                    batch = {k: v[start: start + batch_size] for k, v in attributes.items()}
                    emb_a = self.embedder(self.masker(batch)).cpu().numpy()
                    emb_b = self.embedder(self.masker(batch)).cpu().numpy()
                    chunks_a.append(emb_a)
                    chunks_b.append(emb_b)

            emb_a = np.concatenate(chunks_a, axis=0)
            emb_b = np.concatenate(chunks_b, axis=0)
            diff = emb_a - emb_b
            result["alignment"] = float(np.mean(np.sum(diff ** 2, axis=1)))
        else:
            result["alignment"] = None

        # --- uniformity ---
        emb = self._embeddings
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        z = emb / norms  # unit-normalised

        rng = np.random.default_rng(self.config.seed)
        n = len(z)
        n_pairs = min(10_000, n * (n - 1) // 2)
        i_idx = rng.integers(0, n, size=n_pairs)
        j_idx = rng.integers(0, n, size=n_pairs)
        # Avoid self-pairs
        same = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % n

        diff = z[i_idx] - z[j_idx]
        sq_dists = np.sum(diff ** 2, axis=1)
        result["uniformity"] = float(np.log(np.mean(np.exp(-2.0 * sq_dists))))

        return result

    # ------------------------------------------------------------------
    # Metric: rank correlation
    # ------------------------------------------------------------------

    def rank_correlation(
        self,
        n_pairs: int = 5000,
        schedule_distances: np.ndarray | None = None,
    ) -> float:
        """Spearman rank correlation between embedding and schedule distances.

        Parameters
        ----------
        n_pairs:
            Number of random person pairs to sample.
        schedule_distances:
            Precomputed schedule distances array of shape ``(n_pairs,)``.
            If ``None``, distances are computed via ``self.distance_fn``.

        Returns
        -------
        float
            Spearman ρ in [-1, 1].
        """
        rng = np.random.default_rng(self.config.seed)
        n = len(self.test_pids)
        i_idx = rng.integers(0, n, size=n_pairs)
        j_idx = rng.integers(0, n, size=n_pairs)
        same = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % n

        # Embedding distances
        diff = self._embeddings[i_idx] - self._embeddings[j_idx]
        emb_dists = np.linalg.norm(diff, axis=1)

        # Schedule distances
        if schedule_distances is not None:
            sched_dists = schedule_distances
        else:
            sched_dists = np.array([
                self.distance_fn(
                    self._embeddings[i],  # placeholder — caller must override
                    self._embeddings[j],
                )
                for i, j in zip(i_idx, j_idx)
            ])

        corr, _ = spearmanr(emb_dists, sched_dists)
        return float(corr)

    # ------------------------------------------------------------------
    # Metric: neighbourhood overlap
    # ------------------------------------------------------------------

    def neighbourhood_overlap(
        self,
        k_values: list[int] | None = None,
        schedule_distance_matrix: np.ndarray | None = None,
    ) -> dict[int, float]:
        """Fraction of k nearest neighbours in embedding space also in schedule space.

        Parameters
        ----------
        k_values:
            List of k values to evaluate.  Defaults to ``[5, 10, 20, 50]``.
        schedule_distance_matrix:
            Precomputed square distance matrix of shape ``(N, N)``.  If
            ``None``, this metric cannot be computed and returns 0.0 for all k.

        Returns
        -------
        dict mapping each k to its overlap score in [0, 1].
        """
        if k_values is None:
            k_values = [5, 10, 20, 50]

        if schedule_distance_matrix is None:
            return {k: 0.0 for k in k_values}

        n = len(self.test_pids)
        max_k = max(k_values)

        # kNN in embedding space
        nn_emb = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
        nn_emb.fit(self._embeddings)
        _, emb_indices = nn_emb.kneighbors(self._embeddings)
        # emb_indices[i] includes i itself at position 0; drop it
        emb_indices = emb_indices[:, 1:]  # (N, max_k)

        # kNN in schedule space (from precomputed matrix)
        # argsort each row, exclude diagonal
        sched_sorted = np.argsort(schedule_distance_matrix, axis=1)[:, 1:]  # (N, N-1)

        results: dict[int, float] = {}
        for k in k_values:
            k_eff = min(k, n - 1)
            overlaps = []
            emb_k = emb_indices[:, :k_eff]
            sched_k = sched_sorted[:, :k_eff]
            for i in range(n):
                e_set = set(emb_k[i].tolist())
                s_set = set(sched_k[i].tolist())
                overlaps.append(len(e_set & s_set) / k_eff)
            results[k] = float(np.mean(overlaps))

        return results

    # ------------------------------------------------------------------
    # Metric: source separation
    # ------------------------------------------------------------------

    def source_separation(self, source_column: str = "source") -> dict[str, Any]:
        """Measure how well the embedding separates (or mixes) data sources.

        Lower separation = better cross-source alignment.

        Computes:
        - ``mean_wasserstein``: mean 1-D Wasserstein distance (first PCA
          component) between all source-pair embedding distributions.
        - ``source_accuracy``: accuracy of a linear source classifier trained
          on embeddings.  Lower = less source-decodable (better).
        - ``per_source_pair``: per-pair Wasserstein dict.

        Parameters
        ----------
        source_column:
            Name of the source column in ``test_attributes``.

        Returns
        -------
        dict with keys ``"mean_wasserstein"``, ``"source_accuracy"``,
        ``"per_source_pair"``.
        """
        if source_column not in self.test_attributes.columns:
            return {"mean_wasserstein": 0.0, "source_accuracy": 0.0, "per_source_pair": {}}

        pid_to_source = dict(
            zip(
                self.test_attributes["pid"].to_list(),
                self.test_attributes[source_column].to_list(),
            )
        )
        sources = np.array([pid_to_source.get(p) for p in self.test_pids])
        unique_sources = [s for s in np.unique(sources) if s is not None]

        # Project to 1D via PCA for Wasserstein
        emb = self._embeddings
        pca = PCA(n_components=1, random_state=self.config.seed)
        proj = pca.fit_transform(emb).ravel()  # (N,)

        # Per-source-pair Wasserstein distances
        per_pair: dict[str, float] = {}
        for i, sa in enumerate(unique_sources):
            for sb in unique_sources[i + 1:]:
                mask_a = sources == sa
                mask_b = sources == sb
                if mask_a.sum() == 0 or mask_b.sum() == 0:
                    continue
                dist = float(wasserstein_distance(proj[mask_a], proj[mask_b]))
                per_pair[f"{sa}_vs_{sb}"] = dist

        mean_w = float(np.mean(list(per_pair.values()))) if per_pair else 0.0

        # Source decodability: logistic regression accuracy
        valid = sources != np.array(None)
        labels = sources[valid]
        feats = emb[valid]
        if len(np.unique(labels)) < 2 or len(labels) < 4:
            source_accuracy = 0.0
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                feats, labels, test_size=0.3, random_state=self.config.seed, stratify=labels
            )
            clf = LogisticRegression(max_iter=500, random_state=self.config.seed)
            clf.fit(X_train, y_train)
            source_accuracy = float(clf.score(X_test, y_test))

        return {
            "mean_wasserstein": mean_w,
            "source_accuracy": source_accuracy,
            "per_source_pair": per_pair,
        }

    # ------------------------------------------------------------------
    # Metric: CKA
    # ------------------------------------------------------------------

    def cka_with_schedule_kernel(
        self,
        n_samples: int = 500,
        schedule_distance_matrix: np.ndarray | None = None,
    ) -> float:
        """Centred Kernel Alignment between embedding and schedule kernels.

        Uses an RBF kernel with bandwidth set to the median pairwise distance
        for each kernel separately.

        Parameters
        ----------
        n_samples:
            Number of test points to subsample.
        schedule_distance_matrix:
            Precomputed square schedule distance matrix ``(N, N)``.  Required;
            returns 0.0 if not provided.

        Returns
        -------
        float
            CKA value in [0, 1].  1 = perfect alignment.
        """
        if schedule_distance_matrix is None:
            return 0.0

        rng = np.random.default_rng(self.config.seed)
        n = len(self.test_pids)
        n_s = min(n_samples, n)
        idx = rng.choice(n, size=n_s, replace=False)

        emb = self._embeddings[idx]
        sched_dists = schedule_distance_matrix[np.ix_(idx, idx)]

        # Embedding pairwise squared distances
        emb_sq = np.sum(emb ** 2, axis=1, keepdims=True)
        emb_dist2 = emb_sq + emb_sq.T - 2 * (emb @ emb.T)
        emb_dist2 = np.clip(emb_dist2, 0.0, None)

        # RBF kernels — bandwidth = median pairwise distance
        emb_median = np.median(np.sqrt(emb_dist2[emb_dist2 > 0])) if np.any(emb_dist2 > 0) else 1.0
        sched_median = np.median(sched_dists[sched_dists > 0]) if np.any(sched_dists > 0) else 1.0

        K_emb = np.exp(-emb_dist2 / (2 * emb_median ** 2 + 1e-8))
        K_sched = np.exp(-(sched_dists ** 2) / (2 * sched_median ** 2 + 1e-8))

        return float(_linear_cka(K_emb, K_sched))

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        output_dir: str | None = None,
        schedule_distance_matrix: np.ndarray | None = None,
        source_column: str = "source",
        n_rank_pairs: int = 5000,
        rank_schedule_distances: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run all analyses and save a markdown report with plots.

        Parameters
        ----------
        output_dir:
            Directory for report output.  Falls back to ``config.report_dir``.
            If both are ``None``, plots are not saved.
        schedule_distance_matrix:
            Precomputed ``(N, N)`` distance matrix used by neighbourhood
            overlap and CKA.
        source_column:
            Source column name passed to :meth:`source_separation`.
        n_rank_pairs:
            Number of pairs for :meth:`rank_correlation`.
        rank_schedule_distances:
            Precomputed schedule distances for rank correlation sampling.

        Returns
        -------
        dict
            All metric results keyed by method name.
        """
        results: dict[str, Any] = {}

        results["alignment_uniformity"] = self.alignment_uniformity()
        results["rank_correlation"] = self.rank_correlation(
            n_pairs=n_rank_pairs, schedule_distances=rank_schedule_distances
        )
        results["neighbourhood_overlap"] = self.neighbourhood_overlap(
            schedule_distance_matrix=schedule_distance_matrix
        )
        results["source_separation"] = self.source_separation(source_column=source_column)
        results["cka"] = self.cka_with_schedule_kernel(
            schedule_distance_matrix=schedule_distance_matrix
        )

        out_dir = Path(output_dir) if output_dir else (
            Path(self.config.report_dir) if self.config.report_dir else None
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            figs = self._make_plots(
                schedule_distance_matrix=schedule_distance_matrix,
                source_column=source_column,
                n_rank_pairs=n_rank_pairs,
                rank_schedule_distances=rank_schedule_distances,
            )
            for name, fig in figs.items():
                fig.savefig(out_dir / f"{name}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)

            self._write_markdown(results, out_dir)

        return results

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _make_plots(
        self,
        schedule_distance_matrix: np.ndarray | None,
        source_column: str,
        n_rank_pairs: int,
        rank_schedule_distances: np.ndarray | None,
    ) -> dict[str, plt.Figure]:
        figs: dict[str, plt.Figure] = {}

        # (a) t-SNE coloured by source
        try:
            from sklearn.manifold import TSNE

            emb = self._embeddings
            n = len(emb)
            n_tsne = min(1000, n)
            rng = np.random.default_rng(self.config.seed)
            idx = rng.choice(n, size=n_tsne, replace=False)
            tsne = TSNE(n_components=2, random_state=self.config.seed, perplexity=min(30, n_tsne - 1))
            proj = tsne.fit_transform(emb[idx])

            fig, ax = plt.subplots(figsize=(7, 5))
            if source_column in self.test_attributes.columns:
                pid_to_source = dict(
                    zip(
                        self.test_attributes["pid"].to_list(),
                        self.test_attributes[source_column].to_list(),
                    )
                )
                sources = [pid_to_source.get(self.test_pids[i]) for i in idx]
                unique_s = list(dict.fromkeys(sources))
                cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(unique_s))
                s_to_idx = {s: i for i, s in enumerate(unique_s)}
                colors = [cmap(s_to_idx.get(s, 0)) for s in sources]
                for s in unique_s:
                    mask = [src == s for src in sources]
                    ax.scatter(proj[mask, 0], proj[mask, 1], label=s, s=4, alpha=0.6)
                ax.legend(markerscale=3, fontsize=7)
            else:
                ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.6)
            ax.set_title("t-SNE of embeddings")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            figs["tsne"] = fig
        except Exception:
            pass

        # (b) Embedding distance vs schedule distance scatter
        if rank_schedule_distances is not None or schedule_distance_matrix is not None:
            rng = np.random.default_rng(self.config.seed)
            n = len(self.test_pids)
            n_pairs = min(n_rank_pairs, 2000)
            i_idx = rng.integers(0, n, size=n_pairs)
            j_idx = rng.integers(0, n, size=n_pairs)
            same = i_idx == j_idx
            j_idx[same] = (j_idx[same] + 1) % n

            diff = self._embeddings[i_idx] - self._embeddings[j_idx]
            emb_dists = np.linalg.norm(diff, axis=1)

            if rank_schedule_distances is not None:
                s_dists = rank_schedule_distances[:n_pairs]
            else:
                s_dists = schedule_distance_matrix[i_idx, j_idx]

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(s_dists, emb_dists, s=3, alpha=0.3)
            ax.set_xlabel("Schedule distance")
            ax.set_ylabel("Embedding L2 distance")
            ax.set_title("Embedding vs Schedule distances")
            figs["distance_scatter"] = fig

        # (c) Neighbourhood overlap bar chart
        if schedule_distance_matrix is not None:
            overlap = self.neighbourhood_overlap(schedule_distance_matrix=schedule_distance_matrix)
            fig, ax = plt.subplots(figsize=(5, 4))
            ks = sorted(overlap.keys())
            vals = [overlap[k] for k in ks]
            ax.bar([str(k) for k in ks], vals)
            ax.set_xlabel("k")
            ax.set_ylabel("Neighbourhood overlap")
            ax.set_title("kNN overlap: embedding vs schedule space")
            ax.set_ylim(0, 1)
            figs["neighbourhood_overlap"] = fig

        return figs

    def _write_markdown(self, results: dict[str, Any], out_dir: Path) -> None:
        au = results["alignment_uniformity"]
        rc = results["rank_correlation"]
        no = results["neighbourhood_overlap"]
        ss = results["source_separation"]
        cka = results["cka"]

        lines = [
            "# Geometry Analysis Report\n",
            "## Alignment / Uniformity\n",
            f"- Alignment: {au.get('alignment')}\n",
            f"- Uniformity: {au.get('uniformity'):.4f}\n\n",
            "## Rank Correlation\n",
            f"- Spearman ρ (embedding vs schedule distance): **{rc:.4f}**\n\n",
            "## Neighbourhood Overlap\n",
        ]
        for k, v in sorted(no.items()):
            lines.append(f"- k={k}: {v:.4f}\n")
        lines += [
            "\n## Source Separation\n",
            f"- Mean Wasserstein: {ss['mean_wasserstein']:.4f}\n",
            f"- Source accuracy (lower = better): {ss['source_accuracy']:.4f}\n",
            "\n### Per-source-pair Wasserstein\n",
        ]
        for pair, dist in ss["per_source_pair"].items():
            lines.append(f"- {pair}: {dist:.4f}\n")
        lines += [
            "\n## CKA with Schedule Kernel\n",
            f"- CKA: **{cka:.4f}**\n\n",
            "## Plots\n",
            "![t-SNE](tsne.png)\n",
            "![Distance scatter](distance_scatter.png)\n",
            "![Neighbourhood overlap](neighbourhood_overlap.png)\n",
        ]
        (out_dir / "geometry_report.md").write_text("".join(lines))


# ---------------------------------------------------------------------------
# CKA helper
# ---------------------------------------------------------------------------


def _linear_cka(K: np.ndarray, L: np.ndarray) -> float:
    """Compute linear CKA between two kernel matrices.

    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    where HSIC(K, L) = tr(KHLH) / (n-1)²
    with H = I - (1/n) 11ᵀ.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = K @ H
    LH = L @ H
    hsic_kl = np.trace(KH @ LH) / (n - 1) ** 2
    hsic_kk = np.trace(KH @ KH) / (n - 1) ** 2
    hsic_ll = np.trace(LH @ LH) / (n - 1) ** 2
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)
