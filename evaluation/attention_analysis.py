"""Attention weight analysis for the self-attention embedding model.

Extracts and analyses learned attribute interaction patterns from a trained
:class:`~models.attention.SelfAttentionEmbedder`.

Public API
----------
AttentionAnalyserConfig
    Configuration dataclass.

AttentionAnalyser
    Computes mean attention weights, plots heatmaps, quantifies per-attribute
    importance, analyses source modulation, and checks interaction consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from datasets.encoding import AttributeEncoder
from models.attention import SelfAttentionEmbedder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AttentionAnalyserConfig:
    """Configuration for :class:`AttentionAnalyser`.

    Parameters
    ----------
    device:
        PyTorch device string.
    embed_batch_size:
        Batch size for forward passes.
    report_dir:
        Default output directory.
    top_k_interactions:
        Number of strongest off-diagonal attention pairs to highlight.
    """

    device: str = "cpu"
    embed_batch_size: int = 256
    report_dir: str | None = None
    top_k_interactions: int = 5


# ---------------------------------------------------------------------------
# AttentionAnalyser
# ---------------------------------------------------------------------------


class AttentionAnalyser:
    """Analysis tools for the self-attention attribute embedding model.

    Extracts averaged attention weight matrices from a trained
    :class:`~models.attention.SelfAttentionEmbedder` and provides methods to
    visualise and interpret learned attribute interaction patterns.

    Parameters
    ----------
    attention_embedder:
        A trained :class:`~models.attention.SelfAttentionEmbedder`.
    dataset_attributes:
        Attributes DataFrame for the analysis split.
    dataset_pids:
        Ordered list of person IDs for the analysis split.
    encoder:
        Fitted :class:`~datasets.encoding.AttributeEncoder`.
    config:
        Analyser configuration.
    """

    def __init__(
        self,
        attention_embedder: SelfAttentionEmbedder,
        dataset_attributes: pl.DataFrame,
        dataset_pids: list[str],
        encoder: AttributeEncoder,
        config: AttentionAnalyserConfig | None = None,
    ) -> None:
        self.embedder = attention_embedder
        self.dataset_attributes = dataset_attributes
        self.dataset_pids = dataset_pids
        self.encoder = encoder
        self.config = config or AttentionAnalyserConfig()

        # Ordered list of active attribute names (no CLS)
        self._attr_names: list[str] = [
            cfg.name for cfg in attention_embedder._active_attribute_configs
        ]
        self._use_cls_token: bool = attention_embedder.use_cls_token
        self._n_layers: int = len(attention_embedder.layers)

        # Sequence labels: [CLS, attr0, attr1, ...] or [attr0, attr1, ...]
        self._seq_labels: list[str] = (
            ["CLS"] + self._attr_names if self._use_cls_token else list(self._attr_names)
        )

        # Cache for mean attention weights
        self._cached_mean_attn: np.ndarray | None = None
        self._cached_pids: list[str] | None = None

    # ------------------------------------------------------------------
    # Core: mean attention weights
    # ------------------------------------------------------------------

    def mean_attention_weights(
        self, split_pids: list[str] | None = None
    ) -> np.ndarray:
        """Compute mean attention weights averaged over a set of persons.

        Parameters
        ----------
        split_pids:
            Person IDs to average over.  ``None`` uses all stored pids.

        Returns
        -------
        np.ndarray
            Shape ``(n_layers, seq_len, seq_len)``.
        """
        pids = split_pids if split_pids is not None else self.dataset_pids

        # Use cached result when the same pids are requested
        if self._cached_pids == pids and self._cached_mean_attn is not None:
            return self._cached_mean_attn

        pid_series = pl.Series("pid", pids)
        filtered = (
            pl.DataFrame({"pid": pid_series})
            .join(self.dataset_attributes, on="pid", how="left")
        )
        attributes = self.encoder.transform(filtered)
        device = torch.device(self.config.device)
        attributes = {k: v.to(device) for k, v in attributes.items()}

        self.embedder.eval()
        self.embedder.to(device)
        batch_size = self.config.embed_batch_size
        n = len(pids)

        # Accumulate per-layer sums
        layer_sums: list[np.ndarray] | None = None
        total = 0

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = {k: v[start: start + batch_size] for k, v in attributes.items()}
                _, attn_list = self.embedder(batch, return_attention=True)
                # attn_list: list[Tensor (B, seq, seq)]
                b = attn_list[0].shape[0]
                if layer_sums is None:
                    layer_sums = [np.zeros(a.shape[1:], dtype=np.float64) for a in attn_list]
                for l_idx, attn in enumerate(attn_list):
                    layer_sums[l_idx] += attn.cpu().numpy().sum(axis=0)
                total += b

        if layer_sums is None or total == 0:
            seq_len = len(self._seq_labels)
            return np.zeros((self._n_layers, seq_len, seq_len), dtype=np.float32)

        result = np.stack([s / total for s in layer_sums], axis=0).astype(np.float32)
        self._cached_mean_attn = result
        self._cached_pids = pids
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_attention_heatmap(
        self,
        layer: int,
        attribute_names: list[str] | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot a mean attention weight matrix as a labelled heatmap.

        Parameters
        ----------
        layer:
            Which transformer layer to visualise (0-indexed).
        attribute_names:
            Sequence position labels.  Defaults to ``self._seq_labels``.
        ax:
            Existing Axes to draw into; a new figure is created if ``None``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        attn = self.mean_attention_weights()  # (n_layers, seq, seq)
        mat = attn[layer]  # (seq, seq)

        labels = attribute_names if attribute_names is not None else self._seq_labels

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.45)))
        else:
            fig = ax.get_figure()

        try:
            import seaborn as sns
            sns.heatmap(mat, xticklabels=labels, yticklabels=labels,
                        ax=ax, cmap="viridis", vmin=0,
                        linewidths=0.2, linecolor="white")
        except ImportError:
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            plt.colorbar(im, ax=ax)

        ax.set_title(f"Mean attention weights — layer {layer}")
        ax.set_xlabel("Key (attending to)")
        ax.set_ylabel("Query (attending from)")

        # Annotate top-k off-diagonal interactions
        k = self.config.top_k_interactions
        n = mat.shape[0]
        off_diag = mat.copy()
        np.fill_diagonal(off_diag, -np.inf)
        flat_idx = np.argsort(off_diag.ravel())[::-1][:k]
        for fi in flat_idx:
            row, col = divmod(int(fi), n)
            ax.add_patch(
                plt.Rectangle((col, row), 1, 1, fill=False, edgecolor="red", lw=1.5)
            )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Attribute importance
    # ------------------------------------------------------------------

    def attribute_importance(self, target_attribute: str) -> dict[str, float]:
        """How much do other attributes attend to ``target_attribute``?

        Sums the attention column of ``target_attribute`` across all layers,
        excluding self-attention.

        Parameters
        ----------
        target_attribute:
            Name of the attribute to analyse (must be in ``self._attr_names``).

        Returns
        -------
        dict mapping source attribute name → summed attention weight.
        Raises ``KeyError`` if ``target_attribute`` is not found.
        """
        if target_attribute not in self._seq_labels:
            raise KeyError(
                f"{target_attribute!r} not found in sequence labels {self._seq_labels}"
            )

        attn = self.mean_attention_weights()  # (n_layers, seq, seq)
        target_col = self._seq_labels.index(target_attribute)

        importance: dict[str, float] = {}
        for src_idx, src_name in enumerate(self._seq_labels):
            if src_idx == target_col:
                continue
            # Sum column `target_col` row `src_idx` across layers
            importance[src_name] = float(attn[:, src_idx, target_col].sum())

        return importance

    # ------------------------------------------------------------------
    # Source modulation analysis
    # ------------------------------------------------------------------

    def source_modulation_analysis(
        self,
        source_attribute: str = "source",
        reference_sources: tuple[str, str] | None = None,
    ) -> dict[str, float]:
        """Measure how much changing the source attribute shifts other token representations.

        For two source values A and B, embeds all test persons twice (once with
        source overridden to A, once to B).  For each non-source attribute
        position the mean L2 distance between the two resulting post-attention
        token representations is computed.

        Because the full internal representation after each transformer block is
        not directly exposed, this method uses the attention-weighted output:
        it computes embeddings with ``return_attention=True`` and uses the
        final ``tokens`` tensor reconstructed via a forward hook.

        Parameters
        ----------
        source_attribute:
            Name of the source attribute in the encoder vocab.
        reference_sources:
            A pair ``(source_A, source_B)`` to compare.  If ``None``, the
            first two unique source values in the dataset are used.

        Returns
        -------
        dict mapping attribute name → mean modulation magnitude.  The source
        attribute itself is excluded.  Returns an empty dict if the source
        attribute is not found or fewer than two unique values exist.
        """
        if source_attribute not in self.encoder._vocab:
            return {}

        vocab = self.encoder._vocab[source_attribute]
        unique_sources = [v for v in vocab if v != "unknown"]
        if len(unique_sources) < 2:
            return {}

        if reference_sources is not None:
            src_a, src_b = reference_sources
        else:
            src_a, src_b = unique_sources[0], unique_sources[1]

        src_to_idx = {v: i for i, v in enumerate(vocab)}
        idx_a = src_to_idx.get(src_a)
        idx_b = src_to_idx.get(src_b)
        if idx_a is None or idx_b is None:
            return {}

        # Build attribute tensors for the dataset, overriding source
        pid_series = pl.Series("pid", self.dataset_pids)
        filtered = (
            pl.DataFrame({"pid": pid_series})
            .join(self.dataset_attributes, on="pid", how="left")
        )
        attributes = self.encoder.transform(filtered)
        device = torch.device(self.config.device)
        attributes = {k: v.to(device) for k, v in attributes.items()}

        n = len(self.dataset_pids)
        batch_size = self.config.embed_batch_size

        # We capture token representations using a forward hook on the last layer
        token_reps_a: list[np.ndarray] = []
        token_reps_b: list[np.ndarray] = []

        def _make_hook(storage: list[np.ndarray]):
            def hook(module, input, output):
                # output is (tokens_out, attn_weights)
                # tokens_out shape: (B, seq, d_embed)
                storage.append(output[0].detach().cpu().numpy())
            return hook

        self.embedder.eval()
        self.embedder.to(device)
        last_layer = self.embedder.layers[-1]

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = {k: v[start: start + batch_size] for k, v in attributes.items()}

                # Variant A
                batch_a = dict(batch)
                if source_attribute in batch_a:
                    batch_a[source_attribute] = torch.full_like(
                        batch_a[source_attribute], idx_a
                    )
                h_a = last_layer.register_forward_hook(_make_hook(token_reps_a))
                self.embedder(batch_a)
                h_a.remove()

                # Variant B
                batch_b = dict(batch)
                if source_attribute in batch_b:
                    batch_b[source_attribute] = torch.full_like(
                        batch_b[source_attribute], idx_b
                    )
                h_b = last_layer.register_forward_hook(_make_hook(token_reps_b))
                self.embedder(batch_b)
                h_b.remove()

        tokens_a = np.concatenate(token_reps_a, axis=0)  # (N, seq, d_embed)
        tokens_b = np.concatenate(token_reps_b, axis=0)  # (N, seq, d_embed)

        # seq positions: if use_cls_token, pos 0 = CLS, pos 1+ = attrs
        offset = 1 if self._use_cls_token else 0
        modulation: dict[str, float] = {}
        source_seq_pos = (offset + self._attr_names.index(source_attribute)
                          if source_attribute in self._attr_names else None)

        for attr_idx, attr_name in enumerate(self._attr_names):
            if attr_name == source_attribute:
                continue
            seq_pos = offset + attr_idx
            diff = tokens_a[:, seq_pos, :] - tokens_b[:, seq_pos, :]
            modulation[attr_name] = float(np.mean(np.linalg.norm(diff, axis=1)))

        return modulation

    # ------------------------------------------------------------------
    # Interaction consistency
    # ------------------------------------------------------------------

    def interaction_consistency(
        self,
        expected_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Test whether learned attention interactions match domain expectations.

        Checks that expected high-attention attribute pairs (based on domain
        knowledge) rank highly in the mean attention matrix.

        Default expected pairs
        ----------------------
        - ``("source", "zone")`` — source should attend to zone type
        - ``("employment", "age")`` — employment should attend to age
        - ``("year", "country")`` — year should attend to country (Covid)

        Parameters
        ----------
        expected_pairs:
            List of ``(query_attribute, key_attribute)`` tuples expected to
            have high attention weight.

        Returns
        -------
        dict with keys:
        - ``"consistency_score"``: fraction of expected pairs in the top-k
          off-diagonal attention pairs (across any layer).
        - ``"expected_pairs_attention"``: dict mapping ``"q->k"`` to mean
          attention weight summed over layers.
        - ``"unexpected_top_pairs"``: list of top off-diagonal pairs not in
          the expected set, as ``{"query": str, "key": str, "weight": float}``.
        """
        if expected_pairs is None:
            expected_pairs = [
                ("source", "zone"),
                ("employment", "age"),
                ("year", "country"),
            ]

        attn = self.mean_attention_weights()  # (n_layers, seq, seq)
        # Sum over layers for a single importance matrix
        summed = attn.sum(axis=0)  # (seq, seq)

        n_seq = summed.shape[0]
        labels = self._seq_labels

        # Build set of valid expected pairs (both tokens must exist in labels)
        valid_expected: list[tuple[str, str]] = [
            (q, k) for q, k in expected_pairs
            if q in labels and k in labels
        ]

        # Off-diagonal entries: collect (weight, q_label, k_label)
        off_diag_entries: list[tuple[float, str, str]] = []
        for row in range(n_seq):
            for col in range(n_seq):
                if row == col:
                    continue
                off_diag_entries.append((float(summed[row, col]), labels[row], labels[col]))
        off_diag_entries.sort(key=lambda x: x[0], reverse=True)

        k = max(self.config.top_k_interactions, len(valid_expected))
        top_k_set = {(q, ki) for _, q, ki in off_diag_entries[:k]}

        # Consistency score
        n_found = sum(1 for q, ki in valid_expected if (q, ki) in top_k_set)
        consistency_score = n_found / len(valid_expected) if valid_expected else 0.0

        # Attention weights for expected pairs
        expected_attn: dict[str, float] = {}
        for q, ki in valid_expected:
            row = labels.index(q)
            col = labels.index(ki)
            expected_attn[f"{q}->{ki}"] = float(summed[row, col])

        # Unexpected top pairs
        expected_set_labels = {(q, ki) for q, ki in valid_expected}
        unexpected = [
            {"query": q, "key": ki, "weight": w}
            for w, q, ki in off_diag_entries[:k]
            if (q, ki) not in expected_set_labels
        ]

        return {
            "consistency_score": float(consistency_score),
            "expected_pairs_attention": expected_attn,
            "unexpected_top_pairs": unexpected,
        }
