"""Tests for evaluation/attention_analysis.py.

Uses the shared fixture CSV files (3 persons) and a tiny SelfAttentionEmbedder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datasets.encoding import AttributeEncoder, default_attribute_configs
from distances.data import load_activities, load_attributes
from evaluation.attention_analysis import AttentionAnalyser, AttentionAnalyserConfig
from models import AttributeEmbedderConfig
from models.attention import SelfAttentionEmbedder

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def activities_df():
    return load_activities(_FIXTURES / "activities.csv")


@pytest.fixture
def attributes_df():
    return load_attributes(_FIXTURES / "attributes.csv")


@pytest.fixture
def encoder(attributes_df):
    enc = AttributeEncoder(default_attribute_configs())
    enc.fit(attributes_df)
    return enc


@pytest.fixture
def all_pids(attributes_df):
    return attributes_df["pid"].to_list()


@pytest.fixture
def tiny_attn_config(encoder):
    return AttributeEmbedderConfig.from_encoder(
        encoder,
        d_embed=8,
        d_model=8,
        dropout=0.0,
        n_heads=1,
        n_layers=1,
        use_cls_token=True,
        pooling="cls",
    )


@pytest.fixture
def tiny_attn_embedder(tiny_attn_config):
    model = SelfAttentionEmbedder(tiny_attn_config)
    model.eval()
    return model


@pytest.fixture
def analyser(tiny_attn_embedder, attributes_df, all_pids, encoder):
    return AttentionAnalyser(
        attention_embedder=tiny_attn_embedder,
        dataset_attributes=attributes_df,
        dataset_pids=all_pids,
        encoder=encoder,
        config=AttentionAnalyserConfig(top_k_interactions=2),
    )


# ---------------------------------------------------------------------------
# mean_attention_weights
# ---------------------------------------------------------------------------


class TestMeanAttentionWeights:
    def test_shape(self, analyser):
        attn = analyser.mean_attention_weights()
        n_layers = analyser._n_layers
        seq_len = len(analyser._seq_labels)
        assert attn.shape == (n_layers, seq_len, seq_len)

    def test_rows_sum_to_one(self, analyser):
        """Each row of each attention matrix should sum to approximately 1."""
        attn = analyser.mean_attention_weights()
        row_sums = attn.sum(axis=-1)  # (n_layers, seq_len)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)

    def test_non_negative(self, analyser):
        attn = analyser.mean_attention_weights()
        assert np.all(attn >= -1e-6)

    def test_result_is_cached(self, analyser):
        a1 = analyser.mean_attention_weights()
        a2 = analyser.mean_attention_weights()
        assert a1 is a2  # same object from cache

    def test_subset_pids(self, analyser, all_pids):
        subset = all_pids[:1]
        attn = analyser.mean_attention_weights(split_pids=subset)
        assert attn.shape[0] == analyser._n_layers


# ---------------------------------------------------------------------------
# plot_attention_heatmap
# ---------------------------------------------------------------------------


class TestPlotAttentionHeatmap:
    def test_returns_figure(self, analyser):
        import matplotlib.pyplot as plt
        fig = analyser.plot_attention_heatmap(layer=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_attribute_names(self, analyser):
        import matplotlib.pyplot as plt
        seq_len = len(analyser._seq_labels)
        custom = [f"attr_{i}" for i in range(seq_len)]
        fig = analyser.plot_attention_heatmap(layer=0, attribute_names=custom)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# attribute_importance
# ---------------------------------------------------------------------------


class TestAttributeImportance:
    def test_returns_dict_for_valid_attribute(self, analyser):
        # Use the first non-CLS label
        target = analyser._attr_names[0]
        result = analyser.attribute_importance(target)
        assert isinstance(result, dict)
        # All other seq labels should be keys
        for lbl in analyser._seq_labels:
            if lbl != target:
                assert lbl in result

    def test_values_are_non_negative(self, analyser):
        target = analyser._attr_names[0]
        result = analyser.attribute_importance(target)
        for v in result.values():
            assert v >= -1e-6

    def test_raises_for_unknown_attribute(self, analyser):
        with pytest.raises(KeyError):
            analyser.attribute_importance("nonexistent_attribute_xyz")


# ---------------------------------------------------------------------------
# source_modulation_analysis
# ---------------------------------------------------------------------------


class TestSourceModulationAnalysis:
    def test_returns_dict(self, analyser):
        result = analyser.source_modulation_analysis(source_attribute="source")
        assert isinstance(result, dict)

    def test_source_attribute_not_in_result(self, analyser):
        result = analyser.source_modulation_analysis(source_attribute="source")
        assert "source" not in result

    def test_values_are_non_negative(self, analyser):
        result = analyser.source_modulation_analysis(source_attribute="source")
        for v in result.values():
            assert v >= 0.0

    def test_missing_source_returns_empty(self, analyser):
        result = analyser.source_modulation_analysis(source_attribute="nonexistent_xyz")
        assert result == {}


# ---------------------------------------------------------------------------
# interaction_consistency
# ---------------------------------------------------------------------------


class TestInteractionConsistency:
    def test_returns_required_keys(self, analyser):
        result = analyser.interaction_consistency()
        assert "consistency_score" in result
        assert "expected_pairs_attention" in result
        assert "unexpected_top_pairs" in result

    def test_consistency_score_in_range(self, analyser):
        result = analyser.interaction_consistency()
        score = result["consistency_score"]
        assert 0.0 <= score <= 1.0

    def test_custom_expected_pairs(self, analyser):
        # Use pairs that exist in the attribute names
        labels = analyser._attr_names
        if len(labels) >= 2:
            pairs = [(labels[0], labels[1])]
            result = analyser.interaction_consistency(expected_pairs=pairs)
            assert isinstance(result["consistency_score"], float)

    def test_empty_expected_pairs(self, analyser):
        result = analyser.interaction_consistency(expected_pairs=[])
        assert result["consistency_score"] == 0.0

    def test_unexpected_pairs_are_dicts(self, analyser):
        result = analyser.interaction_consistency()
        for item in result["unexpected_top_pairs"]:
            assert "query" in item
            assert "key" in item
            assert "weight" in item
