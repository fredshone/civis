"""Tests for the models package.

Covers:
- DiscreteEmbedding / ContinuousProjection shared layers
- BaseAttributeEmbedder.get_attribute_tokens (via AdditionEmbedder)
- AdditionEmbedder forward pass
- SelfAttentionEmbedder forward pass, attention weights, pooling variants
- FiLMEmbedder forward pass, graceful degradation, film_stats
- Registry: build_model, count_parameters
"""

from __future__ import annotations

import pytest
import torch

from datasets.encoding import AttributeConfig
from models import (
    AdditionEmbedder,
    AttributeEmbedderConfig,
    FiLMEmbedder,
    SelfAttentionEmbedder,
    build_model,
    count_parameters,
)
from models.base import ContinuousProjection, DiscreteEmbedding


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_configs() -> list[AttributeConfig]:
    return [
        AttributeConfig(name="sex", kind="discrete"),
        AttributeConfig(name="age", kind="continuous"),
        AttributeConfig(name="employment", kind="discrete"),
    ]


@pytest.fixture
def simple_vocab_sizes() -> dict[str, int]:
    return {"sex": 3, "employment": 4}  # index 0 = unknown in both


@pytest.fixture
def base_config(
    simple_configs: list[AttributeConfig],
    simple_vocab_sizes: dict[str, int],
) -> AttributeEmbedderConfig:
    return AttributeEmbedderConfig(
        d_embed=16,
        d_model=32,
        attribute_configs=simple_configs,
        vocab_sizes=simple_vocab_sizes,
        dropout=0.0,
    )


@pytest.fixture
def sample_batch() -> dict[str, torch.Tensor]:
    """Batch of 4 samples covering all three configured attributes."""
    return {
        "sex": torch.tensor([1, 0, 2, 1], dtype=torch.long),
        "age": torch.tensor([0.3, 0.0, 0.8, 0.5], dtype=torch.float32),
        "employment": torch.tensor([2, 1, 0, 3], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# DiscreteEmbedding
# ---------------------------------------------------------------------------


class TestDiscreteEmbedding:
    def test_unknown_token_embeds_to_zero(self):
        emb = DiscreteEmbedding({"x": 5}, d_embed=8)
        out = emb("x", torch.tensor([0]))
        assert out.shape == (1, 8)
        assert torch.allclose(out, torch.zeros(1, 8))

    def test_nonzero_indices_produce_output(self):
        emb = DiscreteEmbedding({"x": 5}, d_embed=8)
        out = emb("x", torch.tensor([1, 2, 3]))
        assert out.shape == (3, 8)

    def test_multiple_attributes_independent(self):
        emb = DiscreteEmbedding({"a": 3, "b": 7}, d_embed=16)
        assert emb("a", torch.tensor([1])).shape == (1, 16)
        assert emb("b", torch.tensor([5])).shape == (1, 16)
        # Weights are independent — unknown in "a" doesn't affect "b"
        a_unk = emb("a", torch.tensor([0]))
        b_unk = emb("b", torch.tensor([0]))
        assert torch.allclose(a_unk, torch.zeros(1, 16))
        assert torch.allclose(b_unk, torch.zeros(1, 16))


# ---------------------------------------------------------------------------
# ContinuousProjection
# ---------------------------------------------------------------------------


class TestContinuousProjection:
    def test_output_shape(self):
        proj = ContinuousProjection(d_embed=16)
        out = proj(torch.rand(8))
        assert out.shape == (8, 16)

    def test_same_input_same_output(self):
        proj = ContinuousProjection(d_embed=16)
        v = torch.tensor([0.5, 0.5])
        out1 = proj(v)
        out2 = proj(v)
        assert torch.allclose(out1, out2)

    def test_scalar_consistency(self):
        proj = ContinuousProjection(d_embed=8)
        single = proj(torch.tensor([0.7]))
        batch = proj(torch.tensor([0.7, 0.7]))
        assert torch.allclose(single, batch[0:1])


# ---------------------------------------------------------------------------
# get_attribute_tokens (tested via AdditionEmbedder)
# ---------------------------------------------------------------------------


class TestGetAttributeTokens:
    def test_output_shape(
        self,
        base_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        model = AdditionEmbedder(base_config)
        model.eval()
        with torch.no_grad():
            tokens, mask = model.get_attribute_tokens(sample_batch)
        # 3 active configs: sex (discrete), age (continuous), employment (discrete)
        assert tokens.shape == (4, 3, 16)
        assert mask.shape == (4, 3)

    def test_missing_attribute_treated_as_unknown(
        self, base_config: AttributeEmbedderConfig
    ):
        model = AdditionEmbedder(base_config)
        model.eval()
        attrs = {"age": torch.tensor([0.5, 0.3])}
        with torch.no_grad():
            tokens, mask = model.get_attribute_tokens(attrs)
        assert tokens.shape == (2, 3, 16)
        assert mask[:, 0].all(), "missing 'sex' should be fully masked"
        assert mask[:, 2].all(), "missing 'employment' should be fully masked"

    def test_unknown_discrete_token_is_zero_vector(
        self, base_config: AttributeEmbedderConfig
    ):
        model = AdditionEmbedder(base_config)
        model.eval()
        attrs = {
            "sex": torch.zeros(2, dtype=torch.long),
            "age": torch.full((2,), 0.5),
            "employment": torch.zeros(2, dtype=torch.long),
        }
        with torch.no_grad():
            tokens, _ = model.get_attribute_tokens(attrs)
        assert torch.allclose(tokens[:, 0], torch.zeros(2, 16)), "sex unknown -> zero"
        assert torch.allclose(tokens[:, 2], torch.zeros(2, 16)), "employment unknown -> zero"

    def test_mask_true_for_index_zero(
        self,
        base_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        # sample_batch sex = [1, 0, 2, 1]; position 1 is unknown (index 0)
        model = AdditionEmbedder(base_config)
        model.eval()
        with torch.no_grad():
            _, mask = model.get_attribute_tokens(sample_batch)
        assert mask[1, 0].item() is True   # sex=0 -> unknown
        assert mask[0, 0].item() is False  # sex=1 -> known

    def test_missing_attribute_token_is_zero_vector(
        self, base_config: AttributeEmbedderConfig
    ):
        model = AdditionEmbedder(base_config)
        model.eval()
        attrs = {"age": torch.tensor([0.5])}
        with torch.no_grad():
            tokens, _ = model.get_attribute_tokens(attrs)
        # Missing discrete attrs embed to index 0 -> zero vector
        assert torch.allclose(tokens[:, 0], torch.zeros(1, 16))  # sex
        assert torch.allclose(tokens[:, 2], torch.zeros(1, 16))  # employment


# ---------------------------------------------------------------------------
# AdditionEmbedder
# ---------------------------------------------------------------------------


class TestAdditionEmbedder:
    def test_forward_shape(
        self,
        base_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        model = AdditionEmbedder(base_config)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert out.shape == (4, 32)

    def test_no_nan_on_fully_masked_input(self, base_config: AttributeEmbedderConfig):
        model = AdditionEmbedder(base_config)
        model.eval()
        attrs = {
            "sex": torch.zeros(3, dtype=torch.long),
            "age": torch.zeros(3, dtype=torch.float32),
            "employment": torch.zeros(3, dtype=torch.long),
        }
        with torch.no_grad():
            out = model(attrs)
        assert not out.isnan().any()

    def test_deterministic_without_dropout(
        self,
        base_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        model = AdditionEmbedder(base_config)
        model.eval()
        with torch.no_grad():
            out1 = model(sample_batch)
            out2 = model(sample_batch)
        assert torch.allclose(out1, out2)

    def test_embed_dim_property(self, base_config: AttributeEmbedderConfig):
        assert AdditionEmbedder(base_config).embed_dim == 32

    def test_partial_attributes(self, base_config: AttributeEmbedderConfig):
        model = AdditionEmbedder(base_config)
        model.eval()
        attrs = {"sex": torch.tensor([1, 2], dtype=torch.long)}
        with torch.no_grad():
            out = model(attrs)
        assert out.shape == (2, 32)
        assert not out.isnan().any()


# ---------------------------------------------------------------------------
# SelfAttentionEmbedder
# ---------------------------------------------------------------------------


class TestSelfAttentionEmbedder:
    @pytest.fixture
    def attn_config(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ) -> AttributeEmbedderConfig:
        return AttributeEmbedderConfig(
            d_embed=16,
            d_model=32,
            attribute_configs=simple_configs,
            vocab_sizes=simple_vocab_sizes,
            dropout=0.0,
            n_heads=2,
            n_layers=2,
            use_cls_token=True,
            pooling="cls",
        )

    def test_forward_shape(
        self,
        attn_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        model = SelfAttentionEmbedder(attn_config)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert out.shape == (4, 32)

    def test_return_attention_shape(
        self,
        attn_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        model = SelfAttentionEmbedder(attn_config)
        model.eval()
        with torch.no_grad():
            out, attn_weights = model(sample_batch, return_attention=True)
        assert out.shape == (4, 32)
        assert len(attn_weights) == 2  # n_layers=2
        # seq_len = 1 (CLS) + 3 (attrs) = 4
        assert attn_weights[0].shape == (4, 4, 4)

    def test_no_nan_with_all_unknown_attrs(
        self, attn_config: AttributeEmbedderConfig
    ):
        model = SelfAttentionEmbedder(attn_config)
        model.eval()
        attrs = {
            "sex": torch.zeros(2, dtype=torch.long),
            "age": torch.zeros(2, dtype=torch.float32),
            "employment": torch.zeros(2, dtype=torch.long),
        }
        with torch.no_grad():
            out = model(attrs)
        assert not out.isnan().any()

    def test_mean_pooling_shape(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
        sample_batch: dict[str, torch.Tensor],
    ):
        config = AttributeEmbedderConfig(
            d_embed=16,
            d_model=32,
            attribute_configs=simple_configs,
            vocab_sizes=simple_vocab_sizes,
            dropout=0.0,
            n_heads=2,
            n_layers=1,
            use_cls_token=False,
            pooling="mean",
        )
        model = SelfAttentionEmbedder(config)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert out.shape == (4, 32)

    def test_sum_pooling_shape(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
        sample_batch: dict[str, torch.Tensor],
    ):
        config = AttributeEmbedderConfig(
            d_embed=16,
            d_model=32,
            attribute_configs=simple_configs,
            vocab_sizes=simple_vocab_sizes,
            dropout=0.0,
            n_heads=2,
            n_layers=1,
            use_cls_token=False,
            pooling="sum",
        )
        model = SelfAttentionEmbedder(config)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert out.shape == (4, 32)

    def test_invalid_config_cls_without_use_cls_token(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        with pytest.raises(ValueError, match="requires use_cls_token=True"):
            SelfAttentionEmbedder(AttributeEmbedderConfig(
                d_embed=16, d_model=32,
                attribute_configs=simple_configs,
                vocab_sizes=simple_vocab_sizes,
                use_cls_token=False, pooling="cls",
            ))

    def test_invalid_config_use_cls_token_with_mean_pooling(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        with pytest.raises(ValueError, match="requires pooling='cls'"):
            SelfAttentionEmbedder(AttributeEmbedderConfig(
                d_embed=16, d_model=32,
                attribute_configs=simple_configs,
                vocab_sizes=simple_vocab_sizes,
                use_cls_token=True, pooling="mean",
            ))

    def test_attribute_groups(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
        sample_batch: dict[str, torch.Tensor],
    ):
        config = AttributeEmbedderConfig(
            d_embed=16, d_model=32,
            attribute_configs=simple_configs,
            vocab_sizes=simple_vocab_sizes,
            dropout=0.0, n_heads=2, n_layers=1,
            use_cls_token=True, pooling="cls",
            attribute_groups={
                "sex": "person",
                "age": "person",
                "employment": "person",
            },
        )
        model = SelfAttentionEmbedder(config)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert out.shape == (4, 32)


# ---------------------------------------------------------------------------
# FiLMEmbedder
# ---------------------------------------------------------------------------


class TestFiLMEmbedder:
    @pytest.fixture
    def film_config(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ) -> AttributeEmbedderConfig:
        return AttributeEmbedderConfig(
            d_embed=16,
            d_model=32,
            attribute_configs=simple_configs,
            vocab_sizes=simple_vocab_sizes,
            dropout=0.0,
            context_attributes=["employment"],
        )

    def test_forward_shape(
        self,
        film_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        model = FiLMEmbedder(film_config)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert out.shape == (4, 32)

    def test_no_nan_all_context_unknown(self, film_config: AttributeEmbedderConfig):
        model = FiLMEmbedder(film_config)
        model.eval()
        attrs = {
            "sex": torch.tensor([1, 2], dtype=torch.long),
            "age": torch.tensor([0.5, 0.3], dtype=torch.float32),
            "employment": torch.zeros(2, dtype=torch.long),  # unknown context
        }
        with torch.no_grad():
            out = model(attrs)
        assert not out.isnan().any()

    def test_no_context_attributes(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
        sample_batch: dict[str, torch.Tensor],
    ):
        config = AttributeEmbedderConfig(
            d_embed=16, d_model=32,
            attribute_configs=simple_configs,
            vocab_sizes=simple_vocab_sizes,
            dropout=0.0,
            context_attributes=[],
        )
        model = FiLMEmbedder(config)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        assert out.shape == (4, 32)
        assert not out.isnan().any()

    def test_film_stats_before_forward(self, film_config: AttributeEmbedderConfig):
        model = FiLMEmbedder(film_config)
        stats = model.film_stats()
        assert stats["mean_gamma_deviation"] == 0.0
        assert stats["mean_beta_magnitude"] == 0.0

    def test_film_stats_after_forward(
        self,
        film_config: AttributeEmbedderConfig,
        sample_batch: dict[str, torch.Tensor],
    ):
        model = FiLMEmbedder(film_config)
        model.eval()
        with torch.no_grad():
            model(sample_batch)
        stats = model.film_stats()
        assert "mean_gamma_deviation" in stats
        assert "mean_beta_magnitude" in stats
        assert stats["mean_gamma_deviation"] >= 0.0
        assert stats["mean_beta_magnitude"] >= 0.0

    def test_no_nan_fully_masked(self, film_config: AttributeEmbedderConfig):
        model = FiLMEmbedder(film_config)
        model.eval()
        attrs = {
            "sex": torch.zeros(3, dtype=torch.long),
            "age": torch.zeros(3, dtype=torch.float32),
            "employment": torch.zeros(3, dtype=torch.long),
        }
        with torch.no_grad():
            out = model(attrs)
        assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_build_addition(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        model = build_model({
            "architecture": "addition",
            "d_embed": 16, "d_model": 32,
            "attribute_configs": simple_configs,
            "vocab_sizes": simple_vocab_sizes,
        })
        assert isinstance(model, AdditionEmbedder)

    def test_build_attention(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        model = build_model({
            "architecture": "attention",
            "d_embed": 16, "d_model": 32,
            "attribute_configs": simple_configs,
            "vocab_sizes": simple_vocab_sizes,
        })
        assert isinstance(model, SelfAttentionEmbedder)

    def test_build_film(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        model = build_model({
            "architecture": "film",
            "d_embed": 16, "d_model": 32,
            "attribute_configs": simple_configs,
            "vocab_sizes": simple_vocab_sizes,
        })
        assert isinstance(model, FiLMEmbedder)

    def test_unknown_architecture_raises(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model({
                "architecture": "no_such_arch",
                "d_embed": 16, "d_model": 32,
                "attribute_configs": simple_configs,
                "vocab_sizes": simple_vocab_sizes,
            })

    def test_missing_architecture_key_raises(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        with pytest.raises(ValueError):
            build_model({
                "d_embed": 16, "d_model": 32,
                "attribute_configs": simple_configs,
                "vocab_sizes": simple_vocab_sizes,
            })

    def test_build_does_not_mutate_input(
        self,
        simple_configs: list[AttributeConfig],
        simple_vocab_sizes: dict[str, int],
    ):
        config = {
            "architecture": "addition",
            "d_embed": 16, "d_model": 32,
            "attribute_configs": simple_configs,
            "vocab_sizes": simple_vocab_sizes,
        }
        original_keys = set(config.keys())
        build_model(config)
        assert set(config.keys()) == original_keys, "build_model must not mutate input dict"

    def test_count_parameters_keys(self, base_config: AttributeEmbedderConfig):
        model = AdditionEmbedder(base_config)
        params = count_parameters(model)
        assert "total" in params
        assert "trainable" in params
        assert "per_component" in params

    def test_count_parameters_positive(self, base_config: AttributeEmbedderConfig):
        model = AdditionEmbedder(base_config)
        params = count_parameters(model)
        assert params["total"] > 0
        assert params["trainable"] > 0

    def test_count_parameters_all_trainable_by_default(
        self, base_config: AttributeEmbedderConfig
    ):
        model = AdditionEmbedder(base_config)
        params = count_parameters(model)
        assert params["trainable"] == params["total"]

    def test_count_parameters_per_component_positive(
        self, base_config: AttributeEmbedderConfig
    ):
        model = AdditionEmbedder(base_config)
        params = count_parameters(model)
        for name, count in params["per_component"].items():
            assert count >= 0, f"component '{name}' has negative param count"
