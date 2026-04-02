"""Tests for datasets/encoding.py."""

from __future__ import annotations

import pickle
from pathlib import Path

import polars as pl
import pytest
import torch

from datasets.encoding import AttributeConfig, AttributeEncoder, default_attribute_configs

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "attributes.csv"


@pytest.fixture()
def attrs_df():
    from distances.data import load_attributes
    return load_attributes(FIXTURE_PATH)


@pytest.fixture()
def encoder(attrs_df):
    enc = AttributeEncoder(default_attribute_configs())
    enc.fit(attrs_df)
    return enc


@pytest.fixture()
def encoded(encoder, attrs_df):
    return encoder.transform(attrs_df)


# ---------------------------------------------------------------------------
# fit / transform basics
# ---------------------------------------------------------------------------

class TestAttributeEncoderFitTransform:
    def test_returns_self(self, attrs_df):
        enc = AttributeEncoder(default_attribute_configs())
        result = enc.fit(attrs_df)
        assert result is enc

    def test_transform_requires_fit(self, attrs_df):
        enc = AttributeEncoder(default_attribute_configs())
        with pytest.raises(RuntimeError, match="fit"):
            enc.transform(attrs_df)

    def test_output_is_dict(self, encoded):
        assert isinstance(encoded, dict)

    def test_all_configured_attrs_present(self, encoder, encoded):
        from distances.data import load_attributes
        df = load_attributes(FIXTURE_PATH)
        for cfg in encoder.configs:
            if cfg.name in df.columns:
                assert cfg.name in encoded, f"Missing key: {cfg.name}"

    def test_shape_matches_dataframe(self, attrs_df, encoded):
        n = len(attrs_df)
        for name, tensor in encoded.items():
            assert tensor.shape == (n,), f"{name}: expected ({n},), got {tensor.shape}"


# ---------------------------------------------------------------------------
# Discrete encoding
# ---------------------------------------------------------------------------

class TestDiscreteEncoding:
    def test_dtype_is_int64(self, encoded):
        discrete_names = {
            cfg.name for cfg in default_attribute_configs() if cfg.kind == "discrete"
        }
        for name in discrete_names:
            if name in encoded:
                assert encoded[name].dtype == torch.int64, f"{name} dtype wrong"

    def test_values_non_negative(self, encoded):
        discrete_names = {
            cfg.name for cfg in default_attribute_configs() if cfg.kind == "discrete"
        }
        for name in discrete_names:
            if name in encoded:
                assert (encoded[name] >= 0).all(), f"{name} has negative index"

    def test_null_maps_to_zero(self, encoder):
        # ktdb_pid_c has many nulls — check ownership is index 0
        from distances.data import load_attributes
        df = load_attributes(FIXTURE_PATH)
        enc_dict = encoder.transform(df)
        # Row 2 is ktdb_pid_c, ownership is null in fixture
        assert enc_dict["ownership"][2].item() == 0

    def test_unknown_always_index_zero(self, encoder):
        for name, vocab in encoder._vocab.items():
            assert vocab[0] == "unknown", f"{name}: index 0 is not 'unknown'"

    def test_unseen_value_maps_to_zero(self, encoder):
        df = pl.DataFrame({"sex": ["unknown_alien_value"]})
        result = encoder.transform(df)
        assert result["sex"][0].item() == 0

    def test_vocab_size_includes_unknown(self, encoder):
        # sex has two known values in fixture (female, male) → vocab_size = 3
        assert encoder.vocab_size("sex") == 3


# ---------------------------------------------------------------------------
# Continuous encoding
# ---------------------------------------------------------------------------

class TestContinuousEncoding:
    def test_dtype_is_float32(self, encoded):
        continuous_names = {
            cfg.name for cfg in default_attribute_configs() if cfg.kind == "continuous"
        }
        for name in continuous_names:
            if name in encoded:
                assert encoded[name].dtype == torch.float32, f"{name} dtype wrong"

    def test_values_in_unit_interval(self, encoded):
        continuous_names = {
            cfg.name for cfg in default_attribute_configs() if cfg.kind == "continuous"
        }
        for name in continuous_names:
            if name in encoded:
                t = encoded[name]
                assert (t >= 0.0).all() and (t <= 1.0).all(), (
                    f"{name}: values outside [0, 1]: {t}"
                )

    def test_null_continuous_maps_to_zero(self, encoder):
        # access_egress_distance is null for cmap_pid_a (row 0 in fixture)
        from distances.data import load_attributes
        df = load_attributes(FIXTURE_PATH)
        enc_dict = encoder.transform(df)
        assert enc_dict["access_egress_distance"][0].item() == pytest.approx(0.0)

    def test_min_maps_near_zero(self, encoder, attrs_df):
        enc_dict = encoder.transform(attrs_df)
        # The minimum age in fixture is 28 — should normalise to 0.0
        ages = enc_dict["age"]
        assert ages.min().item() == pytest.approx(0.0)

    def test_max_maps_near_one(self, encoder, attrs_df):
        enc_dict = encoder.transform(attrs_df)
        ages = enc_dict["age"]
        assert ages.max().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, encoder, attrs_df, tmp_path):
        save_path = tmp_path / "encoder.pkl"
        encoder.save(save_path)
        loaded = AttributeEncoder.load(save_path)
        original_out = encoder.transform(attrs_df)
        loaded_out = loaded.transform(attrs_df)
        for name in original_out:
            assert torch.equal(original_out[name], loaded_out[name]), (
                f"Mismatch for {name} after save/load"
            )

    def test_save_creates_file(self, encoder, tmp_path):
        save_path = tmp_path / "enc.pkl"
        encoder.save(save_path)
        assert save_path.exists()

    def test_load_is_fitted(self, encoder, tmp_path):
        save_path = tmp_path / "enc.pkl"
        encoder.save(save_path)
        loaded = AttributeEncoder.load(save_path)
        assert loaded._fitted


# ---------------------------------------------------------------------------
# default_attribute_configs
# ---------------------------------------------------------------------------

class TestDefaultConfigs:
    def test_returns_list(self):
        configs = default_attribute_configs()
        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_all_kinds_valid(self):
        for cfg in default_attribute_configs():
            assert cfg.kind in ("discrete", "continuous")

    def test_no_duplicates(self):
        names = [cfg.name for cfg in default_attribute_configs()]
        assert len(names) == len(set(names))
