"""Tests for datasets/masking.py."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from datasets.masking import AttributeMasker

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "attributes.csv"


@pytest.fixture()
def attrs_df():
    from distances.data import load_attributes
    return load_attributes(FIXTURE_PATH)


def _make_attrs(n: int = 4) -> dict[str, torch.Tensor]:
    """Create a small synthetic attributes dict for testing."""
    return {
        "sex":        torch.tensor([1, 2, 0, 1], dtype=torch.int64)[:n],
        "age":        torch.tensor([0.5, 0.3, 0.0, 0.8], dtype=torch.float32)[:n],
        "employment": torch.tensor([1, 0, 2, 1], dtype=torch.int64)[:n],
        "country":    torch.tensor([1, 1, 2, 3], dtype=torch.int64)[:n],
        "source":     torch.tensor([1, 1, 2, 2], dtype=torch.int64)[:n],
    }


# ---------------------------------------------------------------------------
# Independent strategy
# ---------------------------------------------------------------------------

class TestIndependentMasking:
    def test_returns_new_dict(self):
        masker = AttributeMasker({"sex": 1.0})
        attrs = _make_attrs()
        result = masker(attrs)
        assert result is not attrs

    def test_input_not_modified(self):
        masker = AttributeMasker({"sex": 1.0})
        attrs = _make_attrs()
        original = attrs["sex"].clone()
        masker(attrs)
        assert torch.equal(attrs["sex"], original)

    def test_always_masked_at_prob_one(self):
        masker = AttributeMasker({"sex": 1.0, "age": 1.0})
        attrs = _make_attrs()
        result = masker(attrs)
        assert torch.all(result["sex"] == 0)
        assert torch.all(result["age"] == 0.0)

    def test_never_masked_at_prob_zero(self):
        masker = AttributeMasker({"sex": 0.0, "age": 0.0})
        attrs = _make_attrs()
        result = masker(attrs)
        assert torch.equal(result["sex"], attrs["sex"])
        assert torch.equal(result["age"], attrs["age"])

    def test_unlisted_attr_not_masked(self):
        masker = AttributeMasker({"sex": 1.0})
        attrs = _make_attrs()
        original_age = attrs["age"].clone()
        result = masker(attrs)
        assert torch.equal(result["age"], original_age)

    def test_masked_value_is_zero(self):
        masker = AttributeMasker({"sex": 1.0, "age": 1.0})
        attrs = _make_attrs()
        result = masker(attrs)
        assert torch.all(result["sex"] == 0)
        assert torch.all(result["age"] == 0.0)


# ---------------------------------------------------------------------------
# Protected attributes
# ---------------------------------------------------------------------------

class TestProtectedAttributes:
    def test_protected_never_masked(self):
        masker = AttributeMasker(
            {"sex": 1.0, "source": 1.0},
            protected=["source"],
        )
        attrs = _make_attrs()
        original_source = attrs["source"].clone()
        for _ in range(20):
            result = masker(attrs)
            assert torch.equal(result["source"], original_source)

    def test_non_protected_still_masked(self):
        masker = AttributeMasker(
            {"sex": 1.0},
            protected=["source"],
        )
        attrs = _make_attrs()
        result = masker(attrs)
        assert torch.all(result["sex"] == 0)


# ---------------------------------------------------------------------------
# Grouped strategy
# ---------------------------------------------------------------------------

class TestGroupedMasking:
    def test_group_all_or_nothing(self):
        groups = {"person": ["sex", "age"], "context": ["country", "source"]}
        masker = AttributeMasker(
            {"sex": 1.0, "age": 1.0, "country": 0.0, "source": 0.0},
            strategy="grouped",
            groups=groups,
        )
        attrs = _make_attrs()
        result = masker(attrs)
        # Person group should be fully masked (prob=1.0)
        assert torch.all(result["sex"] == 0)
        assert torch.all(result["age"] == 0.0)
        # Context group should be untouched (prob=0.0)
        assert torch.equal(result["country"], attrs["country"])
        assert torch.equal(result["source"], attrs["source"])

    def test_grouped_consistency_over_runs(self):
        """Within a group, all attributes are masked or none are."""
        groups = {"person": ["sex", "age"]}
        masker = AttributeMasker(
            {"sex": 0.5, "age": 0.5},
            strategy="grouped",
            groups=groups,
        )
        attrs = _make_attrs()
        for _ in range(50):
            result = masker(attrs)
            sex_masked = torch.all(result["sex"] == 0).item()
            age_masked = torch.all(result["age"] == 0.0).item()
            # Both masked or neither
            assert sex_masked == age_masked, (
                f"Inconsistent group masking: sex_masked={sex_masked}, age_masked={age_masked}"
            )


# ---------------------------------------------------------------------------
# Curriculum strategy
# ---------------------------------------------------------------------------

class TestCurriculumMasking:
    def test_no_masking_at_step_zero(self):
        masker = AttributeMasker(
            {"sex": 1.0, "age": 1.0},
            strategy="curriculum",
            warmup_steps=100,
        )
        masker.set_step(0)
        attrs = _make_attrs()
        for _ in range(20):
            result = masker(attrs)
            # At step 0, effective_prob = 0 → nothing masked
            assert torch.equal(result["sex"], attrs["sex"])
            assert torch.equal(result["age"], attrs["age"])

    def test_full_masking_at_warmup_steps(self):
        masker = AttributeMasker(
            {"sex": 1.0, "age": 1.0},
            strategy="curriculum",
            warmup_steps=100,
        )
        masker.set_step(100)
        attrs = _make_attrs()
        result = masker(attrs)
        assert torch.all(result["sex"] == 0)
        assert torch.all(result["age"] == 0.0)

    def test_partial_masking_at_half_warmup(self):
        # At half warmup, prob = 0.5 × target → never deterministically all-or-nothing
        masker = AttributeMasker(
            {"sex": 1.0},
            strategy="curriculum",
            warmup_steps=100,
        )
        masker.set_step(50)
        attrs = {"sex": torch.tensor([1], dtype=torch.int64)}
        # Just check it doesn't crash and returns a tensor
        result = masker(attrs)
        assert "sex" in result
        assert result["sex"].dtype == torch.int64

    def test_set_step_updates_state(self):
        masker = AttributeMasker({}, strategy="curriculum")
        masker.set_step(42)
        assert masker._step == 42


# ---------------------------------------------------------------------------
# from_data factory
# ---------------------------------------------------------------------------

class TestFromData:
    def test_returns_masker(self, attrs_df):
        masker = AttributeMasker.from_data(attrs_df)
        assert isinstance(masker, AttributeMasker)

    def test_probs_non_negative(self, attrs_df):
        masker = AttributeMasker.from_data(attrs_df)
        for prob in masker.mask_probs.values():
            assert prob >= 0.0

    def test_uniform_when_not_weighted(self, attrs_df):
        base_rate = 0.2
        masker = AttributeMasker.from_data(attrs_df, base_rate=base_rate, missingness_weighted=False)
        for prob in masker.mask_probs.values():
            assert prob == pytest.approx(base_rate)

    def test_higher_missingness_higher_prob(self, attrs_df):
        # access_egress_distance has a null in fixture; age has none
        masker = AttributeMasker.from_data(attrs_df, missingness_weighted=True)
        if "access_egress_distance" in masker.mask_probs and "age" in masker.mask_probs:
            assert masker.mask_probs["access_egress_distance"] >= masker.mask_probs["age"]

    def test_all_zero_missingness_uses_base_rate(self):
        # DataFrame with no nulls
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        masker = AttributeMasker.from_data(df, base_rate=0.1, missingness_weighted=True)
        for prob in masker.mask_probs.values():
            assert prob == pytest.approx(0.1)
