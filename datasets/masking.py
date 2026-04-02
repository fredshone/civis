"""Attribute masking augmentation for contrastive training.

Randomly replaces attribute values with the unknown token (index 0) during
training, acting as attribute-level dropout.  This trains the embedding model
to handle missing attributes at inference time and forces it to learn
cross-attribute interactions.

Masking strategies
------------------
independent
    Each attribute is masked independently with its own probability.
grouped
    Attributes are partitioned into groups (e.g. person-level,
    household-level, context-level); when a group is chosen to be masked,
    *all* attributes in that group are masked together.
curriculum
    Like independent, but masking probability scales from 0 up to the
    target probability over ``warmup_steps`` training steps (set via
    :meth:`AttributeMasker.set_step`).

The unknown token is always integer/float 0, matching the convention set by
:class:`~datasets.encoding.AttributeEncoder`.

Public API
----------
AttributeMasker
    ``__call__(attributes) -> attributes`` applies masking in-place on
    a copy.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
import torch


_DEFAULT_GROUPS: dict[str, list[str]] = {
    "person": [
        "age", "sex", "disability", "education", "can_wfh", "occupation",
        "race", "has_licence", "relationship", "employment",
    ],
    "household": [
        "hid", "hh_size", "hh_income", "dwelling", "ownership", "vehicles",
        "hh_zone",
    ],
    "context": [
        "country", "source", "year", "month", "day", "weight",
        "access_egress_distance", "max_temp_c", "rain", "avg_speed",
    ],
}

_DEFAULT_WARMUP_STEPS: int = 10_000


class AttributeMasker:
    """Apply random masking to encoded attribute dictionaries.

    Parameters
    ----------
    mask_probs:
        Target masking probability per attribute name.  Attributes not listed
        here will never be masked.
    strategy:
        ``"independent"``, ``"grouped"``, or ``"curriculum"``.
    groups:
        Attribute groupings used when *strategy* is ``"grouped"``.  Keys are
        group names; values are lists of attribute names.  Defaults to
        person / household / context groups.
    protected:
        Attribute names that are never masked regardless of *mask_probs*.
    warmup_steps:
        For ``"curriculum"`` strategy only: number of steps over which
        masking probability ramps from 0 to target.
    """

    def __init__(
        self,
        mask_probs: dict[str, float],
        strategy: Literal["independent", "grouped", "curriculum"] = "independent",
        groups: dict[str, list[str]] | None = None,
        protected: list[str] | None = None,
        warmup_steps: int = _DEFAULT_WARMUP_STEPS,
    ) -> None:
        self.mask_probs = dict(mask_probs)
        self.strategy = strategy
        self.groups = groups if groups is not None else _DEFAULT_GROUPS
        self.protected: set[str] = set(protected or [])
        self.warmup_steps = warmup_steps
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core masking
    # ------------------------------------------------------------------

    def __call__(
        self, attributes: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return a copy of *attributes* with random masking applied.

        Masked attribute values are replaced with 0 (the unknown token).
        The input dictionary is not modified.
        """
        out = {k: v.clone() for k, v in attributes.items()}

        if self.strategy == "independent":
            self._apply_independent(out)
        elif self.strategy == "grouped":
            self._apply_grouped(out)
        elif self.strategy == "curriculum":
            self._apply_curriculum(out)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

        return out

    def _effective_prob(self, name: str) -> float:
        """Return the masking probability for *name* (after curriculum scaling)."""
        base = self.mask_probs.get(name, 0.0)
        if self.strategy == "curriculum":
            scale = min(self._step / max(self.warmup_steps, 1), 1.0)
            return base * scale
        return base

    def _should_mask(self, prob: float) -> bool:
        return torch.rand(1).item() < prob

    def _zero_attr(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a zero tensor of the same shape and dtype."""
        return torch.zeros_like(tensor)

    def _apply_independent(self, out: dict[str, torch.Tensor]) -> None:
        for name, tensor in out.items():
            if name in self.protected:
                continue
            if self._should_mask(self._effective_prob(name)):
                out[name] = self._zero_attr(tensor)

    def _apply_grouped(self, out: dict[str, torch.Tensor]) -> None:
        # Build reverse map: attr_name -> group_prob (max prob in group)
        attr_to_group: dict[str, str] = {}
        for grp_name, members in self.groups.items():
            for m in members:
                attr_to_group[m] = grp_name

        # Decide per group once, then apply to all members
        masked_groups: set[str] = set()
        unmasked_groups: set[str] = set()

        for name, tensor in out.items():
            if name in self.protected:
                continue
            grp = attr_to_group.get(name)
            if grp is None:
                # Not in any group — fall back to independent
                if self._should_mask(self._effective_prob(name)):
                    out[name] = self._zero_attr(tensor)
                continue

            if grp in masked_groups:
                out[name] = self._zero_attr(tensor)
            elif grp not in unmasked_groups:
                # First attribute encountered for this group — decide
                grp_prob = max(
                    self.mask_probs.get(m, 0.0)
                    for m in self.groups[grp]
                )
                if self._should_mask(grp_prob):
                    masked_groups.add(grp)
                    out[name] = self._zero_attr(tensor)
                else:
                    unmasked_groups.add(grp)

    def _apply_curriculum(self, out: dict[str, torch.Tensor]) -> None:
        # Same as independent but uses scaled probabilities
        self._apply_independent(out)

    # ------------------------------------------------------------------
    # Curriculum step
    # ------------------------------------------------------------------

    def set_step(self, step: int) -> None:
        """Update the global training step counter for curriculum masking."""
        self._step = step

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        df: pl.DataFrame,
        base_rate: float = 0.15,
        missingness_weighted: bool = True,
        **kwargs,
    ) -> "AttributeMasker":
        """Construct a masker with per-attribute probabilities from a DataFrame.

        Parameters
        ----------
        df:
            Attributes DataFrame (from :func:`~distances.data.load_attributes`).
        base_rate:
            Target mean masking probability across all attributes.
        missingness_weighted:
            If ``True``, weight probabilities proportional to each attribute's
            empirical null rate.  If ``False``, use *base_rate* uniformly.
        **kwargs:
            Forwarded to :class:`AttributeMasker.__init__`.
        """
        n = len(df)
        if n == 0 or not missingness_weighted:
            probs = {col: base_rate for col in df.columns}
        else:
            null_rates = {
                col: df[col].null_count() / n for col in df.columns
            }
            total_null = sum(null_rates.values())
            if total_null == 0:
                probs = {col: base_rate for col in df.columns}
            else:
                # Scale so the mean across columns equals base_rate
                n_cols = len(df.columns)
                probs = {
                    col: float(
                        (null_rates[col] / total_null) * n_cols * base_rate
                    )
                    for col in df.columns
                }
        return cls(probs, **kwargs)
