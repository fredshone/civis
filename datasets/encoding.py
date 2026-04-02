"""Attribute encoding utilities.

Converts raw Polars DataFrames of person/household/context attributes into
tensors suitable for embedding models.

Attribute types
---------------
discrete
    Categorical columns (Utf8).  Encoded as integer indices; index 0 is
    always reserved for the unknown/missing category.

continuous
    Numeric columns (Int32 or Float64).  Normalised to [0, 1] using
    min-max statistics learned during ``fit``; null values map to 0.0.

Public API
----------
AttributeConfig
    Dataclass describing one attribute's name and encoding type.

AttributeEncoder
    sklearn-style encoder: ``fit(df) -> self``, ``transform(df) -> dict``.

default_attribute_configs() -> list[AttributeConfig]
    Returns configs covering all standard foundata attributes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch


@dataclass
class AttributeConfig:
    """Configuration for a single attribute."""

    name: str
    kind: Literal["discrete", "continuous"]


class AttributeEncoder:
    """Encode raw attribute DataFrames into tensors.

    Parameters
    ----------
    configs:
        One :class:`AttributeConfig` per attribute to encode.
    """

    def __init__(self, configs: list[AttributeConfig]) -> None:
        self.configs = configs
        self._vocab: dict[str, list[str]] = {}        # discrete attr -> sorted vocab (index 0 = unknown)
        self._min: dict[str, float] = {}              # continuous attr -> min
        self._max: dict[str, float] = {}              # continuous attr -> max
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> "AttributeEncoder":
        """Learn vocabulary mappings and normalisation statistics.

        Parameters
        ----------
        df:
            DataFrame as returned by :func:`~distances.data.load_attributes`.

        Returns
        -------
        self
        """
        for cfg in self.configs:
            if cfg.name not in df.columns:
                continue
            col = df[cfg.name]
            if cfg.kind == "discrete":
                # Collect unique non-null values; sort for reproducibility
                unique_vals = sorted(
                    v for v in col.drop_nulls().unique().to_list()
                )
                # index 0 is reserved for unknown/missing
                self._vocab[cfg.name] = ["unknown"] + unique_vals
            else:
                non_null = col.drop_nulls().cast(pl.Float64)
                if len(non_null) == 0:
                    self._min[cfg.name] = 0.0
                    self._max[cfg.name] = 1.0
                else:
                    vmin = float(non_null.min())
                    vmax = float(non_null.max())
                    # Guard against zero-range columns
                    if vmin == vmax:
                        vmax = vmin + 1.0
                    self._min[cfg.name] = vmin
                    self._max[cfg.name] = vmax
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, df: pl.DataFrame) -> dict[str, torch.Tensor]:
        """Encode a DataFrame to a dictionary of tensors.

        Parameters
        ----------
        df:
            DataFrame with the same schema as the one passed to :meth:`fit`.

        Returns
        -------
        dict[str, Tensor]
            ``{attr_name: Tensor(shape (N,))}`` for each configured attribute
            present in *df*.  Discrete tensors are ``torch.int64``; continuous
            tensors are ``torch.float32``.  Rows are in the same order as *df*.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        out: dict[str, torch.Tensor] = {}
        for cfg in self.configs:
            if cfg.name not in df.columns:
                continue
            col = df[cfg.name]
            if cfg.kind == "discrete":
                vocab = self._vocab.get(cfg.name)
                if vocab is None:
                    continue
                val_to_idx = {v: i for i, v in enumerate(vocab)}
                indices = [
                    val_to_idx.get(v, 0) if v is not None else 0
                    for v in col.to_list()
                ]
                out[cfg.name] = torch.tensor(indices, dtype=torch.int64)
            else:
                vmin = self._min.get(cfg.name, 0.0)
                vmax = self._max.get(cfg.name, 1.0)
                values = col.cast(pl.Float64).to_list()
                normed = [
                    float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))
                    if v is not None
                    else 0.0
                    for v in values
                ]
                out[cfg.name] = torch.tensor(normed, dtype=torch.float32)
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def vocab_size(self, name: str) -> int:
        """Return vocabulary size (including unknown) for a discrete attribute."""
        if name not in self._vocab:
            raise KeyError(f"No discrete vocabulary for attribute '{name}'")
        return len(self._vocab[name])

    def summary(self) -> None:
        """Print attribute names, types, vocab sizes, and missingness."""
        if not self._fitted:
            print("(Not fitted yet)")
            return
        print(f"{'Attribute':<30} {'Kind':<12} {'Info'}")
        print("-" * 60)
        for cfg in self.configs:
            if cfg.kind == "discrete" and cfg.name in self._vocab:
                info = f"vocab_size={len(self._vocab[cfg.name])}"
            elif cfg.kind == "continuous" and cfg.name in self._min:
                info = f"range=[{self._min[cfg.name]:.3g}, {self._max[cfg.name]:.3g}]"
            else:
                info = "(not seen in data)"
            print(f"{cfg.name:<30} {cfg.kind:<12} {info}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise encoder state to a pickle file."""
        path = Path(path)
        state = {
            "configs": self.configs,
            "vocab": self._vocab,
            "min": self._min,
            "max": self._max,
            "fitted": self._fitted,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, path: str | Path) -> "AttributeEncoder":
        """Load an encoder from a pickle file produced by :meth:`save`."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        enc = cls(state["configs"])
        enc._vocab = state["vocab"]
        enc._min = state["min"]
        enc._max = state["max"]
        enc._fitted = state["fitted"]
        return enc


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def default_attribute_configs() -> list[AttributeConfig]:
    """Return configs covering all standard foundata attribute columns."""
    discrete_attrs = [
        "sex", "dwelling", "ownership", "disability", "education", "can_wfh",
        "occupation", "race", "has_licence", "relationship", "employment",
        "country", "source", "year", "month", "day", "hh_zone", "rain",
    ]
    continuous_attrs = [
        "age", "hh_size", "vehicles", "hh_income",
        "access_egress_distance", "max_temp_c", "avg_speed",
    ]
    configs: list[AttributeConfig] = []
    configs += [AttributeConfig(name=n, kind="discrete") for n in discrete_attrs]
    configs += [AttributeConfig(name=n, kind="continuous") for n in continuous_attrs]
    return configs
