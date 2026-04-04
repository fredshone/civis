"""Base interface and shared components for attribute embedding models.

All models receive a ``dict[str, Tensor]`` of encoded attributes (as produced by
``AttributeEncoder.transform``) and produce a fixed-size embedding of shape
``(batch, d_model)``.

Public API
----------
AttributeEmbedderConfig
    Configuration dataclass for all embedding models.  Contains a union of all
    architecture-specific fields; fields irrelevant to the chosen architecture
    are silently ignored.

DiscreteEmbedding
    Per-attribute lookup tables for discrete attributes.  Unknown token (index 0)
    is zero-initialised so masked attributes embed to the zero vector.

ContinuousProjection
    Small MLP projecting a scalar continuous attribute to ``d_embed`` dimensions.
    Shared across all continuous attributes so they live in the same space as
    discrete embeddings.

BaseAttributeEmbedder
    Abstract ``nn.Module`` base class.  Subclasses implement ``forward`` and
    call ``get_attribute_tokens`` to obtain per-attribute embedding matrices
    before pooling or interaction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from datasets.encoding import AttributeConfig, AttributeEncoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AttributeEmbedderConfig:
    """Configuration for attribute embedding models.

    This is a union of all architecture-specific fields.  Fields not used by
    a given architecture are silently ignored.

    Parameters
    ----------
    d_embed:
        Dimensionality of each per-attribute embedding token.
    d_model:
        Dimensionality of the final output embedding vector.
    attribute_configs:
        One :class:`~datasets.encoding.AttributeConfig` per attribute,
        in fixed order.
    vocab_sizes:
        Vocabulary sizes (including the unknown token at index 0) for each
        discrete attribute that was seen during fitting.  Discrete attributes
        absent from this dict are ignored.
    dropout:
        Dropout rate applied before the final linear projection.
    n_heads:
        Number of attention heads (``SelfAttentionEmbedder`` only).
    n_layers:
        Number of transformer encoder layers (``SelfAttentionEmbedder`` only).
    use_cls_token:
        Whether to prepend a learned ``[CLS]`` token
        (``SelfAttentionEmbedder`` only; requires ``pooling='cls'``).
    pooling:
        Pooling strategy: ``'cls'``, ``'mean'``, or ``'sum'``
        (``SelfAttentionEmbedder`` only).
    attribute_groups:
        Optional mapping from attribute name to group label
        (``'person'``, ``'household'``, ``'context'``, ``'day'``).  Used for
        learned attribute-type positional encodings
        (``SelfAttentionEmbedder`` only).
    context_attributes:
        Names of attributes used as conditioning context
        (``FiLMEmbedder`` only).
    """

    d_embed: int
    d_model: int
    attribute_configs: list[AttributeConfig]
    vocab_sizes: dict[str, int] = field(default_factory=dict)
    dropout: float = 0.1
    # SelfAttentionEmbedder fields
    n_heads: int = 4
    n_layers: int = 2
    use_cls_token: bool = True
    pooling: Literal["cls", "mean", "sum"] = "cls"
    attribute_groups: dict[str, str] | None = None
    # FiLMEmbedder fields
    context_attributes: list[str] = field(default_factory=list)

    @classmethod
    def from_encoder(
        cls,
        encoder: AttributeEncoder,
        d_embed: int,
        d_model: int,
        **kwargs,
    ) -> "AttributeEmbedderConfig":
        """Build config from a fitted :class:`~datasets.encoding.AttributeEncoder`.

        Reads vocabulary sizes for all discrete attributes that were seen
        during ``encoder.fit``.

        Parameters
        ----------
        encoder:
            A fitted ``AttributeEncoder``.
        d_embed:
            Per-attribute embedding dimension.
        d_model:
            Final output embedding dimension.
        **kwargs:
            Additional fields forwarded to :class:`AttributeEmbedderConfig`.
        """
        vocab_sizes: dict[str, int] = {}
        for cfg in encoder.configs:
            if cfg.kind == "discrete":
                try:
                    vocab_sizes[cfg.name] = encoder.vocab_size(cfg.name)
                except KeyError:
                    pass
        return cls(
            d_embed=d_embed,
            d_model=d_model,
            attribute_configs=encoder.configs,
            vocab_sizes=vocab_sizes,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Shared embedding layers
# ---------------------------------------------------------------------------


class DiscreteEmbedding(nn.Module):
    """Per-attribute lookup tables for discrete attributes.

    Each attribute has its own ``nn.Embedding`` of shape
    ``(vocab_size, d_embed)``.  The unknown token at index 0 is
    zero-initialised so masked or missing attributes embed to the zero vector.

    Parameters
    ----------
    vocab_sizes:
        Mapping from attribute name to vocabulary size (index 0 = unknown).
    d_embed:
        Embedding dimension.
    """

    def __init__(self, vocab_sizes: dict[str, int], d_embed: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, d_embed)
                for name, vocab_size in vocab_sizes.items()
            }
        )
        # Zero-initialise the unknown token for each attribute
        with torch.no_grad():
            for emb in self.embeddings.values():
                emb.weight[0].zero_()

    def forward(self, name: str, indices: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for a named attribute.

        Parameters
        ----------
        name:
            Attribute name (must be a key in ``vocab_sizes``).
        indices:
            Integer indices of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_embed)``.
        """
        return self.embeddings[name](indices)


class ContinuousProjection(nn.Module):
    """Projects a scalar continuous attribute to ``d_embed`` dimensions.

    Architecture: ``Linear(1, d_embed) → LayerNorm → ReLU → Linear(d_embed, d_embed)``.
    Shared across all continuous attributes so they occupy the same embedding
    space as discrete tokens.

    When an ``is_known`` flag is provided indicating unknown values, the
    output embedding is zeroed for those samples (matching the zero-initialised
    unknown token for discrete attributes).

    Parameters
    ----------
    d_embed:
        Output embedding dimension.
    """

    def __init__(self, d_embed: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
        )

    def forward(
        self, values: torch.Tensor, is_known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project a batch of scalar values.

        Parameters
        ----------
        values:
            Float tensor of shape ``(batch,)`` with values in ``[0, 1]``.
        is_known:
            Optional bool tensor of shape ``(batch,)`` indicating which values
            are known (True) vs unknown (False).  Unknown values produce
            zero embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_embed)``.
        """
        out = self.net(values.unsqueeze(-1))
        if is_known is not None:
            out = out * is_known.unsqueeze(-1).float()
        return out


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseAttributeEmbedder(ABC, nn.Module):
    """Abstract base class for attribute embedding models.

    Handles attribute encoding infrastructure.  Subclasses implement
    ``forward`` and typically call :meth:`get_attribute_tokens` to obtain the
    ``(batch, n_attrs, d_embed)`` token matrix before pooling or interaction.

    Parameters
    ----------
    config:
        Model configuration.
    """

    def __init__(self, config: AttributeEmbedderConfig) -> None:
        super().__init__()
        self.config = config

        # Only process attributes that have usable embedding layers
        self._active_attribute_configs: list[AttributeConfig] = [
            cfg
            for cfg in config.attribute_configs
            if (cfg.kind == "discrete" and cfg.name in config.vocab_sizes)
            or cfg.kind == "continuous"
        ]

        self.discrete_embedding = DiscreteEmbedding(config.vocab_sizes, config.d_embed)
        self.continuous_projection = ContinuousProjection(config.d_embed)

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension (``d_model``)."""
        return self.config.d_model

    def get_attribute_tokens(
        self,
        attributes: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed all configured attributes into a per-attribute token matrix.

        Attributes present in the dict are encoded normally.  Discrete
        attributes missing from the dict default to index 0 (unknown);
        continuous attributes missing from the dict default to 0.0.

        Parameters
        ----------
        attributes:
            Mapping from attribute name to 1-D tensor of shape ``(batch,)``.
            Discrete attributes are ``int64`` indices; continuous are
            ``float32`` in ``[0, 1]``.

        Returns
        -------
        tokens : torch.Tensor
            Shape ``(batch, n_attrs, d_embed)``.
        mask : torch.Tensor
            Bool tensor of shape ``(batch, n_attrs)``.  ``True`` where the
            attribute is unknown or masked (discrete index == 0, or continuous
            value == 0.0, or key absent from ``attributes``).
        """
        device = (
            next(iter(attributes.values())).device
            if attributes
            else torch.device("cpu")
        )
        batch_size: int = next(iter(attributes.values())).shape[0] if attributes else 1

        token_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []

        for cfg in self._active_attribute_configs:
            name = cfg.name
            if cfg.kind == "discrete":
                if name in attributes:
                    indices = attributes[name].to(device)
                    token = self.discrete_embedding(name, indices)
                    is_unknown = indices == 0
                else:
                    indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                    token = self.discrete_embedding(name, indices)
                    is_unknown = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:  # continuous
                if name in attributes:
                    values = attributes[name].to(device)
                    # Check for parallel is_known flag if available
                    is_known_name = f"{name}_is_known"
                    if is_known_name in attributes:
                        is_known_indices = attributes[is_known_name].to(device)
                        is_known_bool = is_known_indices == 1  # index 1 means known
                    else:
                        is_known_bool = None
                    token = self.continuous_projection(values, is_known=is_known_bool)
                    is_unknown = (
                        values == 0.0 if is_known_bool is None else ~is_known_bool
                    )
                else:
                    values = torch.zeros(batch_size, dtype=torch.float32, device=device)
                    token = self.continuous_projection(values)
                    is_unknown = torch.ones(batch_size, dtype=torch.bool, device=device)

            token_list.append(token)
            mask_list.append(is_unknown)

        if not token_list:
            tokens = torch.zeros(batch_size, 0, self.config.d_embed, device=device)
            mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            return tokens, mask

        tokens = torch.stack(token_list, dim=1)  # (batch, n_attrs, d_embed)
        mask = torch.stack(mask_list, dim=1)  # (batch, n_attrs) bool
        return tokens, mask

    @abstractmethod
    def forward(self, attributes: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed a batch of attributes to a fixed-size vector.

        Parameters
        ----------
        attributes:
            Mapping as returned by ``AttributeEncoder.transform``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_model)``.
        """
        ...
