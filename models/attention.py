"""Self-attention attribute embedding model.

Uses a stack of transformer encoder layers to learn pairwise interactions
between attributes before pooling to a fixed-size vector.

Key design choices
------------------
- Pre-LayerNorm transformer blocks (more stable training than post-LN).
- Learned attribute-type positional encodings that encode the *role* of each
  attribute (person / household / context / day), not sequential position.
- Unknown/masked tokens are excluded from attention via a key-padding mask
  so they neither receive nor contribute information to other tokens.
- Optional ``[CLS]`` token whose final representation is used as the embedding.
- ``return_attention=True`` exposes per-layer attention weight tensors for
  analysis and visualisation.

Public API
----------
SelfAttentionEmbedder
    Transformer-based attribute embedding model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AttributeEmbedderConfig, BaseAttributeEmbedder

# Canonical group labels and their integer indices
_GROUP_LABELS: list[str] = ["person", "household", "context", "day", "unknown"]
_GROUP_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(_GROUP_LABELS)}


class _AttentionEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer encoder block.

    Supports optional return of averaged attention weights for analysis.

    Parameters
    ----------
    d_embed:
        Model dimension.
    n_heads:
        Number of attention heads.
    dropout:
        Dropout applied after attention and feedforward sublayers.
    """

    def __init__(self, d_embed: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_embed, n_heads, dropout=dropout, batch_first=True
        )
        dim_ff = 4 * d_embed
        self.ff = nn.Sequential(
            nn.Linear(d_embed, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_embed),
        )
        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one pre-LN encoder block.

        Parameters
        ----------
        x:
            Input of shape ``(batch, seq, d_embed)``.
        src_key_padding_mask:
            Bool mask of shape ``(batch, seq)``; ``True`` means ignore token.
        return_weights:
            Whether to return averaged attention weights.

        Returns
        -------
        x : torch.Tensor
            Updated representation, shape ``(batch, seq, d_embed)``.
        attn_weights : torch.Tensor or None
            Shape ``(batch, seq, seq)`` if ``return_weights`` else ``None``.
        """
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, (attn_weights if return_weights else None)


class SelfAttentionEmbedder(BaseAttributeEmbedder):
    """Transformer-based attribute embedding model.

    Processes per-attribute token embeddings through a stack of self-attention
    layers, then pools to a fixed-size vector.

    Parameters
    ----------
    config:
        Model configuration.  Uses ``d_embed``, ``d_model``, ``dropout``,
        ``n_heads``, ``n_layers``, ``use_cls_token``, ``pooling``,
        ``attribute_groups``, ``attribute_configs``, and ``vocab_sizes``.
    """

    def __init__(self, config: AttributeEmbedderConfig) -> None:
        super().__init__(config)

        if config.use_cls_token and config.pooling != "cls":
            raise ValueError("use_cls_token=True requires pooling='cls'")
        if not config.use_cls_token and config.pooling == "cls":
            raise ValueError("pooling='cls' requires use_cls_token=True")

        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling

        # Learned [CLS] token
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_embed))

        # Learned attribute-type positional encodings (one per group label)
        self.group_embeddings = nn.Embedding(len(_GROUP_LABELS), config.d_embed)
        self._build_group_ids(config)

        # Transformer layers
        self.layers = nn.ModuleList([
            _AttentionEncoderLayer(config.d_embed, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.d_embed, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def _build_group_ids(self, config: AttributeEmbedderConfig) -> None:
        """Register a buffer of group indices for each active attribute."""
        groups = config.attribute_groups or {}
        unknown_idx = _GROUP_TO_IDX["unknown"]
        group_ids = [
            _GROUP_TO_IDX.get(groups.get(cfg.name, "unknown"), unknown_idx)
            for cfg in self._active_attribute_configs
        ]
        self.register_buffer("group_ids", torch.tensor(group_ids, dtype=torch.long))

    def forward(
        self,
        attributes: dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Embed attributes through a transformer encoder.

        Parameters
        ----------
        attributes:
            Mapping from attribute name to ``(batch,)`` tensor.
        return_attention:
            If ``True``, return ``(embeddings, attention_weights)`` where
            ``attention_weights`` is a list of ``(batch, seq, seq)`` tensors,
            one per layer.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_model)``, or a tuple when
            ``return_attention=True``.
        """
        tokens, mask = self.get_attribute_tokens(attributes)  # (B, n_attrs, d_embed), (B, n_attrs)
        batch_size = tokens.shape[0]

        # Add learned attribute-type positional encodings
        group_enc = self.group_embeddings(self.group_ids)  # (n_attrs, d_embed)
        tokens = tokens + group_enc.unsqueeze(0)           # (B, n_attrs, d_embed)

        # Prepend [CLS] token (never masked)
        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, -1, -1)          # (B, 1, d_embed)
            tokens = torch.cat([cls, tokens], dim=1)                  # (B, n_attrs+1, d_embed)
            cls_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=mask.device
            )
            mask = torch.cat([cls_mask, mask], dim=1)                 # (B, n_attrs+1)

        # Apply transformer layers
        all_attn_weights: list[torch.Tensor] = []
        for layer in self.layers:
            tokens, attn_w = layer(
                tokens,
                src_key_padding_mask=mask,
                return_weights=return_attention,
            )
            if return_attention and attn_w is not None:
                all_attn_weights.append(attn_w)

        # Pool to single vector
        if self.pooling == "cls":
            x = tokens[:, 0]  # (B, d_embed)
        elif self.pooling == "mean":
            valid = ~mask                                              # (B, seq)
            counts = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
            x = (tokens * valid.unsqueeze(-1).float()).sum(dim=1) / counts
        else:  # sum
            x = tokens.sum(dim=1)

        x = self.dropout(x)
        x = self.projection(x)  # (B, d_model)
        x = self.norm(x)

        if return_attention:
            return x, all_attn_weights
        return x
