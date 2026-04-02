"""Addition baseline embedding model.

Embeds each attribute independently, sums the embeddings, then projects to
``d_model`` with a single linear layer followed by layer normalisation.

This is the simplest possible model and serves as the baseline against which
all other architectures are compared.  It has no nonlinearities beyond the
final LayerNorm, directly analogous to the addition model in ActVAE.

Public API
----------
AdditionEmbedder
    Sum-pooling attribute embedding model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AttributeEmbedderConfig, BaseAttributeEmbedder


class AdditionEmbedder(BaseAttributeEmbedder):
    """Sum-pooling attribute embedding model.

    Embeds each attribute to a ``d_embed``-dimensional vector, sums them,
    applies dropout, then projects to ``d_model`` via a linear layer and
    layer normalisation.

    Parameters
    ----------
    config:
        Model configuration.  Only ``d_embed``, ``d_model``, ``dropout``,
        ``attribute_configs``, and ``vocab_sizes`` are used.
    """

    def __init__(self, config: AttributeEmbedderConfig) -> None:
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.d_embed, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, attributes: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed attributes by summing per-attribute embeddings.

        Missing or unknown attributes embed to zero and therefore contribute
        nothing to the sum.

        Parameters
        ----------
        attributes:
            Mapping from attribute name to ``(batch,)`` tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_model)``.
        """
        tokens, _ = self.get_attribute_tokens(attributes)  # (B, n_attrs, d_embed)
        x = tokens.sum(dim=1)                               # (B, d_embed)
        x = self.dropout(x)
        x = self.projection(x)                              # (B, d_model)
        x = self.norm(x)
        return x
