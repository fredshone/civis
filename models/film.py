"""FiLM (Feature-wise Linear Modulation) attribute embedding model.

Unlike the self-attention model, which treats all attributes symmetrically,
FiLM has a directed structure: a designated set of *context* attributes
conditions the processing of all remaining *content* attributes via learned
scale-and-shift (gamma/beta) modulation.

Graceful degradation
--------------------
When all context attributes are unknown or masked, the modulation falls back
to identity (gamma=1, beta=0) so the model behaves like the addition baseline.
This is implemented via a smooth blend controlled by an ``all_context_masked``
flag per sample, so the model never produces NaN under full context masking.

Diagnostics
-----------
After each forward pass the most recent gamma/beta tensors are cached
(detached from the graph).  Call :meth:`film_stats` to inspect the mean
modulation magnitude as a training diagnostic.

Public API
----------
FiLMEmbedder
    Context-conditioned attribute embedding model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AttributeEmbedderConfig, BaseAttributeEmbedder


class FiLMEmbedder(BaseAttributeEmbedder):
    """Context-conditioned attribute embedding model using FiLM modulation.

    Parameters
    ----------
    config:
        Model configuration.  Uses ``d_embed``, ``d_model``, ``dropout``,
        ``context_attributes``, ``attribute_configs``, and ``vocab_sizes``.
    """

    def __init__(self, config: AttributeEmbedderConfig) -> None:
        super().__init__(config)

        context_names = set(config.context_attributes)

        # Split active configs into context and content index lists
        self._context_indices: list[int] = []
        self._content_indices: list[int] = []
        for i, cfg in enumerate(self._active_attribute_configs):
            if cfg.name in context_names:
                self._context_indices.append(i)
            else:
                self._content_indices.append(i)

        n_content = len(self._content_indices)

        # Context combination: addition pool + linear -> d_embed context vector
        self.context_linear = nn.Linear(config.d_embed, config.d_embed)

        # Per-content-attribute FiLM generators: context_vec -> (gamma, beta)
        self.film_generators = nn.ModuleList([
            nn.Linear(config.d_embed, 2 * config.d_embed)
            for _ in range(n_content)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.d_embed, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        # Cached modulation tensors for diagnostics (detached, no grad)
        self._last_gamma: torch.Tensor | None = None
        self._last_beta: torch.Tensor | None = None

    def forward(self, attributes: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed attributes using FiLM-conditioned modulation.

        Context attributes are combined into a single conditioning vector that
        modulates each content attribute embedding via learned gamma/beta
        scale-and-shift.

        Parameters
        ----------
        attributes:
            Mapping from attribute name to ``(batch,)`` tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, d_model)``.
        """
        tokens, mask = self.get_attribute_tokens(attributes)  # (B, n_attrs, d_embed)
        batch_size = tokens.shape[0]
        device = tokens.device

        # --- Build context vector ---
        if self._context_indices:
            ctx_idx = torch.tensor(self._context_indices, dtype=torch.long, device=device)
            context_tokens = tokens[:, ctx_idx]          # (B, n_ctx, d_embed)
            context_mask = mask[:, ctx_idx]              # (B, n_ctx)
            ctx_sum = context_tokens.sum(dim=1)          # (B, d_embed)
            context_vec = self.context_linear(ctx_sum)   # (B, d_embed)
            # Per-sample flag: 1.0 if ALL context attrs are unknown
            all_ctx_masked = context_mask.all(dim=1).float().unsqueeze(-1)  # (B, 1)
        else:
            context_vec = torch.zeros(batch_size, self.config.d_embed, device=device)
            all_ctx_masked = torch.ones(batch_size, 1, device=device)

        # --- FiLM modulation of content attributes ---
        if self._content_indices:
            cnt_idx = torch.tensor(self._content_indices, dtype=torch.long, device=device)
            content_tokens = tokens[:, cnt_idx]  # (B, n_content, d_embed)
        else:
            content_tokens = torch.zeros(batch_size, 0, self.config.d_embed, device=device)

        modulated_list: list[torch.Tensor] = []
        all_gammas: list[torch.Tensor] = []
        all_betas: list[torch.Tensor] = []

        for j, gen in enumerate(self.film_generators):
            film_params = gen(context_vec)                           # (B, 2 * d_embed)
            gamma, beta = film_params.chunk(2, dim=-1)               # each (B, d_embed)
            # Graceful degradation: blend to identity when all context masked
            gamma = gamma * (1.0 - all_ctx_masked) + all_ctx_masked
            beta = beta * (1.0 - all_ctx_masked)
            e = content_tokens[:, j]                                 # (B, d_embed)
            modulated_list.append(gamma * e + beta)
            all_gammas.append(gamma.detach())
            all_betas.append(beta.detach())

        # Cache for diagnostics
        if all_gammas:
            self._last_gamma = torch.stack(all_gammas, dim=1)  # (B, n_content, d_embed)
            self._last_beta = torch.stack(all_betas, dim=1)

        if modulated_list:
            modulated = torch.stack(modulated_list, dim=1)  # (B, n_content, d_embed)
            x = modulated.sum(dim=1)                         # (B, d_embed)
        else:
            x = torch.zeros(batch_size, self.config.d_embed, device=device)

        x = self.dropout(x)
        x = self.projection(x)  # (B, d_model)
        x = self.norm(x)
        return x

    def film_stats(self) -> dict[str, float]:
        """Return FiLM modulation statistics from the most recent forward pass.

        Useful as a training diagnostic to monitor how strongly context
        attributes modulate content embeddings.

        Returns
        -------
        dict
            ``'mean_gamma_deviation'``: mean ``|gamma - 1|`` across all
            content attributes and batch samples.
            ``'mean_beta_magnitude'``: mean ``|beta|``.
            Both are 0.0 if no forward pass has been run or there are no
            content attributes.
        """
        if self._last_gamma is None or self._last_beta is None:
            return {"mean_gamma_deviation": 0.0, "mean_beta_magnitude": 0.0}
        return {
            "mean_gamma_deviation": float((self._last_gamma - 1.0).abs().mean()),
            "mean_beta_magnitude": float(self._last_beta.abs().mean()),
        }
