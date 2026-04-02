"""Adapter for integrating pre-trained attribute embedders with ActVAE (Caveat).

ActVAE is an activity schedule generation model that conditions on a label
encoder.  This module defines the interface that ActVAE expects from a label
encoder and provides a :class:`CaveatAdapter` that wraps any
:class:`~models.base.BaseAttributeEmbedder` to satisfy that interface.

Three transfer modes are supported:

``frozen``
    Pre-trained weights are fixed.  Only the downstream VAE is trained.
    Use this as the primary evaluation mode.

``fine_tuned``
    Pre-trained weights are unfrozen and can be updated during VAE training.
    Intended for a low-learning-rate fine-tuning phase after VAE warm-up.

``random_init``
    Same architecture as the pre-trained model, but with randomly initialised
    weights.  Used as an ablation baseline (equivalent to the current ActVAE
    baseline with no pre-training).

Public API
----------
LabelEncoderProtocol
    Runtime-checkable Protocol defining the interface ActVAE expects.

CaveatAdapterConfig
    Configuration dataclass for :class:`CaveatAdapter`.

CaveatAdapter
    Wraps a :class:`~models.base.BaseAttributeEmbedder` to satisfy
    :class:`LabelEncoderProtocol`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from typing_extensions import Protocol, runtime_checkable

from models.base import BaseAttributeEmbedder


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LabelEncoderProtocol(Protocol):
    """Interface that ActVAE/Caveat expects from a label encoder.

    Any object satisfying this protocol can be passed to ActVAE as its
    label encoder.  :class:`CaveatAdapter` implements this protocol.

    Methods
    -------
    encode(attributes) -> Tensor
        Map a batch of encoded attributes to a label embedding vector.
        Shape: ``(batch, label_dim)``.
    label_dim : int
        Dimensionality of the label embedding (output of ``encode``).
    """

    def encode(self, attributes: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @property
    def label_dim(self) -> int: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CaveatAdapterConfig:
    """Configuration for :class:`CaveatAdapter`.

    Parameters
    ----------
    transfer_mode:
        One of ``'frozen'``, ``'fine_tuned'``, or ``'random_init'``.
        See module docstring for semantics.
    fine_tune_lr_factor:
        Learning rate multiplier relative to the VAE learning rate used
        when ``transfer_mode == 'fine_tuned'``.  Ignored for other modes.
    """

    transfer_mode: Literal["frozen", "fine_tuned", "random_init"] = "frozen"
    fine_tune_lr_factor: float = 0.1


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CaveatAdapter(nn.Module):
    """Wraps a :class:`~models.base.BaseAttributeEmbedder` for use in ActVAE.

    Satisfies :class:`LabelEncoderProtocol`.

    Parameters
    ----------
    embedder:
        A trained :class:`~models.base.BaseAttributeEmbedder`.  In
        ``'random_init'`` mode, only the architecture config is used;
        the weights are re-randomised.
    config:
        Transfer mode and fine-tuning settings.
    """

    def __init__(
        self,
        embedder: BaseAttributeEmbedder,
        config: CaveatAdapterConfig,
    ) -> None:
        super().__init__()
        self.adapter_config = config

        if config.transfer_mode == "random_init":
            # Build a fresh model with the same architecture but new weights
            self._model = type(embedder)(embedder.config)
        else:
            self._model = embedder

        # Freeze or unfreeze depending on transfer mode
        requires_grad = config.transfer_mode == "fine_tuned"
        for param in self._model.parameters():
            param.requires_grad_(requires_grad)

    def encode(self, attributes: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed a batch of attributes.

        Parameters
        ----------
        attributes:
            Attribute dict as produced by ``AttributeEncoder.transform``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, label_dim)``.
        """
        return self._model(attributes)

    @property
    def label_dim(self) -> int:
        """Dimensionality of the label embedding."""
        return self._model.embed_dim

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return parameters that require gradients.

        For ``'frozen'`` mode this is empty.  For ``'fine_tuned'`` mode
        this returns all embedder parameters, intended to be passed to the
        VAE optimiser with a scaled learning rate
        (``config.fine_tune_lr_factor``).
        """
        return [p for p in self._model.parameters() if p.requires_grad]
