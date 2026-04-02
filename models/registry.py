"""Model registry and factory for attribute embedding models.

Provides config-driven instantiation via :func:`build_model` and utilities
for parameter counting and model inspection.

Public API
----------
MODEL_REGISTRY
    Dict mapping architecture name strings to model classes.
build_model(config)
    Instantiate a model from a configuration dictionary.
count_parameters(model)
    Count total, trainable, and per-component parameters.
model_summary(model, sample_batch)
    Print architecture, parameter counts, and forward-pass tensor shapes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .addition import AdditionEmbedder
from .attention import SelfAttentionEmbedder
from .base import AttributeEmbedderConfig, BaseAttributeEmbedder
from .film import FiLMEmbedder

MODEL_REGISTRY: dict[str, type[BaseAttributeEmbedder]] = {
    "addition": AdditionEmbedder,
    "attention": SelfAttentionEmbedder,
    "film": FiLMEmbedder,
}


def build_model(config: dict[str, Any]) -> BaseAttributeEmbedder:
    """Instantiate a model from a configuration dictionary.

    Parameters
    ----------
    config:
        Must contain an ``'architecture'`` key whose value matches a name in
        :data:`MODEL_REGISTRY`.  All remaining keys are forwarded to
        :class:`~models.base.AttributeEmbedderConfig`.  The input dict is
        not mutated.

    Returns
    -------
    BaseAttributeEmbedder

    Raises
    ------
    ValueError
        If ``'architecture'`` is missing or not in the registry.
    """
    config = dict(config)  # copy — do not mutate caller's dict
    arch = config.pop("architecture", None)
    if arch is None:
        raise ValueError("config must contain an 'architecture' key")
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    model_cls = MODEL_REGISTRY[arch]
    embedder_config = AttributeEmbedderConfig(**config)
    return model_cls(embedder_config)


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Count parameters in a model.

    Parameters
    ----------
    model:
        Any ``nn.Module``.

    Returns
    -------
    dict
        ``'total'``: total parameter count.
        ``'trainable'``: number of parameters with ``requires_grad=True``.
        ``'per_component'``: dict mapping top-level child module names to
        their recursive parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_component: dict[str, int] = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    return {"total": total, "trainable": trainable, "per_component": per_component}


def model_summary(
    model: BaseAttributeEmbedder,
    sample_batch: dict[str, torch.Tensor],
) -> None:
    """Print a model summary with architecture info and forward-pass shapes.

    Registers forward hooks on top-level child modules to capture output
    tensor shapes during a single forward pass.

    Parameters
    ----------
    model:
        A fitted embedding model.
    sample_batch:
        Sample attribute dict with ``batch_size >= 1``, used for the shape
        trace forward pass.
    """
    arch = type(model).__name__
    params = count_parameters(model)

    print("=" * 60)
    print(f"Architecture : {arch}")
    print(f"Output dim   : {model.embed_dim}")
    print(f"Parameters   : {params['total']:,} total, {params['trainable']:,} trainable")
    print()
    print("Per-component parameter counts:")
    for name, count in params["per_component"].items():
        print(f"  {name:<30} {count:>10,}")

    # Forward pass with shape tracing via hooks on top-level children
    shapes: list[tuple[str, tuple[int, ...]]] = []
    hooks = []

    def _make_hook(name: str):
        def _hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                shapes.append((name, tuple(t.shape)))
        return _hook

    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(_make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(sample_batch)
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    print()
    print("Forward pass tensor shapes:")
    for name, shape in shapes:
        print(f"  {name:<30} -> {shape}")
    final_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else tuple(out[0].shape)
    print(f"  {'output':<30} -> {final_shape}")
    print("=" * 60)
