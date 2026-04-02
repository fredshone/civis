"""Attribute embedding model architectures.

All models implement :class:`BaseAttributeEmbedder`, accepting a
``dict[str, Tensor]`` of encoded attributes (as returned by
``AttributeEncoder.transform``) and producing a ``(batch, d_model)``
embedding tensor.

Quick start
-----------
>>> from models import AdditionEmbedder, AttributeEmbedderConfig
>>> from datasets.encoding import AttributeEncoder, default_attribute_configs
>>> encoder = AttributeEncoder(default_attribute_configs()).fit(df)
>>> config = AttributeEmbedderConfig.from_encoder(encoder, d_embed=64, d_model=128)
>>> model = AdditionEmbedder(config)
>>> embeddings = model(encoder.transform(df))   # shape (N, 128)

Or use the factory for config-driven instantiation:

>>> from models import build_model
>>> embeddings = build_model({"architecture": "addition", **config_dict})(attrs)
"""

from .addition import AdditionEmbedder
from .attention import SelfAttentionEmbedder
from .base import AttributeEmbedderConfig, BaseAttributeEmbedder
from .film import FiLMEmbedder
from .registry import MODEL_REGISTRY, build_model, count_parameters, model_summary

__all__ = [
    "BaseAttributeEmbedder",
    "AttributeEmbedderConfig",
    "AdditionEmbedder",
    "SelfAttentionEmbedder",
    "FiLMEmbedder",
    "MODEL_REGISTRY",
    "build_model",
    "count_parameters",
    "model_summary",
]
