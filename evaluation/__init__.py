"""Downstream and intrinsic evaluation for schedule embedding models.

Public API
----------
DownstreamEvaluatorConfig, DownstreamEvaluator
    Shared base class and configuration.

LinearHead, MLPHead
    Task heads backed by sklearn.

compare_embeddings, random_baseline, frozen_attribute_baseline
    Multi-embedder comparison and baseline utilities.

WorkParticipationConfig, WorkParticipationEvaluator
    Binary classification: does this person go to work today?

WorkDurationConfig, WorkDurationEvaluator
    Regression: total time spent working (minutes).

TripCountConfig, TripCountEvaluator
    Regression: number of trips (activity transitions).

GenerativeEvaluatorConfig, GenerativeEvaluator
    Schedule generation evaluation (requires ActVAE integration for fit()).

CaveatAdapterConfig, CaveatAdapter, LabelEncoderProtocol
    ActVAE label encoder adapter and interface.

GeometryAnalyserConfig, GeometryAnalyser
    Intrinsic geometry analysis: alignment/uniformity, rank correlation,
    neighbourhood overlap, source separation, CKA.

AttentionAnalyserConfig, AttentionAnalyser
    Attention weight analysis for SelfAttentionEmbedder.
"""

from evaluation.base import (
    DownstreamEvaluator,
    DownstreamEvaluatorConfig,
    LinearHead,
    MLPHead,
    compare_embeddings,
    frozen_attribute_baseline,
    random_baseline,
)
from evaluation.caveat_adapter import (
    CaveatAdapter,
    CaveatAdapterConfig,
    LabelEncoderProtocol,
)
from evaluation.continuous import (
    TripCountConfig,
    TripCountEvaluator,
    WorkDurationConfig,
    WorkDurationEvaluator,
)
from evaluation.discrete import WorkParticipationConfig, WorkParticipationEvaluator
from evaluation.generative import GenerativeEvaluator, GenerativeEvaluatorConfig
from evaluation.geometry import GeometryAnalyser, GeometryAnalyserConfig
from evaluation.attention_analysis import AttentionAnalyser, AttentionAnalyserConfig

__all__ = [
    "DownstreamEvaluator",
    "DownstreamEvaluatorConfig",
    "LinearHead",
    "MLPHead",
    "compare_embeddings",
    "random_baseline",
    "frozen_attribute_baseline",
    "WorkParticipationConfig",
    "WorkParticipationEvaluator",
    "WorkDurationConfig",
    "WorkDurationEvaluator",
    "TripCountConfig",
    "TripCountEvaluator",
    "GenerativeEvaluator",
    "GenerativeEvaluatorConfig",
    "CaveatAdapter",
    "CaveatAdapterConfig",
    "LabelEncoderProtocol",
    "GeometryAnalyser",
    "GeometryAnalyserConfig",
    "AttentionAnalyser",
    "AttentionAnalyserConfig",
]
