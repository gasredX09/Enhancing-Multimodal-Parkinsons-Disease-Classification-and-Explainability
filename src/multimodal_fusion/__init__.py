"""Multimodal late fusion for Parkinson's Disease classification."""
from .fusion import LateFusionModel, StackingFusionModel
from .loaders import (
    load_all_from_embeddings,
    compute_oof_predictions,
    load_gait_from_embeddings,
    load_handwriting_from_embeddings,
    load_speech_from_embeddings,
)
from .explainability import FusionExplainer, GaitEmbeddingExplainer
from . import loaders

__all__ = [
    "LateFusionModel",
    "StackingFusionModel",
    "load_all_from_embeddings",
    "compute_oof_predictions",
    "load_gait_from_embeddings",
    "load_handwriting_from_embeddings",
    "load_speech_from_embeddings",
    "FusionExplainer",
    "GaitEmbeddingExplainer",
    "loaders",
]
