"""
VL-JEPA Source Package

This package contains the core components for the VL-JEPA model:
- loss.py: Loss functions (CosineSimilarityLoss, JEPALoss)
- train.py: Training utilities (train_step, validate_step, train_epoch)
"""

from .loss import cosine_similarity_loss, CosineSimilarityLoss, JEPALoss
from .train import train_step, validate_step, train_epoch

__all__ = [
    'cosine_similarity_loss',
    'CosineSimilarityLoss',
    'JEPALoss',
    'train_step',
    'validate_step',
    'train_epoch',
]
