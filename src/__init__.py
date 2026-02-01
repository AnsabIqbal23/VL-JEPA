"""
VL-JEPA Source Package

This package contains the core components for the VL-JEPA model:
- loss.py: Loss functions (CosineSimilarityLoss, JEPALoss)
- train.py: Training utilities (train_step, validate_step, train_epoch)
- predictor_network.py: Predictor network implementation (Abdullah's PredictorNetwork)
"""

from .loss import cosine_similarity_loss, CosineSimilarityLoss, JEPALoss
from .train import train_step, validate_step, train_epoch
from .predictor_network import PredictorNetwork, PredictorNetworkConfig

__all__ = [
    'cosine_similarity_loss',
    'CosineSimilarityLoss',
    'JEPALoss',
    'train_step',
    'validate_step',
    'train_epoch',
    'PredictorNetwork',
    'PredictorNetworkConfig',
]

__version__ = '0.1.0'
__author__ = 'Abdullah (Phase 1)'
