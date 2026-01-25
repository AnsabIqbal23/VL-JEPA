"""
VL-JEPA Source Package

This package contains the core components for the VL-JEPA project.

Modules:
    - predictor_network: Abdullah's PredictorNetwork implementation
"""

from .predictor_network import PredictorNetwork, PredictorNetworkConfig

__all__ = ['PredictorNetwork', 'PredictorNetworkConfig']
__version__ = '0.1.0'
__author__ = 'Abdullah (Phase 1)'
