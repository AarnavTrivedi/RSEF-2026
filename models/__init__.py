"""
PulmoTrace Models Module

Contains:
    - MPPDSurrogate: Differentiable neural surrogate for MPPD
    - PIVAE: Physics-Informed Variational Autoencoder
    - DepositionCalculator: High-level interface for deposition predictions
"""

from .surrogate import (
    MPPDSurrogate,
    MPPDSurrogateTrainer,
    MPPDDataGenerator,
    create_and_train_surrogate
)

from .pivae import (
    PIVAE,
    Encoder,
    BiologicalDecoder,
    PhysicsDecoder,
    load_pivae
)

# New ICRP-based surrogate with benchmark data
from .mppd_surrogate import (
    MPPDSurrogate as ICRPSurrogate,
    MPPDSurrogateTrainer as ICRPSurrogateTrainer,
    DepositionCalculator,
    create_trained_surrogate,
    ICRP66_REST,
    ICRP66_EXERCISE,
    ICRP_PHYSIOLOGY
)

__all__ = [
    # Original surrogate
    'MPPDSurrogate',
    'MPPDSurrogateTrainer',
    'MPPDDataGenerator',
    'create_and_train_surrogate',
    # PI-VAE
    'PIVAE',
    'Encoder',
    'BiologicalDecoder',
    'PhysicsDecoder',
    'load_pivae',
    # ICRP-based surrogate
    'ICRPSurrogate',
    'ICRPSurrogateTrainer',
    'DepositionCalculator',
    'create_trained_surrogate',
    'ICRP66_REST',
    'ICRP66_EXERCISE',
    'ICRP_PHYSIOLOGY'
]
