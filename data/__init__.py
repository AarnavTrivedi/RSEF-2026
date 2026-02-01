"""
PulmoTrace Data Module

Contains utilities for:
    - GEO dataset downloading and parsing
    - Gene expression normalization
    - Batch correction (ComBat-seq)
    - Spatial deconvolution
    - Physics-informed feature lists
    - PyTorch dataset creation
"""

from .processing import (
    GEODataLoader,
    ExpressionNormalizer,
    BatchCorrector,
    SpatialDeconvolution,
    PulmoTraceDataset,
    prepare_training_data,
    prepare_geo_training_data,
    PlatformAnnotation
)

from .features import (
    PHYSICS_FEATURES,
    VALIDATED_DATASETS,
    BRONCHIAL_MARKERS,
    ALVEOLAR_MARKERS,
    get_human_genes,
    get_mouse_genes,
    get_ortholog_map,
    get_dataset_info,
    list_validated_datasets,
    GeneOrthologs,
    DatasetMetadata
)

__all__ = [
    # Processing
    'GEODataLoader',
    'ExpressionNormalizer',
    'BatchCorrector',
    'SpatialDeconvolution',
    'PulmoTraceDataset',
    'prepare_training_data',
    'prepare_geo_training_data',
    'PlatformAnnotation',
    # Features
    'PHYSICS_FEATURES',
    'VALIDATED_DATASETS',
    'BRONCHIAL_MARKERS',
    'ALVEOLAR_MARKERS',
    'get_human_genes',
    'get_mouse_genes',
    'get_ortholog_map',
    'get_dataset_info',
    'list_validated_datasets',
    'GeneOrthologs',
    'DatasetMetadata',
]


