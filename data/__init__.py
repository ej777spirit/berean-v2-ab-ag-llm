"""
BEREAN Data Module
==================

Status: ORIGINAL
Source: berean_phase1

Handles data loading, preprocessing, and split strategies for antibody-antigen datasets.
Implements the four split strategies from Barkan et al. (2025):
    1. Lenient (random pairs) - Target: 0.91-0.92 AUROC
    2. HA-exclusive (novel strains) - Target: 0.90 AUROC
    3. mAb-exclusive (novel antibodies) - Target: 0.73 AUROC
    4. mAb-cluster-exclusive (divergent antibodies) - Target: 0.63-0.66 AUROC

These modules were kept from Phase 1 as they correctly implement
the paper's split strategies and data structures.

Note: For converting data to MAMMAL prompt format, use the NEW task
modules (berean.tasks.antibody_ha_binding) which call the OFFICIAL
MAMMAL API.
"""

from .dataset import (
    AntibodyHADataset,
    AntibodyHAPair,
    DatasetConfig,
    load_mammal_dataset
)
from .splits import (
    SplitStrategy,
    DataSplitter,
    create_cv_folds
)
from .preprocessing import (
    DataPreprocessor,
    normalize_sequences,
    augment_dataset
)
from .loaders import (
    create_dataloader,
    collate_fn
)

__all__ = [
    "AntibodyHADataset",
    "AntibodyHAPair",
    "DatasetConfig",
    "load_mammal_dataset",
    "SplitStrategy",
    "DataSplitter",
    "create_cv_folds",
    "DataPreprocessor",
    "normalize_sequences",
    "augment_dataset",
    "create_dataloader",
    "collate_fn",
]
