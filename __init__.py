"""
BEREAN Protocol v2.0
Bioinformatic Exploration and Rational Engineering of Antibodies via Neural Networks

Built on IBM's official biomed-multi-alignment (MAMMAL) foundation model.

Architecture Overview:
======================
OFFICIAL (from biomed-multi-alignment):
    - mammal.model.Mammal - Foundation model
    - mammal.keys - Standard key definitions
    - fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp - Tokenization
    - mammal training infrastructure (Hydra configs, FuseMedML)

ORIGINAL (kept from Phase 1 build):
    - berean.data.dataset - AntibodyHAPair, VariantRecord dataclasses
    - berean.data.splits - 4 split strategies from Cleveland Clinic paper
    - berean.data.preprocessing - Sequence validation, numbering
    - berean.data.loaders - DataLoader utilities
    - berean.evaluation.metrics - AUROC benchmarks, CI calculation

NEW (added in this refactor):
    - berean.tasks.antibody_ha_binding - Custom MAMMAL task for Ab-HA binding
    - berean.tasks.hai_prediction - HAI titer regression task
    - berean.core.mammal_adapter - Adapter wrapping official MAMMAL API
    - berean.extensions.esm2_ensemble - Optional ESM-2 embedding augmentation
    - berean.extensions.attention_analysis - Cross-attention interpretation
    - berean.surveillance.gisaid - GISAID influenza integration

Reference:
    Barkan et al. (2025) Computational and Structural Biotechnology Journal 27

Installation:
    pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]

Author: Ej (Cleveland Clinic Lerner College of Medicine / Sautto Lab)
"""

__version__ = "2.0.0"
__author__ = "Ej"
__lab__ = "Sautto Lab, Cleveland Clinic Florida"
__mammal_model__ = "ibm/biomed.omics.bl.sm.ma-ted-458m"

# Module provenance tracking
MODULE_PROVENANCE = {
    # === OFFICIAL: From biomed-multi-alignment ===
    "mammal.model.Mammal": {
        "status": "OFFICIAL",
        "source": "biomed-multi-alignment",
        "description": "Foundation model with 458M parameters"
    },
    "mammal.keys": {
        "status": "OFFICIAL", 
        "source": "biomed-multi-alignment",
        "description": "Standard keys for sample dictionaries"
    },
    "fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp": {
        "status": "OFFICIAL",
        "source": "biomed-multi-alignment",
        "description": "Multi-modal tokenizer for AA/SMILES/scalars"
    },
    
    # === ORIGINAL: From Phase 1 build ===
    "berean.data.dataset": {
        "status": "ORIGINAL",
        "source": "berean_phase1",
        "description": "Data structures for Ab-HA pairs"
    },
    "berean.data.splits": {
        "status": "ORIGINAL",
        "source": "berean_phase1", 
        "description": "4 split strategies from paper"
    },
    "berean.data.preprocessing": {
        "status": "ORIGINAL",
        "source": "berean_phase1",
        "description": "Sequence validation and normalization"
    },
    "berean.data.loaders": {
        "status": "ORIGINAL",
        "source": "berean_phase1",
        "description": "PyTorch DataLoader utilities"
    },
    "berean.evaluation.metrics": {
        "status": "ORIGINAL",
        "source": "berean_phase1",
        "description": "AUROC, PR-AUC with bootstrap CI"
    },
    
    # === NEW: Added in this refactor ===
    "berean.tasks.antibody_ha_binding": {
        "status": "NEW",
        "source": "berean_v2_refactor",
        "description": "MAMMAL task for binding classification"
    },
    "berean.tasks.hai_prediction": {
        "status": "NEW",
        "source": "berean_v2_refactor",
        "description": "MAMMAL task for HAI titer regression"
    },
    "berean.core.mammal_adapter": {
        "status": "NEW",
        "source": "berean_v2_refactor",
        "description": "High-level adapter for official MAMMAL"
    },
    "berean.extensions.esm2_ensemble": {
        "status": "NEW",
        "source": "berean_v2_refactor",
        "description": "ESM-2 embedding augmentation"
    },
    "berean.extensions.attention_analysis": {
        "status": "NEW",
        "source": "berean_v2_refactor",
        "description": "Attention weight interpretation"
    },
    "berean.surveillance.gisaid": {
        "status": "NEW",
        "source": "berean_v2_refactor",
        "description": "GISAID influenza monitoring"
    },
}

def print_module_report():
    """Print comprehensive module status report."""
    print("=" * 70)
    print("BEREAN Protocol v2.0 - Module Provenance Report")
    print("=" * 70)
    
    for status in ["OFFICIAL", "ORIGINAL", "NEW"]:
        modules = {k: v for k, v in MODULE_PROVENANCE.items() if v["status"] == status}
        if modules:
            print(f"\n{status} ({len(modules)} modules):")
            print("-" * 50)
            for name, info in modules.items():
                print(f"  {name}")
                print(f"    Source: {info['source']}")
                print(f"    {info['description']}")
    
    print("\n" + "=" * 70)

from . import core
from . import data
from . import tasks
from . import extensions
from . import surveillance
from . import evaluation

__all__ = [
    "core",
    "data",
    "tasks",
    "extensions",
    "surveillance", 
    "evaluation",
    "print_module_report",
    "MODULE_PROVENANCE",
]
