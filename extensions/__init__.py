"""
BEREAN Extensions Module
========================

Status: NEW (v2.0 Enhanced)
Source: berean_v2_refactor

Extensions that go beyond the official MAMMAL functionality:
    - ESM-2 ensemble for improved generalization
    - Attention analysis for epitope/paratope mapping (v1 and v2)
    - Multi-modal fusion approaches

These extensions aim to address the 0.63-0.73 AUROC gap on
mAb-exclusive splits observed in the Cleveland Clinic paper.

v2.0 Enhancements:
    - Proper CDR detection using IMGT/Kabat/Chothia numbering
    - IEDB epitope database integration for validation
    - Improved attention extraction aligned with MAMMAL tokenizer
    - Benchmark module comparing to crystal structure epitopes
    - Interactive Plotly visualization
"""

# Original attention analyzer (v1)
from .attention_analysis import AttentionAnalyzer, EpitopePrediction

# Enhanced attention analyzer (v2)
from .attention_analysis_v2 import (
    # Core classes
    EnhancedAttentionAnalyzer,
    EnhancedEpitopePrediction,
    
    # CDR annotation
    CDRAnnotator,
    CDRAnnotation,
    NumberingScheme,
    
    # Validation
    IEDBEpitopeDatabase,
    EpitopeValidation,
    
    # Attention extraction
    MAMMALAttentionExtractor,
    
    # Benchmarking
    EpitopeBenchmark,
    
    # Reference data
    HA_ANTIGENIC_SITES,
    HA_STEM_EPITOPES,
    CDR_DEFINITIONS,
)

# ESM-2 ensemble
from .esm2_ensemble import ESM2Ensemble, EnsemblePredictor

__all__ = [
    # ESM-2 Ensemble
    "ESM2Ensemble",
    "EnsemblePredictor",
    
    # Attention Analyzer v1 (legacy)
    "AttentionAnalyzer", 
    "EpitopePrediction",
    
    # Enhanced Attention Analyzer v2
    "EnhancedAttentionAnalyzer",
    "EnhancedEpitopePrediction",
    "CDRAnnotator",
    "CDRAnnotation",
    "NumberingScheme",
    "IEDBEpitopeDatabase",
    "EpitopeValidation",
    "MAMMALAttentionExtractor",
    "EpitopeBenchmark",
    
    # Reference data
    "HA_ANTIGENIC_SITES",
    "HA_STEM_EPITOPES",
    "CDR_DEFINITIONS",
]
