"""
BEREAN Prediction Heads
=======================

Status: NEW (v1.1 - with Integrated Binding Explainer)
Source: berean_v2_refactor (OmniSynapticBindingPredictor implementation)

This module contains specialized prediction heads that sit on top of the
MAMMAL foundation model embeddings. These heads correspond to the Core
Processing components in the BEREAN architecture schematic.

Heads:
    - BindingHead: Binary binding classification (Ab binds HA: yes/no)
    - AffinityHead: Quantitative affinity prediction (IC50, Kd values)
    - CrossReactivityHead: Multi-strain breadth prediction

Explainability:
    - IntegratedBindingExplainer: Shows which residues contributed to binding decision

Architecture:
    
    MAMMAL Embeddings (768-dim) + Attention Weights
            │                          │
            ▼                          │
    ┌───────────────────────────────────────────────────────────────┐
    │         OmniSynapticBindingPredictor                          │
    │  ┌─────────┐ ┌─────────┐ ┌─────────────┐                      │
    │  │ Binding │ │Affinity │ │Cross-React. │                      │
    │  │  Head   │ │  Head   │ │    Head     │                      │
    │  └────┬────┘ └────┬────┘ └──────┬──────┘                      │
    │       │           │             │                             │
    │       ▼           ▼             ▼                             │
    │   [0/1]       [IC50]      [strain1..N]                        │
    │       │                                                       │
    │       └───────────────────────────────┐                       │
    │                                       ▼                       │
    │                          ┌────────────────────────┐           │
    │                          │ IntegratedBindingExpl. │◄──────────┘
    │                          │ - Key Ab residues      │  (attention)
    │                          │ - Key Ag residues      │
    │                          │ - Interaction pairs    │
    │                          │ - CDR contributions    │
    │                          └────────────────────────┘           │
    └───────────────────────────────────────────────────────────────┘

Reference:
    Barkan et al. (2025) CSBJ - Cleveland Clinic MAMMAL paper
"""

__version__ = "1.1.0"
__author__ = "Ej"
__status__ = "NEW"

from .binding_head import (
    BindingHead,
    BindingHeadConfig,
    BindingPrediction,
)

from .affinity_head import (
    AffinityHead,
    AffinityHeadConfig,
    AffinityPrediction,
    AffinityType,
)

from .cross_reactivity_head import (
    CrossReactivityHead,
    CrossReactivityHeadConfig,
    CrossReactivityPrediction,
    StrainCoverage,
)

from .omni_synaptic_predictor import (
    OmniSynapticBindingPredictor,
    OmniSynapticConfig,
    UnifiedPrediction,
)

from .binding_explainer import (
    IntegratedBindingExplainer,
    BindingExplanation,
    ResidueContribution,
    InteractionPair,
    explain_binding,
)

from .queryable_attention import (
    QueryableAttentionMatrix,
    AntibodyResidueQuery,
    AntigenResidueQuery,
    ResidueResiduePairQuery,
    AttentionTarget,
    create_queryable_attention,
)

__all__ = [
    # Binding Head
    "BindingHead",
    "BindingHeadConfig",
    "BindingPrediction",
    # Affinity Head
    "AffinityHead",
    "AffinityHeadConfig", 
    "AffinityPrediction",
    "AffinityType",
    # Cross-Reactivity Head
    "CrossReactivityHead",
    "CrossReactivityHeadConfig",
    "CrossReactivityPrediction",
    "StrainCoverage",
    # Unified Predictor
    "OmniSynapticBindingPredictor",
    "OmniSynapticConfig",
    "UnifiedPrediction",
    # Binding Explainer (NEW)
    "IntegratedBindingExplainer",
    "BindingExplanation",
    "ResidueContribution",
    "InteractionPair",
    "explain_binding",
    # Queryable Attention (NEW)
    "QueryableAttentionMatrix",
    "AntibodyResidueQuery",
    "AntigenResidueQuery",
    "ResidueResiduePairQuery",
    "AttentionTarget",
    "create_queryable_attention",
]
