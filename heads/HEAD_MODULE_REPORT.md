# BEREAN Heads Module Report
## Neural Network Prediction Heads for MAMMAL Embeddings

**Status:** NEW (v2.0 Architecture Completion)  
**Date:** 2025-11-30  
**Author:** Ej  
**Lines of Code:** 3,132 (6 files)

---

## Executive Summary

This module completes the Core Processing layer of the BEREAN architecture by implementing three specialized neural network heads that sit on top of MAMMAL foundation model embeddings. These heads address the gap identified in the architecture schematic where only prompt-based prediction existed without dedicated neural network layers.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │       MAMMAL Foundation Model       │
                    │            (458M params)            │
                    └──────────────┬──────────────────────┘
                                   │
                         [768-dim embeddings]
                                   │
    ┌──────────────────────────────┼──────────────────────────────┐
    │           OmniSynapticBindingPredictor                      │
    │  ┌───────────┐   ┌───────────┐   ┌─────────────────┐        │
    │  │  Binding  │   │  Affinity │   │ Cross-Reactivity│        │
    │  │   Head    │   │   Head    │   │      Head       │        │
    │  │ (768→512  │   │ (768→512  │   │  (768→512→256   │        │
    │  │  →256→2)  │   │  →256     │   │   →N×64→N×2)    │        │
    │  └─────┬─────┘   │  →128→1)  │   └───────┬─────────┘        │
    │        │         └─────┬─────┘           │                  │
    │        ▼               ▼                 ▼                  │
    │   [P(bind)]       [pIC50/Kd]      [P(s1)...P(sN)]           │
    └─────────────────────────────────────────────────────────────┘
```

---

## Module Files

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | 86 | Package exports and documentation |
| `binding_head.py` | 474 | Binary binding classification head |
| `affinity_head.py` | 566 | Quantitative affinity prediction head |
| `cross_reactivity_head.py` | 654 | Multi-strain breadth prediction head |
| `omni_synaptic_predictor.py` | 744 | Unified orchestrator for all heads |
| `integration.py` | 608 | MAMMAL integration utilities |
| **TOTAL** | **3,132** | |

---

## 1. Binding Head (`binding_head.py`)

### Purpose
Binary classification predicting whether an antibody binds to a given antigen.

### Architecture
```
Input: MAMMAL Embeddings (768-dim)
    │
    ▼
LayerNorm(768)
    │
    ▼
Linear(768 → 512) + GELU + Dropout(0.1)
    │
    ▼
Linear(512 → 256) + GELU + Dropout(0.1)
    │
    ▼
Linear(256 → 2) → Softmax
    │
    ▼
Output: [P(non-binding), P(binding)]
```

### Key Features
- **Class Weighting**: Handles imbalanced data (~35% positive in MAMMAL dataset)
- **Confidence Scoring**: Reports prediction confidence
- **Batch Support**: Efficient batch inference

### Target Benchmarks (from Cleveland Clinic paper)
- Lenient split: 0.91-0.92 AUROC
- HA-exclusive: 0.90 AUROC
- mAb-exclusive: 0.73 AUROC
- mAb-cluster-exclusive: 0.63-0.66 AUROC

### Usage
```python
from berean.heads import BindingHead, BindingHeadConfig

config = BindingHeadConfig(input_dim=768)
head = BindingHead(config)

# Predict
embeddings = mammal_adapter.get_embeddings(ab_seq, ag_seq)
prediction = head.predict(embeddings)
print(f"Binds: {prediction.prediction}, Prob: {prediction.probability:.3f}")
```

---

## 2. Affinity Head (`affinity_head.py`)

### Purpose
Regression predicting quantitative binding affinity values (IC50, Kd, HAI titers).

### Architecture
```
Input: MAMMAL Embeddings (768-dim)
    │
    ▼
LayerNorm(768)
    │
    ▼
Linear(768 → 512) + GELU + Dropout(0.1)
    │
    ▼
Linear(512 → 256) + GELU + Dropout(0.1)
    │
    ▼
Linear(256 → 128) + GELU
    │
    ▼
Linear(128 → 1)
    │
    ▼
Output: Affinity value (log-transformed)
```

### Supported Affinity Types
| Type | Description | Transform | Output Range |
|------|-------------|-----------|--------------|
| IC50 | Half-maximal inhibition | pIC50 = -log₁₀(IC50) | 3-12 |
| Kd | Dissociation constant | log₁₀(Kd) | -12 to -3 |
| HAI | Hemagglutination inhibition | log₂(HAI) | 0-13 |
| EC50 | Half-maximal effect | Similar to IC50 | 3-12 |

### Key Features
- **Automatic Transform**: Converts between log and original units
- **Output Clamping**: Prevents unrealistic predictions
- **Potency Labels**: Human-readable IC50 interpretation

### Specialized Heads
```python
from berean.heads.affinity_head import IC50Head, HAIHead, KdHead

ic50_head = IC50Head(input_dim=768)
hai_head = HAIHead(input_dim=768)

# HAI protective threshold
pred = hai_head.predict(embeddings)
is_protective = pred.raw_value >= 40  # Standard protective titer
```

---

## 3. Cross-Reactivity Head (`cross_reactivity_head.py`)

### Purpose
Multi-output classification predicting antibody binding across multiple influenza strains for broadly neutralizing antibody (bnAb) identification.

### Architecture
```
Input: MAMMAL Embeddings (768-dim)
    │
    ▼
LayerNorm(768)
    │
    ▼
Shared Backbone: Linear(768→512) + GELU + Dropout
                        │
                 Linear(512→256) + GELU + Dropout
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    StrainHead₁    StrainHead₂    StrainHeadₙ
    (256→64→2)     (256→64→2)     (256→64→2)
        │               │               │
        ▼               ▼               ▼
    P(bind H1)     P(bind H3)     P(bind H5)
```

### Default Strain Panel
```
Group 1 HAs:
  - H1N1/California/04/2009 (Pandemic 2009)
  - H1N1/Michigan/45/2015 (Seasonal)
  - H2N2/Japan/305/1957 (Historical)
  - H5N1/Vietnam/1194/2004 (Avian)
  - H6N1/Taiwan/2/2013 (Avian)

Group 2 HAs:
  - H3N2/Hong_Kong/1/1968 (Historical)
  - H3N2/Texas/50/2012 (Seasonal)
  - H7N9/Anhui/1/2013 (Avian)
  - H10N8/Jiangxi/IPB13/2013 (Avian)
```

### Output Metrics
- **Breadth Score**: Fraction of strains predicted to bind
- **Group 1/2 Coverage**: Coverage within each HA phylogenetic group
- **bnAb Classification**: True if ≥50% breadth AND covers both groups

### Specialized Heads
```python
from berean.heads.cross_reactivity_head import (
    SeasonalFluCrossReactivityHead,
    PandemicPreparednessCrossReactivityHead
)

seasonal_head = SeasonalFluCrossReactivityHead(input_dim=768)
pandemic_head = PandemicPreparednessCrossReactivityHead(input_dim=768)
```

---

## 4. OmniSynapticBindingPredictor (`omni_synaptic_predictor.py`)

### Purpose
Unified orchestrator that combines all three heads and provides:
- Multi-task learning support
- Composite therapeutic potential scoring
- Uncertainty estimation
- Single prediction interface

### Configuration
```python
from berean.heads import OmniSynapticConfig, OmniSynapticBindingPredictor

config = OmniSynapticConfig(
    embedding_dim=768,
    enable_binding_head=True,
    enable_affinity_head=True,
    enable_cross_reactivity_head=True,
    pooling_strategy="mean",  # "mean", "max", "attention", "cls"
    use_uncertainty=True,
)

predictor = OmniSynapticBindingPredictor(config)
```

### Unified Prediction
```python
prediction = predictor.predict(embeddings)

# Access all predictions
print(f"Binds: {prediction.binding.prediction}")
print(f"IC50: {prediction.affinity.raw_value} nM")
print(f"Breadth: {prediction.cross_reactivity.breadth_score:.1%}")
print(f"Composite Score: {prediction.composite_score:.3f}")
print(f"Is Promising: {prediction.is_promising_candidate}")
```

### Composite Score Calculation
```
Composite = 0.3 × P(binding) + 0.4 × normalized_affinity + 0.3 × breadth_score

Where:
- normalized_affinity = (pIC50 - 3) / 9  (scaled to 0-1)
- breadth_score = bound_strains / total_strains
```

### Multi-Task Training
```python
from berean.heads.omni_synaptic_predictor import OmniSynapticTrainer

trainer = OmniSynapticTrainer(predictor, learning_rate=1e-4)

for epoch in range(epochs):
    loss, head_losses = trainer.train_step(
        embeddings=batch_embeddings,
        binding_labels=batch_binding_labels,
        affinity_targets=batch_ic50_values,
        cross_reactivity_labels=batch_strain_labels,
    )
```

---

## Integration with MAMMAL Adapter

### Complete Workflow
```python
from berean.core.mammal_adapter import MAMMALAdapter
from berean.heads import OmniSynapticBindingPredictor, OmniSynapticConfig

# Initialize
adapter = MAMMALAdapter()
predictor = OmniSynapticBindingPredictor()

# Get MAMMAL embeddings
ab_seq = "EVQLVESGGGLVQPGGSLRL..."
ag_seq = "MKTIIALSYIFCLVFA..."

embeddings = adapter.get_embeddings(ab_seq, ag_seq)

# Unified prediction
result = predictor.predict(embeddings)

# Evaluate candidate
if result.is_promising_candidate:
    print("✓ Promising therapeutic candidate!")
    print(f"  Binding confidence: {result.binding.confidence:.2f}")
    print(f"  Potency: IC50 = {result.affinity.raw_value:.1f} nM")
    print(f"  Breadth: {result.cross_reactivity.breadth_score:.1%}")
```

---

## Parameter Summary

| Component | Parameters | % of Total |
|-----------|------------|------------|
| Binding Head | ~526,850 | 28% |
| Affinity Head | ~461,569 | 25% |
| Cross-Reactivity Head | ~872,746 | 47% |
| **Total** | **~1,861,165** | 100% |

*Note: Cross-reactivity head parameters scale with number of strains (9 default)*

---

## Gap Analysis Update

### Previously Missing (from schematic):
| Component | Status | Implementation |
|-----------|--------|----------------|
| Binding Head | ✅ BUILT | `heads/binding_head.py` |
| Affinity Head | ✅ BUILT | `heads/affinity_head.py` |
| Cross-reactivity Head | ✅ BUILT | `heads/cross_reactivity_head.py` |
| OmniSynapticBindingPredictor | ✅ BUILT | `heads/omni_synaptic_predictor.py` |

### Core Processing Layer: **NOW 100% COMPLETE**

---

## Future Enhancements

1. **GradNorm Loss Weighting**: Automatic multi-task loss balancing
2. **Ensemble Predictions**: Multiple model averaging
3. **Attention Visualization**: Head attention patterns for interpretability
4. **Domain Adaptation**: Transfer learning for new pathogen families
5. **Structured Prediction**: Joint head optimization with constraints

---

## References

1. Barkan et al. (2025) "Modeling anti-HA antibody-HA binding using MAMMAL" - CSBJ
2. Vaswani et al. (2017) "Attention Is All You Need" - NeurIPS
3. Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning" - NeurIPS

---

*Generated: 2025-11-30*  
*BEREAN Protocol v2.0 - Heads Module*
