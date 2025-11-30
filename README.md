# BEREAN Protocol v2.0

**Bioinformatic Exploration and Rational Engineering of Antibodies via Neural Networks**

Built on IBM's official [biomed-multi-alignment](https://github.com/BiomedSciAI/biomed-multi-alignment) (MAMMAL) foundation model.

## Module Provenance Report

### Summary

| Status | Count | Lines | Description |
|--------|-------|-------|-------------|
| **OFFICIAL** | 5 | N/A | External biomed-multi-alignment package |
| **ORIGINAL** | 6 | 2,866 | Kept from Phase 1 build |
| **NEW** | 6 | 3,167 | Added in v2.0 refactor |
| **REMOVED** | 2 | - | Replaced by official API |
| **TOTAL** | - | **6,033** | (excluding external) |

---

## OFFICIAL Modules (from biomed-multi-alignment)

These are **external dependencies** from IBM's official package. Install with:

```bash
pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]
```

| Module | Source | Description |
|--------|--------|-------------|
| `mammal.model.Mammal` | biomed-multi-alignment | Foundation model (458M parameters) |
| `mammal.keys` | biomed-multi-alignment | Standard key definitions (ENCODER_INPUTS_STR, etc.) |
| `fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp` | fuse-med-ml | Multi-modal tokenizer |
| `mammal.main_finetune` | biomed-multi-alignment | Official training with Hydra |
| `mammal.examples.*` | biomed-multi-alignment | Example tasks (DTI, solubility, etc.) |

---

## ORIGINAL Modules (from Phase 1)

These modules were **kept without modification** from the Phase 1 build as they correctly implement paper-specific functionality.

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| `berean.data.dataset` | `data/dataset.py` | 590 | AntibodyHAPair, VariantRecord dataclasses |
| `berean.data.splits` | `data/splits.py` | 557 | 4 split strategies from Cleveland Clinic paper |
| `berean.data.preprocessing` | `data/preprocessing.py` | 485 | Sequence validation, normalization |
| `berean.data.loaders` | `data/loaders.py` | 273 | PyTorch DataLoader utilities |
| `berean.evaluation.metrics` | `evaluation/metrics.py` | 534 | AUROC, PR-AUC, MCC with bootstrap CI |
| `berean.core.embeddings` | `core/embeddings.py` | 427 | ESM-2 embedding extraction (for ensemble) |

**Total ORIGINAL: 2,866 lines**

---

## NEW Modules (v2.0 refactor)

These modules were **added** to integrate with the official MAMMAL API and extend functionality.

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| `berean.core.mammal_adapter` | `core/mammal_adapter.py` | 734 | Adapter wrapping official MAMMAL API |
| `berean.tasks.antibody_ha_binding` | `tasks/antibody_ha_binding/__init__.py` | 509 | MAMMAL task for Ab-HA binding classification |
| `berean.tasks.hai_prediction` | `tasks/hai_prediction/__init__.py` | 471 | MAMMAL task for HAI titer regression |
| `berean.extensions.esm2_ensemble` | `extensions/esm2_ensemble.py` | 545 | ESM-2 + MAMMAL ensemble predictor |
| `berean.extensions.attention_analysis` | `extensions/attention_analysis.py` | 457 | Epitope/paratope prediction from attention |
| `berean.surveillance.gisaid` | `surveillance/gisaid.py` | 451 | GISAID influenza integration |

**Total NEW: 3,167 lines**

---

## REMOVED Modules

These were **replaced** by the official biomed-multi-alignment API:

| Old Module | Replaced By | Reason |
|------------|-------------|--------|
| `berean.core.mammal_wrapper` (458 lines) | `mammal.model.Mammal` (OFFICIAL) | Custom wrapper unnecessary |
| `berean.core.tokenizer` (412 lines) | `ModularTokenizerOp` (OFFICIAL) | Official tokenizer required for proper prompts |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BEREAN Protocol v2.0                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    OFFICIAL (biomed-multi-alignment)            │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │   Mammal    │  │ ModularTokenizer │  │  main_finetune   │   │   │
│  │  │   Model     │  │       Op         │  │   (training)     │   │   │
│  │  └─────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ wraps                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    NEW (berean v2.0) - 3,167 lines              │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │   MAMMAL    │  │  Ab-HA Binding   │  │  HAI Prediction  │   │   │
│  │  │   Adapter   │  │     Task         │  │     Task         │   │   │
│  │  └─────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │ ESM-2       │  │   Attention      │  │    GISAID        │   │   │
│  │  │ Ensemble    │  │   Analysis       │  │  Surveillance    │   │   │
│  │  └─────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ uses                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ORIGINAL (Phase 1) - 2,866 lines             │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │   Dataset   │  │     Splits       │  │    Metrics       │   │   │
│  │  │  Classes    │  │  (4 strategies)  │  │  (AUROC, etc.)   │   │   │
│  │  └─────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```python
# Install dependencies
# pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]

from berean.core import MAMMALAdapter          # NEW (wraps OFFICIAL)
from berean.tasks import AntibodyHABindingTask # NEW
from berean.data import DataSplitter           # ORIGINAL
from berean.evaluation import compute_auroc    # ORIGINAL

# Initialize adapter (uses OFFICIAL MAMMAL API)
adapter = MAMMALAdapter()

# Create task (NEW - follows OFFICIAL task pattern)
task = AntibodyHABindingTask()

# Preprocess data
sample = task.data_preprocessing({
    "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
    "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKR",
    "ha_sequence": "MKTIIALSYILCLVFAQKIPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEF",
    "binding_label": 1
})

# Predict
result = adapter.predict_binding(
    heavy_chain=sample["heavy_chain"],
    light_chain=sample["light_chain"],
    ha_sequence=sample["ha_sequence"]
)
print(f"Binding probability: {result['probability']:.3f}")
```

---

## Target Benchmarks (from Cleveland Clinic Paper)

| Split Strategy | Target AUROC | Description |
|---------------|--------------|-------------|
| Lenient | 0.91-0.92 | Random train/test |
| HA-exclusive | 0.90 | Novel strains in test |
| mAb-exclusive | 0.73 | Novel antibodies in test |
| mAb-cluster-exclusive | 0.63-0.66 | Divergent antibodies in test |

---

## References

- **MAMMAL Paper**: [arXiv:2410.22367](https://arxiv.org/abs/2410.22367)
- **Cleveland Clinic Paper**: Barkan et al. (2025) "Leveraging large language models to predict antibody biological activity against influenza A hemagglutinin" CSBJ 27
- **GitHub**: [BiomedSciAI/biomed-multi-alignment](https://github.com/BiomedSciAI/biomed-multi-alignment)
- **HuggingFace**: [ibm/biomed.omics.bl.sm.ma-ted-458m](https://huggingface.co/ibm/biomed.omics.bl.sm.ma-ted-458m)

---

## Author

**Ej** - PhD Student, Cleveland Clinic Lerner College of Medicine  
**Lab**: Dr. Giuseppe Sautto, Cleveland Clinic Florida Research and Innovation Center
