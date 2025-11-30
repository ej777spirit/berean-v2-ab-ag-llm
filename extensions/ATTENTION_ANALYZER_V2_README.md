# Enhanced Attention Analyzer v2.0
## Epitope/Paratope Prediction with CDR Detection, IEDB Validation, and Interactive Visualization

**Status:** NEW (Enhanced)  
**Lines:** 1,650  
**Author:** Ej  
**Date:** 2025-11-30  

---

## Overview

The Enhanced Attention Analyzer v2.0 is a complete rewrite of the original attention analysis module, adding:

1. **Proper CDR Detection** using IMGT/Kabat/Chothia numbering schemes
2. **IEDB Epitope Database** integration for validation
3. **Improved Attention Extraction** aligned with MAMMAL tokenizer
4. **Benchmark Module** comparing to crystal structure epitopes
5. **Interactive Visualization** using Plotly (with matplotlib fallback)
6. **PyMOL Script Generation** for 3D structure visualization

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                  EnhancedAttentionAnalyzer                     │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ CDRAnnotator │  │ MAMMALAttn   │  │  EpitopeValidator    │ │
│  │ (IMGT/Kabat) │  │  Extractor   │  │  (IEDB/PDB)          │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                 │                     │             │
│         └────────────┬────┴─────────────────────┘             │
│                      │                                        │
│                      ▼                                        │
│         ┌────────────────────────┐                           │
│         │  EnhancedEpitopePred   │                           │
│         │  - epitope_positions   │                           │
│         │  - cdr_contacts        │                           │
│         │  - confidence          │                           │
│         │  - validation_score    │                           │
│         │  - antigenic_mapping   │                           │
│         └────────────────────────┘                           │
└────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. CDRAnnotator (Lines 248-400)

Annotates CDR regions using multiple numbering schemes.

**Supported Schemes:**
| Scheme | Description |
|--------|-------------|
| IMGT | International ImMunoGeneTics (default) |
| Kabat | Traditional numbering |
| Chothia | Structure-based |
| Martin | Enhanced Chothia |

**Features:**
- Auto-detection of heavy vs. light chain
- Conserved anchor refinement for CDR3
- Optional ANARCI integration for accurate numbering
- Fallback to regex-based detection

```python
from berean.extensions import CDRAnnotator, NumberingScheme

annotator = CDRAnnotator(scheme=NumberingScheme.IMGT)
cdr = annotator.annotate("EVQLVESGGGLVQPGGSLRL...")

print(f"CDR-H3: {cdr.cdr3}")  # (start, end, sequence)
print(f"All CDR positions: {cdr.get_cdr_positions()}")
```

### 2. IEDBEpitopeDatabase (Lines 402-490)

Curated influenza HA epitopes from IEDB for validation.

**Includes:**
- Linear epitopes with antibody annotations
- Discontinuous epitope patterns
- Stem vs. head classification

**Known Epitopes:**
| ID | Antibody | Type | Positions |
|----|----------|------|-----------|
| IEDB_1 | C179 | Linear | 91-108 |
| IEDB_2 | 5J8 | Linear | 140-150 |
| IEDB_D1 | CR6261 | Stem (discontinuous) | 18-21, 38-42, 291-292 |
| IEDB_D2 | CH65 | RBS (discontinuous) | 153-158, 186-194 |

### 3. MAMMALAttentionExtractor (Lines 492-590)

Handles attention weight extraction from MAMMAL model.

**Features:**
- Layer selection (default: last 4 layers)
- Head aggregation (mean, max)
- Prompt structure estimation
- Token-to-residue mapping

### 4. EnhancedAttentionAnalyzer (Lines 592-1100)

Main analysis class combining all components.

**Analysis Pipeline:**
1. Annotate CDRs in heavy and light chains
2. Extract attention weights from MAMMAL
3. Compute cross-attention between Ab and HA
4. Identify top epitope/paratope positions
5. Map paratope positions to specific CDRs
6. Map epitope to known antigenic sites
7. Validate against IEDB database
8. Calculate confidence score

### 5. EpitopeBenchmark (Lines 1100-1180)

Benchmark predictions against crystal structures.

**Benchmark Structures:**
| Complex | PDB | Epitope Type |
|---------|-----|--------------|
| CR6261-H1 | 3GBN | Stem |
| CH65-H1 | 5UGY | Head (RBS) |
| C179-H1 | 4HLZ | Head |
| FI6v3-H1 | 3ZTN | Stem |

---

## Reference Data

### HA Antigenic Sites

```python
HA_ANTIGENIC_SITES = {
    "H1": {
        "Sa": {"positions": [128-131, 156-159], "accessibility": "high"},
        "Sb": {"positions": [187-197], "accessibility": "high"},
        "Ca1": {"positions": [169-172, 206-208], "accessibility": "medium"},
        "Ca2": {"positions": [140-144, 224-226], "accessibility": "medium"},
        "Cb": {"positions": [74-78], "accessibility": "low"},
    },
    "H3": {
        "A": {"positions": [122-145], "accessibility": "high"},
        "B": {"positions": [155-159, 188-197], "accessibility": "high"},
        ...
    }
}
```

### Stem Epitopes (bnAb targets)

```python
HA_STEM_EPITOPES = {
    "group1_stem": {
        "positions": [18-20, 38-45, 318-321],
        "antibodies": ["CR6261", "F10", "FI6v3"],
        "conservation": "high",
    },
    "group2_stem": {
        "positions": [19-20, 38-45, 291-292],
        "antibodies": ["CR8020", "CR8043"],
        "conservation": "high",
    },
}
```

---

## Usage Examples

### Basic Analysis

```python
from berean.extensions import EnhancedAttentionAnalyzer, NumberingScheme

analyzer = EnhancedAttentionAnalyzer(
    numbering_scheme=NumberingScheme.IMGT,
    validate_with_iedb=True,
)

result = analyzer.analyze(
    heavy_chain="EVQLVESGGGLVQPGGSLRL...",
    light_chain="DIQMTQSPSSLSASVGDR...",
    ha_sequence="MKTIIALSYIFCLVFA...",
    subtype="H1N1",
)

# Access results
print(f"Top epitope positions: {result.epitope_positions[:5]}")
print(f"Dominant CDR: {result.get_dominant_cdr()}")
print(f"Confidence: {result.confidence:.2f}")

if result.validation:
    print(f"IEDB F1 Score: {result.validation.f1_score:.2f}")
```

### Interactive Visualization

```python
# Generate Plotly visualization
fig = analyzer.visualize_interactive(
    result,
    ha_sequence=ha_sequence,
    heavy_chain=heavy_chain,
    light_chain=light_chain,
    output_path="epitope_analysis.html",
)

# Or matplotlib fallback
fig = analyzer.visualize_matplotlib(
    result,
    ha_sequence,
    output_path="epitope_analysis.png",
)
```

### PyMOL Script Generation

```python
# Generate PyMOL script for 3D visualization
script = analyzer.generate_pymol_script(
    result,
    pdb_id="4HMG",
    ha_chain="A",
    output_path="epitope_visualization.pml",
)

# Run in PyMOL: @epitope_visualization.pml
```

### Benchmarking

```python
from berean.extensions import EpitopeBenchmark

benchmark = EpitopeBenchmark(analyzer)
results = benchmark.run_benchmark()

print(f"Mean F1: {results['overall']['mean_f1']:.2f}")
```

---

## Output Data Structures

### EnhancedEpitopePrediction

```python
@dataclass
class EnhancedEpitopePrediction:
    ha_attention_scores: np.ndarray      # Per-residue HA attention
    antibody_attention_scores: np.ndarray # Per-residue Ab attention
    epitope_positions: List[int]         # Top HA positions
    epitope_residues: List[Dict]         # Position, residue, score
    paratope_positions: List[int]        # Top Ab positions
    cdr_contacts: Dict[str, float]       # CDR contribution scores
    cross_attention_matrix: np.ndarray   # Full cross-attention
    confidence: float                    # Overall confidence
    cdr_annotations: Dict[str, CDRAnnotation]  # CDR boundaries
    validation: EpitopeValidation        # IEDB validation metrics
    antigenic_site_mapping: Dict[str, float]   # Site contributions
```

### CDRAnnotation

```python
@dataclass
class CDRAnnotation:
    chain_type: str              # 'H' or 'L'
    numbering_scheme: NumberingScheme
    cdr1: Tuple[int, int, str]   # (start, end, sequence)
    cdr2: Tuple[int, int, str]
    cdr3: Tuple[int, int, str]
    framework_regions: Dict[str, Tuple]
    full_numbering: Dict[int, str]
```

### EpitopeValidation

```python
@dataclass
class EpitopeValidation:
    source: str                  # "IEDB"
    matched_epitopes: List[Dict] # Matched known epitopes
    precision: float             # True positives / predicted
    recall: float                # True positives / actual
    f1_score: float              # Harmonic mean
    distance_to_known: float     # Avg distance to nearest known
```

---

## Improvements Over v1

| Feature | v1 | v2 |
|---------|----|----|
| CDR Detection | Hardcoded positions | IMGT/Kabat/Chothia schemes |
| CDR Annotation | None | Full ANARCI integration |
| Epitope Validation | None | IEDB database |
| Benchmarking | None | Crystal structure comparison |
| Visualization | matplotlib only | Plotly interactive + matplotlib |
| 3D Visualization | Basic | PyMOL script generation |
| Antigenic Sites | Basic | H1/H3 + stem epitopes |
| Confidence | Entropy only | Multi-factor scoring |

---

## Dependencies

**Required:**
- numpy

**Optional:**
- anarci (for accurate CDR numbering)
- plotly (for interactive visualization)
- matplotlib (fallback visualization)
- biopython (for structure mapping)

---

## File Statistics

- **Total Lines:** 1,650
- **Classes:** 9
- **Dataclasses:** 3
- **Methods:** ~50

---

*BEREAN Protocol v2.0 - Enhanced Attention Analyzer*
