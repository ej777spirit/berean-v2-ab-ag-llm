================================================================================
BEREAN Protocol v2.0 - Module Provenance Report
================================================================================

SUMMARY
----------------------------------------
  OFFICIAL: 5 modules
  ORIGINAL: 6 modules
  NEW: 6 modules
  REMOVED (replaced): 2 modules


================================================================================
OFFICIAL MODULES
================================================================================
Source: biomed-multi-alignment (IBM Research)
Install: pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]

Module: mammal.model.Mammal
  Status: OFFICIAL
  Source: biomed-multi-alignment (IBM)
  Description: Foundation model with 458M parameters for multi-task biomedical prediction

Module: mammal.keys
  Status: OFFICIAL
  Source: biomed-multi-alignment (IBM)
  Description: Standard key definitions for sample dictionaries (ENCODER_INPUTS_STR, etc.)

Module: fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp
  Status: OFFICIAL
  Source: biomed-multi-alignment / fuse-med-ml (IBM)
  Description: Multi-modal tokenizer supporting AA sequences, SMILES, scalars

Module: mammal.main_finetune
  Status: OFFICIAL
  Source: biomed-multi-alignment (IBM)
  Description: Official training infrastructure with Hydra configuration

Module: mammal.examples.*
  Status: OFFICIAL
  Source: biomed-multi-alignment (IBM)
  Description: Example tasks: protein_solubility, carcinogenicity, dti_bindingdb_kd


================================================================================
ORIGINAL MODULES
================================================================================
Source: Phase 1 build (kept without modification)

Module: berean.data.dataset
  Status: ORIGINAL
  Source: berean_phase1
  Description: AntibodyHAPair, VariantRecord dataclasses; dataset loading
  File: data/dataset.py
  Lines: 590

Module: berean.data.splits
  Status: ORIGINAL
  Source: berean_phase1
  Description: 4 split strategies from Cleveland Clinic paper (lenient, HA-exclusive, mAb-exclusive, mAb-cluster-exclusive)
  File: data/splits.py
  Lines: 557

Module: berean.data.preprocessing
  Status: ORIGINAL
  Source: berean_phase1
  Description: Sequence validation, normalization, Fv numbering preparation
  File: data/preprocessing.py
  Lines: 485

Module: berean.data.loaders
  Status: ORIGINAL
  Source: berean_phase1
  Description: PyTorch DataLoader utilities, collate functions
  File: data/loaders.py
  Lines: 273

Module: berean.evaluation.metrics
  Status: ORIGINAL
  Source: berean_phase1
  Description: AUROC, PR-AUC, MCC, F1 with bootstrap confidence intervals
  File: evaluation/metrics.py
  Lines: 534

Module: berean.core.embeddings
  Status: ORIGINAL
  Source: berean_phase1
  Description: Embedding extraction for ESM-2 (kept for ensemble approaches)
  File: core/embeddings.py
  Lines: 427


================================================================================
NEW MODULES
================================================================================
Source: v2.0 refactor (added to integrate with official MAMMAL)

Module: berean.core.mammal_adapter
  Status: NEW
  Source: berean_v2_refactor
  Description: High-level adapter wrapping official MAMMAL API; creates prompts, handles inference
  File: core/mammal_adapter.py
  Lines: 734

Module: berean.tasks.antibody_ha_binding
  Status: NEW
  Source: berean_v2_refactor
  Description: Custom MAMMAL task for Ab-HA binding classification; follows official task pattern
  File: tasks/antibody_ha_binding/__init__.py
  Lines: 509

Module: berean.tasks.hai_prediction
  Status: NEW
  Source: berean_v2_refactor
  Description: MAMMAL task for HAI titer regression; log-transform and normalization
  File: tasks/hai_prediction/__init__.py
  Lines: 471

Module: berean.extensions.esm2_ensemble
  Status: NEW
  Source: berean_v2_refactor
  Description: ESM-2 + MAMMAL ensemble for improved generalization on novel antibodies
  File: extensions/esm2_ensemble.py
  Lines: 545

Module: berean.extensions.attention_analysis
  Status: NEW
  Source: berean_v2_refactor
  Description: Attention weight analysis for epitope/paratope prediction
  File: extensions/attention_analysis.py
  Lines: 457

Module: berean.surveillance.gisaid
  Status: NEW
  Source: berean_v2_refactor
  Description: GISAID influenza integration for real-time variant surveillance
  File: surveillance/gisaid.py
  Lines: 451


================================================================================
REMOVED MODULES
================================================================================
Status: Replaced by official biomed-multi-alignment components

Module: berean.core.mammal_wrapper (old)
  Status: REMOVED
  Source: berean_phase1
  Description: Custom MAMMAL wrapper - REPLACED by official API via mammal_adapter

Module: berean.core.tokenizer (old)
  Status: REMOVED
  Source: berean_phase1
  Description: Custom tokenizer - REPLACED by official ModularTokenizerOp


================================================================================
ARCHITECTURE OVERVIEW
================================================================================

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
    │  │                    NEW (berean v2.0)                            │   │
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
    │  │                    ORIGINAL (Phase 1)                           │   │
    │  │  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
    │  │  │   Dataset   │  │     Splits       │  │    Metrics       │   │   │
    │  │  │  Classes    │  │  (4 strategies)  │  │  (AUROC, etc.)   │   │   │
    │  │  └─────────────┘  └──────────────────┘  └──────────────────┘   │   │
    │  │  ┌─────────────┐  ┌──────────────────┐                         │   │
    │  │  │Preprocessing│  │    Loaders       │                         │   │
    │  │  └─────────────┘  └──────────────────┘                         │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘


================================================================================
QUICK START EXAMPLE
================================================================================

    # Install dependencies
    pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]
    
    # Use BEREAN with official MAMMAL
    from berean.core import MAMMALAdapter
    from berean.tasks import AntibodyHABindingTask
    from berean.data import DataSplitter, SplitStrategy
    from berean.evaluation import compute_auroc
    
    # Initialize adapter (wraps OFFICIAL MAMMAL)
    adapter = MAMMALAdapter()
    
    # Create task (NEW, uses OFFICIAL prompt format)
    task = AntibodyHABindingTask()
    
    # Preprocess sample
    sample = task.data_preprocessing({
        "heavy_chain": "EVQLVESGG...",
        "light_chain": "DIQMTQSPS...",
        "ha_sequence": "MKTIIALSYI...",
        "binding_label": 1
    })
    
    # Predict
    result = adapter.predict_binding(
        heavy_chain="EVQLVESGG...",
        light_chain="DIQMTQSPS...",
        ha_sequence="MKTIIALSYI..."
    )
    print(f"Binding probability: {result['probability']:.3f}")
