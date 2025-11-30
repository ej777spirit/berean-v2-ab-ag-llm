#!/usr/bin/env python3
"""
BEREAN Protocol v2.0 - Module Provenance Report
================================================

This script generates a comprehensive report of all modules,
showing their status (OFFICIAL, ORIGINAL, NEW) and source.

Run:
    python -m berean.module_report
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    status: str  # OFFICIAL, ORIGINAL, NEW
    source: str
    description: str
    file_path: str = ""
    lines: int = 0


# =============================================================================
# MODULE DEFINITIONS
# =============================================================================

MODULES = {
    # OFFICIAL: From biomed-multi-alignment (external dependency)
    "OFFICIAL": [
        ModuleInfo(
            name="mammal.model.Mammal",
            status="OFFICIAL",
            source="biomed-multi-alignment (IBM)",
            description="Foundation model with 458M parameters for multi-task biomedical prediction",
        ),
        ModuleInfo(
            name="mammal.keys",
            status="OFFICIAL", 
            source="biomed-multi-alignment (IBM)",
            description="Standard key definitions for sample dictionaries (ENCODER_INPUTS_STR, etc.)",
        ),
        ModuleInfo(
            name="fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp",
            status="OFFICIAL",
            source="biomed-multi-alignment / fuse-med-ml (IBM)",
            description="Multi-modal tokenizer supporting AA sequences, SMILES, scalars",
        ),
        ModuleInfo(
            name="mammal.main_finetune",
            status="OFFICIAL",
            source="biomed-multi-alignment (IBM)",
            description="Official training infrastructure with Hydra configuration",
        ),
        ModuleInfo(
            name="mammal.examples.*",
            status="OFFICIAL",
            source="biomed-multi-alignment (IBM)",
            description="Example tasks: protein_solubility, carcinogenicity, dti_bindingdb_kd",
        ),
    ],
    
    # ORIGINAL: Kept from Phase 1 build
    "ORIGINAL": [
        ModuleInfo(
            name="berean.data.dataset",
            status="ORIGINAL",
            source="berean_phase1",
            description="AntibodyHAPair, VariantRecord dataclasses; dataset loading",
            file_path="data/dataset.py",
        ),
        ModuleInfo(
            name="berean.data.splits",
            status="ORIGINAL",
            source="berean_phase1",
            description="4 split strategies from Cleveland Clinic paper (lenient, HA-exclusive, mAb-exclusive, mAb-cluster-exclusive)",
            file_path="data/splits.py",
        ),
        ModuleInfo(
            name="berean.data.preprocessing",
            status="ORIGINAL",
            source="berean_phase1",
            description="Sequence validation, normalization, Fv numbering preparation",
            file_path="data/preprocessing.py",
        ),
        ModuleInfo(
            name="berean.data.loaders",
            status="ORIGINAL",
            source="berean_phase1",
            description="PyTorch DataLoader utilities, collate functions",
            file_path="data/loaders.py",
        ),
        ModuleInfo(
            name="berean.evaluation.metrics",
            status="ORIGINAL",
            source="berean_phase1",
            description="AUROC, PR-AUC, MCC, F1 with bootstrap confidence intervals",
            file_path="evaluation/metrics.py",
        ),
        ModuleInfo(
            name="berean.core.embeddings",
            status="ORIGINAL",
            source="berean_phase1",
            description="Embedding extraction for ESM-2 (kept for ensemble approaches)",
            file_path="core/embeddings.py",
        ),
    ],
    
    # NEW: Added in v2.0 refactor
    "NEW": [
        ModuleInfo(
            name="berean.core.mammal_adapter",
            status="NEW",
            source="berean_v2_refactor",
            description="High-level adapter wrapping official MAMMAL API; creates prompts, handles inference",
            file_path="core/mammal_adapter.py",
        ),
        ModuleInfo(
            name="berean.tasks.antibody_ha_binding",
            status="NEW",
            source="berean_v2_refactor",
            description="Custom MAMMAL task for Ab-HA binding classification; follows official task pattern",
            file_path="tasks/antibody_ha_binding/__init__.py",
        ),
        ModuleInfo(
            name="berean.tasks.hai_prediction",
            status="NEW",
            source="berean_v2_refactor",
            description="MAMMAL task for HAI titer regression; log-transform and normalization",
            file_path="tasks/hai_prediction/__init__.py",
        ),
        ModuleInfo(
            name="berean.extensions.esm2_ensemble",
            status="NEW",
            source="berean_v2_refactor",
            description="ESM-2 + MAMMAL ensemble for improved generalization on novel antibodies",
            file_path="extensions/esm2_ensemble.py",
        ),
        ModuleInfo(
            name="berean.extensions.attention_analysis",
            status="NEW",
            source="berean_v2_refactor",
            description="Attention weight analysis for epitope/paratope prediction",
            file_path="extensions/attention_analysis.py",
        ),
        ModuleInfo(
            name="berean.surveillance.gisaid",
            status="NEW",
            source="berean_v2_refactor",
            description="GISAID influenza integration for real-time variant surveillance",
            file_path="surveillance/gisaid.py",
        ),
    ],
    
    # REMOVED: Replaced by official MAMMAL
    "REMOVED": [
        ModuleInfo(
            name="berean.core.mammal_wrapper (old)",
            status="REMOVED",
            source="berean_phase1",
            description="Custom MAMMAL wrapper - REPLACED by official API via mammal_adapter",
        ),
        ModuleInfo(
            name="berean.core.tokenizer (old)",
            status="REMOVED",
            source="berean_phase1",
            description="Custom tokenizer - REPLACED by official ModularTokenizerOp",
        ),
    ],
}


def count_lines(file_path: str, base_dir: str) -> int:
    """Count lines in a file."""
    full_path = Path(base_dir) / file_path
    if full_path.exists():
        with open(full_path, 'r') as f:
            return sum(1 for _ in f)
    return 0


def generate_report(base_dir: str = ".") -> str:
    """Generate comprehensive module report."""
    lines = []
    lines.append("=" * 80)
    lines.append("BEREAN Protocol v2.0 - Module Provenance Report")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    stats = {status: len(modules) for status, modules in MODULES.items()}
    lines.append("SUMMARY")
    lines.append("-" * 40)
    for status, count in stats.items():
        if status != "REMOVED":
            lines.append(f"  {status}: {count} modules")
    lines.append(f"  REMOVED (replaced): {stats.get('REMOVED', 0)} modules")
    lines.append("")
    
    # Detailed listing
    for status in ["OFFICIAL", "ORIGINAL", "NEW", "REMOVED"]:
        modules = MODULES.get(status, [])
        if not modules:
            continue
            
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"{status} MODULES")
        lines.append("=" * 80)
        
        if status == "OFFICIAL":
            lines.append("Source: biomed-multi-alignment (IBM Research)")
            lines.append("Install: pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]")
        elif status == "ORIGINAL":
            lines.append("Source: Phase 1 build (kept without modification)")
        elif status == "NEW":
            lines.append("Source: v2.0 refactor (added to integrate with official MAMMAL)")
        elif status == "REMOVED":
            lines.append("Status: Replaced by official biomed-multi-alignment components")
        
        lines.append("")
        
        for module in modules:
            lines.append(f"Module: {module.name}")
            lines.append(f"  Status: {module.status}")
            lines.append(f"  Source: {module.source}")
            lines.append(f"  Description: {module.description}")
            
            if module.file_path:
                loc = count_lines(module.file_path, base_dir)
                lines.append(f"  File: {module.file_path}")
                lines.append(f"  Lines: {loc}")
            
            lines.append("")
    
    # Architecture diagram
    lines.append("")
    lines.append("=" * 80)
    lines.append("ARCHITECTURE OVERVIEW")
    lines.append("=" * 80)
    lines.append("""
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
""")
    
    # Usage example
    lines.append("")
    lines.append("=" * 80)
    lines.append("QUICK START EXAMPLE")
    lines.append("=" * 80)
    lines.append("""
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
""")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    # Find base directory
    script_dir = Path(__file__).parent if "__file__" in dir() else Path(".")
    base_dir = script_dir
    
    report = generate_report(str(base_dir))
    print(report)
    
    # Also save to file
    output_path = base_dir / "MODULE_REPORT.md"
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
