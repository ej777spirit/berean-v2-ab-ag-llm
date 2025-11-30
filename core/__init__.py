"""
BEREAN Core Module
==================

This module provides the adapter layer between BEREAN and the official
biomed-multi-alignment (MAMMAL) package.

Status: NEW (adapter wrapping OFFICIAL components)

Official MAMMAL components used:
    - mammal.model.Mammal
    - mammal.keys (ENCODER_INPUTS_STR, etc.)
    - fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp
"""

from .mammal_adapter import (
    MAMMALAdapter,
    load_mammal_model,
    load_mammal_tokenizer,
    create_binding_prompt,
    create_regression_prompt,
)

from .embeddings import (
    EmbeddingExtractor,
    ESM2Embeddings,
)

__all__ = [
    "MAMMALAdapter",
    "load_mammal_model", 
    "load_mammal_tokenizer",
    "create_binding_prompt",
    "create_regression_prompt",
    "EmbeddingExtractor",
    "ESM2Embeddings",
]
