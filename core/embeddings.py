"""
Embedding Extraction Module
===========================

Status: ORIGINAL (kept from Phase 1 build)
Source: berean_phase1

Provides unified interface for extracting embeddings from:
- IBM MAMMAL (primary) - via MAMMALAdapter (NEW)
- Meta ESM-2 (for ensemble approaches)
- Combined/concatenated embeddings

Extension beyond paper: ESM-2 integration for improved generalization
on novel antibodies (addressing the 0.63-0.73 AUROC gap).

Note: For MAMMAL embeddings, prefer using the MAMMALAdapter.get_embeddings()
method from core/mammal_adapter.py (NEW), which properly uses the OFFICIAL
biomed-multi-alignment API.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple
from enum import Enum
import logging

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Supported embedding models."""
    MAMMAL = "mammal"      # IBM MAMMAL (ibm/biomed.omics.bl.sm.ma-ted-458m)
    ESM2 = "esm2"          # Meta ESM-2 (facebook/esm2_t33_650M_UR50D)
    ESM2_LARGE = "esm2_lg" # Meta ESM-2 large (facebook/esm2_t36_3B_UR50D)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction.
    
    Attributes:
        model_type: Which embedding model to use
        model_name: HuggingFace model identifier (optional override)
        pooling: Pooling strategy ('cls', 'mean', 'max')
        layer: Which layer to extract from (-1 for last, -2 for second-to-last)
        normalize: Whether to L2-normalize embeddings
        device: Compute device
    """
    model_type: EmbeddingModel = EmbeddingModel.MAMMAL
    model_name: Optional[str] = None
    pooling: str = "cls"  # 'cls', 'mean', 'max'
    layer: int = -1
    normalize: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        # Set default model names
        if self.model_name is None:
            model_names = {
                EmbeddingModel.MAMMAL: "ibm/biomed.omics.bl.sm.ma-ted-458m",
                EmbeddingModel.ESM2: "facebook/esm2_t33_650M_UR50D",
                EmbeddingModel.ESM2_LARGE: "facebook/esm2_t36_3B_UR50D"
            }
            self.model_name = model_names.get(self.model_type)


class EmbeddingExtractor:
    """Extract sequence embeddings from pre-trained protein language models.
    
    Supports multiple backends (MAMMAL, ESM-2) with consistent interface.
    """
    
    # Model embedding dimensions
    EMBEDDING_DIMS = {
        EmbeddingModel.MAMMAL: 768,
        EmbeddingModel.ESM2: 1280,
        EmbeddingModel.ESM2_LARGE: 2560
    }
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        self.embedding_dim = self.EMBEDDING_DIMS.get(config.model_type, 768)
        
    def load(self) -> None:
        """Load the embedding model and tokenizer."""
        try:
            if self.config.model_type == EmbeddingModel.MAMMAL:
                self._load_mammal()
            elif self.config.model_type in [EmbeddingModel.ESM2, EmbeddingModel.ESM2_LARGE]:
                self._load_esm2()
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
                
            self._loaded = True
            logger.info(f"Loaded {self.config.model_type.value} embeddings model")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
    def _load_mammal(self) -> None:
        """Load IBM MAMMAL model."""
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        
    def _load_esm2(self) -> None:
        """Load Meta ESM-2 model."""
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply pooling strategy to sequence embeddings.
        
        Args:
            hidden_states: Model output (batch, seq_len, hidden_dim)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Pooled embeddings (batch, hidden_dim)
        """
        if self.config.pooling == "cls":
            # Use [CLS] token (first position)
            return hidden_states[:, 0, :]
            
        elif self.config.pooling == "mean":
            # Mean pooling over non-padding tokens
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
            
        elif self.config.pooling == "max":
            # Max pooling over non-padding tokens
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states[~mask.bool()] = float('-inf')
            return torch.max(hidden_states, dim=1).values
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")
            
    def extract(
        self,
        sequences: Union[str, List[str]],
        max_length: int = 512,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Extract embeddings for one or more sequences.
        
        Args:
            sequences: Single sequence or list of sequences
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            Embeddings tensor of shape (num_sequences, embedding_dim)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Handle single sequence
        if isinstance(sequences, str):
            sequences = [sequences]
            
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                max_length=max_length,
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
            # Get specified layer
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[self.config.layer]
            else:
                hidden_states = outputs.last_hidden_state
                
            # Pool
            embeddings = self._pool_embeddings(
                hidden_states, 
                inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
            )
            
            # Normalize if requested
            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                
            all_embeddings.append(embeddings)
            
        return torch.cat(all_embeddings, dim=0)
        
    def extract_with_attention(
        self,
        sequence: str,
        max_length: int = 512
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings along with attention weights.
        
        Useful for epitope/paratope mapping analysis.
        
        Args:
            sequence: Single sequence
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (embedding, list of attention matrices per layer)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
            
        # Get embedding
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[self.config.layer]
        else:
            hidden_states = outputs.last_hidden_state
            
        embedding = self._pool_embeddings(
            hidden_states,
            inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        )
        
        # Get attention weights
        attentions = list(outputs.attentions) if outputs.attentions else []
        
        return embedding, attentions


class CombinedEmbeddings(nn.Module):
    """Combine embeddings from multiple models for ensemble prediction.
    
    This is an EXTENSION beyond the paper to improve generalization.
    
    Strategies:
    - Concatenation: [MAMMAL; ESM2] 
    - Weighted sum: w1*MAMMAL + w2*ESM2
    - Attention fusion: Learn to combine based on sequence
    - Projection: Project both to shared space then combine
    """
    
    def __init__(
        self,
        extractors: List[EmbeddingExtractor],
        fusion_strategy: str = "concat",
        output_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.extractors = extractors
        self.fusion_strategy = fusion_strategy
        
        # Calculate total embedding dimension
        total_dim = sum(e.embedding_dim for e in extractors)
        
        if fusion_strategy == "concat":
            self.output_dim = total_dim
            self.fusion_layer = nn.Identity()
            
        elif fusion_strategy == "weighted":
            self.output_dim = extractors[0].embedding_dim  # Assume same dim
            weights = torch.ones(len(extractors)) / len(extractors)
            self.weights = nn.Parameter(weights)
            
        elif fusion_strategy == "project":
            self.output_dim = output_dim or 512
            self.projections = nn.ModuleList([
                nn.Linear(e.embedding_dim, self.output_dim)
                for e in extractors
            ])
            
        elif fusion_strategy == "attention":
            self.output_dim = output_dim or 512
            # Project all to same dimension
            self.projections = nn.ModuleList([
                nn.Linear(e.embedding_dim, self.output_dim)
                for e in extractors
            ])
            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 4),
                nn.ReLU(),
                nn.Linear(self.output_dim // 4, 1)
            )
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
    def forward(
        self,
        sequences: List[str],
        max_length: int = 512
    ) -> torch.Tensor:
        """Extract and combine embeddings from all models.
        
        Args:
            sequences: List of sequences
            max_length: Maximum sequence length
            
        Returns:
            Combined embeddings (batch, output_dim)
        """
        # Extract from all models
        embeddings = [
            extractor.extract(sequences, max_length=max_length)
            for extractor in self.extractors
        ]
        
        # Fuse based on strategy
        if self.fusion_strategy == "concat":
            return torch.cat(embeddings, dim=-1)
            
        elif self.fusion_strategy == "weighted":
            weights = torch.softmax(self.weights, dim=0)
            combined = sum(w * e for w, e in zip(weights, embeddings))
            return combined
            
        elif self.fusion_strategy == "project":
            projected = [proj(e) for proj, e in zip(self.projections, embeddings)]
            return sum(projected) / len(projected)
            
        elif self.fusion_strategy == "attention":
            # Project all to same dimension
            projected = [proj(e) for proj, e in zip(self.projections, embeddings)]
            stacked = torch.stack(projected, dim=1)  # (batch, num_models, dim)
            
            # Compute attention scores
            scores = self.attention(stacked).squeeze(-1)  # (batch, num_models)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, num_models, 1)
            
            # Weighted combination
            combined = (stacked * weights).sum(dim=1)  # (batch, dim)
            return combined


def create_mammal_extractor(
    pooling: str = "cls",
    device: str = "cuda"
) -> EmbeddingExtractor:
    """Create MAMMAL embedding extractor."""
    config = EmbeddingConfig(
        model_type=EmbeddingModel.MAMMAL,
        pooling=pooling,
        device=device
    )
    extractor = EmbeddingExtractor(config)
    extractor.load()
    return extractor


def create_esm2_extractor(
    pooling: str = "mean",
    large: bool = False,
    device: str = "cuda"
) -> EmbeddingExtractor:
    """Create ESM-2 embedding extractor."""
    config = EmbeddingConfig(
        model_type=EmbeddingModel.ESM2_LARGE if large else EmbeddingModel.ESM2,
        pooling=pooling,
        device=device
    )
    extractor = EmbeddingExtractor(config)
    extractor.load()
    return extractor


def create_ensemble_extractor(
    include_esm2: bool = True,
    fusion: str = "concat",
    device: str = "cuda"
) -> CombinedEmbeddings:
    """Create ensemble embedding extractor (MAMMAL + ESM-2).
    
    This combination aims to improve generalization on novel antibodies
    beyond the 0.63-0.73 AUROC achieved with MAMMAL alone.
    """
    extractors = [create_mammal_extractor(device=device)]
    
    if include_esm2:
        extractors.append(create_esm2_extractor(device=device))
        
    return CombinedEmbeddings(extractors, fusion_strategy=fusion)
