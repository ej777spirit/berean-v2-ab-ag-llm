"""
ESM-2 Ensemble Extension
========================

Status: NEW
Source: berean_v2_refactor

Combines MAMMAL predictions with ESM-2 embeddings for improved
generalization on novel antibody sequences.

Rationale:
    The Cleveland Clinic paper shows AUROC drops from 0.91 (lenient) to
    0.73 (mAb-exclusive) when predicting on unseen antibodies. This
    extension aims to improve generalization by combining:
    
    1. MAMMAL: Task-specific pre-training, prompt-based prediction
    2. ESM-2: Large-scale protein language model with broader coverage
    
    The ensemble approach may capture complementary information.

Usage:
    >>> from berean.extensions import ESM2Ensemble
    >>> ensemble = ESM2Ensemble()
    >>> prediction = ensemble.predict(
    ...     heavy_chain="EVQLVESGG...",
    ...     light_chain="DIQMTQSPS...",
    ...     ha_sequence="MKTIIALSYI..."
    ... )
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ESM2Config:
    """Configuration for ESM-2 model.
    
    Attributes:
        model_name: HuggingFace model identifier
        pooling: Pooling strategy
        layer: Which layer to extract from
        device: Compute device
    """
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    pooling: str = "mean"  # "mean", "cls", "max"
    layer: int = -1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ESM2Embeddings:
    """Extract embeddings from Meta ESM-2 model.
    
    Status: NEW
    
    Provides ESM-2 embeddings to complement MAMMAL predictions.
    ESM-2 was trained on >250M protein sequences and may provide
    better coverage for novel antibody sequences.
    """
    
    EMBEDDING_DIM = {
        "facebook/esm2_t33_650M_UR50D": 1280,
        "facebook/esm2_t36_3B_UR50D": 2560,
        "facebook/esm2_t30_150M_UR50D": 640,
    }
    
    def __init__(self, config: Optional[ESM2Config] = None):
        """Initialize ESM-2 embedding extractor.
        
        Args:
            config: ESM-2 configuration
        """
        self.config = config or ESM2Config()
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        self.embedding_dim = self.EMBEDDING_DIM.get(
            self.config.model_name, 1280
        )
    
    def load(self) -> None:
        """Load ESM-2 model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading ESM-2: {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )
            self.model = AutoModel.from_pretrained(
                self.config.model_name
            )
            self.model.to(self.config.device)
            self.model.eval()
            
            self._loaded = True
            logger.info("ESM-2 loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ESM-2: {e}")
            raise
    
    def extract(
        self,
        sequence: str,
        max_length: int = 1024,
    ) -> torch.Tensor:
        """Extract embedding for a single sequence.
        
        Args:
            sequence: Amino acid sequence
            max_length: Maximum sequence length
            
        Returns:
            Embedding tensor (embedding_dim,)
        """
        if not self._loaded:
            self.load()
        
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Extract hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.config.layer]
        
        # Pool
        if self.config.pooling == "cls":
            embedding = hidden_states[:, 0, :]
        elif self.config.pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            sum_hidden = (hidden_states * mask).sum(dim=1)
            embedding = sum_hidden / mask.sum(dim=1)
        elif self.config.pooling == "max":
            embedding = hidden_states.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")
        
        return embedding.squeeze(0)
    
    def extract_batch(
        self,
        sequences: List[str],
        max_length: int = 1024,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Extract embeddings for multiple sequences.
        
        Args:
            sequences: List of amino acid sequences
            max_length: Maximum sequence length
            batch_size: Batch size
            
        Returns:
            Embeddings tensor (num_sequences, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.config.layer]
            
            # Pool
            if self.config.pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1)
                sum_hidden = (hidden_states * mask).sum(dim=1)
                embeddings = sum_hidden / mask.sum(dim=1).clamp(min=1e-9)
            else:
                embeddings = hidden_states[:, 0, :]
            
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)


class FusionHead(nn.Module):
    """Fusion head for combining MAMMAL and ESM-2 features.
    
    Status: NEW
    
    Simple MLP that takes concatenated embeddings from both models
    and produces binding predictions.
    """
    
    def __init__(
        self,
        mammal_dim: int = 768,
        esm2_dim: int = 1280,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize fusion head.
        
        Args:
            mammal_dim: MAMMAL embedding dimension
            esm2_dim: ESM-2 embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        input_dim = mammal_dim + esm2_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        mammal_embedding: torch.Tensor,
        esm2_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            mammal_embedding: MAMMAL embedding (batch, mammal_dim)
            esm2_embedding: ESM-2 embedding (batch, esm2_dim)
            
        Returns:
            Logits (batch, num_classes)
        """
        combined = torch.cat([mammal_embedding, esm2_embedding], dim=-1)
        return self.fusion(combined)


class ESM2Ensemble:
    """Ensemble predictor combining MAMMAL and ESM-2.
    
    Status: NEW
    
    This extension combines:
    1. MAMMAL predictions via official API
    2. ESM-2 embeddings for additional sequence features
    3. Trainable fusion head
    
    The goal is to improve generalization on novel antibodies
    beyond what MAMMAL alone achieves.
    
    Example:
        >>> ensemble = ESM2Ensemble()
        >>> ensemble.load()
        >>> 
        >>> result = ensemble.predict(
        ...     heavy_chain="EVQLVESGG...",
        ...     light_chain="DIQMTQSPS...",
        ...     ha_sequence="MKTIIALSYI..."
        ... )
        >>> print(f"Binding probability: {result['probability']:.3f}")
    """
    
    def __init__(
        self,
        mammal_model: str = "ibm/biomed.omics.bl.sm.ma-ted-458m",
        esm2_model: str = "facebook/esm2_t33_650M_UR50D",
        fusion_checkpoint: Optional[str] = None,
        device: str = "auto",
    ):
        """Initialize ensemble.
        
        Args:
            mammal_model: MAMMAL model identifier
            esm2_model: ESM-2 model identifier
            fusion_checkpoint: Path to trained fusion head
            device: Compute device
        """
        self.mammal_model_name = mammal_model
        self.esm2_model_name = esm2_model
        self.fusion_checkpoint = fusion_checkpoint
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.mammal_adapter = None
        self.esm2_embeddings = None
        self.fusion_head = None
        
        self._loaded = False
    
    def load(self) -> None:
        """Load all models."""
        logger.info("Loading ensemble models...")
        
        # Load MAMMAL via adapter (uses OFFICIAL API)
        from berean.core import MAMMALAdapter, MAMMALConfig
        
        config = MAMMALConfig(
            model_name=self.mammal_model_name,
            device=self.device,
        )
        self.mammal_adapter = MAMMALAdapter(config)
        
        # Load ESM-2
        esm2_config = ESM2Config(
            model_name=self.esm2_model_name,
            device=self.device,
        )
        self.esm2_embeddings = ESM2Embeddings(esm2_config)
        self.esm2_embeddings.load()
        
        # Initialize or load fusion head
        self.fusion_head = FusionHead(
            mammal_dim=768,
            esm2_dim=self.esm2_embeddings.embedding_dim,
        )
        
        if self.fusion_checkpoint:
            state_dict = torch.load(self.fusion_checkpoint, map_location=self.device)
            self.fusion_head.load_state_dict(state_dict)
            logger.info(f"Loaded fusion head from {self.fusion_checkpoint}")
        
        self.fusion_head.to(self.device)
        self.fusion_head.eval()
        
        self._loaded = True
        logger.info("Ensemble loaded successfully")
    
    def get_mammal_embedding(
        self,
        heavy_chain: str,
        light_chain: str,
        ha_sequence: str,
    ) -> torch.Tensor:
        """Get MAMMAL embedding for antibody-HA pair.
        
        Uses the MAMMALAdapter which wraps the OFFICIAL API.
        """
        result = self.mammal_adapter.get_embeddings(
            heavy_chain, light_chain, ha_sequence
        )
        return result["pooled"]
    
    def get_esm2_embedding(
        self,
        heavy_chain: str,
        light_chain: str,
        ha_sequence: str,
    ) -> torch.Tensor:
        """Get ESM-2 embedding for antibody-HA pair.
        
        Extracts embeddings for antibody and HA separately,
        then concatenates or averages.
        """
        # Concatenate antibody chains
        antibody_seq = heavy_chain + light_chain
        
        # Extract embeddings
        ab_emb = self.esm2_embeddings.extract(antibody_seq)
        ha_emb = self.esm2_embeddings.extract(ha_sequence)
        
        # Combine (average or concatenate)
        # Using average to maintain dimension
        combined = (ab_emb + ha_emb) / 2
        
        return combined
    
    def predict(
        self,
        heavy_chain: str,
        light_chain: str,
        ha_sequence: str,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Predict binding using ensemble.
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            ha_sequence: HA sequence
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary with prediction results
        """
        if not self._loaded:
            self.load()
        
        # Get embeddings from both models
        mammal_emb = self.get_mammal_embedding(
            heavy_chain, light_chain, ha_sequence
        )
        esm2_emb = self.get_esm2_embedding(
            heavy_chain, light_chain, ha_sequence
        )
        
        # Ensure correct shape (batch dimension)
        if mammal_emb.dim() == 1:
            mammal_emb = mammal_emb.unsqueeze(0)
        if esm2_emb.dim() == 1:
            esm2_emb = esm2_emb.unsqueeze(0)
        
        # Fusion prediction
        with torch.no_grad():
            logits = self.fusion_head(mammal_emb, esm2_emb)
            probs = torch.softmax(logits, dim=-1)
        
        result = {
            "probability": probs[0, 1].item(),
            "prediction": int(probs[0, 1] > 0.5),
            "logits": logits.cpu().numpy(),
        }
        
        if return_embeddings:
            result["mammal_embedding"] = mammal_emb.cpu().numpy()
            result["esm2_embedding"] = esm2_emb.cpu().numpy()
        
        return result
    
    def predict_batch(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Batch prediction.
        
        Args:
            samples: List of dicts with heavy_chain, light_chain, ha_sequence
            batch_size: Batch size
            
        Returns:
            List of prediction results
        """
        results = []
        
        for sample in samples:
            result = self.predict(
                sample["heavy_chain"],
                sample["light_chain"],
                sample["ha_sequence"],
            )
            results.append(result)
        
        return results
    
    def save_fusion_head(self, path: str) -> None:
        """Save fusion head checkpoint.
        
        Args:
            path: Output path
        """
        torch.save(self.fusion_head.state_dict(), path)
        logger.info(f"Saved fusion head to {path}")


class EnsemblePredictor:
    """Simplified ensemble predictor using averaging.
    
    Status: NEW
    
    Simpler approach that just averages MAMMAL and ESM-2 based
    predictions without a trainable fusion layer.
    """
    
    def __init__(
        self,
        mammal_weight: float = 0.7,
        esm2_weight: float = 0.3,
    ):
        """Initialize predictor.
        
        Args:
            mammal_weight: Weight for MAMMAL predictions
            esm2_weight: Weight for ESM-2 based predictions
        """
        self.mammal_weight = mammal_weight
        self.esm2_weight = esm2_weight
        
        # Normalize weights
        total = mammal_weight + esm2_weight
        self.mammal_weight /= total
        self.esm2_weight /= total
    
    def ensemble_probability(
        self,
        mammal_prob: float,
        esm2_prob: float,
    ) -> float:
        """Compute ensemble probability.
        
        Args:
            mammal_prob: MAMMAL binding probability
            esm2_prob: ESM-2 based binding probability
            
        Returns:
            Ensemble probability
        """
        return (
            self.mammal_weight * mammal_prob +
            self.esm2_weight * esm2_prob
        )


def example_usage():
    """Demonstrate ESM-2 ensemble."""
    print("=" * 60)
    print("ESM-2 Ensemble Extension")
    print("=" * 60)
    print("\nStatus: NEW")
    print("Purpose: Improve generalization on novel antibodies")
    print("\nComponents:")
    print("  - MAMMAL (OFFICIAL): Task-specific predictions")
    print("  - ESM-2 (NEW): Additional sequence features")
    print("  - FusionHead (NEW): Trainable combination layer")
    print("\nTarget: Beat 0.73 AUROC on mAb-exclusive split")


if __name__ == "__main__":
    example_usage()
