"""
BEREAN Integrated Predictor
============================

Status: NEW
Source: berean_v2_refactor

This module provides end-to-end integration between the MAMMAL adapter
and the prediction heads. It handles:
    1. Sequence tokenization via MAMMAL
    2. Embedding extraction from MAMMAL model
    3. Prediction via OmniSynapticBindingPredictor heads

Architecture:
    
    ┌────────────────────────────────────────────────────────────────┐
    │                    BEREANPredictor                             │
    │                                                                │
    │   Input: (antibody_seq, antigen_seq)                          │
    │              │                                                 │
    │              ▼                                                 │
    │   ┌─────────────────────┐                                     │
    │   │   MAMMALAdapter     │  ← Official biomed-multi-alignment   │
    │   │   (458M params)     │                                     │
    │   └──────────┬──────────┘                                     │
    │              │                                                 │
    │              ▼                                                 │
    │        [Embeddings]                                            │
    │              │                                                 │
    │              ▼                                                 │
    │   ┌─────────────────────────────────────────────┐             │
    │   │   OmniSynapticBindingPredictor              │             │
    │   │   ┌─────────┐ ┌─────────┐ ┌─────────────┐   │             │
    │   │   │ Binding │ │Affinity │ │Cross-React. │   │             │
    │   │   │  Head   │ │  Head   │ │    Head     │   │             │
    │   │   └─────────┘ └─────────┘ └─────────────┘   │             │
    │   └──────────┬──────────────────────────────────┘             │
    │              │                                                 │
    │              ▼                                                 │
    │   Output: UnifiedPrediction                                    │
    └────────────────────────────────────────────────────────────────┘

Usage:
    >>> from berean.integration import BEREANPredictor
    >>> 
    >>> predictor = BEREANPredictor.from_pretrained()
    >>> result = predictor.predict(
    ...     antibody_sequence="EVQLVESGGGLVQPGG...",
    ...     antigen_sequence="MKAIIVLLMVVTSNA..."
    ... )
    >>> print(result.summary())
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn

# Import MAMMAL adapter
try:
    from ..core.mammal_adapter import (
        MAMMALAdapter,
        load_mammal_model,
        load_mammal_tokenizer,
        check_mammal_installation,
    )
    MAMMAL_AVAILABLE = True
except ImportError:
    MAMMAL_AVAILABLE = False

# Import prediction heads
from .omni_synaptic_predictor import (
    OmniSynapticBindingPredictor,
    OmniSynapticConfig,
    UnifiedPrediction,
)
from .binding_head import BindingHeadConfig
from .affinity_head import AffinityHeadConfig, AffinityType
from .cross_reactivity_head import CrossReactivityHeadConfig, DEFAULT_STRAIN_PANEL

logger = logging.getLogger(__name__)


@dataclass
class BEREANConfig:
    """Configuration for BEREAN integrated predictor.
    
    Attributes:
        mammal_model_path: Path to MAMMAL model or HuggingFace ID
        embedding_dim: Dimension of MAMMAL embeddings
        enable_binding: Enable binding classification head
        enable_affinity: Enable affinity prediction head
        enable_cross_reactivity: Enable cross-reactivity head
        affinity_type: Type of affinity to predict
        cross_reactivity_strains: Strains for cross-reactivity assessment
        device: Target device
        use_fp16: Use half precision for inference
    """
    mammal_model_path: str = "ibm/biomed.omics.bl.sm.ma-ted-458m"
    embedding_dim: int = 768
    enable_binding: bool = True
    enable_affinity: bool = True
    enable_cross_reactivity: bool = True
    affinity_type: AffinityType = AffinityType.IC50
    cross_reactivity_strains: Optional[List[str]] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = False
    
    def __post_init__(self):
        if self.cross_reactivity_strains is None:
            self.cross_reactivity_strains = DEFAULT_STRAIN_PANEL


class BEREANPredictor(nn.Module):
    """
    BEREAN Integrated Predictor - End-to-End Antibody Analysis.
    
    Status: NEW
    
    This class integrates:
        1. MAMMAL adapter for embedding extraction
        2. OmniSynapticBindingPredictor for multi-head prediction
    
    Example:
        >>> predictor = BEREANPredictor.from_pretrained()
        >>> 
        >>> # Single prediction
        >>> result = predictor.predict(
        ...     antibody_sequence="EVQLVESGGGLVQPGG...",
        ...     antigen_sequence="MKAIIVLLMVVTSNA..."
        ... )
        >>> print(result.summary())
        >>>
        >>> # Batch prediction
        >>> results = predictor.predict_batch([
        ...     ("ab_seq_1", "ag_seq_1"),
        ...     ("ab_seq_2", "ag_seq_2"),
        ... ])
    """
    
    def __init__(
        self,
        config: Optional[BEREANConfig] = None,
        mammal_adapter: Optional["MAMMALAdapter"] = None,
    ):
        """Initialize BEREAN Predictor.
        
        Args:
            config: Configuration object
            mammal_adapter: Pre-initialized MAMMAL adapter (optional)
        """
        super().__init__()
        
        self.config = config or BEREANConfig()
        
        # Initialize MAMMAL adapter
        if mammal_adapter is not None:
            self.mammal_adapter = mammal_adapter
            logger.info("Using provided MAMMAL adapter")
        elif MAMMAL_AVAILABLE:
            self.mammal_adapter = MAMMALAdapter(
                model_path=self.config.mammal_model_path,
                device=self.config.device,
            )
            logger.info(f"Initialized MAMMAL adapter from {self.config.mammal_model_path}")
        else:
            self.mammal_adapter = None
            logger.warning("MAMMAL not available - using embedding-only mode")
        
        # Build OmniSynaptic config
        omni_config = OmniSynapticConfig(
            embedding_dim=self.config.embedding_dim,
            enable_binding_head=self.config.enable_binding,
            enable_affinity_head=self.config.enable_affinity,
            enable_cross_reactivity_head=self.config.enable_cross_reactivity,
            binding_config=BindingHeadConfig(input_dim=self.config.embedding_dim),
            affinity_config=AffinityHeadConfig(
                input_dim=self.config.embedding_dim,
                affinity_type=self.config.affinity_type,
            ),
            cross_reactivity_config=CrossReactivityHeadConfig(
                input_dim=self.config.embedding_dim,
                strains=self.config.cross_reactivity_strains,
            ),
            device=self.config.device,
        )
        
        # Initialize prediction heads
        self.predictor = OmniSynapticBindingPredictor(omni_config)
        
        # Half precision if requested
        if self.config.use_fp16 and self.config.device == "cuda":
            self.predictor = self.predictor.half()
        
        logger.info("BEREAN Predictor initialized")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "ibm/biomed.omics.bl.sm.ma-ted-458m",
        heads_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "BEREANPredictor":
        """Load BEREAN predictor from pretrained components.
        
        Args:
            model_path: Path to MAMMAL model
            heads_checkpoint: Path to prediction heads checkpoint
            device: Target device
            **kwargs: Additional config arguments
            
        Returns:
            Initialized BEREANPredictor
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        config = BEREANConfig(
            mammal_model_path=model_path,
            device=device,
            **kwargs,
        )
        
        predictor = cls(config)
        
        # Load heads checkpoint if provided
        if heads_checkpoint is not None:
            predictor.load_heads(heads_checkpoint)
        
        return predictor
    
    def get_embeddings(
        self,
        antibody_sequence: str,
        antigen_sequence: str,
    ) -> torch.Tensor:
        """Extract embeddings from MAMMAL model.
        
        Args:
            antibody_sequence: Antibody amino acid sequence
            antigen_sequence: Antigen amino acid sequence
            
        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        if self.mammal_adapter is None:
            raise RuntimeError("MAMMAL adapter not available")
        
        embeddings = self.mammal_adapter.get_embeddings(
            antibody_seq=antibody_sequence,
            antigen_seq=antigen_sequence,
        )
        
        return embeddings
    
    def predict(
        self,
        antibody_sequence: str,
        antigen_sequence: str,
    ) -> UnifiedPrediction:
        """Make end-to-end prediction from sequences.
        
        Args:
            antibody_sequence: Antibody amino acid sequence
            antigen_sequence: Antigen amino acid sequence
            
        Returns:
            UnifiedPrediction object
        """
        # Get embeddings from MAMMAL
        embeddings = self.get_embeddings(antibody_sequence, antigen_sequence)
        
        # Run through prediction heads
        prediction = self.predictor.predict(
            embeddings,
            antibody_sequence=antibody_sequence,
            antigen_sequence=antigen_sequence,
        )
        
        return prediction
    
    def predict_from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> UnifiedPrediction:
        """Make prediction from pre-computed embeddings.
        
        Args:
            embeddings: Pre-computed MAMMAL embeddings
            
        Returns:
            UnifiedPrediction object
        """
        return self.predictor.predict(embeddings)
    
    def predict_batch(
        self,
        sequence_pairs: List[Tuple[str, str]],
    ) -> List[UnifiedPrediction]:
        """Make batch predictions from sequence pairs.
        
        Args:
            sequence_pairs: List of (antibody_seq, antigen_seq) tuples
            
        Returns:
            List of UnifiedPrediction objects
        """
        # Extract embeddings for all pairs
        embeddings_list = []
        for ab_seq, ag_seq in sequence_pairs:
            emb = self.get_embeddings(ab_seq, ag_seq)
            embeddings_list.append(emb)
        
        # Stack embeddings
        embeddings = torch.stack(embeddings_list)
        
        # Batch prediction
        predictions = self.predictor.predict_batch(embeddings)
        
        # Add sequences to predictions
        for i, pred in enumerate(predictions):
            pred.antibody_sequence = sequence_pairs[i][0]
            pred.antigen_sequence = sequence_pairs[i][1]
        
        return predictions
    
    def rank_candidates(
        self,
        antibody_sequences: List[str],
        antigen_sequence: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, UnifiedPrediction]]:
        """Rank antibody candidates against a single antigen.
        
        Args:
            antibody_sequences: List of antibody sequences to rank
            antigen_sequence: Target antigen sequence
            top_k: Return only top K candidates
            
        Returns:
            List of (original_index, antibody_seq, prediction) tuples
        """
        # Create sequence pairs
        sequence_pairs = [(ab_seq, antigen_sequence) for ab_seq in antibody_sequences]
        
        # Get predictions
        predictions = self.predict_batch(sequence_pairs)
        
        # Create indexed list
        indexed = [(i, antibody_sequences[i], pred) for i, pred in enumerate(predictions)]
        
        # Sort by composite score
        indexed.sort(key=lambda x: x[2].composite_score, reverse=True)
        
        if top_k is not None:
            indexed = indexed[:top_k]
        
        return indexed
    
    def screen_library(
        self,
        antibody_library: List[str],
        antigen_sequences: List[str],
        binding_threshold: float = 0.5,
        breadth_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Screen antibody library against multiple antigens.
        
        Args:
            antibody_library: Library of antibody sequences
            antigen_sequences: List of target antigen sequences
            binding_threshold: Threshold for binding classification
            breadth_threshold: Threshold for breadth score
            
        Returns:
            Screening results dictionary
        """
        results = {
            "total_antibodies": len(antibody_library),
            "total_antigens": len(antigen_sequences),
            "hits": [],
            "broad_hits": [],
            "summary": {},
        }
        
        for i, ab_seq in enumerate(antibody_library):
            hit_count = 0
            predictions = []
            
            for ag_seq in antigen_sequences:
                pred = self.predict(ab_seq, ag_seq)
                predictions.append(pred)
                
                if pred.binding is not None and pred.binding.probability >= binding_threshold:
                    hit_count += 1
            
            # Check if this is a hit
            breadth = hit_count / len(antigen_sequences)
            
            if hit_count > 0:
                results["hits"].append({
                    "index": i,
                    "sequence": ab_seq[:50] + "..." if len(ab_seq) > 50 else ab_seq,
                    "hit_count": hit_count,
                    "breadth": breadth,
                })
            
            if breadth >= breadth_threshold:
                results["broad_hits"].append({
                    "index": i,
                    "sequence": ab_seq[:50] + "..." if len(ab_seq) > 50 else ab_seq,
                    "hit_count": hit_count,
                    "breadth": breadth,
                })
        
        results["summary"] = {
            "total_hits": len(results["hits"]),
            "broad_hits": len(results["broad_hits"]),
            "hit_rate": len(results["hits"]) / len(antibody_library),
            "broad_rate": len(results["broad_hits"]) / len(antibody_library),
        }
        
        return results
    
    def load_heads(self, checkpoint_path: str):
        """Load prediction heads from checkpoint.
        
        Args:
            checkpoint_path: Path to heads checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.predictor.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded prediction heads from {checkpoint_path}")
    
    def save_heads(self, checkpoint_path: str):
        """Save prediction heads to checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        self.predictor.save(checkpoint_path)
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts."""
        params = {
            "prediction_heads": self.predictor.get_num_parameters(),
        }
        
        if self.mammal_adapter is not None:
            # MAMMAL parameters (frozen)
            params["mammal_backbone"] = 458_000_000  # Approximate
        
        params["total_trainable"] = params["prediction_heads"]
        
        return params
    
    def summary(self) -> str:
        """Get model summary."""
        lines = [
            "=" * 70,
            "BEREAN INTEGRATED PREDICTOR",
            "=" * 70,
            "",
            "BACKBONE:",
            f"  MAMMAL Model: {self.config.mammal_model_path}",
            f"  Embedding Dim: {self.config.embedding_dim}",
            f"  Device: {self.config.device}",
            "",
            "PREDICTION HEADS:",
        ]
        
        lines.append(self.predictor.summary())
        
        params = self.get_num_parameters()
        lines.extend([
            "",
            "PARAMETERS:",
            f"  Prediction Heads: {params['prediction_heads']:,}",
            f"  Total Trainable: {params['total_trainable']:,}",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_predict(
    antibody_sequence: str,
    antigen_sequence: str,
    device: str = "cpu",
) -> UnifiedPrediction:
    """Quick one-off prediction without persistent model.
    
    Note: For multiple predictions, use BEREANPredictor directly
    to avoid reloading the model each time.
    
    Args:
        antibody_sequence: Antibody sequence
        antigen_sequence: Antigen sequence
        device: Target device
        
    Returns:
        UnifiedPrediction
    """
    predictor = BEREANPredictor.from_pretrained(device=device)
    return predictor.predict(antibody_sequence, antigen_sequence)


def create_minimal_predictor(device: str = "cpu") -> BEREANPredictor:
    """Create minimal predictor with only binding head.
    
    Useful for memory-constrained environments.
    """
    config = BEREANConfig(
        enable_binding=True,
        enable_affinity=False,
        enable_cross_reactivity=False,
        device=device,
    )
    return BEREANPredictor(config)


def create_hai_predictor(device: str = "cpu") -> BEREANPredictor:
    """Create predictor optimized for HAI titer prediction."""
    config = BEREANConfig(
        enable_binding=True,
        enable_affinity=True,
        enable_cross_reactivity=False,
        affinity_type=AffinityType.HAI,
        device=device,
    )
    return BEREANPredictor(config)


if __name__ == "__main__":
    print("Testing BEREAN Integrated Predictor...")
    print("=" * 70)
    
    # Test without MAMMAL (embedding-only mode)
    print("\nCreating predictor (embedding-only mode for testing)...")
    
    config = BEREANConfig(
        enable_binding=True,
        enable_affinity=True,
        enable_cross_reactivity=True,
        cross_reactivity_strains=["H1N1/Test", "H3N2/Test", "H5N1/Test"],
        device="cpu",
    )
    
    # Create predictor without MAMMAL adapter
    predictor = BEREANPredictor.__new__(BEREANPredictor)
    predictor.config = config
    predictor.mammal_adapter = None
    
    # Build OmniSynaptic config
    omni_config = OmniSynapticConfig(
        embedding_dim=config.embedding_dim,
        enable_binding_head=config.enable_binding,
        enable_affinity_head=config.enable_affinity,
        enable_cross_reactivity_head=config.enable_cross_reactivity,
        cross_reactivity_config=CrossReactivityHeadConfig(
            input_dim=config.embedding_dim,
            strains=config.cross_reactivity_strains,
        ),
        device=config.device,
    )
    predictor.predictor = OmniSynapticBindingPredictor(omni_config)
    
    print(predictor.summary())
    
    # Test prediction from embeddings
    print("\n" + "=" * 70)
    print("PREDICTION FROM EMBEDDINGS TEST")
    print("=" * 70)
    
    test_embedding = torch.randn(768)
    result = predictor.predict_from_embeddings(test_embedding)
    print(result.summary())
    
    # Test batch prediction from embeddings
    print("\n" + "=" * 70)
    print("BATCH PREDICTION TEST")
    print("=" * 70)
    
    batch_embeddings = torch.randn(5, 768)
    batch_results = predictor.predictor.predict_batch(batch_embeddings)
    
    for i, res in enumerate(batch_results):
        print(f"Sample {i}: score={res.composite_score:.3f}, rank={res.candidate_rank}")
    
    # Test ranking
    print("\n" + "=" * 70)
    print("CANDIDATE RANKING TEST")
    print("=" * 70)
    
    embeddings_list = [torch.randn(768) for _ in range(10)]
    rankings = predictor.predictor.rank_candidates(embeddings_list, top_k=5)
    
    print("Top 5 candidates:")
    for rank, (orig_idx, pred) in enumerate(rankings):
        print(f"  #{rank+1}: idx={orig_idx}, score={pred.composite_score:.3f}")
    
    print("\n✓ BEREAN Integrated Predictor tests passed!")
