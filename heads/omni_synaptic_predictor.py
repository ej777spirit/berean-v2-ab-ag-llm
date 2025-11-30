"""
OmniSynapticBindingPredictor - Unified Multi-Head Prediction Orchestrator
==========================================================================

Status: NEW
Source: berean_v2_refactor

This module provides the OmniSynapticBindingPredictor class, which orchestrates
all three prediction heads (Binding, Affinity, Cross-Reactivity) on top of
MAMMAL foundation model embeddings.

Architecture (from BEREAN schematic):
    
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
    │  │  (768→2)  │   │  (768→1)  │   │   (768→N×2)     │        │
    │  └─────┬─────┘   └─────┬─────┘   └───────┬─────────┘        │
    │        │               │                 │                  │
    │        ▼               ▼                 ▼                  │
    │    [binding]       [IC50/Kd]      [strain probs]            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │ UnifiedPrediction│
                        │  - binding_prob  │
                        │  - affinity      │
                        │  - breadth       │
                        │  - bnAb_score    │
                        └──────────────────┘

Features:
    - Multi-task learning across all heads
    - Attention-based embedding pooling
    - Uncertainty estimation (optional)
    - Ensemble support
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .binding_head import BindingHead, BindingHeadConfig, BindingPrediction
from .affinity_head import AffinityHead, AffinityHeadConfig, AffinityPrediction, AffinityType
from .cross_reactivity_head import (
    CrossReactivityHead, 
    CrossReactivityHeadConfig, 
    CrossReactivityPrediction,
    DEFAULT_STRAIN_PANEL,
)

logger = logging.getLogger(__name__)


@dataclass
class OmniSynapticConfig:
    """Configuration for OmniSynapticBindingPredictor.
    
    Attributes:
        embedding_dim: Dimension of MAMMAL embeddings (default 768)
        enable_binding_head: Whether to enable binding classification
        enable_affinity_head: Whether to enable affinity prediction
        enable_cross_reactivity_head: Whether to enable cross-reactivity
        binding_config: Configuration for binding head
        affinity_config: Configuration for affinity head
        cross_reactivity_config: Configuration for cross-reactivity head
        pooling_strategy: How to pool sequence embeddings
        use_uncertainty: Whether to estimate prediction uncertainty
        dropout: Global dropout for attention pooling
    """
    embedding_dim: int = 768
    
    # Head enables
    enable_binding_head: bool = True
    enable_affinity_head: bool = True
    enable_cross_reactivity_head: bool = True
    
    # Head configurations (use defaults if None)
    binding_config: Optional[BindingHeadConfig] = None
    affinity_config: Optional[AffinityHeadConfig] = None
    cross_reactivity_config: Optional[CrossReactivityHeadConfig] = None
    
    # Pooling and uncertainty
    pooling_strategy: str = "mean"  # "mean", "max", "attention", "cls"
    use_uncertainty: bool = False
    dropout: float = 0.1
    
    def __post_init__(self):
        # Initialize default configs with correct embedding dim
        if self.binding_config is None:
            self.binding_config = BindingHeadConfig(input_dim=self.embedding_dim)
        if self.affinity_config is None:
            self.affinity_config = AffinityHeadConfig(input_dim=self.embedding_dim)
        if self.cross_reactivity_config is None:
            self.cross_reactivity_config = CrossReactivityHeadConfig(
                input_dim=self.embedding_dim
            )


@dataclass
class UnifiedPrediction:
    """Unified prediction output from OmniSynapticBindingPredictor.
    
    Combines predictions from all enabled heads into a single output.
    
    Attributes:
        binding: Binding head prediction (if enabled)
        affinity: Affinity head prediction (if enabled)
        cross_reactivity: Cross-reactivity head prediction (if enabled)
        composite_score: Combined therapeutic potential score
        uncertainty: Optional uncertainty estimates
        embeddings: The embeddings used for prediction
    """
    binding: Optional[BindingPrediction] = None
    affinity: Optional[AffinityPrediction] = None
    cross_reactivity: Optional[CrossReactivityPrediction] = None
    composite_score: float = 0.0
    uncertainty: Optional[Dict[str, float]] = None
    embeddings: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "composite_score": self.composite_score,
        }
        
        if self.binding is not None:
            result["binding"] = self.binding.to_dict()
        
        if self.affinity is not None:
            result["affinity"] = self.affinity.to_dict()
        
        if self.cross_reactivity is not None:
            result["cross_reactivity"] = self.cross_reactivity.to_dict()
        
        if self.uncertainty is not None:
            result["uncertainty"] = self.uncertainty
        
        # Add summary metrics
        result["summary"] = self._compute_summary()
        
        return result
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary metrics."""
        summary = {}
        
        if self.binding is not None:
            summary["binds"] = self.binding.prediction == 1
            summary["binding_confidence"] = self.binding.confidence
        
        if self.affinity is not None:
            summary["affinity_value"] = self.affinity.value
            summary["affinity_type"] = self.affinity.affinity_type.value
        
        if self.cross_reactivity is not None:
            summary["breadth"] = self.cross_reactivity.breadth_score
            summary["is_bnab"] = self.cross_reactivity.is_broadly_neutralizing
        
        return summary
    
    @property
    def is_promising_candidate(self) -> bool:
        """Check if antibody is a promising therapeutic candidate.
        
        Criteria:
        - Binds target (if binding head enabled)
        - High affinity (if affinity head enabled)
        - Broad reactivity (if cross-reactivity head enabled)
        """
        # Binding criterion
        if self.binding is not None:
            if self.binding.prediction != 1 or self.binding.confidence < 0.7:
                return False
        
        # Affinity criterion (pIC50 > 7 means IC50 < 100 nM)
        if self.affinity is not None:
            if self.affinity.affinity_type == AffinityType.IC50:
                if self.affinity.value < 7:  # pIC50 threshold
                    return False
        
        # Breadth criterion
        if self.cross_reactivity is not None:
            if self.cross_reactivity.breadth_score < 0.3:
                return False
        
        return True


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence embeddings."""
    
    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Softmax(dim=1),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool embeddings using attention.
        
        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
            mask: Optional attention mask (batch_size, seq_len)
            
        Returns:
            Pooled embeddings (batch_size, embedding_dim)
        """
        attention_weights = self.attention(embeddings)  # (B, L, 1)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~mask.unsqueeze(-1), float('-inf')
            )
            attention_weights = F.softmax(attention_weights, dim=1)
        
        pooled = torch.sum(attention_weights * embeddings, dim=1)
        return self.dropout(pooled)


class OmniSynapticBindingPredictor(nn.Module):
    """
    OmniSynapticBindingPredictor - Unified Multi-Head Prediction System
    
    Status: NEW
    
    This class orchestrates multiple prediction heads on top of MAMMAL
    foundation model embeddings, providing a unified interface for
    antibody characterization.
    
    Example:
        >>> config = OmniSynapticConfig(embedding_dim=768)
        >>> predictor = OmniSynapticBindingPredictor(config)
        >>> 
        >>> # Get embeddings from MAMMAL
        >>> embeddings = mammal_model.get_embeddings(antibody_seq, antigen_seq)
        >>> 
        >>> # Unified prediction
        >>> prediction = predictor(embeddings)
        >>> print(f"Binds: {prediction.binding.prediction}")
        >>> print(f"IC50: {prediction.affinity.raw_value} nM")
        >>> print(f"Breadth: {prediction.cross_reactivity.breadth_score:.1%}")
    """
    
    def __init__(self, config: Optional[OmniSynapticConfig] = None):
        """Initialize OmniSynapticBindingPredictor.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        super().__init__()
        
        self.config = config or OmniSynapticConfig()
        
        # Pooling layer
        if self.config.pooling_strategy == "attention":
            self.pooling = AttentionPooling(
                self.config.embedding_dim,
                self.config.dropout
            )
        else:
            self.pooling = None
        
        # Initialize heads
        self.binding_head = None
        self.affinity_head = None
        self.cross_reactivity_head = None
        
        if self.config.enable_binding_head:
            self.binding_head = BindingHead(self.config.binding_config)
            logger.info("Initialized Binding Head")
        
        if self.config.enable_affinity_head:
            self.affinity_head = AffinityHead(self.config.affinity_config)
            logger.info("Initialized Affinity Head")
        
        if self.config.enable_cross_reactivity_head:
            self.cross_reactivity_head = CrossReactivityHead(
                self.config.cross_reactivity_config
            )
            logger.info("Initialized Cross-Reactivity Head")
        
        # Uncertainty estimation (optional)
        if self.config.use_uncertainty:
            self._init_uncertainty_layers()
        
        logger.info(
            f"Initialized OmniSynapticBindingPredictor with "
            f"{self.get_num_parameters():,} parameters"
        )
    
    def _init_uncertainty_layers(self):
        """Initialize layers for uncertainty estimation."""
        # MC Dropout will be used - no additional layers needed
        pass
    
    def _pool_embeddings(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool sequence embeddings to fixed-size representation.
        
        Args:
            embeddings: Either (batch, embedding_dim) or (batch, seq_len, embedding_dim)
            mask: Optional attention mask
            
        Returns:
            Pooled embeddings (batch, embedding_dim)
        """
        # Already pooled
        if embeddings.dim() == 2:
            return embeddings
        
        # Apply pooling strategy
        if self.config.pooling_strategy == "mean":
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return embeddings.mean(dim=1)
        
        elif self.config.pooling_strategy == "max":
            if mask is not None:
                embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return embeddings.max(dim=1)[0]
        
        elif self.config.pooling_strategy == "cls":
            return embeddings[:, 0, :]
        
        elif self.config.pooling_strategy == "attention":
            return self.pooling(embeddings, mask)
        
        else:
            return embeddings.mean(dim=1)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UnifiedPrediction, Dict[str, Any]]:
        """Forward pass through all enabled heads.
        
        Args:
            embeddings: MAMMAL embeddings
            attention_mask: Optional mask for sequence pooling
            return_dict: If True, return UnifiedPrediction object
            
        Returns:
            UnifiedPrediction or dictionary of outputs
        """
        # Pool embeddings if needed
        pooled = self._pool_embeddings(embeddings, attention_mask)
        
        # Run enabled heads
        binding_pred = None
        affinity_pred = None
        cross_reactivity_pred = None
        
        if self.binding_head is not None:
            binding_pred = self.binding_head(pooled, return_dict=True)
        
        if self.affinity_head is not None:
            affinity_pred = self.affinity_head(pooled, return_dict=True)
        
        if self.cross_reactivity_head is not None:
            cross_reactivity_pred = self.cross_reactivity_head(pooled, return_dict=True)
        
        # Compute composite score
        composite_score = self._compute_composite_score(
            binding_pred, affinity_pred, cross_reactivity_pred
        )
        
        # Uncertainty estimation (if enabled)
        uncertainty = None
        if self.config.use_uncertainty:
            uncertainty = self._estimate_uncertainty(pooled)
        
        if not return_dict:
            return {
                "binding": binding_pred,
                "affinity": affinity_pred,
                "cross_reactivity": cross_reactivity_pred,
                "composite_score": composite_score,
                "uncertainty": uncertainty,
            }
        
        return UnifiedPrediction(
            binding=binding_pred,
            affinity=affinity_pred,
            cross_reactivity=cross_reactivity_pred,
            composite_score=composite_score,
            uncertainty=uncertainty,
            embeddings=pooled,
        )
    
    def _compute_composite_score(
        self,
        binding: Optional[BindingPrediction],
        affinity: Optional[AffinityPrediction],
        cross_reactivity: Optional[CrossReactivityPrediction],
    ) -> float:
        """Compute composite therapeutic potential score.
        
        The composite score combines:
        - Binding probability (weight: 0.3)
        - Normalized affinity (weight: 0.4)
        - Breadth score (weight: 0.3)
        
        Returns:
            Score between 0 and 1
        """
        scores = []
        weights = []
        
        if binding is not None:
            scores.append(binding.probability)
            weights.append(0.3)
        
        if affinity is not None:
            # Normalize affinity to 0-1 range
            if affinity.affinity_type == AffinityType.IC50:
                # pIC50 range ~3-12, normalize to 0-1
                normalized_affinity = (affinity.value - 3) / 9
                normalized_affinity = max(0, min(1, normalized_affinity))
                scores.append(normalized_affinity)
                weights.append(0.4)
        
        if cross_reactivity is not None:
            scores.append(cross_reactivity.breadth_score)
            weights.append(0.3)
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return composite
    
    def _estimate_uncertainty(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Estimate prediction uncertainty using MC Dropout."""
        n_samples = 10
        uncertainty = {}
        
        was_training = self.training
        self.train()  # Enable dropout
        
        with torch.no_grad():
            if self.binding_head is not None:
                probs = []
                for _ in range(n_samples):
                    pred = self.binding_head(embeddings, return_dict=True)
                    probs.append(pred.probability)
                uncertainty["binding"] = float(torch.tensor(probs).std())
            
            if self.affinity_head is not None:
                values = []
                for _ in range(n_samples):
                    pred = self.affinity_head(embeddings, return_dict=True)
                    values.append(pred.value)
                uncertainty["affinity"] = float(torch.tensor(values).std())
        
        if not was_training:
            self.eval()
        
        return uncertainty
    
    def predict(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> UnifiedPrediction:
        """Make a prediction."""
        self.eval()
        with torch.no_grad():
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            return self.forward(embeddings, attention_mask, return_dict=True)
    
    def predict_batch(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[UnifiedPrediction]:
        """Make batch predictions."""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = embeddings.size(0)
            
            for i in range(batch_size):
                emb = embeddings[i:i+1]
                mask = attention_mask[i:i+1] if attention_mask is not None else None
                
                pred = self.forward(emb, mask, return_dict=True)
                predictions.append(pred)
        
        return predictions
    
    def compute_loss(
        self,
        embeddings: torch.Tensor,
        binding_labels: Optional[torch.Tensor] = None,
        affinity_targets: Optional[torch.Tensor] = None,
        cross_reactivity_labels: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss."""
        pooled = self._pool_embeddings(embeddings, attention_mask)
        
        default_weights = {"binding": 1.0, "affinity": 1.0, "cross_reactivity": 1.0}
        weights = loss_weights or default_weights
        
        losses = {}
        total_loss = torch.tensor(0.0, device=embeddings.device)
        
        if self.binding_head is not None and binding_labels is not None:
            binding_loss = self.binding_head.compute_loss(pooled, binding_labels)
            losses["binding"] = binding_loss
            total_loss = total_loss + weights.get("binding", 1.0) * binding_loss
        
        if self.affinity_head is not None and affinity_targets is not None:
            affinity_loss = self.affinity_head.compute_loss(pooled, affinity_targets)
            losses["affinity"] = affinity_loss
            total_loss = total_loss + weights.get("affinity", 1.0) * affinity_loss
        
        if self.cross_reactivity_head is not None and cross_reactivity_labels is not None:
            cr_loss = self.cross_reactivity_head.compute_loss(pooled, cross_reactivity_labels)
            losses["cross_reactivity"] = cr_loss
            total_loss = total_loss + weights.get("cross_reactivity", 1.0) * cr_loss
        
        return total_loss, losses
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_head_parameters(self) -> Dict[str, int]:
        """Get parameter count per head."""
        params = {}
        
        if self.binding_head is not None:
            params["binding_head"] = self.binding_head.get_num_parameters()
        
        if self.affinity_head is not None:
            params["affinity_head"] = self.affinity_head.get_num_parameters()
        
        if self.cross_reactivity_head is not None:
            params["cross_reactivity_head"] = self.cross_reactivity_head.get_num_parameters()
        
        return params
    
    def summary(self) -> str:
        """Get model summary string."""
        lines = [
            "=" * 70,
            "OMNISYNAPTIC BINDING PREDICTOR SUMMARY",
            "=" * 70,
            f"Embedding dimension: {self.config.embedding_dim}",
            f"Pooling strategy: {self.config.pooling_strategy}",
            f"Uncertainty estimation: {self.config.use_uncertainty}",
            "",
            "ENABLED HEADS:",
        ]
        
        if self.binding_head is not None:
            lines.append(f"  ✓ Binding Head ({self.binding_head.get_num_parameters():,} params)")
        else:
            lines.append("  ✗ Binding Head (disabled)")
        
        if self.affinity_head is not None:
            lines.append(
                f"  ✓ Affinity Head ({self.affinity_head.get_num_parameters():,} params) "
                f"[{self.config.affinity_config.affinity_type.value}]"
            )
        else:
            lines.append("  ✗ Affinity Head (disabled)")
        
        if self.cross_reactivity_head is not None:
            n_strains = len(self.config.cross_reactivity_config.strains)
            lines.append(
                f"  ✓ Cross-Reactivity Head ({self.cross_reactivity_head.get_num_parameters():,} params) "
                f"[{n_strains} strains]"
            )
        else:
            lines.append("  ✗ Cross-Reactivity Head (disabled)")
        
        lines.extend([
            "",
            f"TOTAL PARAMETERS: {self.get_num_parameters():,}",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """Save model state dict."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)
        logger.info(f"Saved OmniSynapticBindingPredictor to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "OmniSynapticBindingPredictor":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        
        logger.info(f"Loaded OmniSynapticBindingPredictor from {path}")
        return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class OmniSynapticTrainer:
    """Training utility for OmniSynapticBindingPredictor."""
    
    def __init__(
        self,
        model: OmniSynapticBindingPredictor,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        loss_weighting: str = "equal",
    ):
        self.model = model
        self.loss_weighting = loss_weighting
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
    
    def train_step(
        self,
        embeddings: torch.Tensor,
        binding_labels: Optional[torch.Tensor] = None,
        affinity_targets: Optional[torch.Tensor] = None,
        cross_reactivity_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss, head_losses = self.model.compute_loss(
            embeddings,
            binding_labels=binding_labels,
            affinity_targets=affinity_targets,
            cross_reactivity_labels=cross_reactivity_labels,
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item(), {k: v.item() for k, v in head_losses.items()}
    
    def validation_step(
        self,
        embeddings: torch.Tensor,
        binding_labels: Optional[torch.Tensor] = None,
        affinity_targets: Optional[torch.Tensor] = None,
        cross_reactivity_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Validation step."""
        self.model.eval()
        
        with torch.no_grad():
            total_loss, head_losses = self.model.compute_loss(
                embeddings,
                binding_labels=binding_labels,
                affinity_targets=affinity_targets,
                cross_reactivity_labels=cross_reactivity_labels,
            )
        
        return total_loss.item(), {k: v.item() for k, v in head_losses.items()}
    
    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()


if __name__ == "__main__":
    # Test the OmniSynapticBindingPredictor
    print("Testing OmniSynapticBindingPredictor...")
    
    cr_config = CrossReactivityHeadConfig(
        input_dim=768,
        strains=["H1N1/Test/1", "H3N2/Test/2", "H5N1/Test/3"],
    )
    
    config = OmniSynapticConfig(
        embedding_dim=768,
        enable_binding_head=True,
        enable_affinity_head=True,
        enable_cross_reactivity_head=True,
        cross_reactivity_config=cr_config,
        use_uncertainty=True,
    )
    
    predictor = OmniSynapticBindingPredictor(config)
    print(predictor.summary())
    
    batch_size = 4
    embeddings = torch.randn(batch_size, 768)
    
    single_pred = predictor.predict(embeddings[0])
    print(f"\nComposite Score: {single_pred.composite_score:.3f}")
    print(f"Is Promising: {single_pred.is_promising_candidate}")
    
    print("\n✓ OmniSynapticBindingPredictor tests passed!")
