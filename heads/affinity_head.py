"""
Affinity Head - Quantitative Binding Affinity Prediction
=========================================================

Status: NEW
Source: berean_v2_refactor

This head predicts quantitative binding affinity values including:
    - IC50: Half-maximal inhibitory concentration
    - Kd: Dissociation constant
    - HAI: Hemagglutination inhibition titer
    - EC50: Half-maximal effective concentration

Architecture:
    
    MAMMAL Embeddings (768-dim)
            │
            ▼
    ┌─────────────────┐
    │   Layer Norm    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Dense (768→512)│
    │     + GELU      │
    │    + Dropout    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Dense (512→256)│
    │     + GELU      │
    │    + Dropout    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Dense (256→128)│
    │     + GELU      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Dense (128→1)  │
    │   (regression)  │
    └────────┬────────┘
             │
             ▼
        [affinity value]

Output transformations:
    - IC50/Kd: Log-transformed (pIC50 = -log10(IC50))
    - HAI: Log2-transformed titers
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AffinityType(Enum):
    """Types of affinity measurements."""
    IC50 = "ic50"           # Half-maximal inhibitory concentration
    KD = "kd"               # Dissociation constant
    HAI = "hai"             # Hemagglutination inhibition titer
    EC50 = "ec50"           # Half-maximal effective concentration
    KA = "ka"               # Association constant
    CUSTOM = "custom"       # Custom affinity metric


@dataclass
class AffinityHeadConfig:
    """Configuration for Affinity Head.
    
    Attributes:
        input_dim: Dimension of MAMMAL embeddings (default 768)
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        affinity_type: Type of affinity being predicted
        log_transform: Whether output is log-transformed
        output_activation: Final activation (None, 'relu', 'softplus')
        min_value: Minimum valid affinity value (for clamping)
        max_value: Maximum valid affinity value (for clamping)
    """
    input_dim: int = 768
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    affinity_type: AffinityType = AffinityType.IC50
    log_transform: bool = True
    output_activation: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    use_layer_norm: bool = True
    
    def __post_init__(self):
        # Set default bounds based on affinity type
        if self.min_value is None or self.max_value is None:
            bounds = {
                AffinityType.IC50: (-3.0, 6.0),      # pIC50 range: 1nM to 1mM
                AffinityType.KD: (-12.0, -3.0),     # log10(Kd) range
                AffinityType.HAI: (0.0, 13.0),      # log2(HAI) range: 1 to 8192
                AffinityType.EC50: (-3.0, 6.0),      # Similar to IC50
                AffinityType.KA: (3.0, 12.0),        # log10(Ka) range
                AffinityType.CUSTOM: (-10.0, 10.0),  # Wide range
            }
            default_min, default_max = bounds.get(self.affinity_type, (-10.0, 10.0))
            self.min_value = self.min_value or default_min
            self.max_value = self.max_value or default_max


@dataclass
class AffinityPrediction:
    """Output from Affinity Head prediction.
    
    Attributes:
        value: Predicted affinity value (in transformed space if applicable)
        raw_value: Value in original units (e.g., nM for IC50)
        affinity_type: Type of affinity predicted
        uncertainty: Optional uncertainty estimate
        embedding: Optional - the embedding used for prediction
    """
    value: float                           # Transformed value (e.g., pIC50)
    raw_value: float                       # Original units (e.g., nM)
    affinity_type: AffinityType
    uncertainty: Optional[float] = None
    embedding: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "value": self.value,
            "raw_value": self.raw_value,
            "affinity_type": self.affinity_type.value,
        }
        
        # Add interpretable labels based on affinity type
        if self.affinity_type == AffinityType.IC50:
            result["pIC50"] = self.value
            result["IC50_nM"] = self.raw_value
            result["potency"] = self._get_potency_label()
        elif self.affinity_type == AffinityType.HAI:
            result["log2_HAI"] = self.value
            result["HAI_titer"] = self.raw_value
            result["protective"] = self.raw_value >= 40
        elif self.affinity_type == AffinityType.KD:
            result["log10_Kd"] = self.value
            result["Kd_M"] = self.raw_value
        
        if self.uncertainty is not None:
            result["uncertainty"] = self.uncertainty
        
        return result
    
    def _get_potency_label(self) -> str:
        """Get human-readable potency label for IC50."""
        if self.raw_value < 1:
            return "Very High (sub-nM)"
        elif self.raw_value < 10:
            return "High (1-10 nM)"
        elif self.raw_value < 100:
            return "Moderate (10-100 nM)"
        elif self.raw_value < 1000:
            return "Low (100-1000 nM)"
        else:
            return "Very Low (>1 µM)"


class AffinityHead(nn.Module):
    """
    Affinity Head for Quantitative Binding Affinity Prediction.
    
    Status: NEW
    
    This neural network head takes MAMMAL embeddings and predicts
    quantitative binding affinity values (IC50, Kd, HAI, etc.).
    
    Example:
        >>> config = AffinityHeadConfig(affinity_type=AffinityType.IC50)
        >>> head = AffinityHead(config)
        >>> embeddings = torch.randn(batch_size, 768)
        >>> prediction = head(embeddings)
        >>> print(f"pIC50: {prediction.value}, IC50: {prediction.raw_value} nM")
    """
    
    def __init__(self, config: Optional[AffinityHeadConfig] = None):
        """Initialize Affinity Head.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        super().__init__()
        
        self.config = config or AffinityHeadConfig()
        
        # Layer normalization
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.config.input_dim)
        else:
            self.layer_norm = nn.Identity()
        
        # Build MLP layers
        layers = []
        prev_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            ])
            prev_dim = hidden_dim
        
        # Final regression layer (single output)
        layers.append(nn.Linear(prev_dim, 1))
        
        # Optional output activation
        if self.config.output_activation == "relu":
            layers.append(nn.ReLU())
        elif self.config.output_activation == "softplus":
            layers.append(nn.Softplus())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized AffinityHead: {self.config.input_dim} → "
            f"{self.config.hidden_dims} → 1 ({self.config.affinity_type.value})"
        )
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[AffinityPrediction, torch.Tensor]:
        """Forward pass through Affinity Head.
        
        Args:
            embeddings: MAMMAL embeddings of shape (batch_size, input_dim)
            return_dict: If True, return AffinityPrediction object
            
        Returns:
            AffinityPrediction or raw output tensor
        """
        # Ensure embeddings are 2D
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)
        
        # Layer normalization
        normalized = self.layer_norm(embeddings)
        
        # MLP forward pass
        output = self.mlp(normalized).squeeze(-1)
        
        # Clamp to valid range
        output = torch.clamp(
            output,
            min=self.config.min_value,
            max=self.config.max_value
        )
        
        if not return_dict:
            return output
        
        # Convert to prediction object (single sample)
        if embeddings.size(0) == 1:
            value = output[0].item()
            raw_value = self._inverse_transform(value)
            
            return AffinityPrediction(
                value=value,
                raw_value=raw_value,
                affinity_type=self.config.affinity_type,
                embedding=embeddings[0],
            )
        else:
            # Return first sample for compatibility
            value = output[0].item()
            raw_value = self._inverse_transform(value)
            
            return AffinityPrediction(
                value=value,
                raw_value=raw_value,
                affinity_type=self.config.affinity_type,
            )
    
    def _inverse_transform(self, transformed_value: float) -> float:
        """Convert from transformed space back to original units.
        
        Args:
            transformed_value: Value in transformed space
            
        Returns:
            Value in original units
        """
        if not self.config.log_transform:
            return transformed_value
        
        affinity_type = self.config.affinity_type
        
        if affinity_type == AffinityType.IC50:
            # pIC50 = -log10(IC50 in M), want nM
            # IC50 (M) = 10^(-pIC50)
            # IC50 (nM) = 10^(-pIC50) * 10^9 = 10^(9 - pIC50)
            return math.pow(10, 9 - transformed_value)
        
        elif affinity_type == AffinityType.HAI:
            # log2(HAI) → HAI
            return math.pow(2, transformed_value)
        
        elif affinity_type == AffinityType.KD:
            # log10(Kd) → Kd
            return math.pow(10, transformed_value)
        
        elif affinity_type == AffinityType.EC50:
            # Same as IC50
            return math.pow(10, 9 - transformed_value)
        
        elif affinity_type == AffinityType.KA:
            # log10(Ka) → Ka
            return math.pow(10, transformed_value)
        
        else:
            return transformed_value
    
    def _forward_transform(self, raw_value: float) -> float:
        """Convert from original units to transformed space.
        
        Args:
            raw_value: Value in original units
            
        Returns:
            Value in transformed space
        """
        if not self.config.log_transform:
            return raw_value
        
        affinity_type = self.config.affinity_type
        
        if affinity_type == AffinityType.IC50:
            # IC50 (nM) → pIC50
            # pIC50 = -log10(IC50 in M) = -log10(IC50_nM * 10^-9) = 9 - log10(IC50_nM)
            return 9 - math.log10(max(raw_value, 1e-6))
        
        elif affinity_type == AffinityType.HAI:
            # HAI → log2(HAI)
            return math.log2(max(raw_value, 1))
        
        elif affinity_type == AffinityType.KD:
            # Kd → log10(Kd)
            return math.log10(max(raw_value, 1e-15))
        
        elif affinity_type == AffinityType.EC50:
            return 9 - math.log10(max(raw_value, 1e-6))
        
        elif affinity_type == AffinityType.KA:
            return math.log10(max(raw_value, 1))
        
        else:
            return raw_value
    
    def predict(self, embeddings: torch.Tensor) -> AffinityPrediction:
        """Make a single prediction.
        
        Args:
            embeddings: Single embedding vector
            
        Returns:
            AffinityPrediction object
        """
        self.eval()
        with torch.no_grad():
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            return self.forward(embeddings, return_dict=True)
    
    def predict_batch(self, embeddings: torch.Tensor) -> List[AffinityPrediction]:
        """Make batch predictions.
        
        Args:
            embeddings: Batch of embeddings
            
        Returns:
            List of AffinityPrediction objects
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(embeddings, return_dict=False)
            
            predictions = []
            for i in range(embeddings.size(0)):
                value = outputs[i].item()
                raw_value = self._inverse_transform(value)
                
                predictions.append(AffinityPrediction(
                    value=value,
                    raw_value=raw_value,
                    affinity_type=self.config.affinity_type,
                ))
            
            return predictions
    
    def compute_loss(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        transform_targets: bool = True,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """Compute regression loss.
        
        Args:
            embeddings: Input embeddings
            targets: Ground truth affinity values (in original units if transform_targets=True)
            transform_targets: Whether to transform targets to log space
            loss_type: Type of loss ('mse', 'mae', 'huber')
            
        Returns:
            Loss tensor
        """
        predictions = self.forward(embeddings, return_dict=False)
        
        # Transform targets if needed
        if transform_targets and self.config.log_transform:
            transformed_targets = torch.tensor([
                self._forward_transform(t.item()) for t in targets
            ], dtype=predictions.dtype, device=predictions.device)
        else:
            transformed_targets = targets
        
        # Compute loss
        if loss_type == "mse":
            loss = F.mse_loss(predictions, transformed_targets)
        elif loss_type == "mae":
            loss = F.l1_loss(predictions, transformed_targets)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predictions, transformed_targets)
        else:
            loss = F.mse_loss(predictions, transformed_targets)
        
        return loss
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get model summary string."""
        lines = [
            "=" * 60,
            "AFFINITY HEAD SUMMARY",
            "=" * 60,
            f"Affinity type: {self.config.affinity_type.value}",
            f"Input dimension: {self.config.input_dim}",
            f"Hidden dimensions: {self.config.hidden_dims}",
            f"Dropout: {self.config.dropout}",
            f"Log transform: {self.config.log_transform}",
            f"Output range: [{self.config.min_value}, {self.config.max_value}]",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# SPECIALIZED AFFINITY HEADS
# =============================================================================

class IC50Head(AffinityHead):
    """Specialized head for IC50 prediction."""
    
    def __init__(self, input_dim: int = 768, **kwargs):
        config = AffinityHeadConfig(
            input_dim=input_dim,
            affinity_type=AffinityType.IC50,
            log_transform=True,
            **kwargs
        )
        super().__init__(config)


class HAIHead(AffinityHead):
    """Specialized head for HAI titer prediction."""
    
    def __init__(self, input_dim: int = 768, **kwargs):
        config = AffinityHeadConfig(
            input_dim=input_dim,
            affinity_type=AffinityType.HAI,
            log_transform=True,
            min_value=0.0,
            max_value=13.0,  # log2(8192)
            **kwargs
        )
        super().__init__(config)
    
    def is_protective(self, prediction: AffinityPrediction) -> bool:
        """Check if HAI titer is protective (≥40)."""
        return prediction.raw_value >= 40


class KdHead(AffinityHead):
    """Specialized head for Kd (dissociation constant) prediction."""
    
    def __init__(self, input_dim: int = 768, **kwargs):
        config = AffinityHeadConfig(
            input_dim=input_dim,
            affinity_type=AffinityType.KD,
            log_transform=True,
            **kwargs
        )
        super().__init__(config)


if __name__ == "__main__":
    # Test the Affinity Head
    print("Testing Affinity Head...")
    
    # Test IC50 Head
    ic50_head = IC50Head(input_dim=768)
    print(ic50_head.summary())
    
    batch_size = 4
    embeddings = torch.randn(batch_size, 768)
    
    # Single prediction
    single_pred = ic50_head.predict(embeddings[0])
    print(f"\nIC50 prediction: {single_pred.to_dict()}")
    
    # Batch prediction
    batch_preds = ic50_head.predict_batch(embeddings)
    print(f"\nBatch IC50 predictions:")
    for i, pred in enumerate(batch_preds):
        print(f"  Sample {i}: pIC50={pred.value:.2f}, IC50={pred.raw_value:.1f} nM")
    
    # Test HAI Head
    print("\n" + "=" * 60)
    hai_head = HAIHead(input_dim=768)
    print(hai_head.summary())
    
    hai_pred = hai_head.predict(embeddings[0])
    print(f"\nHAI prediction: {hai_pred.to_dict()}")
    print(f"Protective: {hai_head.is_protective(hai_pred)}")
    
    # Test loss computation
    targets = torch.tensor([100.0, 50.0, 200.0, 10.0])  # IC50 in nM
    loss = ic50_head.compute_loss(embeddings, targets)
    print(f"\nIC50 Loss: {loss.item():.4f}")
    
    print("\n✓ Affinity Head tests passed!")
