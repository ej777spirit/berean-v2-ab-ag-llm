"""
Binding Head - Binary Antibody-Antigen Binding Classification
==============================================================

Status: NEW
Source: berean_v2_refactor

This head predicts whether an antibody binds to a given antigen (HA).
It takes MAMMAL embeddings and outputs a binary classification.

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
    │  Dense (256→2)  │
    │    (logits)     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Softmax      │
    │  [P(neg), P(pos)]│
    └─────────────────┘

Target Benchmarks (from Cleveland Clinic paper):
    - Lenient split: 0.91-0.92 AUROC
    - HA-exclusive: 0.90 AUROC
    - mAb-exclusive: 0.73 AUROC
    - mAb-cluster-exclusive: 0.63-0.66 AUROC
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class BindingHeadConfig:
    """Configuration for Binding Head.
    
    Attributes:
        input_dim: Dimension of MAMMAL embeddings (default 768)
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        num_classes: Number of output classes (2 for binary)
        use_layer_norm: Whether to apply layer normalization
        activation: Activation function name
        class_weights: Optional class weights for imbalanced data
    """
    input_dim: int = 768
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    num_classes: int = 2
    use_layer_norm: bool = True
    activation: str = "gelu"
    class_weights: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.class_weights is None:
            # Default weights for imbalanced binding data (~35% positive)
            self.class_weights = [1.0, 1.86]  # Upweight positive class


@dataclass
class BindingPrediction:
    """Output from Binding Head prediction.
    
    Attributes:
        probability: Probability of binding (positive class)
        prediction: Binary prediction (0=no binding, 1=binding)
        logits: Raw logit values [neg_logit, pos_logit]
        confidence: Confidence score (max probability)
        embedding: Optional - the embedding used for prediction
    """
    probability: float
    prediction: int
    logits: torch.Tensor
    confidence: float
    embedding: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "probability": self.probability,
            "prediction": self.prediction,
            "logits": self.logits.cpu().numpy().tolist() if isinstance(self.logits, torch.Tensor) else self.logits,
            "confidence": self.confidence,
            "binding": bool(self.prediction),
            "binding_label": "BINDING" if self.prediction == 1 else "NON-BINDING",
        }


class BindingHead(nn.Module):
    """
    Binding Head for Binary Antibody-Antigen Binding Classification.
    
    Status: NEW
    
    This neural network head takes MAMMAL embeddings and predicts
    whether an antibody binds to a given antigen.
    
    Example:
        >>> config = BindingHeadConfig(input_dim=768)
        >>> head = BindingHead(config)
        >>> embeddings = torch.randn(batch_size, 768)
        >>> prediction = head(embeddings)
        >>> print(prediction.probability)
    """
    
    def __init__(self, config: Optional[BindingHeadConfig] = None):
        """Initialize Binding Head.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        super().__init__()
        
        self.config = config or BindingHeadConfig()
        
        # Build layers
        layers = []
        
        # Optional layer normalization
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.config.input_dim)
        else:
            self.layer_norm = nn.Identity()
        
        # Build MLP layers
        prev_dim = self.config.input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.config.dropout),
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, self.config.num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Class weights for loss computation
        if self.config.class_weights:
            self.register_buffer(
                'class_weights',
                torch.tensor(self.config.class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized BindingHead: {self.config.input_dim} → {self.config.hidden_dims} → {self.config.num_classes}")
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(self.config.activation.lower(), nn.GELU())
    
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
    ) -> Union[BindingPrediction, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through Binding Head.
        
        Args:
            embeddings: MAMMAL embeddings of shape (batch_size, input_dim)
            return_dict: If True, return BindingPrediction object
            
        Returns:
            BindingPrediction or tuple of (logits, probabilities)
        """
        # Ensure embeddings are 2D
        if embeddings.dim() == 3:
            # Pool over sequence dimension (mean pooling)
            embeddings = embeddings.mean(dim=1)
        
        # Layer normalization
        normalized = self.layer_norm(embeddings)
        
        # MLP forward pass
        logits = self.mlp(normalized)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        if not return_dict:
            return logits, probabilities
        
        # Create prediction object (for single sample or batch)
        if embeddings.size(0) == 1:
            prob_positive = probabilities[0, 1].item()
            prediction = int(prob_positive > 0.5)
            confidence = max(prob_positive, 1 - prob_positive)
            
            return BindingPrediction(
                probability=prob_positive,
                prediction=prediction,
                logits=logits[0],
                confidence=confidence,
                embedding=embeddings[0] if embeddings.size(0) == 1 else None,
            )
        else:
            # Batch prediction - return first sample's prediction
            # For batch processing, use predict_batch instead
            prob_positive = probabilities[0, 1].item()
            prediction = int(prob_positive > 0.5)
            confidence = max(prob_positive, 1 - prob_positive)
            
            return BindingPrediction(
                probability=prob_positive,
                prediction=prediction,
                logits=logits[0],
                confidence=confidence,
            )
    
    def predict(self, embeddings: torch.Tensor) -> BindingPrediction:
        """Make a single prediction.
        
        Args:
            embeddings: Single embedding vector (input_dim,) or (1, input_dim)
            
        Returns:
            BindingPrediction object
        """
        self.eval()
        with torch.no_grad():
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            return self.forward(embeddings, return_dict=True)
    
    def predict_batch(
        self,
        embeddings: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[BindingPrediction]:
        """Make batch predictions.
        
        Args:
            embeddings: Batch of embeddings (batch_size, input_dim)
            threshold: Classification threshold
            
        Returns:
            List of BindingPrediction objects
        """
        self.eval()
        with torch.no_grad():
            logits, probabilities = self.forward(embeddings, return_dict=False)
            
            predictions = []
            for i in range(embeddings.size(0)):
                prob_positive = probabilities[i, 1].item()
                prediction = int(prob_positive > threshold)
                confidence = max(prob_positive, 1 - prob_positive)
                
                predictions.append(BindingPrediction(
                    probability=prob_positive,
                    prediction=prediction,
                    logits=logits[i],
                    confidence=confidence,
                ))
            
            return predictions
    
    def compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute cross-entropy loss with optional class weighting.
        
        Args:
            embeddings: Input embeddings (batch_size, input_dim)
            labels: Ground truth labels (batch_size,)
            reduction: Loss reduction method
            
        Returns:
            Loss tensor
        """
        logits, _ = self.forward(embeddings, return_dict=False)
        
        if self.class_weights is not None:
            loss = F.cross_entropy(
                logits, labels,
                weight=self.class_weights,
                reduction=reduction
            )
        else:
            loss = F.cross_entropy(logits, labels, reduction=reduction)
        
        return loss
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get model summary string."""
        lines = [
            "=" * 60,
            "BINDING HEAD SUMMARY",
            "=" * 60,
            f"Input dimension: {self.config.input_dim}",
            f"Hidden dimensions: {self.config.hidden_dims}",
            f"Output classes: {self.config.num_classes}",
            f"Dropout: {self.config.dropout}",
            f"Activation: {self.config.activation}",
            f"Layer norm: {self.config.use_layer_norm}",
            f"Class weights: {self.config.class_weights}",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class BindingHeadTrainer:
    """Training utility for Binding Head.
    
    Example:
        >>> trainer = BindingHeadTrainer(head, learning_rate=1e-4)
        >>> for epoch in range(10):
        ...     loss = trainer.train_step(embeddings, labels)
    """
    
    def __init__(
        self,
        head: BindingHead,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.head = head
        self.optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
    
    def train_step(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Single training step.
        
        Args:
            embeddings: Input embeddings
            labels: Ground truth labels
            
        Returns:
            Loss value
        """
        self.head.train()
        self.optimizer.zero_grad()
        
        loss = self.head.compute_loss(embeddings, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validation_step(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """Validation step.
        
        Args:
            embeddings: Input embeddings
            labels: Ground truth labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.head.eval()
        with torch.no_grad():
            loss = self.head.compute_loss(embeddings, labels)
            
            predictions = self.head.predict_batch(embeddings)
            preds = torch.tensor([p.prediction for p in predictions])
            accuracy = (preds == labels.cpu()).float().mean().item()
        
        return loss.item(), accuracy
    
    def update_scheduler(self, val_loss: float):
        """Update learning rate scheduler."""
        self.scheduler.step(val_loss)


if __name__ == "__main__":
    # Test the Binding Head
    print("Testing Binding Head...")
    
    config = BindingHeadConfig(input_dim=768)
    head = BindingHead(config)
    print(head.summary())
    
    # Test forward pass
    batch_size = 4
    embeddings = torch.randn(batch_size, 768)
    
    # Single prediction
    single_pred = head.predict(embeddings[0])
    print(f"\nSingle prediction: {single_pred.to_dict()}")
    
    # Batch prediction
    batch_preds = head.predict_batch(embeddings)
    print(f"\nBatch predictions:")
    for i, pred in enumerate(batch_preds):
        print(f"  Sample {i}: prob={pred.probability:.3f}, pred={pred.prediction}")
    
    # Test loss computation
    labels = torch.tensor([0, 1, 1, 0])
    loss = head.compute_loss(embeddings, labels)
    print(f"\nLoss: {loss.item():.4f}")
    
    print("\n✓ Binding Head tests passed!")
