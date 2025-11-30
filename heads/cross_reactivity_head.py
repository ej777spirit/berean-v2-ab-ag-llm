"""
Cross-Reactivity Head - Multi-Strain Breadth Prediction
========================================================

Status: NEW
Source: berean_v2_refactor

This head predicts antibody cross-reactivity across multiple influenza strains.
It's critical for identifying broadly neutralizing antibodies (bnAbs) that
can protect against diverse viral variants.

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
             ├──────────────────┬─────────────────┐
             │                  │                 │
             ▼                  ▼                 ▼
    ┌─────────────┐    ┌─────────────┐   ┌─────────────┐
    │ Strain Head │    │ Strain Head │   │ Strain Head │
    │   (H1N1)    │    │   (H3N2)    │   │    (H5N1)   │
    └──────┬──────┘    └──────┬──────┘   └──────┬──────┘
           │                  │                 │
           ▼                  ▼                 ▼
    [P(binds H1N1)]   [P(binds H3N2)]  [P(binds H5N1)]

Outputs:
    - Per-strain binding probabilities
    - Breadth score (fraction of strains bound)
    - Strain coverage pattern

Use Cases:
    - Identify broadly neutralizing antibodies (bnAbs)
    - Predict coverage for universal flu vaccine candidates
    - Assess pandemic preparedness
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# STRAIN DEFINITIONS
# =============================================================================

class InfluenzaSubtype(Enum):
    """Influenza A subtypes."""
    H1N1 = "H1N1"
    H2N2 = "H2N2"
    H3N2 = "H3N2"
    H5N1 = "H5N1"
    H7N9 = "H7N9"
    H9N2 = "H9N2"
    H1N1_PANDEMIC = "H1N1pdm09"
    INFLUENZA_B = "B"


# Default strain panel for cross-reactivity assessment
DEFAULT_STRAIN_PANEL = [
    # Group 1 HAs
    "H1N1/California/04/2009",      # Pandemic 2009
    "H1N1/Michigan/45/2015",        # Seasonal
    "H2N2/Japan/305/1957",          # Historical
    "H5N1/Vietnam/1194/2004",       # Avian
    "H6N1/Taiwan/2/2013",           # Avian
    
    # Group 2 HAs
    "H3N2/Hong_Kong/1/1968",        # Historical
    "H3N2/Texas/50/2012",           # Seasonal
    "H7N9/Anhui/1/2013",            # Avian
    "H10N8/Jiangxi/IPB13/2013",     # Avian
]

# Strain groupings for group-level analysis
STRAIN_GROUPS = {
    "group1": ["H1N1", "H2N2", "H5N1", "H6N1", "H9N2"],
    "group2": ["H3N2", "H7N9", "H10N8", "H4N6", "H14N5"],
    "seasonal": ["H1N1", "H3N2", "B"],
    "pandemic_potential": ["H5N1", "H7N9", "H9N2", "H2N2"],
}


@dataclass
class StrainCoverage:
    """Detailed coverage information for a strain.
    
    Attributes:
        strain_name: Name/identifier of the strain
        subtype: Influenza subtype
        probability: Predicted binding probability
        binds: Binary binding prediction
        group: HA phylogenetic group (1 or 2)
    """
    strain_name: str
    subtype: str
    probability: float
    binds: bool
    group: int = 1  # HA group 1 or 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strain": self.strain_name,
            "subtype": self.subtype,
            "probability": self.probability,
            "binds": self.binds,
            "group": self.group,
        }


@dataclass
class CrossReactivityHeadConfig:
    """Configuration for Cross-Reactivity Head.
    
    Attributes:
        input_dim: Dimension of MAMMAL embeddings
        hidden_dims: Shared hidden layer dimensions
        strain_head_dim: Dimension of per-strain classification heads
        strains: List of strain names to predict
        dropout: Dropout probability
        threshold: Classification threshold for binding
        use_shared_backbone: Whether to share layers across strains
        multi_task_weighting: How to weight losses across strains
    """
    input_dim: int = 768
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    strain_head_dim: int = 64
    strains: List[str] = field(default_factory=lambda: DEFAULT_STRAIN_PANEL.copy())
    dropout: float = 0.1
    threshold: float = 0.5
    use_shared_backbone: bool = True
    multi_task_weighting: str = "equal"  # "equal", "uncertainty", "gradnorm"
    use_layer_norm: bool = True


@dataclass
class CrossReactivityPrediction:
    """Output from Cross-Reactivity Head prediction.
    
    Attributes:
        strain_predictions: Per-strain binding predictions
        breadth_score: Fraction of strains predicted to bind
        breadth_count: Number of strains predicted to bind
        total_strains: Total number of strains evaluated
        group1_coverage: Coverage of Group 1 HAs
        group2_coverage: Coverage of Group 2 HAs
        is_broadly_neutralizing: Whether antibody qualifies as bnAb
        embedding: Optional input embedding
    """
    strain_predictions: List[StrainCoverage]
    breadth_score: float
    breadth_count: int
    total_strains: int
    group1_coverage: float
    group2_coverage: float
    is_broadly_neutralizing: bool
    embedding: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "breadth_score": self.breadth_score,
            "breadth_count": self.breadth_count,
            "total_strains": self.total_strains,
            "group1_coverage": self.group1_coverage,
            "group2_coverage": self.group2_coverage,
            "is_broadly_neutralizing": self.is_broadly_neutralizing,
            "strain_predictions": [s.to_dict() for s in self.strain_predictions],
            "bound_strains": [s.strain_name for s in self.strain_predictions if s.binds],
        }
    
    def get_coverage_by_subtype(self) -> Dict[str, float]:
        """Get coverage grouped by subtype."""
        subtype_probs = {}
        subtype_counts = {}
        
        for pred in self.strain_predictions:
            subtype = pred.subtype
            if subtype not in subtype_probs:
                subtype_probs[subtype] = 0.0
                subtype_counts[subtype] = 0
            subtype_probs[subtype] += pred.probability
            subtype_counts[subtype] += 1
        
        return {k: v / subtype_counts[k] for k, v in subtype_probs.items()}


class CrossReactivityHead(nn.Module):
    """
    Cross-Reactivity Head for Multi-Strain Breadth Prediction.
    
    Status: NEW
    
    This head predicts antibody binding across multiple influenza strains,
    enabling identification of broadly neutralizing antibodies.
    
    Example:
        >>> config = CrossReactivityHeadConfig(strains=["H1N1", "H3N2", "H5N1"])
        >>> head = CrossReactivityHead(config)
        >>> embeddings = torch.randn(batch_size, 768)
        >>> prediction = head(embeddings)
        >>> print(f"Breadth: {prediction.breadth_score:.1%}")
    """
    
    def __init__(self, config: Optional[CrossReactivityHeadConfig] = None):
        """Initialize Cross-Reactivity Head.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        super().__init__()
        
        self.config = config or CrossReactivityHeadConfig()
        self.num_strains = len(self.config.strains)
        
        # Build strain metadata
        self._build_strain_metadata()
        
        # Layer normalization
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.config.input_dim)
        else:
            self.layer_norm = nn.Identity()
        
        # Shared backbone
        backbone_layers = []
        prev_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        self.backbone_out_dim = prev_dim
        
        # Per-strain classification heads
        self.strain_heads = nn.ModuleDict()
        for strain in self.config.strains:
            strain_key = self._sanitize_strain_name(strain)
            self.strain_heads[strain_key] = nn.Sequential(
                nn.Linear(self.backbone_out_dim, self.config.strain_head_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout / 2),
                nn.Linear(self.config.strain_head_dim, 2),  # Binary classification
            )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized CrossReactivityHead: {self.config.input_dim} → "
            f"{self.config.hidden_dims} → {self.num_strains} strain heads"
        )
    
    def _sanitize_strain_name(self, strain: str) -> str:
        """Convert strain name to valid module key."""
        return strain.replace("/", "_").replace(" ", "_").replace("-", "_")
    
    def _build_strain_metadata(self):
        """Build metadata for each strain (group, subtype)."""
        self.strain_metadata = {}
        
        for strain in self.config.strains:
            # Parse subtype from strain name
            subtype = self._parse_subtype(strain)
            group = self._get_ha_group(subtype)
            
            self.strain_metadata[strain] = {
                "subtype": subtype,
                "group": group,
            }
    
    def _parse_subtype(self, strain: str) -> str:
        """Parse subtype from strain name."""
        # Handle formats like "H1N1/California/04/2009" or just "H1N1"
        parts = strain.split("/")
        subtype = parts[0]
        
        # Handle "H1N1pdm09" style
        if "pdm" in subtype:
            subtype = subtype.split("pdm")[0]
        
        return subtype
    
    def _get_ha_group(self, subtype: str) -> int:
        """Get HA phylogenetic group (1 or 2)."""
        group1_subtypes = {"H1", "H2", "H5", "H6", "H8", "H9", "H11", "H12", "H13", "H16", "H17", "H18"}
        group2_subtypes = {"H3", "H4", "H7", "H10", "H14", "H15"}
        
        # Extract HA number
        ha = subtype.split("N")[0] if "N" in subtype else subtype
        
        if ha in group1_subtypes:
            return 1
        elif ha in group2_subtypes:
            return 2
        else:
            return 1  # Default to group 1
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[CrossReactivityPrediction, Dict[str, torch.Tensor]]:
        """Forward pass through Cross-Reactivity Head.
        
        Args:
            embeddings: MAMMAL embeddings of shape (batch_size, input_dim)
            return_dict: If True, return CrossReactivityPrediction object
            
        Returns:
            CrossReactivityPrediction or dict of per-strain logits
        """
        # Ensure embeddings are 2D
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)
        
        batch_size = embeddings.size(0)
        
        # Layer normalization
        normalized = self.layer_norm(embeddings)
        
        # Shared backbone
        backbone_out = self.backbone(normalized)
        
        # Per-strain predictions
        strain_logits = {}
        strain_probs = {}
        
        for strain in self.config.strains:
            strain_key = self._sanitize_strain_name(strain)
            logits = self.strain_heads[strain_key](backbone_out)
            probs = F.softmax(logits, dim=-1)
            
            strain_logits[strain] = logits
            strain_probs[strain] = probs[:, 1]  # Positive class probability
        
        if not return_dict:
            return strain_logits
        
        # Build prediction object (for first sample in batch)
        strain_predictions = []
        group1_bound = 0
        group1_total = 0
        group2_bound = 0
        group2_total = 0
        
        for strain in self.config.strains:
            prob = strain_probs[strain][0].item()
            binds = prob > self.config.threshold
            metadata = self.strain_metadata[strain]
            
            strain_predictions.append(StrainCoverage(
                strain_name=strain,
                subtype=metadata["subtype"],
                probability=prob,
                binds=binds,
                group=metadata["group"],
            ))
            
            # Track group coverage
            if metadata["group"] == 1:
                group1_total += 1
                if binds:
                    group1_bound += 1
            else:
                group2_total += 1
                if binds:
                    group2_bound += 1
        
        # Calculate breadth metrics
        breadth_count = sum(1 for s in strain_predictions if s.binds)
        breadth_score = breadth_count / self.num_strains
        group1_coverage = group1_bound / max(group1_total, 1)
        group2_coverage = group2_bound / max(group2_total, 1)
        
        # Broadly neutralizing criteria: binds ≥50% of strains AND covers both groups
        is_bnab = (breadth_score >= 0.5) and (group1_coverage > 0) and (group2_coverage > 0)
        
        return CrossReactivityPrediction(
            strain_predictions=strain_predictions,
            breadth_score=breadth_score,
            breadth_count=breadth_count,
            total_strains=self.num_strains,
            group1_coverage=group1_coverage,
            group2_coverage=group2_coverage,
            is_broadly_neutralizing=is_bnab,
            embedding=embeddings[0] if batch_size == 1 else None,
        )
    
    def predict(self, embeddings: torch.Tensor) -> CrossReactivityPrediction:
        """Make a single prediction."""
        self.eval()
        with torch.no_grad():
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            return self.forward(embeddings, return_dict=True)
    
    def predict_batch(self, embeddings: torch.Tensor) -> List[CrossReactivityPrediction]:
        """Make batch predictions."""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            strain_logits = self.forward(embeddings, return_dict=False)
            
            batch_size = embeddings.size(0)
            
            for i in range(batch_size):
                strain_predictions = []
                group1_bound = 0
                group1_total = 0
                group2_bound = 0
                group2_total = 0
                
                for strain in self.config.strains:
                    logits = strain_logits[strain][i]
                    prob = F.softmax(logits, dim=-1)[1].item()
                    binds = prob > self.config.threshold
                    metadata = self.strain_metadata[strain]
                    
                    strain_predictions.append(StrainCoverage(
                        strain_name=strain,
                        subtype=metadata["subtype"],
                        probability=prob,
                        binds=binds,
                        group=metadata["group"],
                    ))
                    
                    if metadata["group"] == 1:
                        group1_total += 1
                        if binds:
                            group1_bound += 1
                    else:
                        group2_total += 1
                        if binds:
                            group2_bound += 1
                
                breadth_count = sum(1 for s in strain_predictions if s.binds)
                breadth_score = breadth_count / self.num_strains
                group1_coverage = group1_bound / max(group1_total, 1)
                group2_coverage = group2_bound / max(group2_total, 1)
                is_bnab = (breadth_score >= 0.5) and (group1_coverage > 0) and (group2_coverage > 0)
                
                predictions.append(CrossReactivityPrediction(
                    strain_predictions=strain_predictions,
                    breadth_score=breadth_score,
                    breadth_count=breadth_count,
                    total_strains=self.num_strains,
                    group1_coverage=group1_coverage,
                    group2_coverage=group2_coverage,
                    is_broadly_neutralizing=is_bnab,
                ))
        
        return predictions
    
    def compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: Dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute multi-task loss across all strains.
        
        Args:
            embeddings: Input embeddings
            labels: Dictionary mapping strain names to binary labels
            reduction: Loss reduction method
            
        Returns:
            Combined loss tensor
        """
        strain_logits = self.forward(embeddings, return_dict=False)
        
        total_loss = 0.0
        num_strains_with_labels = 0
        
        for strain, logits in strain_logits.items():
            if strain in labels:
                strain_labels = labels[strain]
                loss = F.cross_entropy(logits, strain_labels, reduction=reduction)
                total_loss += loss
                num_strains_with_labels += 1
        
        if num_strains_with_labels > 0:
            total_loss = total_loss / num_strains_with_labels
        
        return total_loss
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get model summary string."""
        lines = [
            "=" * 60,
            "CROSS-REACTIVITY HEAD SUMMARY",
            "=" * 60,
            f"Input dimension: {self.config.input_dim}",
            f"Hidden dimensions: {self.config.hidden_dims}",
            f"Number of strains: {self.num_strains}",
            f"Strains:",
        ]
        
        for strain in self.config.strains:
            meta = self.strain_metadata[strain]
            lines.append(f"  - {strain} (Group {meta['group']})")
        
        lines.extend([
            f"Threshold: {self.config.threshold}",
            f"Total parameters: {self.get_num_parameters():,}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# =============================================================================
# SPECIALIZED CROSS-REACTIVITY HEADS
# =============================================================================

class SeasonalFluCrossReactivityHead(CrossReactivityHead):
    """Cross-reactivity head focused on seasonal influenza strains."""
    
    SEASONAL_STRAINS = [
        "H1N1/California/04/2009",
        "H1N1/Michigan/45/2015",
        "H1N1/Brisbane/02/2018",
        "H3N2/Hong_Kong/4801/2014",
        "H3N2/Kansas/14/2017",
        "H3N2/Darwin/9/2021",
    ]
    
    def __init__(self, input_dim: int = 768, **kwargs):
        config = CrossReactivityHeadConfig(
            input_dim=input_dim,
            strains=self.SEASONAL_STRAINS,
            **kwargs
        )
        super().__init__(config)


class PandemicPreparednessCrossReactivityHead(CrossReactivityHead):
    """Cross-reactivity head for pandemic preparedness assessment."""
    
    PANDEMIC_STRAINS = [
        # Historical pandemics
        "H1N1/Brevig_Mission/1/1918",
        "H2N2/Japan/305/1957",
        "H3N2/Hong_Kong/1/1968",
        "H1N1/California/04/2009",
        # Pandemic potential
        "H5N1/Vietnam/1194/2004",
        "H7N9/Anhui/1/2013",
        "H9N2/Hong_Kong/1073/99",
        "H10N8/Jiangxi/IPB13/2013",
    ]
    
    def __init__(self, input_dim: int = 768, **kwargs):
        config = CrossReactivityHeadConfig(
            input_dim=input_dim,
            strains=self.PANDEMIC_STRAINS,
            threshold=0.5,
            **kwargs
        )
        super().__init__(config)


if __name__ == "__main__":
    # Test the Cross-Reactivity Head
    print("Testing Cross-Reactivity Head...")
    
    # Simple test panel
    test_strains = ["H1N1/Test/1", "H3N2/Test/2", "H5N1/Test/3"]
    config = CrossReactivityHeadConfig(
        input_dim=768,
        strains=test_strains,
    )
    head = CrossReactivityHead(config)
    print(head.summary())
    
    batch_size = 4
    embeddings = torch.randn(batch_size, 768)
    
    # Single prediction
    single_pred = head.predict(embeddings[0])
    print(f"\nCross-reactivity prediction:")
    print(f"  Breadth: {single_pred.breadth_score:.1%} ({single_pred.breadth_count}/{single_pred.total_strains})")
    print(f"  Group 1 coverage: {single_pred.group1_coverage:.1%}")
    print(f"  Group 2 coverage: {single_pred.group2_coverage:.1%}")
    print(f"  Is bnAb: {single_pred.is_broadly_neutralizing}")
    print(f"  Per-strain:")
    for sp in single_pred.strain_predictions:
        print(f"    {sp.strain_name}: {sp.probability:.2f} ({'✓' if sp.binds else '✗'})")
    
    # Batch prediction
    batch_preds = head.predict_batch(embeddings)
    print(f"\nBatch predictions:")
    for i, pred in enumerate(batch_preds):
        print(f"  Sample {i}: breadth={pred.breadth_score:.1%}, bnAb={pred.is_broadly_neutralizing}")
    
    # Test loss computation
    labels = {
        "H1N1/Test/1": torch.tensor([1, 0, 1, 0]),
        "H3N2/Test/2": torch.tensor([1, 1, 0, 0]),
        "H5N1/Test/3": torch.tensor([0, 0, 1, 1]),
    }
    loss = head.compute_loss(embeddings, labels)
    print(f"\nLoss: {loss.item():.4f}")
    
    # Test specialized heads
    print("\n" + "=" * 60)
    print("Testing Seasonal Flu Head...")
    seasonal_head = SeasonalFluCrossReactivityHead(input_dim=768)
    print(f"Seasonal strains: {seasonal_head.num_strains}")
    
    print("\n✓ Cross-Reactivity Head tests passed!")
