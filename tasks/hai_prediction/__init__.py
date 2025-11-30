"""
HAI Titer Prediction Task (Regression)
======================================

Status: NEW
Source: berean_v2_refactor

Custom MAMMAL task for predicting HAI (Hemagglutination Inhibition) titers.
This is a regression task following the MAMMAL pattern.

Official Components Used:
    - mammal.keys (OFFICIAL)
    - Task pattern from mammal.examples (OFFICIAL)

HAI titers indicate antibody neutralization strength:
    - Titer ≥40: Considered protective
    - Higher titers indicate stronger neutralization
    - Values typically: 10, 20, 40, 80, 160, 320, 640, 1280, 2560+

From Cleveland Clinic paper:
    - 5,035 HAI pairs (11% positive, titer ≥40)
    - Can be used for both classification (positive/negative) and regression

Usage:
    >>> from berean.tasks import HAIPredictionTask
    >>> task = HAIPredictionTask()
    >>> sample = task.data_preprocessing({
    ...     "heavy_chain": "EVQLVESGG...",
    ...     "light_chain": "DIQMTQSPS...",
    ...     "ha_sequence": "MKTIIALSYI...",
    ...     "hai_titer": 160
    ... })
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# =============================================================================
# MAMMAL PROMPT TOKENS - From Official Documentation
# =============================================================================

TOKENIZER_TYPE_AA = "<@TOKENIZER-TYPE=AA>"
BINDING_AFFINITY_VALUE = "<BINDING_AFFINITY_VALUE>"  # Regression task token
SENTINEL_ID_0 = "<SENTINEL_ID_0>"
MOLECULAR_ENTITY = "<MOLECULAR_ENTITY>"
MOLECULAR_ENTITY_ANTIBODY = "<MOLECULAR_ENTITY_ANTIBODY>"
MOLECULAR_ENTITY_GENERAL_PROTEIN = "<MOLECULAR_ENTITY_GENERAL_PROTEIN>"
SEQUENCE_NATURAL_START = "<SEQUENCE_NATURAL_START>"
SEQUENCE_NATURAL_END = "<SEQUENCE_NATURAL_END>"
EOS = "<EOS>"


@dataclass
class HAIPredictionConfig:
    """Configuration for HAI prediction task.
    
    Attributes:
        max_antibody_length: Max length for antibody sequence
        max_ha_length: Max length for HA sequence
        log_transform: Whether to log-transform HAI titers
        normalize: Whether to normalize titer values
        titer_min: Minimum titer value for normalization
        titer_max: Maximum titer value for normalization
        classification_threshold: HAI titer threshold for positive (default 40)
    """
    max_antibody_length: int = 600
    max_ha_length: int = 600
    log_transform: bool = True      # log2 transform is standard for titers
    normalize: bool = True
    titer_min: float = 5.0          # log2(10) ≈ 3.3, but allow some margin
    titer_max: float = 2560.0       # Typical max titer
    classification_threshold: float = 40.0  # Standard protective threshold


class HAIPredictionTask:
    """MAMMAL task for HAI titer regression prediction.
    
    Status: NEW
    
    Predicts antibody HAI (Hemagglutination Inhibition) titers,
    which indicate neutralization strength against influenza.
    
    Attributes:
        config: Task configuration
        
    Example:
        >>> task = HAIPredictionTask()
        >>> sample = {
        ...     "heavy_chain": "EVQLVESGGGLVQPGG...",
        ...     "light_chain": "DIQMTQSPSSLSASV...",
        ...     "ha_sequence": "MKTIIALSYILCLVF...",
        ...     "hai_titer": 160
        ... }
        >>> processed = task.data_preprocessing(sample)
    """
    
    # Task identification
    TASK_NAME = "hai_prediction"
    TASK_TYPE = "regression"
    
    def __init__(self, config: Optional[HAIPredictionConfig] = None):
        """Initialize task.
        
        Args:
            config: Task configuration (uses defaults if None)
        """
        self.config = config or HAIPredictionConfig()
        
    @staticmethod
    def data_preprocessing(
        sample_dict: Dict[str, Any],
        config: Optional[HAIPredictionConfig] = None,
    ) -> Dict[str, Any]:
        """Preprocess a sample for MAMMAL model input.
        
        Converts antibody-HA pair to MAMMAL prompt format for
        regression prediction of HAI titers.
        
        Args:
            sample_dict: Dictionary with keys:
                - heavy_chain: Heavy chain amino acid sequence
                - light_chain: Light chain amino acid sequence
                - ha_sequence: HA amino acid sequence
                - hai_titer: (optional) HAI titer for training
            config: Task configuration
            
        Returns:
            Modified sample_dict with "encoder_inputs_str" key added
        """
        config = config or HAIPredictionConfig()
        
        # Extract sequences
        heavy = sample_dict.get("heavy_chain", "")
        light = sample_dict.get("light_chain", "")
        ha_seq = sample_dict.get("ha_sequence", "")
        
        # Concatenate antibody chains
        antibody_seq = heavy + light
        
        # Truncate if needed
        if len(antibody_seq) > config.max_antibody_length:
            antibody_seq = antibody_seq[:config.max_antibody_length]
        if len(ha_seq) > config.max_ha_length:
            ha_seq = ha_seq[:config.max_ha_length]
        
        # Construct MAMMAL prompt for regression
        encoder_input = (
            f"{TOKENIZER_TYPE_AA}"
            f"{BINDING_AFFINITY_VALUE}"  # Regression task
            f"{SENTINEL_ID_0}"
            f"{MOLECULAR_ENTITY}{MOLECULAR_ENTITY_ANTIBODY}"
            f"{SEQUENCE_NATURAL_START}{antibody_seq}{SEQUENCE_NATURAL_END}"
            f"{MOLECULAR_ENTITY}{MOLECULAR_ENTITY_GENERAL_PROTEIN}"
            f"{SEQUENCE_NATURAL_START}{ha_seq}{SEQUENCE_NATURAL_END}"
            f"{EOS}"
        )
        
        sample_dict["encoder_inputs_str"] = encoder_input
        
        # Process label for training
        if "hai_titer" in sample_dict:
            titer = float(sample_dict["hai_titer"])
            
            # Log transform (standard for titers)
            if config.log_transform:
                titer = math.log2(max(titer, 1.0))  # Avoid log(0)
                
            # Normalize
            if config.normalize:
                if config.log_transform:
                    min_val = math.log2(max(config.titer_min, 1.0))
                    max_val = math.log2(config.titer_max)
                else:
                    min_val = config.titer_min
                    max_val = config.titer_max
                    
                titer = (titer - min_val) / (max_val - min_val)
                titer = max(0.0, min(1.0, titer))  # Clamp to [0, 1]
            
            sample_dict["target"] = titer
            sample_dict["target_raw"] = sample_dict["hai_titer"]
            
            # Also create classification label for auxiliary task
            is_protective = sample_dict["hai_titer"] >= config.classification_threshold
            sample_dict["protective_label"] = int(is_protective)
        
        return sample_dict
    
    @staticmethod
    def batch_preprocessing(
        batch: List[Dict[str, Any]],
        config: Optional[HAIPredictionConfig] = None,
    ) -> List[Dict[str, Any]]:
        """Preprocess a batch of samples.
        
        Args:
            batch: List of sample dictionaries
            config: Task configuration
            
        Returns:
            List of preprocessed samples
        """
        return [
            HAIPredictionTask.data_preprocessing(sample, config)
            for sample in batch
        ]
    
    def inverse_transform(self, normalized_value: float) -> float:
        """Convert normalized prediction back to HAI titer.
        
        Args:
            normalized_value: Model output (normalized, log-transformed)
            
        Returns:
            Predicted HAI titer
        """
        if self.config.normalize:
            if self.config.log_transform:
                min_val = math.log2(max(self.config.titer_min, 1.0))
                max_val = math.log2(self.config.titer_max)
            else:
                min_val = self.config.titer_min
                max_val = self.config.titer_max
                
            value = normalized_value * (max_val - min_val) + min_val
        else:
            value = normalized_value
            
        if self.config.log_transform:
            value = 2 ** value
            
        return value
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regression loss.
        
        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth targets (batch,)
            
        Returns:
            Loss tensor
        """
        loss_fn = nn.MSELoss()
        return loss_fn(predictions.squeeze(), targets.squeeze())
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute regression metrics.
        
        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth targets (batch,)
            
        Returns:
            Dictionary of metrics
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # MSE
        mse = ((predictions - targets) ** 2).mean().item()
        
        # RMSE
        rmse = math.sqrt(mse)
        
        # MAE
        mae = (predictions - targets).abs().mean().item()
        
        # Pearson correlation
        try:
            pred_mean = predictions.mean()
            target_mean = targets.mean()
            
            pred_centered = predictions - pred_mean
            target_centered = targets - target_mean
            
            numerator = (pred_centered * target_centered).sum()
            denominator = (pred_centered ** 2).sum().sqrt() * (target_centered ** 2).sum().sqrt()
            
            pearson_r = (numerator / (denominator + 1e-8)).item()
        except Exception:
            pearson_r = 0.0
        
        # R² (coefficient of determination)
        ss_res = ((predictions - targets) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r_squared = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "pearson_r": pearson_r,
            "r_squared": r_squared,
        }
    
    def compute_classification_metrics(
        self,
        predictions: torch.Tensor,
        targets_raw: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute classification metrics using protective threshold.
        
        Converts regression predictions to binary classification
        (protective vs non-protective) for additional evaluation.
        
        Args:
            predictions: Model predictions (normalized)
            targets_raw: Raw HAI titer values
            
        Returns:
            Classification metrics
        """
        # Convert predictions back to titers
        pred_titers = torch.tensor([
            self.inverse_transform(p.item()) 
            for p in predictions.squeeze()
        ])
        
        # Binary classification
        pred_protective = (pred_titers >= self.config.classification_threshold).float()
        actual_protective = (targets_raw >= self.config.classification_threshold).float()
        
        # Accuracy
        accuracy = (pred_protective == actual_protective).float().mean().item()
        
        # AUROC if possible
        try:
            from sklearn.metrics import roc_auc_score
            
            auroc = roc_auc_score(
                actual_protective.cpu().numpy(),
                pred_titers.cpu().numpy()  # Use continuous predictions
            )
        except Exception:
            auroc = 0.0
        
        return {
            "protective_accuracy": accuracy,
            "protective_auroc": auroc,
        }


def create_hydra_config(
    output_dir: str,
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    batch_size: int = 16,
    max_epochs: int = 20,
    learning_rate: float = 1e-5,
) -> Dict[str, Any]:
    """Create Hydra configuration for HAI regression fine-tuning.
    
    Status: NEW
    
    Args:
        output_dir: Directory for checkpoints and logs
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Optional path to test data
        batch_size: Batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate
        
    Returns:
        Configuration dictionary
    """
    config = {
        "task": {
            "name": "hai_prediction",
            "type": "regression",
            "metrics": ["mse", "rmse", "mae", "pearson_r", "r_squared"],
        },
        
        "model": {
            "pretrained_kwargs": {
                "pretrained_model_name_or_path": "ibm/biomed.omics.bl.sm.ma-ted-458m",
            },
        },
        
        "data": {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "batch_size": batch_size,
            "num_workers": 4,
            "preprocessing_fn": "berean.tasks.hai_prediction.HAIPredictionTask.data_preprocessing",
        },
        
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "auto",
            "devices": 1,
            "precision": "16-mixed",
            "default_root_dir": output_dir,
            "callbacks": [
                {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                 "init_args": {
                     "monitor": "val_pearson_r",
                     "mode": "max",
                     "save_top_k": 3,
                 }},
                {"class_path": "pytorch_lightning.callbacks.EarlyStopping",
                 "init_args": {
                     "monitor": "val_pearson_r",
                     "patience": 5,
                     "mode": "max",
                 }},
            ],
        },
        
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": learning_rate,
                "weight_decay": 0.01,
            },
        },
    }
    
    return config


def example_usage():
    """Demonstrate task usage."""
    print("=" * 60)
    print("HAIPredictionTask Example")
    print("=" * 60)
    
    task = HAIPredictionTask()
    
    # Example with HAI titer
    sample = {
        "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKR",
        "ha_sequence": "MKTIIALSYILCLVFAQKIPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEF",
        "hai_titer": 160,
    }
    
    processed = task.data_preprocessing(sample)
    
    print(f"\nRaw HAI titer: {sample['hai_titer']}")
    print(f"Normalized target: {processed['target']:.4f}")
    print(f"Protective (≥40): {processed['protective_label']}")
    
    # Test inverse transform
    reconstructed = task.inverse_transform(processed['target'])
    print(f"Reconstructed titer: {reconstructed:.1f}")
    
    print(f"\nEncoder input (first 150 chars):")
    print(processed["encoder_inputs_str"][:150])


if __name__ == "__main__":
    example_usage()
