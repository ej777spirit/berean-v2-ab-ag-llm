"""
Antibody-HA Binding Classification Task
========================================

Status: NEW
Source: berean_v2_refactor

Custom MAMMAL task for predicting antibody-hemagglutinin binding.
Follows the official MAMMAL task pattern from biomed-multi-alignment.

Official Components Used:
    - mammal.keys (OFFICIAL)
    - Task pattern from mammal.examples (OFFICIAL)
    
Based on:
    - DTI BindingDB example from biomed-multi-alignment
    - Protein-protein interaction pre-training task
    
Target Benchmarks (from Cleveland Clinic paper):
    - Lenient split: 0.91-0.92 AUROC
    - HA-exclusive: 0.90 AUROC
    - mAb-exclusive: 0.73 AUROC
    - mAb-cluster-exclusive: 0.63-0.66 AUROC

Usage:
    >>> from berean.tasks import AntibodyHABindingTask
    >>> task = AntibodyHABindingTask()
    >>> sample = task.data_preprocessing({
    ...     "heavy_chain": "EVQLVESGG...",
    ...     "light_chain": "DIQMTQSPS...",
    ...     "ha_sequence": "MKTIIALSYI...",
    ...     "binding_label": 1
    ... })
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# =============================================================================
# MAMMAL PROMPT TOKENS - From Official Documentation
# =============================================================================

# These match the official MAMMAL prompt syntax
TOKENIZER_TYPE_AA = "<@TOKENIZER-TYPE=AA>"
BINDING_AFFINITY_CLASS = "<BINDING_AFFINITY_CLASS>"
SENTINEL_ID_0 = "<SENTINEL_ID_0>"
MOLECULAR_ENTITY = "<MOLECULAR_ENTITY>"
MOLECULAR_ENTITY_ANTIBODY = "<MOLECULAR_ENTITY_ANTIBODY>"
MOLECULAR_ENTITY_GENERAL_PROTEIN = "<MOLECULAR_ENTITY_GENERAL_PROTEIN>"
SEQUENCE_NATURAL_START = "<SEQUENCE_NATURAL_START>"
SEQUENCE_NATURAL_END = "<SEQUENCE_NATURAL_END>"
EOS = "<EOS>"

# Label tokens for classification
LABEL_POSITIVE = "<1>"
LABEL_NEGATIVE = "<0>"


@dataclass
class AntibodyHABindingConfig:
    """Configuration for antibody-HA binding task.
    
    Attributes:
        max_antibody_length: Max length for antibody sequence (H+L)
        max_ha_length: Max length for HA sequence
        include_germline: Whether to include germline information
        sequence_format: How to format antibody ("concat" or "paired")
    """
    max_antibody_length: int = 600  # ~300 H + ~300 L
    max_ha_length: int = 600        # HA1+HA2 or HA head
    include_germline: bool = False
    sequence_format: str = "concat"  # "concat" or "paired"
    
    # Training settings
    label_smoothing: float = 0.0
    class_weights: Optional[List[float]] = None


class AntibodyHABindingTask:
    """MAMMAL task for antibody-HA binding classification.
    
    Status: NEW
    
    This task follows the official MAMMAL task pattern and can be used
    with the official training infrastructure.
    
    Attributes:
        config: Task configuration
        
    Example:
        >>> task = AntibodyHABindingTask()
        >>> 
        >>> # Preprocess a sample
        >>> sample = {
        ...     "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTY...",
        ...     "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYL...",
        ...     "ha_sequence": "MKTIIALSYILCLVFAQKIPGNDNSTATLCLGH...",
        ...     "binding_label": 1
        ... }
        >>> processed = task.data_preprocessing(sample)
        >>> 
        >>> # Get encoder input string
        >>> print(processed["encoder_inputs_str"][:100])
    """
    
    # Task identification
    TASK_NAME = "antibody_ha_binding"
    TASK_TYPE = "classification"
    NUM_CLASSES = 2
    
    def __init__(self, config: Optional[AntibodyHABindingConfig] = None):
        """Initialize task.
        
        Args:
            config: Task configuration (uses defaults if None)
        """
        self.config = config or AntibodyHABindingConfig()
        
    @staticmethod
    def data_preprocessing(
        sample_dict: Dict[str, Any],
        config: Optional[AntibodyHABindingConfig] = None,
    ) -> Dict[str, Any]:
        """Preprocess a sample for MAMMAL model input.
        
        Converts antibody-HA pair to MAMMAL prompt format following
        the official syntax for protein-protein binding prediction.
        
        Args:
            sample_dict: Dictionary with keys:
                - heavy_chain: Heavy chain amino acid sequence
                - light_chain: Light chain amino acid sequence
                - ha_sequence: HA amino acid sequence
                - binding_label: (optional) 0 or 1 for training
            config: Task configuration
            
        Returns:
            Modified sample_dict with "encoder_inputs_str" key added
        """
        config = config or AntibodyHABindingConfig()
        
        # Extract sequences
        heavy = sample_dict.get("heavy_chain", "")
        light = sample_dict.get("light_chain", "")
        ha_seq = sample_dict.get("ha_sequence", "")
        
        # Concatenate antibody chains
        # MAMMAL expects single sequence per molecular entity
        if config.sequence_format == "concat":
            antibody_seq = heavy + light
        else:
            # Alternative: use linker
            antibody_seq = heavy + "GGGGS" + light
        
        # Truncate if needed
        if len(antibody_seq) > config.max_antibody_length:
            antibody_seq = antibody_seq[:config.max_antibody_length]
        if len(ha_seq) > config.max_ha_length:
            ha_seq = ha_seq[:config.max_ha_length]
        
        # Construct MAMMAL prompt
        # Following official syntax from biomed-multi-alignment examples
        encoder_input = (
            f"{TOKENIZER_TYPE_AA}"
            f"{BINDING_AFFINITY_CLASS}"
            f"{SENTINEL_ID_0}"
            f"{MOLECULAR_ENTITY}{MOLECULAR_ENTITY_ANTIBODY}"
            f"{SEQUENCE_NATURAL_START}{antibody_seq}{SEQUENCE_NATURAL_END}"
            f"{MOLECULAR_ENTITY}{MOLECULAR_ENTITY_GENERAL_PROTEIN}"
            f"{SEQUENCE_NATURAL_START}{ha_seq}{SEQUENCE_NATURAL_END}"
            f"{EOS}"
        )
        
        sample_dict["encoder_inputs_str"] = encoder_input
        
        # Add label string for training
        if "binding_label" in sample_dict:
            label = sample_dict["binding_label"]
            sample_dict["label"] = int(label)
            sample_dict["label_str"] = LABEL_POSITIVE if label else LABEL_NEGATIVE
        
        return sample_dict
    
    @staticmethod
    def batch_preprocessing(
        batch: List[Dict[str, Any]],
        config: Optional[AntibodyHABindingConfig] = None,
    ) -> List[Dict[str, Any]]:
        """Preprocess a batch of samples.
        
        Args:
            batch: List of sample dictionaries
            config: Task configuration
            
        Returns:
            List of preprocessed samples
        """
        return [
            AntibodyHABindingTask.data_preprocessing(sample, config)
            for sample in batch
        ]
    
    def create_dataset_entry(
        self,
        heavy_chain: str,
        light_chain: str,
        ha_sequence: str,
        binding_label: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a dataset entry from raw inputs.
        
        Convenience method for creating properly formatted samples.
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            ha_sequence: HA sequence
            binding_label: Optional label (0 or 1)
            metadata: Optional additional metadata
            
        Returns:
            Preprocessed sample dictionary
        """
        sample = {
            "heavy_chain": heavy_chain,
            "light_chain": light_chain,
            "ha_sequence": ha_sequence,
        }
        
        if binding_label is not None:
            sample["binding_label"] = binding_label
            
        if metadata:
            sample["metadata"] = metadata
            
        return self.data_preprocessing(sample, self.config)
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss.
        
        Args:
            logits: Model output logits (batch, num_classes)
            labels: Ground truth labels (batch,)
            
        Returns:
            Loss tensor
        """
        if self.config.class_weights is not None:
            weight = torch.tensor(
                self.config.class_weights, 
                device=logits.device
            )
        else:
            weight = None
            
        loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.config.label_smoothing,
        )
        
        return loss_fn(logits, labels)
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            predictions: Model predictions (batch,) or (batch, num_classes)
            labels: Ground truth labels (batch,)
            
        Returns:
            Dictionary of metrics
        """
        if predictions.dim() > 1:
            # Convert logits to predictions
            probs = torch.softmax(predictions, dim=-1)
            pred_labels = predictions.argmax(dim=-1)
            positive_probs = probs[:, 1]
        else:
            pred_labels = predictions
            positive_probs = predictions.float()
        
        # Basic metrics
        accuracy = (pred_labels == labels).float().mean().item()
        
        # For AUROC, need probabilities
        try:
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
            
            labels_np = labels.cpu().numpy()
            probs_np = positive_probs.cpu().numpy()
            
            auroc = roc_auc_score(labels_np, probs_np)
            
            # PR-AUC
            precision, recall, _ = precision_recall_curve(labels_np, probs_np)
            pr_auc = auc(recall, precision)
            
        except Exception as e:
            logger.warning(f"Could not compute AUROC/PR-AUC: {e}")
            auroc = 0.0
            pr_auc = 0.0
        
        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "pr_auc": pr_auc,
        }
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "AntibodyHABindingTask":
        """Load task from YAML config file.
        
        Args:
            config_path: Path to config YAML
            
        Returns:
            Initialized task
        """
        import yaml
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = AntibodyHABindingConfig(**config_dict.get("task", {}))
        return cls(config)


def create_hydra_config(
    output_dir: str,
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    batch_size: int = 16,
    max_epochs: int = 20,
    learning_rate: float = 1e-5,
    warmup_steps: int = 100,
) -> Dict[str, Any]:
    """Create Hydra configuration for MAMMAL fine-tuning.
    
    Status: NEW
    
    Generates configuration compatible with official
    mammal.main_finetune training script.
    
    Args:
        output_dir: Directory for checkpoints and logs
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Optional path to test data
        batch_size: Batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate
        warmup_steps: Warmup steps for scheduler
        
    Returns:
        Configuration dictionary for Hydra
    """
    config = {
        "defaults": [
            {"override hydra/launcher": "basic"},
        ],
        
        "task": {
            "name": "antibody_ha_binding",
            "type": "classification",
            "num_classes": 2,
            "metrics": ["auroc", "accuracy", "pr_auc"],
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
            "preprocessing_fn": "berean.tasks.antibody_ha_binding.AntibodyHABindingTask.data_preprocessing",
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
                     "monitor": "val_auroc",
                     "mode": "max",
                     "save_top_k": 3,
                 }},
                {"class_path": "pytorch_lightning.callbacks.EarlyStopping",
                 "init_args": {
                     "monitor": "val_auroc",
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
        
        "scheduler": {
            "class_path": "transformers.get_linear_schedule_with_warmup",
            "init_args": {
                "num_warmup_steps": warmup_steps,
            },
        },
        
        "hydra": {
            "run": {
                "dir": output_dir,
            },
        },
    }
    
    return config


def save_config(config: Dict, path: str) -> None:
    """Save configuration to YAML file."""
    import yaml
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved config to {path}")


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def example_usage():
    """Demonstrate task usage."""
    print("=" * 60)
    print("AntibodyHABindingTask Example")
    print("=" * 60)
    
    # Create task
    task = AntibodyHABindingTask()
    
    # Example antibody-HA pair
    sample = {
        "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKR",
        "ha_sequence": "MKTIIALSYILCLVFAQKIPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFINEGFNWTGVTQNGGSNACKRGPDSGFFSRLNWLHKSEYKYPALNVTMPNNGKFDKLYIWGVHHPGTDNDQISLYAQASGRITVSTKRSQQTVSPNIGSRPRVRDIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAADLKSTQAAIDQINGKLNRLIGKTNEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTKKQLRENAEDMGNGCFKIYHKCDNACIGSIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISCFLLCVALLGFIMWACQKGNIRCNICI",
        "binding_label": 1,
    }
    
    # Preprocess
    processed = task.data_preprocessing(sample)
    
    print(f"\nOriginal heavy chain length: {len(sample['heavy_chain'])}")
    print(f"Original light chain length: {len(sample['light_chain'])}")
    print(f"Original HA length: {len(sample['ha_sequence'])}")
    print(f"\nEncoder input (first 200 chars):")
    print(processed["encoder_inputs_str"][:200])
    print(f"\nLabel: {processed['label']}")
    print(f"Label string: {processed['label_str']}")
    
    # Show config generation
    print("\n" + "=" * 60)
    print("Hydra Config for Fine-tuning")
    print("=" * 60)
    
    config = create_hydra_config(
        output_dir="./checkpoints/ab_ha_binding",
        train_path="./data/train.json",
        val_path="./data/val.json",
    )
    
    import yaml
    print(yaml.dump(config, default_flow_style=False)[:500])
    print("...")


if __name__ == "__main__":
    example_usage()
