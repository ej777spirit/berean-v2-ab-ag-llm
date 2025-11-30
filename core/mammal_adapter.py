"""
MAMMAL Adapter - Wrapper for Official biomed-multi-alignment Package
=====================================================================

Status: NEW
Source: berean_v2_refactor

This module provides a high-level interface to the OFFICIAL MAMMAL components
from IBM's biomed-multi-alignment repository.

Official Components Used:
    - mammal.model.Mammal (OFFICIAL)
    - mammal.keys.* (OFFICIAL)
    - fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp (OFFICIAL)

Usage:
    >>> from berean.core import MAMMALAdapter
    >>> adapter = MAMMALAdapter()
    >>> prediction = adapter.predict_binding(antibody_seq, ha_seq)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default model identifier on HuggingFace
DEFAULT_MODEL = "ibm/biomed.omics.bl.sm.ma-ted-458m"

# =============================================================================
# MAMMAL PROMPT SYNTAX - Based on Official Documentation
# =============================================================================

# Special tokens from official MAMMAL prompt format
MAMMAL_TOKENS = {
    # Tokenizer type selectors
    "AA_TOKENIZER": "<@TOKENIZER-TYPE=AA>",
    "SMILES_TOKENIZER": "<@TOKENIZER-TYPE=SMILES>",
    
    # Task type tokens
    "BINDING_AFFINITY_CLASS": "<BINDING_AFFINITY_CLASS>",
    "BINDING_AFFINITY_VALUE": "<BINDING_AFFINITY_VALUE>",
    
    # Sentinel/separator tokens
    "SENTINEL_0": "<SENTINEL_ID_0>",
    "SENTINEL_1": "<SENTINEL_ID_1>",
    
    # Molecular entity markers
    "MOLECULAR_ENTITY": "<MOLECULAR_ENTITY>",
    "ENTITY_ANTIBODY": "<MOLECULAR_ENTITY_ANTIBODY>",
    "ENTITY_PROTEIN": "<MOLECULAR_ENTITY_GENERAL_PROTEIN>",
    "ENTITY_ANTIGEN": "<MOLECULAR_ENTITY_ANTIGEN>",
    
    # Sequence delimiters
    "SEQ_START": "<SEQUENCE_NATURAL_START>",
    "SEQ_END": "<SEQUENCE_NATURAL_END>",
    
    # Control tokens
    "EOS": "<EOS>",
    "PAD": "<PAD>",
    "UNK": "<UNK>",
}


@dataclass
class MAMMALConfig:
    """Configuration for MAMMAL adapter.
    
    Attributes:
        model_name: HuggingFace model identifier
        device: Computation device
        max_length: Maximum sequence length for tokenization
        task_type: Default task type (classification/regression)
    """
    model_name: str = DEFAULT_MODEL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 1024
    task_type: str = "classification"
    cache_dir: Optional[str] = None


def load_mammal_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    cache_dir: Optional[str] = None,
) -> Any:
    """Load official MAMMAL model from HuggingFace.
    
    Status: NEW (wrapper around OFFICIAL mammal.model.Mammal)
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        cache_dir: Optional cache directory
        
    Returns:
        Loaded MAMMAL model instance
        
    Example:
        >>> model = load_mammal_model()
        >>> model = load_mammal_model("ibm/biomed.omics.bl.sm.ma-ted-458m")
    """
    try:
        # OFFICIAL: Import from biomed-multi-alignment
        from mammal.model import Mammal
        
        logger.info(f"Loading MAMMAL model: {model_name}")
        
        model = Mammal.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"MAMMAL model loaded on {device}")
        return model
        
    except ImportError:
        logger.error(
            "biomed-multi-alignment not installed. Install with:\n"
            "pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]"
        )
        raise


def load_mammal_tokenizer(
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[str] = None,
) -> Any:
    """Load official MAMMAL tokenizer.
    
    Status: NEW (wrapper around OFFICIAL ModularTokenizerOp)
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Optional cache directory
        
    Returns:
        Loaded tokenizer instance
    """
    try:
        # OFFICIAL: Import from fuse-med-ml
        from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
        
        logger.info(f"Loading MAMMAL tokenizer: {model_name}")
        
        tokenizer = ModularTokenizerOp.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        return tokenizer
        
    except ImportError:
        logger.error(
            "fuse-med-ml not installed. Install with:\n"
            "pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]"
        )
        raise


def create_binding_prompt(
    antibody_sequence: str,
    antigen_sequence: str,
    antibody_type: str = "antibody",
) -> str:
    """Create MAMMAL prompt for binding classification.
    
    Status: NEW
    
    Constructs prompt following official MAMMAL syntax for protein-protein
    binding prediction task.
    
    Args:
        antibody_sequence: Antibody amino acid sequence (H+L concatenated or single chain)
        antigen_sequence: Antigen (e.g., HA) amino acid sequence
        antibody_type: Type marker ("antibody" or "protein")
        
    Returns:
        Formatted prompt string for MAMMAL model
        
    Example:
        >>> prompt = create_binding_prompt(
        ...     "EVQLVESGGGLVQPGG...",
        ...     "MKTIIALSYILCLVFA..."
        ... )
    """
    # Select entity type for antibody
    if antibody_type == "antibody":
        ab_entity = MAMMAL_TOKENS["ENTITY_ANTIBODY"]
    else:
        ab_entity = MAMMAL_TOKENS["ENTITY_PROTEIN"]
    
    # Construct prompt following official syntax
    prompt = (
        f"{MAMMAL_TOKENS['AA_TOKENIZER']}"
        f"{MAMMAL_TOKENS['BINDING_AFFINITY_CLASS']}"
        f"{MAMMAL_TOKENS['SENTINEL_0']}"
        f"{MAMMAL_TOKENS['MOLECULAR_ENTITY']}{ab_entity}"
        f"{MAMMAL_TOKENS['SEQ_START']}{antibody_sequence}{MAMMAL_TOKENS['SEQ_END']}"
        f"{MAMMAL_TOKENS['MOLECULAR_ENTITY']}{MAMMAL_TOKENS['ENTITY_PROTEIN']}"
        f"{MAMMAL_TOKENS['SEQ_START']}{antigen_sequence}{MAMMAL_TOKENS['SEQ_END']}"
        f"{MAMMAL_TOKENS['EOS']}"
    )
    
    return prompt


def create_regression_prompt(
    antibody_sequence: str,
    antigen_sequence: str,
    task_name: str = "binding_affinity",
) -> str:
    """Create MAMMAL prompt for regression tasks (HAI, IC50).
    
    Status: NEW
    
    Args:
        antibody_sequence: Antibody amino acid sequence
        antigen_sequence: Antigen amino acid sequence
        task_name: Task identifier
        
    Returns:
        Formatted prompt string for regression prediction
    """
    prompt = (
        f"{MAMMAL_TOKENS['AA_TOKENIZER']}"
        f"{MAMMAL_TOKENS['BINDING_AFFINITY_VALUE']}"
        f"{MAMMAL_TOKENS['SENTINEL_0']}"
        f"{MAMMAL_TOKENS['MOLECULAR_ENTITY']}{MAMMAL_TOKENS['ENTITY_ANTIBODY']}"
        f"{MAMMAL_TOKENS['SEQ_START']}{antibody_sequence}{MAMMAL_TOKENS['SEQ_END']}"
        f"{MAMMAL_TOKENS['MOLECULAR_ENTITY']}{MAMMAL_TOKENS['ENTITY_PROTEIN']}"
        f"{MAMMAL_TOKENS['SEQ_START']}{antigen_sequence}{MAMMAL_TOKENS['SEQ_END']}"
        f"{MAMMAL_TOKENS['EOS']}"
    )
    
    return prompt


def create_antibody_ha_prompt(
    heavy_chain: str,
    light_chain: str,
    ha_sequence: str,
    task: str = "binding",
) -> str:
    """Create prompt specifically for antibody-HA binding prediction.
    
    Status: NEW
    
    This is the main function for BEREAN's influenza application.
    Concatenates heavy and light chains as per MAMMAL antibody convention.
    
    Args:
        heavy_chain: Heavy chain amino acid sequence
        light_chain: Light chain amino acid sequence
        ha_sequence: Hemagglutinin amino acid sequence
        task: "binding" for classification, "hai" or "ic50" for regression
        
    Returns:
        Formatted MAMMAL prompt
    """
    # Concatenate chains (MAMMAL expects single sequence per entity)
    # Using standard linker or simple concatenation
    antibody_seq = heavy_chain + light_chain
    
    if task == "binding":
        return create_binding_prompt(antibody_seq, ha_sequence)
    else:
        return create_regression_prompt(antibody_seq, ha_sequence, task)


class MAMMALAdapter:
    """High-level adapter for official MAMMAL model.
    
    Status: NEW (wraps OFFICIAL components)
    
    This class provides a simplified interface for antibody-antigen
    binding prediction using the official biomed-multi-alignment package.
    
    Attributes:
        model: Official MAMMAL model instance
        tokenizer: Official ModularTokenizerOp instance
        config: Adapter configuration
        
    Example:
        >>> adapter = MAMMALAdapter()
        >>> result = adapter.predict_binding(
        ...     heavy_chain="EVQLVESGGGLVQPGG...",
        ...     light_chain="DIQMTQSPSSLSASV...",
        ...     ha_sequence="MKTIIALSYILCLVFA..."
        ... )
        >>> print(f"Binding probability: {result['probability']:.3f}")
    """
    
    def __init__(
        self,
        config: Optional[MAMMALConfig] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize MAMMAL adapter.
        
        Args:
            config: Configuration object (uses defaults if None)
            model: Pre-loaded model (loads from HF if None)
            tokenizer: Pre-loaded tokenizer (loads from HF if None)
        """
        self.config = config or MAMMALConfig()
        
        # Load official MAMMAL components
        self.model = model or load_mammal_model(
            self.config.model_name,
            self.config.device,
            self.config.cache_dir,
        )
        
        self.tokenizer = tokenizer or load_mammal_tokenizer(
            self.config.model_name,
            self.config.cache_dir,
        )
        
        self._device = self.config.device
        logger.info("MAMMALAdapter initialized")
    
    def predict_binding(
        self,
        heavy_chain: str,
        light_chain: str,
        ha_sequence: str,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """Predict antibody-HA binding.
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            ha_sequence: HA sequence
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with prediction results:
                - probability: Binding probability
                - prediction: Binary prediction (0/1)
                - logits: Raw model output
                - attention: Attention weights (if requested)
        """
        try:
            # OFFICIAL: Import keys
            from mammal.keys import ENCODER_INPUTS_STR
            
            # Create prompt using our helper
            prompt = create_antibody_ha_prompt(
                heavy_chain, light_chain, ha_sequence, task="binding"
            )
            
            # Create sample dict following official pattern
            sample_dict = {
                ENCODER_INPUTS_STR: prompt
            }
            
            # Tokenize using official tokenizer
            self.tokenizer(
                sample_dict,
                key_in=ENCODER_INPUTS_STR,
                key_out_tokens_ids="encoder_input_ids",
                key_out_attention_mask="encoder_attention_mask",
                max_len=self.config.max_length,
            )
            
            # Run inference using official model
            with torch.no_grad():
                batch_dict = self.model.generate(
                    [sample_dict],
                    output_scores=True,
                    output_attentions=return_attention,
                )
            
            # Parse output
            result = self._parse_binding_output(batch_dict[0])
            
            if return_attention and "attentions" in batch_dict[0]:
                result["attention"] = batch_dict[0]["attentions"]
            
            return result
            
        except ImportError:
            raise ImportError(
                "Official MAMMAL package required. Install with:\n"
                "pip install git+https://github.com/BiomedSciAI/biomed-multi-alignment.git#egg=mammal[examples]"
            )
    
    def predict_binding_batch(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Batch prediction for multiple antibody-HA pairs.
        
        Args:
            samples: List of dicts with keys: heavy_chain, light_chain, ha_sequence
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        from mammal.keys import ENCODER_INPUTS_STR
        
        results = []
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            # Create sample dicts
            sample_dicts = []
            for sample in batch:
                prompt = create_antibody_ha_prompt(
                    sample["heavy_chain"],
                    sample["light_chain"],
                    sample["ha_sequence"],
                )
                sample_dict = {ENCODER_INPUTS_STR: prompt}
                
                self.tokenizer(
                    sample_dict,
                    key_in=ENCODER_INPUTS_STR,
                    key_out_tokens_ids="encoder_input_ids",
                    key_out_attention_mask="encoder_attention_mask",
                    max_len=self.config.max_length,
                )
                sample_dicts.append(sample_dict)
            
            # Batch inference
            with torch.no_grad():
                batch_results = self.model.generate(sample_dicts)
            
            # Parse results
            for result_dict in batch_results:
                results.append(self._parse_binding_output(result_dict))
        
        return results
    
    def predict_hai(
        self,
        heavy_chain: str,
        light_chain: str,
        ha_sequence: str,
    ) -> Dict[str, Any]:
        """Predict HAI titer (regression).
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            ha_sequence: HA sequence
            
        Returns:
            Dictionary with predicted HAI titer
        """
        from mammal.keys import ENCODER_INPUTS_STR
        
        prompt = create_antibody_ha_prompt(
            heavy_chain, light_chain, ha_sequence, task="hai"
        )
        
        sample_dict = {ENCODER_INPUTS_STR: prompt}
        
        self.tokenizer(
            sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids="encoder_input_ids",
            key_out_attention_mask="encoder_attention_mask",
            max_len=self.config.max_length,
        )
        
        with torch.no_grad():
            batch_dict = self.model.generate([sample_dict])
        
        return self._parse_regression_output(batch_dict[0])
    
    def _parse_binding_output(self, output_dict: Dict) -> Dict[str, Any]:
        """Parse binding classification output."""
        # Extract prediction from generated tokens
        # MAMMAL outputs task-specific tokens for classification
        
        result = {
            "probability": 0.5,  # Default
            "prediction": 0,
            "logits": None,
        }
        
        if "scores" in output_dict:
            scores = output_dict["scores"]
            if isinstance(scores, torch.Tensor):
                # Apply softmax for probability
                probs = torch.softmax(scores, dim=-1)
                result["probability"] = probs[..., 1].item()  # Positive class
                result["prediction"] = int(result["probability"] > 0.5)
                result["logits"] = scores.cpu().numpy()
        
        if "generated_text" in output_dict:
            # Parse text output (some tasks use text generation)
            text = output_dict["generated_text"]
            if "1" in text or "positive" in text.lower() or "yes" in text.lower():
                result["prediction"] = 1
                result["probability"] = 0.9  # High confidence placeholder
            elif "0" in text or "negative" in text.lower() or "no" in text.lower():
                result["prediction"] = 0
                result["probability"] = 0.1
        
        return result
    
    def _parse_regression_output(self, output_dict: Dict) -> Dict[str, Any]:
        """Parse regression output (HAI, IC50)."""
        result = {
            "value": None,
            "raw_output": output_dict.get("generated_text", ""),
        }
        
        if "scores" in output_dict:
            scores = output_dict["scores"]
            if isinstance(scores, torch.Tensor):
                result["value"] = scores.item()
        
        # Try to parse from generated text
        if result["value"] is None and "generated_text" in output_dict:
            try:
                # Extract numeric value from text
                text = output_dict["generated_text"]
                import re
                numbers = re.findall(r"[-+]?\d*\.?\d+", text)
                if numbers:
                    result["value"] = float(numbers[0])
            except (ValueError, IndexError):
                pass
        
        return result
    
    def get_embeddings(
        self,
        heavy_chain: str,
        light_chain: str,
        ha_sequence: str,
        layer: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Extract embeddings from MAMMAL model.
        
        Status: NEW (extends OFFICIAL model usage)
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            ha_sequence: HA sequence
            layer: Which layer to extract from (-1 for last)
            
        Returns:
            Dictionary with embedding tensors
        """
        from mammal.keys import ENCODER_INPUTS_STR
        
        prompt = create_antibody_ha_prompt(
            heavy_chain, light_chain, ha_sequence
        )
        
        sample_dict = {ENCODER_INPUTS_STR: prompt}
        
        self.tokenizer(
            sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids="encoder_input_ids",
            key_out_attention_mask="encoder_attention_mask",
            max_len=self.config.max_length,
        )
        
        # Get hidden states
        with torch.no_grad():
            # Access underlying transformer for embeddings
            input_ids = torch.tensor([sample_dict["encoder_input_ids"]]).to(self._device)
            attention_mask = torch.tensor([sample_dict["encoder_attention_mask"]]).to(self._device)
            
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            hidden_states = outputs.hidden_states
            
        return {
            "last_hidden_state": hidden_states[-1],
            "pooled": hidden_states[-1].mean(dim=1),
            "layer_embeddings": hidden_states[layer] if layer != -1 else hidden_states[-1],
        }
    
    @property
    def device(self) -> str:
        """Get current device."""
        return self._device
    
    def to(self, device: str) -> "MAMMALAdapter":
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self


# =============================================================================
# FINE-TUNING SUPPORT (Uses Official MAMMAL Infrastructure)
# =============================================================================

def create_finetuning_config(
    output_dir: str,
    train_data: str,
    val_data: str,
    task_type: str = "classification",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
) -> Dict[str, Any]:
    """Create configuration for MAMMAL fine-tuning.
    
    Status: NEW (generates config for OFFICIAL training infrastructure)
    
    This creates a Hydra-compatible configuration dict for fine-tuning
    the MAMMAL model using the official mammal.main_finetune script.
    
    Args:
        output_dir: Directory for checkpoints
        train_data: Path to training data
        val_data: Path to validation data
        task_type: "classification" or "regression"
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Configuration dictionary
    """
    config = {
        "model": {
            "pretrained_kwargs": {
                "pretrained_model_name_or_path": DEFAULT_MODEL,
            }
        },
        "data": {
            "train_path": train_data,
            "val_path": val_data,
            "batch_size": batch_size,
        },
        "trainer": {
            "max_epochs": epochs,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "default_root_dir": output_dir,
        },
        "optimizer": {
            "lr": learning_rate,
        },
        "task": {
            "type": task_type,
        }
    }
    
    return config


def save_hydra_config(config: Dict, config_path: str) -> None:
    """Save configuration in Hydra YAML format.
    
    Args:
        config: Configuration dictionary
        config_path: Output path for config.yaml
    """
    import yaml
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved config to {config_path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_mammal_installation() -> bool:
    """Check if official MAMMAL package is installed.
    
    Returns:
        True if installed, False otherwise
    """
    try:
        import mammal
        from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
        return True
    except ImportError:
        return False


def get_mammal_version() -> Optional[str]:
    """Get version of installed MAMMAL package."""
    try:
        import mammal
        return getattr(mammal, '__version__', 'unknown')
    except ImportError:
        return None


def print_adapter_info():
    """Print information about the MAMMAL adapter."""
    print("=" * 60)
    print("BEREAN MAMMAL Adapter")
    print("=" * 60)
    print(f"\nStatus: NEW (wraps OFFICIAL components)")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"\nOfficial components used:")
    print("  - mammal.model.Mammal (OFFICIAL)")
    print("  - mammal.keys (OFFICIAL)")
    print("  - fuse.data.tokenizers.modular_tokenizer.op.ModularTokenizerOp (OFFICIAL)")
    print(f"\nMAMMAL installed: {check_mammal_installation()}")
    if check_mammal_installation():
        print(f"MAMMAL version: {get_mammal_version()}")
    print("=" * 60)


if __name__ == "__main__":
    print_adapter_info()
