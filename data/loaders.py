"""
Data Loaders Module

PyTorch DataLoader utilities for efficient batch processing
of antibody-antigen datasets.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np

logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for antibody-HA batches.
    
    Handles variable-length sequences by returning lists
    (tokenization happens in model forward pass).
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Collated batch dictionary
    """
    collated = {
        'antibody_sequences': [item['antibody_sequence'] for item in batch],
        'ha_sequences': [item['ha_sequence'] for item in batch],
        'antibody_ids': [item['antibody_id'] for item in batch],
        'ha_ids': [item['ha_id'] for item in batch]
    }
    
    # Handle labels (convert to tensors if present)
    if 'binding_label' in batch[0]:
        labels = [item['binding_label'] for item in batch]
        if None not in labels:
            collated['binding_labels'] = torch.tensor(labels, dtype=torch.long)
        else:
            collated['binding_labels'] = labels
            
    if 'hai_label' in batch[0]:
        labels = [item['hai_label'] for item in batch]
        if None not in labels:
            collated['hai_labels'] = torch.tensor(labels, dtype=torch.long)
        else:
            collated['hai_labels'] = labels
            
    if 'binding_value' in batch[0]:
        values = [item['binding_value'] for item in batch]
        if None not in values:
            collated['binding_values'] = torch.tensor(values, dtype=torch.float)
            
    if 'hai_value' in batch[0]:
        values = [item['hai_value'] for item in batch]
        if None not in values:
            collated['hai_values'] = torch.tensor(values, dtype=torch.float)
            
    return collated


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    indices: Optional[List[int]] = None,
    weighted_sampling: bool = False,
    task: str = "binding"
) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: AntibodyHADataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if weighted_sampling)
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        indices: If provided, only use these indices (for splits)
        weighted_sampling: Use weighted sampling for class balance
        task: Task for weighted sampling ('binding' or 'hai')
        
    Returns:
        PyTorch DataLoader
    """
    # Create subset if indices provided
    if indices is not None:
        dataset = Subset(dataset, indices)
        
    # Set up sampler
    sampler = None
    
    if weighted_sampling:
        # Calculate sample weights for class balance
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            sample_indices = dataset.indices
        else:
            base_dataset = dataset
            sample_indices = range(len(dataset))
            
        weights = []
        for idx in sample_indices:
            pair = base_dataset.pairs[idx]
            label = pair.binding_label if task == "binding" else pair.hai_label
            
            if label is None:
                weights.append(1.0)
            elif label == 1:
                # Oversample positive class (typically minority)
                weights.append(2.0)
            else:
                weights.append(1.0)
                
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        shuffle = False  # Mutually exclusive with sampler
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=False
    )
    
    return loader


def create_train_val_loaders(
    dataset,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    weighted_sampling: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.
    
    Args:
        dataset: AntibodyHADataset instance
        train_indices: Training indices
        val_indices: Validation indices
        batch_size: Batch size
        num_workers: Number of workers
        weighted_sampling: Use weighted sampling for training
        
    Returns:
        (train_loader, val_loader)
    """
    train_loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=not weighted_sampling,
        num_workers=num_workers,
        indices=train_indices,
        weighted_sampling=weighted_sampling
    )
    
    val_loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        indices=val_indices,
        weighted_sampling=False
    )
    
    return train_loader, val_loader


class DataModule:
    """Lightning-style data module for experiment management.
    
    Encapsulates all data loading logic for reproducible experiments.
    """
    
    def __init__(
        self,
        dataset,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: Optional[List[int]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        weighted_sampling: bool = True
    ):
        self.dataset = dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sampling = weighted_sampling
        
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if self._train_loader is None:
            self._train_loader = create_dataloader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=not self.weighted_sampling,
                num_workers=self.num_workers,
                indices=self.train_indices,
                weighted_sampling=self.weighted_sampling
            )
        return self._train_loader
        
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        if self._val_loader is None:
            self._val_loader = create_dataloader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                indices=self.val_indices,
                weighted_sampling=False
            )
        return self._val_loader
        
    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test DataLoader."""
        if self.test_indices is None:
            return None
            
        if self._test_loader is None:
            self._test_loader = create_dataloader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                indices=self.test_indices,
                weighted_sampling=False
            )
        return self._test_loader
        
    def get_class_weights(self, task: str = "binding") -> torch.Tensor:
        """Get class weights for loss function."""
        labels = []
        for idx in self.train_indices:
            pair = self.dataset.pairs[idx]
            label = pair.binding_label if task == "binding" else pair.hai_label
            if label is not None:
                labels.append(label)
                
        if not labels:
            return torch.tensor([1.0, 1.0])
            
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        # Inverse frequency
        if pos_count > 0 and neg_count > 0:
            weights = [len(labels) / (2 * neg_count), len(labels) / (2 * pos_count)]
        else:
            weights = [1.0, 1.0]
            
        return torch.tensor(weights)
