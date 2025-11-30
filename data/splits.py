"""
Data Split Strategies Module

Implements the four split strategies from Barkan et al. (2025):

1. Lenient (random pairs): Random train/test split
   - Expected AUROC: 0.91-0.92
   - Tests: General prediction capability
   
2. HA-exclusive (novel strains): Hold out entire HA sequences
   - Expected AUROC: 0.90
   - Tests: Generalization to unseen viral strains
   
3. mAb-exclusive (novel antibodies): Hold out entire antibody sequences
   - Expected AUROC: 0.73
   - Tests: Prediction for novel antibodies (key challenge!)
   
4. mAb-cluster-exclusive: Hold out antibody clusters at 50% sequence identity
   - Expected AUROC: 0.63-0.66
   - Tests: Prediction for truly divergent antibodies

Critical insight from paper:
- Performance drops significantly for novel antibodies (0.91 → 0.63-0.73)
- This is the key limitation BEREAN aims to address
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import logging
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


class SplitStrategy(Enum):
    """Split strategy types from the paper."""
    LENIENT = "lenient"                    # Random pairs
    HA_EXCLUSIVE = "ha_exclusive"          # Novel strains
    MAB_EXCLUSIVE = "mab_exclusive"        # Novel antibodies
    MAB_CLUSTER_EXCLUSIVE = "mab_cluster"  # Divergent antibodies (50% identity)


@dataclass
class SplitConfig:
    """Configuration for data splitting.
    
    Attributes:
        strategy: Split strategy to use
        n_folds: Number of cross-validation folds
        test_size: Fraction for test set (if not using CV)
        cluster_threshold: Sequence identity threshold for clustering (default 0.5)
        seed: Random seed for reproducibility
        stratify: Whether to stratify by label
    """
    strategy: SplitStrategy = SplitStrategy.LENIENT
    n_folds: int = 5
    test_size: float = 0.2
    cluster_threshold: float = 0.5  # 50% sequence identity
    seed: int = 42
    stratify: bool = True


class DataSplitter:
    """Implements the four split strategies from the paper.
    
    Usage:
        splitter = DataSplitter(config)
        train_idx, test_idx = splitter.split(dataset)
        
        # Or for cross-validation
        for fold, (train_idx, test_idx) in enumerate(splitter.cv_split(dataset)):
            ...
    """
    
    def __init__(self, config: SplitConfig):
        self.config = config
        np.random.seed(config.seed)
        
    def split(
        self,
        dataset,
        return_indices: bool = True
    ) -> Tuple[List[int], List[int]]:
        """Create a single train/test split.
        
        Args:
            dataset: AntibodyHADataset instance
            return_indices: If True, return indices. Otherwise return pair lists.
            
        Returns:
            (train_indices, test_indices) or (train_pairs, test_pairs)
        """
        if self.config.strategy == SplitStrategy.LENIENT:
            train_idx, test_idx = self._lenient_split(dataset)
        elif self.config.strategy == SplitStrategy.HA_EXCLUSIVE:
            train_idx, test_idx = self._ha_exclusive_split(dataset)
        elif self.config.strategy == SplitStrategy.MAB_EXCLUSIVE:
            train_idx, test_idx = self._mab_exclusive_split(dataset)
        elif self.config.strategy == SplitStrategy.MAB_CLUSTER_EXCLUSIVE:
            train_idx, test_idx = self._mab_cluster_exclusive_split(dataset)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.strategy}")
            
        self._log_split_statistics(dataset, train_idx, test_idx)
        
        if return_indices:
            return train_idx, test_idx
        else:
            train_pairs = [dataset.pairs[i] for i in train_idx]
            test_pairs = [dataset.pairs[i] for i in test_idx]
            return train_pairs, test_pairs
            
    def cv_split(self, dataset):
        """Generate cross-validation folds.
        
        Yields:
            (fold_num, train_indices, test_indices) for each fold
        """
        if self.config.strategy == SplitStrategy.LENIENT:
            yield from self._lenient_cv(dataset)
        elif self.config.strategy == SplitStrategy.HA_EXCLUSIVE:
            yield from self._ha_exclusive_cv(dataset)
        elif self.config.strategy == SplitStrategy.MAB_EXCLUSIVE:
            yield from self._mab_exclusive_cv(dataset)
        elif self.config.strategy == SplitStrategy.MAB_CLUSTER_EXCLUSIVE:
            yield from self._mab_cluster_exclusive_cv(dataset)
            
    # ==================== LENIENT SPLIT ====================
    
    def _lenient_split(self, dataset) -> Tuple[List[int], List[int]]:
        """Random split of pairs (easiest setting)."""
        n = len(dataset)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        test_size = int(n * self.config.test_size)
        test_idx = indices[:test_size].tolist()
        train_idx = indices[test_size:].tolist()
        
        return train_idx, test_idx
        
    def _lenient_cv(self, dataset):
        """Stratified K-fold cross-validation on pairs."""
        n = len(dataset)
        indices = np.arange(n)
        
        if self.config.stratify:
            labels = [dataset.pairs[i].binding_label for i in indices]
            # Handle None labels
            labels = [l if l is not None else 0 for l in labels]
            kfold = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.seed
            )
            splitter = kfold.split(indices, labels)
        else:
            kfold = KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.seed
            )
            splitter = kfold.split(indices)
            
        for fold, (train_idx, test_idx) in enumerate(splitter):
            yield fold, train_idx.tolist(), test_idx.tolist()
            
    # ==================== HA-EXCLUSIVE SPLIT ====================
    
    def _ha_exclusive_split(self, dataset) -> Tuple[List[int], List[int]]:
        """Hold out entire HA sequences (novel strain generalization)."""
        # Group pairs by HA
        ha_to_pairs = defaultdict(list)
        for i, pair in enumerate(dataset.pairs):
            ha_to_pairs[pair.ha_id].append(i)
            
        # Randomly select HAs for test set
        ha_ids = list(ha_to_pairs.keys())
        np.random.shuffle(ha_ids)
        
        test_ha_count = max(1, int(len(ha_ids) * self.config.test_size))
        test_has = set(ha_ids[:test_ha_count])
        train_has = set(ha_ids[test_ha_count:])
        
        # Assign pairs
        train_idx = []
        test_idx = []
        
        for ha_id, pair_indices in ha_to_pairs.items():
            if ha_id in test_has:
                test_idx.extend(pair_indices)
            else:
                train_idx.extend(pair_indices)
                
        logger.info(f"HA-exclusive split: {len(train_has)} train HAs, {len(test_has)} test HAs")
        
        return train_idx, test_idx
        
    def _ha_exclusive_cv(self, dataset):
        """K-fold CV with HA-based splitting."""
        ha_to_pairs = defaultdict(list)
        for i, pair in enumerate(dataset.pairs):
            ha_to_pairs[pair.ha_id].append(i)
            
        ha_ids = list(ha_to_pairs.keys())
        
        kfold = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed
        )
        
        for fold, (train_ha_idx, test_ha_idx) in enumerate(kfold.split(ha_ids)):
            train_has = {ha_ids[i] for i in train_ha_idx}
            test_has = {ha_ids[i] for i in test_ha_idx}
            
            train_idx = []
            test_idx = []
            
            for ha_id, pair_indices in ha_to_pairs.items():
                if ha_id in test_has:
                    test_idx.extend(pair_indices)
                else:
                    train_idx.extend(pair_indices)
                    
            yield fold, train_idx, test_idx
            
    # ==================== mAb-EXCLUSIVE SPLIT ====================
    
    def _mab_exclusive_split(self, dataset) -> Tuple[List[int], List[int]]:
        """Hold out entire antibody sequences (novel antibody generalization).
        
        This is the KEY challenging setting - predicting for completely unseen antibodies.
        Paper achieves 0.73 AUROC here.
        """
        # Group pairs by antibody
        ab_to_pairs = defaultdict(list)
        for i, pair in enumerate(dataset.pairs):
            ab_to_pairs[pair.antibody_id].append(i)
            
        # Randomly select antibodies for test set
        ab_ids = list(ab_to_pairs.keys())
        np.random.shuffle(ab_ids)
        
        test_ab_count = max(1, int(len(ab_ids) * self.config.test_size))
        test_abs = set(ab_ids[:test_ab_count])
        train_abs = set(ab_ids[test_ab_count:])
        
        # Assign pairs
        train_idx = []
        test_idx = []
        
        for ab_id, pair_indices in ab_to_pairs.items():
            if ab_id in test_abs:
                test_idx.extend(pair_indices)
            else:
                train_idx.extend(pair_indices)
                
        logger.info(f"mAb-exclusive split: {len(train_abs)} train mAbs, {len(test_abs)} test mAbs")
        
        return train_idx, test_idx
        
    def _mab_exclusive_cv(self, dataset):
        """K-fold CV with antibody-based splitting."""
        ab_to_pairs = defaultdict(list)
        for i, pair in enumerate(dataset.pairs):
            ab_to_pairs[pair.antibody_id].append(i)
            
        ab_ids = list(ab_to_pairs.keys())
        
        kfold = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed
        )
        
        for fold, (train_ab_idx, test_ab_idx) in enumerate(kfold.split(ab_ids)):
            train_abs = {ab_ids[i] for i in train_ab_idx}
            test_abs = {ab_ids[i] for i in test_ab_idx}
            
            train_idx = []
            test_idx = []
            
            for ab_id, pair_indices in ab_to_pairs.items():
                if ab_id in test_abs:
                    test_idx.extend(pair_indices)
                else:
                    train_idx.extend(pair_indices)
                    
            yield fold, train_idx, test_idx
            
    # ==================== mAb-CLUSTER-EXCLUSIVE SPLIT ====================
    
    def _mab_cluster_exclusive_split(self, dataset) -> Tuple[List[int], List[int]]:
        """Hold out antibody clusters at 50% sequence identity.
        
        Most challenging setting - predicting for antibodies with no similar
        sequences in training. Paper achieves 0.63-0.66 AUROC here.
        
        This simulates real-world discovery where we want to predict
        for genuinely novel antibody designs.
        """
        # Get unique antibody sequences
        ab_to_pairs = defaultdict(list)
        ab_sequences = {}
        
        for i, pair in enumerate(dataset.pairs):
            ab_to_pairs[pair.antibody_id].append(i)
            if pair.antibody_id not in ab_sequences:
                ab_sequences[pair.antibody_id] = pair.antibody_sequence
                
        ab_ids = list(ab_sequences.keys())
        sequences = [ab_sequences[ab_id] for ab_id in ab_ids]
        
        # Cluster antibodies by sequence similarity
        clusters = self._cluster_sequences(sequences, self.config.cluster_threshold)
        
        # Map antibodies to clusters
        ab_to_cluster = {ab_id: clusters[i] for i, ab_id in enumerate(ab_ids)}
        cluster_to_abs = defaultdict(list)
        for ab_id, cluster in ab_to_cluster.items():
            cluster_to_abs[cluster].append(ab_id)
            
        # Split clusters
        cluster_ids = list(cluster_to_abs.keys())
        np.random.shuffle(cluster_ids)
        
        test_cluster_count = max(1, int(len(cluster_ids) * self.config.test_size))
        test_clusters = set(cluster_ids[:test_cluster_count])
        
        # Assign pairs
        train_idx = []
        test_idx = []
        
        for ab_id, pair_indices in ab_to_pairs.items():
            if ab_to_cluster[ab_id] in test_clusters:
                test_idx.extend(pair_indices)
            else:
                train_idx.extend(pair_indices)
                
        logger.info(f"mAb-cluster split: {len(cluster_ids)} total clusters, "
                   f"{test_cluster_count} test clusters at {self.config.cluster_threshold:.0%} identity")
        
        return train_idx, test_idx
        
    def _mab_cluster_exclusive_cv(self, dataset):
        """K-fold CV with cluster-based splitting."""
        ab_to_pairs = defaultdict(list)
        ab_sequences = {}
        
        for i, pair in enumerate(dataset.pairs):
            ab_to_pairs[pair.antibody_id].append(i)
            if pair.antibody_id not in ab_sequences:
                ab_sequences[pair.antibody_id] = pair.antibody_sequence
                
        ab_ids = list(ab_sequences.keys())
        sequences = [ab_sequences[ab_id] for ab_id in ab_ids]
        
        # Cluster antibodies
        clusters = self._cluster_sequences(sequences, self.config.cluster_threshold)
        
        ab_to_cluster = {ab_id: clusters[i] for i, ab_id in enumerate(ab_ids)}
        cluster_to_abs = defaultdict(list)
        for ab_id, cluster in ab_to_cluster.items():
            cluster_to_abs[cluster].append(ab_id)
            
        cluster_ids = list(cluster_to_abs.keys())
        
        kfold = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed
        )
        
        for fold, (train_cluster_idx, test_cluster_idx) in enumerate(kfold.split(cluster_ids)):
            test_clusters = {cluster_ids[i] for i in test_cluster_idx}
            
            train_idx = []
            test_idx = []
            
            for ab_id, pair_indices in ab_to_pairs.items():
                if ab_to_cluster[ab_id] in test_clusters:
                    test_idx.extend(pair_indices)
                else:
                    train_idx.extend(pair_indices)
                    
            yield fold, train_idx, test_idx
            
    def _cluster_sequences(
        self,
        sequences: List[str],
        threshold: float
    ) -> np.ndarray:
        """Cluster sequences by identity threshold.
        
        Uses hierarchical clustering with simple identity metric.
        For production, consider CD-HIT or MMseqs2.
        
        Args:
            sequences: List of amino acid sequences
            threshold: Identity threshold (0.5 = 50% identity)
            
        Returns:
            Cluster assignments for each sequence
        """
        n = len(sequences)
        
        if n <= 1:
            return np.zeros(n, dtype=int)
            
        # Compute pairwise distances (1 - identity)
        # Using simple edit distance approximation for speed
        def sequence_distance(s1, s2):
            """Approximate sequence distance (1 - identity)."""
            # Align and compute identity
            # Simplified: using common k-mers as proxy
            k = 3
            kmers1 = set(s1[i:i+k] for i in range(len(s1)-k+1))
            kmers2 = set(s2[i:i+k] for i in range(len(s2)-k+1))
            
            if not kmers1 or not kmers2:
                return 1.0
                
            intersection = len(kmers1 & kmers2)
            union = len(kmers1 | kmers2)
            
            jaccard = intersection / union if union > 0 else 0
            return 1 - jaccard
            
        # Compute condensed distance matrix
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                d = sequence_distance(sequences[i], sequences[j])
                distances.append(d)
                
        if not distances:
            return np.zeros(n, dtype=int)
            
        distances = np.array(distances)
        
        # Hierarchical clustering
        Z = linkage(distances, method='average')
        
        # Cut tree at identity threshold
        # threshold of 0.5 identity = 0.5 distance
        clusters = fcluster(Z, t=1-threshold, criterion='distance')
        
        return clusters
        
    def _log_split_statistics(
        self,
        dataset,
        train_idx: List[int],
        test_idx: List[int]
    ) -> None:
        """Log statistics about the split."""
        train_labels = [dataset.pairs[i].binding_label for i in train_idx]
        test_labels = [dataset.pairs[i].binding_label for i in test_idx]
        
        train_labels = [l for l in train_labels if l is not None]
        test_labels = [l for l in test_labels if l is not None]
        
        train_pos = sum(train_labels) / len(train_labels) if train_labels else 0
        test_pos = sum(test_labels) / len(test_labels) if test_labels else 0
        
        logger.info(f"Split: {len(train_idx)} train ({train_pos:.1%} pos), "
                   f"{len(test_idx)} test ({test_pos:.1%} pos)")


def create_cv_folds(
    dataset,
    strategy: str = "lenient",
    n_folds: int = 5,
    cluster_threshold: float = 0.5,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Convenience function to create CV folds.
    
    Args:
        dataset: AntibodyHADataset instance
        strategy: Split strategy ('lenient', 'ha_exclusive', 'mab_exclusive', 'mab_cluster')
        n_folds: Number of CV folds
        cluster_threshold: Sequence identity threshold for clustering
        seed: Random seed
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    config = SplitConfig(
        strategy=SplitStrategy(strategy),
        n_folds=n_folds,
        cluster_threshold=cluster_threshold,
        seed=seed
    )
    
    splitter = DataSplitter(config)
    
    folds = []
    for fold, train_idx, test_idx in splitter.cv_split(dataset):
        folds.append((train_idx, test_idx))
        
    return folds


def evaluate_split_leakage(
    dataset,
    train_idx: List[int],
    test_idx: List[int]
) -> Dict[str, float]:
    """Check for data leakage between train and test sets.
    
    Reports:
    - Antibody overlap: Fraction of test antibodies seen in training
    - HA overlap: Fraction of test HAs seen in training
    - Pair overlap: Fraction of exact pairs duplicated
    
    Args:
        dataset: AntibodyHADataset instance
        train_idx: Training indices
        test_idx: Test indices
        
    Returns:
        Dictionary of leakage metrics
    """
    train_pairs = [dataset.pairs[i] for i in train_idx]
    test_pairs = [dataset.pairs[i] for i in test_idx]
    
    # Get unique entities
    train_abs = set(p.antibody_id for p in train_pairs)
    test_abs = set(p.antibody_id for p in test_pairs)
    
    train_has = set(p.ha_id for p in train_pairs)
    test_has = set(p.ha_id for p in test_pairs)
    
    train_pair_keys = set((p.antibody_id, p.ha_id) for p in train_pairs)
    test_pair_keys = set((p.antibody_id, p.ha_id) for p in test_pairs)
    
    # Calculate overlaps
    ab_overlap = len(test_abs & train_abs) / len(test_abs) if test_abs else 0
    ha_overlap = len(test_has & train_has) / len(test_has) if test_has else 0
    pair_overlap = len(test_pair_keys & train_pair_keys) / len(test_pair_keys) if test_pair_keys else 0
    
    return {
        'antibody_overlap': ab_overlap,
        'ha_overlap': ha_overlap,
        'pair_overlap': pair_overlap,
        'test_antibodies': len(test_abs),
        'test_has': len(test_has),
        'test_pairs': len(test_pairs)
    }
