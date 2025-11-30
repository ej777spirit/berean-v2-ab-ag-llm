"""
Data Preprocessing Module

Handles sequence normalization, augmentation, and feature engineering
for antibody-antigen datasets.

Includes novel data augmentation strategies (EXTENSION beyond paper):
- Sequence perturbation
- Synthetic pair generation
- Cross-species augmentation
"""

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import random
import string

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing.
    
    Attributes:
        normalize_case: Convert sequences to uppercase
        remove_gaps: Remove gap characters
        validate_aa: Validate amino acid characters
        unknown_aa_replace: Replace unknown AAs with this character
        max_length: Truncate sequences longer than this
    """
    normalize_case: bool = True
    remove_gaps: bool = True
    validate_aa: bool = True
    unknown_aa_replace: str = "X"
    max_length: Optional[int] = None


# Standard amino acid alphabet
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
AMINO_ACIDS_WITH_X = set('ACDEFGHIKLMNPQRSTVWYX')


def normalize_sequences(
    sequences: List[str],
    config: Optional[PreprocessConfig] = None
) -> List[str]:
    """Normalize a list of sequences.
    
    Args:
        sequences: List of amino acid sequences
        config: Preprocessing configuration
        
    Returns:
        List of normalized sequences
    """
    if config is None:
        config = PreprocessConfig()
        
    normalized = []
    
    for seq in sequences:
        # Normalize case
        if config.normalize_case:
            seq = seq.upper()
            
        # Remove gaps
        if config.remove_gaps:
            seq = seq.replace('-', '').replace('.', '')
            
        # Remove whitespace
        seq = ''.join(seq.split())
        
        # Validate amino acids
        if config.validate_aa:
            valid_seq = []
            for aa in seq:
                if aa in AMINO_ACIDS:
                    valid_seq.append(aa)
                else:
                    valid_seq.append(config.unknown_aa_replace)
            seq = ''.join(valid_seq)
            
        # Truncate if needed
        if config.max_length and len(seq) > config.max_length:
            seq = seq[:config.max_length]
            
        normalized.append(seq)
        
    return normalized


class DataPreprocessor:
    """Comprehensive data preprocessor for antibody-antigen datasets.
    
    Handles:
    - Sequence normalization
    - Feature extraction
    - Data augmentation
    - Quality filtering
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        
    def preprocess_antibody(self, heavy_chain: str, light_chain: str) -> Tuple[str, str]:
        """Preprocess antibody heavy and light chains.
        
        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            
        Returns:
            (normalized_hc, normalized_lc)
        """
        hc, lc = normalize_sequences([heavy_chain, light_chain], self.config)
        return hc, lc
        
    def preprocess_antigen(self, sequence: str) -> str:
        """Preprocess antigen (HA) sequence.
        
        Args:
            sequence: HA sequence
            
        Returns:
            Normalized sequence
        """
        return normalize_sequences([sequence], self.config)[0]
        
    def compute_features(self, sequence: str) -> Dict[str, float]:
        """Compute sequence-level features.
        
        Returns:
            Dictionary of computed features
        """
        if not sequence:
            return {}
            
        # Basic composition
        aa_counts = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
        length = len(sequence)
        
        # Amino acid fractions
        aa_fractions = {f"frac_{aa}": count/length for aa, count in aa_counts.items()}
        
        # Physicochemical properties
        # Hydrophobic residues
        hydrophobic = set('AILMFVW')
        hydrophobic_frac = sum(aa_counts.get(aa, 0) for aa in hydrophobic) / length
        
        # Charged residues
        positive = set('RKH')
        negative = set('DE')
        positive_frac = sum(aa_counts.get(aa, 0) for aa in positive) / length
        negative_frac = sum(aa_counts.get(aa, 0) for aa in negative) / length
        charge_frac = positive_frac - negative_frac
        
        # Aromatic residues
        aromatic = set('FWY')
        aromatic_frac = sum(aa_counts.get(aa, 0) for aa in aromatic) / length
        
        # Small residues
        small = set('AGST')
        small_frac = sum(aa_counts.get(aa, 0) for aa in small) / length
        
        features = {
            'length': length,
            'hydrophobic_frac': hydrophobic_frac,
            'positive_frac': positive_frac,
            'negative_frac': negative_frac,
            'charge_frac': charge_frac,
            'aromatic_frac': aromatic_frac,
            'small_frac': small_frac,
            **aa_fractions
        }
        
        return features
        
    def filter_quality(
        self,
        pairs: List,
        min_ab_length: int = 50,
        max_ab_length: int = 300,
        min_ha_length: int = 400,
        max_ha_length: int = 700,
        require_label: bool = True
    ) -> List:
        """Filter pairs by quality criteria.
        
        Args:
            pairs: List of AntibodyHAPair objects
            min_ab_length: Minimum antibody length
            max_ab_length: Maximum antibody length
            min_ha_length: Minimum HA length
            max_ha_length: Maximum HA length
            require_label: Require binding or HAI label
            
        Returns:
            Filtered list of pairs
        """
        filtered = []
        
        for pair in pairs:
            # Length checks
            ab_len = len(pair.heavy_chain) + len(pair.light_chain)
            if ab_len < min_ab_length or ab_len > max_ab_length:
                continue
                
            ha_len = len(pair.ha_sequence)
            if ha_len < min_ha_length or ha_len > max_ha_length:
                continue
                
            # Label check
            if require_label:
                if pair.binding_label is None and pair.hai_label is None:
                    continue
                    
            filtered.append(pair)
            
        logger.info(f"Quality filter: {len(pairs)} -> {len(filtered)} pairs")
        return filtered


# ==================== DATA AUGMENTATION ====================

class SequenceAugmenter:
    """Augment sequences with biologically plausible perturbations.
    
    EXTENSION beyond paper: Novel data augmentation to improve
    generalization on limited training data (188 antibodies).
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.01,
        insertion_rate: float = 0.005,
        deletion_rate: float = 0.005,
        seed: int = 42
    ):
        """
        Args:
            mutation_rate: Probability of point mutation per residue
            insertion_rate: Probability of insertion per position
            deletion_rate: Probability of deletion per residue
            seed: Random seed
        """
        self.mutation_rate = mutation_rate
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Amino acid substitution matrix (simplified BLOSUM-like)
        self.substitution_probs = self._build_substitution_matrix()
        
    def _build_substitution_matrix(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build simplified substitution probabilities."""
        # Group similar amino acids
        groups = [
            'AILMV',   # Hydrophobic
            'FWY',     # Aromatic
            'ST',      # Hydroxyl
            'NQ',      # Amide
            'DE',      # Acidic
            'RKH',     # Basic
            'GP',      # Special
            'C'        # Cysteine
        ]
        
        aa_to_group = {}
        for i, group in enumerate(groups):
            for aa in group:
                aa_to_group[aa] = i
                
        matrix = {}
        for aa in AMINO_ACIDS:
            # Prefer substitutions within same group
            group_idx = aa_to_group.get(aa)
            if group_idx is not None:
                same_group = [a for a in groups[group_idx] if a != aa]
                other_aas = [a for a in AMINO_ACIDS if a not in groups[group_idx]]
            else:
                same_group = []
                other_aas = [a for a in AMINO_ACIDS if a != aa]
                
            probs = []
            # Same group: higher probability
            for sub in same_group:
                probs.append((sub, 0.7 / len(same_group) if same_group else 0))
            # Other groups: lower probability
            for sub in other_aas:
                probs.append((sub, 0.3 / len(other_aas) if other_aas else 0))
                
            matrix[aa] = probs
            
        return matrix
        
    def mutate(self, sequence: str) -> str:
        """Apply point mutations to sequence.
        
        Args:
            sequence: Original amino acid sequence
            
        Returns:
            Mutated sequence
        """
        mutated = list(sequence)
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                aa = mutated[i]
                if aa in self.substitution_probs:
                    subs, probs = zip(*self.substitution_probs[aa])
                    probs = np.array(probs)
                    probs = probs / probs.sum()  # Normalize
                    mutated[i] = np.random.choice(list(subs), p=probs)
                    
        return ''.join(mutated)
        
    def insert(self, sequence: str) -> str:
        """Apply random insertions.
        
        Args:
            sequence: Original sequence
            
        Returns:
            Sequence with insertions
        """
        result = []
        
        for aa in sequence:
            result.append(aa)
            if random.random() < self.insertion_rate:
                # Insert random amino acid
                result.append(random.choice(list(AMINO_ACIDS)))
                
        return ''.join(result)
        
    def delete(self, sequence: str) -> str:
        """Apply random deletions.
        
        Args:
            sequence: Original sequence
            
        Returns:
            Sequence with deletions
        """
        result = []
        
        for aa in sequence:
            if random.random() >= self.deletion_rate:
                result.append(aa)
                
        return ''.join(result)
        
    def augment(self, sequence: str, n_augments: int = 1) -> List[str]:
        """Generate augmented versions of a sequence.
        
        Args:
            sequence: Original sequence
            n_augments: Number of augmented versions to generate
            
        Returns:
            List of augmented sequences
        """
        augmented = []
        
        for _ in range(n_augments):
            seq = sequence
            
            # Apply augmentations
            seq = self.mutate(seq)
            seq = self.insert(seq)
            seq = self.delete(seq)
            
            augmented.append(seq)
            
        return augmented


def augment_dataset(
    pairs: List,
    augmentation_factor: int = 2,
    mutation_rate: float = 0.01,
    augment_positive_only: bool = True,
    seed: int = 42
) -> List:
    """Augment dataset with perturbed sequences.
    
    EXTENSION beyond paper: Addresses limited training data (188 antibodies)
    through biologically plausible sequence perturbations.
    
    Args:
        pairs: List of AntibodyHAPair objects
        augmentation_factor: Number of augmented copies per pair
        mutation_rate: Mutation rate for augmentation
        augment_positive_only: Only augment positive examples (for class balance)
        seed: Random seed
        
    Returns:
        Augmented list of pairs (original + augmented)
    """
    from .dataset import AntibodyHAPair
    
    augmenter = SequenceAugmenter(mutation_rate=mutation_rate, seed=seed)
    
    augmented_pairs = list(pairs)  # Keep originals
    
    for pair in pairs:
        # Optionally only augment positives
        if augment_positive_only and pair.binding_label != 1:
            continue
            
        for _ in range(augmentation_factor):
            # Augment antibody sequences
            aug_hc = augmenter.augment(pair.heavy_chain, 1)[0]
            aug_lc = augmenter.augment(pair.light_chain, 1)[0]
            
            # Create augmented pair
            aug_pair = AntibodyHAPair(
                antibody_id=f"{pair.antibody_id}_aug",
                ha_id=pair.ha_id,
                heavy_chain=aug_hc,
                light_chain=aug_lc,
                ha_sequence=pair.ha_sequence,
                binding_label=pair.binding_label,
                binding_value=pair.binding_value,
                hai_label=pair.hai_label,
                hai_value=pair.hai_value,
                antibody_species=pair.antibody_species,
                ha_subtype=pair.ha_subtype,
                ha_source="augmented",
                epitope_type=pair.epitope_type
            )
            augmented_pairs.append(aug_pair)
            
    logger.info(f"Data augmentation: {len(pairs)} -> {len(augmented_pairs)} pairs")
    
    return augmented_pairs


# ==================== SEQUENCE TRANSFORMS ====================

class SequenceTransform:
    """Composable sequence transformations."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
        
    def __call__(self, antibody_seq: str, antigen_seq: str) -> Tuple[str, str]:
        for transform in self.transforms:
            antibody_seq, antigen_seq = transform(antibody_seq, antigen_seq)
        return antibody_seq, antigen_seq


def create_standard_transform() -> SequenceTransform:
    """Create standard preprocessing transform."""
    def normalize(ab, ag):
        ab = normalize_sequences([ab])[0]
        ag = normalize_sequences([ag])[0]
        return ab, ag
        
    return SequenceTransform([normalize])


def create_augmentation_transform(
    mutation_rate: float = 0.01,
    apply_to_antibody: bool = True,
    apply_to_antigen: bool = False
) -> SequenceTransform:
    """Create transform with augmentation."""
    augmenter = SequenceAugmenter(mutation_rate=mutation_rate)
    
    def augment(ab, ag):
        if apply_to_antibody:
            ab = augmenter.augment(ab, 1)[0]
        if apply_to_antigen:
            ag = augmenter.augment(ag, 1)[0]
        return ab, ag
        
    return SequenceTransform([augment])
