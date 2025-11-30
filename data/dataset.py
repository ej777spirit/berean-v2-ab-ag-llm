"""
Antibody-HA Dataset Module

Implements data structures and loading for the antibody-hemagglutinin binding dataset
used in Barkan et al. (2025).

Dataset characteristics:
- 188 monoclonal antibodies × 79 HA antigens
- 4,922 binding pairs (ELISA, 35% positive)
- 5,035 HAI pairs (11% positive)
- Species: 82-85% human, 15-18% mouse
- Subtypes: 88-95% H1N1/H3N2, 21% COBRA designs
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any
from pathlib import Path
import json
import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class AntibodyHAPair:
    """Single antibody-HA interaction data point.
    
    Attributes:
        antibody_id: Unique antibody identifier
        ha_id: Unique HA identifier (strain name)
        heavy_chain: Heavy chain variable region sequence
        light_chain: Light chain variable region sequence
        ha_sequence: HA amino acid sequence
        
        # Labels
        binding_label: Binary binding (ELISA AUC > 1)
        binding_value: Raw ELISA AUC value
        hai_label: Binary HAI activity (<10 μg/mL)
        hai_value: Raw HAI titer value
        ic50_value: Neutralization IC50 (if available)
        
        # Metadata
        antibody_species: Source species (human/mouse)
        ha_subtype: HA subtype (H1N1, H3N2, etc.)
        ha_source: Sequence source (GISAID, NCBI, COBRA)
        epitope_type: Epitope classification (conformational, linear, unknown)
    """
    antibody_id: str
    ha_id: str
    heavy_chain: str
    light_chain: str
    ha_sequence: str
    
    # Labels
    binding_label: Optional[int] = None  # 0 or 1
    binding_value: Optional[float] = None
    hai_label: Optional[int] = None  # 0 or 1
    hai_value: Optional[float] = None
    ic50_value: Optional[float] = None
    
    # Metadata
    antibody_species: str = "human"
    ha_subtype: str = "unknown"
    ha_source: str = "unknown"
    epitope_type: str = "unknown"
    
    @property
    def antibody_sequence(self) -> str:
        """Concatenated HC + LC sequence."""
        return self.heavy_chain + self.light_chain
        
    @property
    def antibody_length(self) -> int:
        """Total antibody length."""
        return len(self.heavy_chain) + len(self.light_chain)
        
    @property
    def ha_length(self) -> int:
        """HA sequence length."""
        return len(self.ha_sequence)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'antibody_id': self.antibody_id,
            'ha_id': self.ha_id,
            'heavy_chain': self.heavy_chain,
            'light_chain': self.light_chain,
            'ha_sequence': self.ha_sequence,
            'binding_label': self.binding_label,
            'binding_value': self.binding_value,
            'hai_label': self.hai_label,
            'hai_value': self.hai_value,
            'ic50_value': self.ic50_value,
            'antibody_species': self.antibody_species,
            'ha_subtype': self.ha_subtype,
            'ha_source': self.ha_source,
            'epitope_type': self.epitope_type
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing.
    
    Attributes:
        task: Prediction task ('binding', 'hai', 'both')
        binding_threshold: ELISA AUC threshold for positive class (default: 1.0)
        hai_threshold: HAI titer threshold for positive class (default: 10 μg/mL)
        min_ab_length: Minimum antibody length filter
        max_ab_length: Maximum antibody length
        min_ha_length: Minimum HA length filter
        max_ha_length: Maximum HA length
        species_filter: Filter by species (None for all)
        subtype_filter: Filter by HA subtype (None for all)
        include_cobra: Include COBRA-designed antigens
    """
    task: str = "binding"
    binding_threshold: float = 1.0
    hai_threshold: float = 10.0
    min_ab_length: int = 50
    max_ab_length: int = 300
    min_ha_length: int = 400
    max_ha_length: int = 700
    species_filter: Optional[List[str]] = None
    subtype_filter: Optional[List[str]] = None
    include_cobra: bool = True


class AntibodyHADataset(Dataset):
    """PyTorch Dataset for antibody-HA interaction prediction.
    
    Implements data loading compatible with the MAMMAL paper format.
    """
    
    def __init__(
        self,
        pairs: List[AntibodyHAPair],
        config: DatasetConfig,
        transform: Optional[Any] = None
    ):
        """
        Args:
            pairs: List of AntibodyHAPair objects
            config: Dataset configuration
            transform: Optional sequence transform
        """
        self.config = config
        self.transform = transform
        
        # Filter pairs based on config
        self.pairs = self._filter_pairs(pairs)
        
        # Build indices for efficient lookup
        self._build_indices()
        
        logger.info(f"Dataset initialized with {len(self.pairs)} pairs")
        self._log_statistics()
        
    def _filter_pairs(self, pairs: List[AntibodyHAPair]) -> List[AntibodyHAPair]:
        """Filter pairs based on configuration."""
        filtered = []
        
        for pair in pairs:
            # Length filters
            if pair.antibody_length < self.config.min_ab_length:
                continue
            if pair.antibody_length > self.config.max_ab_length:
                continue
            if pair.ha_length < self.config.min_ha_length:
                continue
            if pair.ha_length > self.config.max_ha_length:
                continue
                
            # Species filter
            if self.config.species_filter:
                if pair.antibody_species not in self.config.species_filter:
                    continue
                    
            # Subtype filter
            if self.config.subtype_filter:
                if pair.ha_subtype not in self.config.subtype_filter:
                    continue
                    
            # COBRA filter
            if not self.config.include_cobra:
                if pair.ha_source == "COBRA":
                    continue
                    
            # Task-specific label check
            if self.config.task == "binding":
                if pair.binding_label is None:
                    continue
            elif self.config.task == "hai":
                if pair.hai_label is None:
                    continue
            elif self.config.task == "both":
                if pair.binding_label is None and pair.hai_label is None:
                    continue
                    
            filtered.append(pair)
            
        return filtered
        
    def _build_indices(self) -> None:
        """Build lookup indices for antibodies and antigens."""
        self.antibody_ids = list(set(p.antibody_id for p in self.pairs))
        self.ha_ids = list(set(p.ha_id for p in self.pairs))
        
        self.ab_to_idx = {ab: i for i, ab in enumerate(self.antibody_ids)}
        self.ha_to_idx = {ha: i for i, ha in enumerate(self.ha_ids)}
        
        # Group pairs by antibody and HA
        self.pairs_by_antibody = {}
        self.pairs_by_ha = {}
        
        for i, pair in enumerate(self.pairs):
            if pair.antibody_id not in self.pairs_by_antibody:
                self.pairs_by_antibody[pair.antibody_id] = []
            self.pairs_by_antibody[pair.antibody_id].append(i)
            
            if pair.ha_id not in self.pairs_by_ha:
                self.pairs_by_ha[pair.ha_id] = []
            self.pairs_by_ha[pair.ha_id].append(i)
            
    def _log_statistics(self) -> None:
        """Log dataset statistics."""
        n_antibodies = len(self.antibody_ids)
        n_has = len(self.ha_ids)
        
        # Label distribution
        if self.config.task in ["binding", "both"]:
            binding_labels = [p.binding_label for p in self.pairs if p.binding_label is not None]
            if binding_labels:
                pos_rate = sum(binding_labels) / len(binding_labels)
                logger.info(f"Binding: {len(binding_labels)} pairs, {pos_rate:.1%} positive")
                
        if self.config.task in ["hai", "both"]:
            hai_labels = [p.hai_label for p in self.pairs if p.hai_label is not None]
            if hai_labels:
                pos_rate = sum(hai_labels) / len(hai_labels)
                logger.info(f"HAI: {len(hai_labels)} pairs, {pos_rate:.1%} positive")
                
        # Species distribution
        species_counts = {}
        for pair in self.pairs:
            species_counts[pair.antibody_species] = species_counts.get(pair.antibody_species, 0) + 1
        logger.info(f"Species distribution: {species_counts}")
        
        # Subtype distribution
        subtype_counts = {}
        for pair in self.pairs:
            subtype_counts[pair.ha_subtype] = subtype_counts.get(pair.ha_subtype, 0) + 1
        logger.info(f"Top subtypes: {dict(sorted(subtype_counts.items(), key=lambda x: -x[1])[:5])}")
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data point.
        
        Returns:
            Dictionary with:
            - antibody_sequence: Concatenated HC + LC
            - ha_sequence: HA sequence
            - binding_label: Binary binding label (if available)
            - hai_label: Binary HAI label (if available)
            - metadata: Additional info
        """
        pair = self.pairs[idx]
        
        # Get sequences
        antibody_seq = pair.antibody_sequence
        ha_seq = pair.ha_sequence
        
        # Apply transforms if provided
        if self.transform:
            antibody_seq, ha_seq = self.transform(antibody_seq, ha_seq)
            
        item = {
            'antibody_sequence': antibody_seq,
            'ha_sequence': ha_seq,
            'antibody_id': pair.antibody_id,
            'ha_id': pair.ha_id
        }
        
        # Add labels based on task
        if self.config.task in ["binding", "both"]:
            item['binding_label'] = pair.binding_label
            item['binding_value'] = pair.binding_value
            
        if self.config.task in ["hai", "both"]:
            item['hai_label'] = pair.hai_label
            item['hai_value'] = pair.hai_value
            
        return item
        
    def get_antibody_sequence(self, antibody_id: str) -> Optional[str]:
        """Get antibody sequence by ID."""
        if antibody_id in self.pairs_by_antibody:
            idx = self.pairs_by_antibody[antibody_id][0]
            return self.pairs[idx].antibody_sequence
        return None
        
    def get_ha_sequence(self, ha_id: str) -> Optional[str]:
        """Get HA sequence by ID."""
        if ha_id in self.pairs_by_ha:
            idx = self.pairs_by_ha[ha_id][0]
            return self.pairs[idx].ha_sequence
        return None
        
    def get_label_weights(self, task: str = "binding") -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        if task == "binding":
            labels = [p.binding_label for p in self.pairs if p.binding_label is not None]
        else:
            labels = [p.hai_label for p in self.pairs if p.hai_label is not None]
            
        if not labels:
            return torch.tensor([1.0, 1.0])
            
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        # Inverse frequency weighting
        weights = [len(labels) / (2 * neg_count), len(labels) / (2 * pos_count)]
        
        return torch.tensor(weights)


def load_mammal_dataset(
    data_path: Union[str, Path],
    config: Optional[DatasetConfig] = None
) -> AntibodyHADataset:
    """Load dataset from file(s) in MAMMAL paper format.
    
    Supports:
    - CSV/TSV files with columns: mAb_ID, HA_ID, VH, VL, HA_seq, binding, HAI
    - JSON files with list of pair dictionaries
    - Parquet files
    
    Args:
        data_path: Path to data file or directory
        config: Dataset configuration
        
    Returns:
        AntibodyHADataset instance
    """
    if config is None:
        config = DatasetConfig()
        
    data_path = Path(data_path)
    pairs = []
    
    if data_path.is_file():
        pairs = _load_single_file(data_path, config)
    elif data_path.is_dir():
        # Load all files in directory
        for file_path in data_path.glob("*.csv"):
            pairs.extend(_load_single_file(file_path, config))
        for file_path in data_path.glob("*.json"):
            pairs.extend(_load_single_file(file_path, config))
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")
        
    return AntibodyHADataset(pairs, config)


def _load_single_file(
    file_path: Path,
    config: DatasetConfig
) -> List[AntibodyHAPair]:
    """Load pairs from a single file."""
    pairs = []
    
    suffix = file_path.suffix.lower()
    
    if suffix in ['.csv', '.tsv']:
        sep = '\t' if suffix == '.tsv' else ','
        df = pd.read_csv(file_path, sep=sep)
        pairs = _parse_dataframe(df, config)
        
    elif suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        pairs = _parse_json(data, config)
        
    elif suffix == '.parquet':
        df = pd.read_parquet(file_path)
        pairs = _parse_dataframe(df, config)
        
    else:
        logger.warning(f"Unknown file format: {suffix}")
        
    logger.info(f"Loaded {len(pairs)} pairs from {file_path.name}")
    return pairs


def _parse_dataframe(
    df: pd.DataFrame,
    config: DatasetConfig
) -> List[AntibodyHAPair]:
    """Parse DataFrame into AntibodyHAPair objects."""
    pairs = []
    
    # Column name mapping (handle different naming conventions)
    col_map = {
        'antibody_id': ['mAb_ID', 'antibody_id', 'ab_id', 'mab_id', 'Ab_ID'],
        'ha_id': ['HA_ID', 'ha_id', 'antigen_id', 'HA_strain', 'Strain'],
        'heavy_chain': ['VH', 'heavy_chain', 'HC', 'VH_seq', 'heavy'],
        'light_chain': ['VL', 'light_chain', 'LC', 'VL_seq', 'light'],
        'ha_sequence': ['HA_seq', 'ha_sequence', 'antigen_seq', 'HA', 'target_seq'],
        'binding_value': ['AUC_ELISA', 'binding_value', 'ELISA', 'binding', 'AUC'],
        'hai_value': ['HAI', 'hai_value', 'HAI_titer'],
        'species': ['species', 'antibody_species', 'Species'],
        'subtype': ['subtype', 'ha_subtype', 'Subtype', 'HA_subtype']
    }
    
    def get_col(df, names):
        for name in names:
            if name in df.columns:
                return name
        return None
        
    # Find actual column names
    ab_id_col = get_col(df, col_map['antibody_id'])
    ha_id_col = get_col(df, col_map['ha_id'])
    hc_col = get_col(df, col_map['heavy_chain'])
    lc_col = get_col(df, col_map['light_chain'])
    ha_col = get_col(df, col_map['ha_sequence'])
    binding_col = get_col(df, col_map['binding_value'])
    hai_col = get_col(df, col_map['hai_value'])
    species_col = get_col(df, col_map['species'])
    subtype_col = get_col(df, col_map['subtype'])
    
    # Validate required columns
    required = [ab_id_col, ha_id_col, hc_col, lc_col, ha_col]
    if None in required:
        missing = [col_map[k][0] for k, v in zip(
            ['antibody_id', 'ha_id', 'heavy_chain', 'light_chain', 'ha_sequence'],
            required
        ) if v is None]
        raise ValueError(f"Missing required columns: {missing}")
        
    for _, row in df.iterrows():
        # Get binding label
        binding_value = row.get(binding_col) if binding_col else None
        binding_label = None
        if binding_value is not None and not pd.isna(binding_value):
            binding_label = 1 if float(binding_value) > config.binding_threshold else 0
            
        # Get HAI label
        hai_value = row.get(hai_col) if hai_col else None
        hai_label = None
        if hai_value is not None and not pd.isna(hai_value):
            hai_label = 1 if float(hai_value) < config.hai_threshold else 0
            
        pair = AntibodyHAPair(
            antibody_id=str(row[ab_id_col]),
            ha_id=str(row[ha_id_col]),
            heavy_chain=str(row[hc_col]),
            light_chain=str(row[lc_col]),
            ha_sequence=str(row[ha_col]),
            binding_label=binding_label,
            binding_value=binding_value,
            hai_label=hai_label,
            hai_value=hai_value,
            antibody_species=str(row.get(species_col, 'unknown')) if species_col else 'unknown',
            ha_subtype=str(row.get(subtype_col, 'unknown')) if subtype_col else 'unknown'
        )
        pairs.append(pair)
        
    return pairs


def _parse_json(
    data: Union[List, Dict],
    config: DatasetConfig
) -> List[AntibodyHAPair]:
    """Parse JSON data into AntibodyHAPair objects."""
    if isinstance(data, dict):
        data = data.get('pairs', data.get('data', [data]))
        
    pairs = []
    for item in data:
        # Get binding label
        binding_value = item.get('binding_value') or item.get('AUC_ELISA')
        binding_label = None
        if binding_value is not None:
            binding_label = 1 if float(binding_value) > config.binding_threshold else 0
            
        # Get HAI label
        hai_value = item.get('hai_value') or item.get('HAI')
        hai_label = None
        if hai_value is not None:
            hai_label = 1 if float(hai_value) < config.hai_threshold else 0
            
        pair = AntibodyHAPair(
            antibody_id=item.get('antibody_id', item.get('mAb_ID')),
            ha_id=item.get('ha_id', item.get('HA_ID')),
            heavy_chain=item.get('heavy_chain', item.get('VH')),
            light_chain=item.get('light_chain', item.get('VL')),
            ha_sequence=item.get('ha_sequence', item.get('HA_seq')),
            binding_label=binding_label,
            binding_value=binding_value,
            hai_label=hai_label,
            hai_value=hai_value,
            antibody_species=item.get('species', item.get('antibody_species', 'unknown')),
            ha_subtype=item.get('subtype', item.get('ha_subtype', 'unknown')),
            ha_source=item.get('ha_source', 'unknown'),
            epitope_type=item.get('epitope_type', 'unknown')
        )
        pairs.append(pair)
        
    return pairs


def create_synthetic_dataset(
    n_antibodies: int = 50,
    n_antigens: int = 20,
    n_pairs: int = 500,
    pos_rate: float = 0.35,
    seed: int = 42
) -> AntibodyHADataset:
    """Create synthetic dataset for testing.
    
    Generates random sequences with specified positive rate.
    For development/testing only - not for actual experiments.
    
    Args:
        n_antibodies: Number of unique antibodies
        n_antigens: Number of unique antigens
        n_pairs: Total number of pairs
        pos_rate: Fraction of positive pairs
        seed: Random seed
        
    Returns:
        AntibodyHADataset with synthetic data
    """
    np.random.seed(seed)
    
    # Generate random sequences
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    def random_seq(length):
        return ''.join(np.random.choice(amino_acids, length))
        
    antibodies = {
        f"Ab_{i}": (random_seq(108), random_seq(122))  # HC, LC
        for i in range(n_antibodies)
    }
    
    antigens = {
        f"HA_{i}": random_seq(np.random.randint(530, 590))
        for i in range(n_antigens)
    }
    
    # Generate pairs
    pairs = []
    ab_ids = list(antibodies.keys())
    ag_ids = list(antigens.keys())
    
    for _ in range(n_pairs):
        ab_id = np.random.choice(ab_ids)
        ag_id = np.random.choice(ag_ids)
        hc, lc = antibodies[ab_id]
        ha_seq = antigens[ag_id]
        
        # Generate label
        binding_label = 1 if np.random.random() < pos_rate else 0
        binding_value = np.random.uniform(1.5, 5.0) if binding_label else np.random.uniform(0.1, 0.9)
        
        pair = AntibodyHAPair(
            antibody_id=ab_id,
            ha_id=ag_id,
            heavy_chain=hc,
            light_chain=lc,
            ha_sequence=ha_seq,
            binding_label=binding_label,
            binding_value=binding_value
        )
        pairs.append(pair)
        
    config = DatasetConfig(task="binding")
    return AntibodyHADataset(pairs, config)
