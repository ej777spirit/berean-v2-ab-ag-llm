"""
Queryable Attention Module
==========================

Status: NEW
Source: berean_v2_refactor

Provides arbitrary residue querying and on-demand attention score lookup,
matching the functionality shown in the BEREAN visualization interface.

Features:
    a) Arbitrary residue query: Click any residue → see what it attends to
    b) Attention score for ANY residue pair on demand

Architecture:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      QueryableAttentionMatrix                           │
    │                                                                         │
    │   Stores: Full (Ab_len × Ag_len) attention matrix                       │
    │                                                                         │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    Attention Matrix                              │   │
    │   │                                                                  │   │
    │   │            Antigen Positions (columns)                          │   │
    │   │         0    1    2    3   ...  N                               │   │
    │   │      ┌────────────────────────────┐                             │   │
    │   │    0 │ 0.02 0.01 0.03 0.01 ... 0.02│  ← Heavy chain             │   │
    │   │    1 │ 0.01 0.04 0.02 0.01 ... 0.01│     residues               │   │
    │   │  A   │ ...                         │                             │   │
    │   │  b H │ 0.05 0.12 0.78 0.15 ... 0.03│  ← CDR-H3 (high attention) │   │
    │   │    . │ ...                         │                             │   │
    │   │  r   ├────────────────────────────┤                             │   │
    │   │  o L │ 0.02 0.01 0.01 0.02 ... 0.01│  ← Light chain             │   │
    │   │  w   │ ...                         │     residues               │   │
    │   │  s   │ 0.03 0.08 0.45 0.12 ... 0.02│  ← CDR-L3                  │   │
    │   │      └────────────────────────────┘                             │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │   Query Methods:                                                        │
    │     • query_antibody_residue(chain, pos) → top antigen targets          │
    │     • query_antigen_residue(pos) → top antibody contacts                │
    │     • get_attention_score(ab_chain, ab_pos, ag_pos) → float             │
    │     • get_all_contacts_for_residue(chain, pos) → full attention row     │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    >>> matrix = QueryableAttentionMatrix(heavy, light, antigen, attention)
    >>> 
    >>> # Query what antibody residue G56 (CDR2) attends to
    >>> result = matrix.query_antibody_residue("H", 56)
    >>> print(f"H:G56 → {result.top_target}")  # "AG:H68 (0.798)"
    >>> 
    >>> # Get attention score for specific pair
    >>> score = matrix.get_attention_score("H", 56, 68)
    >>> print(f"Attention: {score:.3f}")  # 0.798
    >>> 
    >>> # Query which antibody residues contact antigen position 68
    >>> contacts = matrix.query_antigen_residue(68)
    >>> for c in contacts.top_contacts[:5]:
    ...     print(f"  {c}")  # "H:G56 (CDR2) - 0.798"
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CDR DEFINITIONS
# =============================================================================

CDR_REGIONS = {
    "H": {
        "CDR1": (26, 35, "CDR1 Loop"),
        "CDR2": (50, 65, "CDR2 Loop"),
        "CDR3": (95, 102, "CDR3 (Hypervariable)"),
    },
    "L": {
        "CDR1": (24, 34, "CDR1 Loop"),
        "CDR2": (50, 56, "CDR2 Loop"),
        "CDR3": (89, 97, "CDR3 (Hypervariable)"),
    },
}

FRAMEWORK_REGIONS = {
    "H": {
        "FR1": (0, 25),
        "FR2": (36, 49),
        "FR3": (66, 94),
        "FR4": (103, 130),
    },
    "L": {
        "FR1": (0, 23),
        "FR2": (35, 49),
        "FR3": (57, 88),
        "FR4": (98, 115),
    },
}


# =============================================================================
# DATA CLASSES FOR QUERY RESULTS
# =============================================================================

@dataclass
class AttentionTarget:
    """Result of querying what a residue attends to.
    
    Attributes:
        position: Target position in sequence
        residue: Amino acid at that position
        attention_score: Attention weight (0-1)
        rank: Rank among all targets (1 = highest)
    """
    position: int
    residue: str
    attention_score: float
    rank: int = 1
    
    def __repr__(self):
        return f"{self.residue}{self.position + 1} ({self.attention_score:.3f})"
    
    def to_dict(self) -> Dict:
        return {
            "position": self.position,
            "residue": self.residue,
            "attention_score": self.attention_score,
            "rank": self.rank,
        }


@dataclass
class AntibodyResidueQuery:
    """Result of querying an antibody residue.
    
    Shows what antigen positions this antibody residue attends to.
    
    Attributes:
        chain: 'H' for heavy, 'L' for light
        position: Position in chain (0-indexed)
        residue: Amino acid
        region: CDR or Framework region name
        region_type: 'CDR1', 'CDR2', 'CDR3', or 'Framework'
        top_target: Highest attention antigen target
        all_targets: All antigen targets sorted by attention
        attention_vector: Raw attention scores to all antigen positions
    """
    chain: str
    position: int
    residue: str
    region: str
    region_type: str
    top_target: AttentionTarget
    all_targets: List[AttentionTarget]
    attention_vector: np.ndarray
    
    @property
    def top_target_str(self) -> str:
        """Formatted string for top target."""
        return f"AG:{self.top_target.residue}{self.top_target.position + 1} ({self.top_target.attention_score:.3f})"
    
    @property
    def residue_id(self) -> str:
        """Formatted residue identifier."""
        return f"{self.chain}:{self.residue}{self.position + 1}"
    
    def __repr__(self):
        return f"{self.residue_id} ({self.region}) → {self.top_target_str}"
    
    def to_dict(self) -> Dict:
        return {
            "chain": self.chain,
            "position": self.position,
            "residue": self.residue,
            "region": self.region,
            "region_type": self.region_type,
            "top_target": self.top_target.to_dict(),
            "top_5_targets": [t.to_dict() for t in self.all_targets[:5]],
        }
    
    def format_display(self) -> str:
        """Generate formatted display string."""
        lines = [
            f"┌─ ANTIBODY RESIDUE QUERY ─────────────────────────",
            f"│ Residue: {self.residue_id}",
            f"│ Region:  {self.region}",
            f"│",
            f"│ Attends to Antigen:",
        ]
        
        for i, target in enumerate(self.all_targets[:5], 1):
            bar_len = int(target.attention_score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"│   {i}. AG:{target.residue}{target.position + 1:3d}  {bar} {target.attention_score:.3f}")
        
        lines.append(f"└──────────────────────────────────────────────────")
        
        return "\n".join(lines)


@dataclass
class AntigenResidueQuery:
    """Result of querying an antigen residue.
    
    Shows which antibody residues attend to this antigen position.
    
    Attributes:
        position: Position in antigen (0-indexed)
        residue: Amino acid
        top_contacts: Antibody residues with highest attention to this position
        total_attention: Sum of all antibody attention to this position
        attention_vector: Raw attention scores from all antibody positions
    """
    position: int
    residue: str
    top_contacts: List[Dict[str, Any]]
    total_attention: float
    attention_vector: np.ndarray
    
    @property
    def residue_id(self) -> str:
        return f"AG:{self.residue}{self.position + 1}"
    
    def __repr__(self):
        top = self.top_contacts[0] if self.top_contacts else {"id": "none", "score": 0}
        return f"{self.residue_id} ← {top['id']} ({top['score']:.3f})"
    
    def to_dict(self) -> Dict:
        return {
            "position": self.position,
            "residue": self.residue,
            "total_attention": self.total_attention,
            "top_contacts": self.top_contacts[:10],
        }
    
    def format_display(self) -> str:
        """Generate formatted display string."""
        lines = [
            f"┌─ ANTIGEN RESIDUE QUERY ──────────────────────────",
            f"│ Residue: {self.residue_id}",
            f"│ Total Attention: {self.total_attention:.3f}",
            f"│",
            f"│ Contacted by Antibody:",
        ]
        
        for i, contact in enumerate(self.top_contacts[:5], 1):
            bar_len = int(contact["score"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            region_str = f" ({contact['region']})" if contact.get('region') else ""
            lines.append(f"│   {i}. {contact['id']}{region_str}  {bar} {contact['score']:.3f}")
        
        lines.append(f"└──────────────────────────────────────────────────")
        
        return "\n".join(lines)


@dataclass
class ResidueResiduePairQuery:
    """Result of querying a specific residue-residue pair.
    
    Attributes:
        ab_chain: Antibody chain ('H' or 'L')
        ab_position: Antibody position
        ab_residue: Antibody amino acid
        ag_position: Antigen position
        ag_residue: Antigen amino acid
        attention_score: Attention weight between them
        ab_region: CDR or Framework region
        percentile: Where this score ranks (e.g., 95 = top 5%)
    """
    ab_chain: str
    ab_position: int
    ab_residue: str
    ag_position: int
    ag_residue: str
    attention_score: float
    ab_region: str
    percentile: float
    
    @property
    def ab_id(self) -> str:
        return f"{self.ab_chain}:{self.ab_residue}{self.ab_position + 1}"
    
    @property
    def ag_id(self) -> str:
        return f"AG:{self.ag_residue}{self.ag_position + 1}"
    
    def __repr__(self):
        return f"{self.ab_id} ↔ {self.ag_id}: {self.attention_score:.3f} (top {100 - self.percentile:.1f}%)"
    
    def to_dict(self) -> Dict:
        return {
            "antibody": {
                "chain": self.ab_chain,
                "position": self.ab_position,
                "residue": self.ab_residue,
                "region": self.ab_region,
            },
            "antigen": {
                "position": self.ag_position,
                "residue": self.ag_residue,
            },
            "attention_score": self.attention_score,
            "percentile": self.percentile,
        }
    
    def format_display(self) -> str:
        """Generate formatted display string."""
        bar_len = int(self.attention_score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        
        lines = [
            f"┌─ RESIDUE PAIR ATTENTION ─────────────────────────",
            f"│",
            f"│ Antibody:  {self.ab_id} ({self.ab_region})",
            f"│            ↓",
            f"│ Antigen:   {self.ag_id}",
            f"│",
            f"│ Attention: {bar} {self.attention_score:.4f}",
            f"│ Percentile: Top {100 - self.percentile:.1f}% of all pairs",
            f"│",
            f"└──────────────────────────────────────────────────",
        ]
        
        return "\n".join(lines)


# =============================================================================
# QUERYABLE ATTENTION MATRIX
# =============================================================================

class QueryableAttentionMatrix:
    """Stores full attention matrix and enables arbitrary queries.
    
    This class provides the core functionality for:
        a) Arbitrary residue query: Click any residue → see what it attends to
        b) Attention score for ANY residue pair on demand
    
    Example:
        >>> # Create from sequences and attention matrix
        >>> qam = QueryableAttentionMatrix(
        ...     heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS...",
        ...     light_chain="DIQMTQSPSSLSASVGDRVTITC...",
        ...     antigen="MKAKLLVLLCTFTATYA...",
        ...     attention_matrix=attention,  # (Ab_len × Ag_len)
        ... )
        >>> 
        >>> # Query antibody residue G56 in heavy chain
        >>> result = qam.query_antibody_residue("H", 56)
        >>> print(result.format_display())
        >>> 
        >>> # Get attention between specific pair
        >>> score = qam.get_attention_score("H", 56, 68)
        >>> print(f"H:G56 → AG:68 attention = {score:.3f}")
        >>> 
        >>> # Query which antibody residues contact antigen position 68
        >>> contacts = qam.query_antigen_residue(68)
        >>> print(contacts.format_display())
    """
    
    def __init__(
        self,
        heavy_chain: str,
        light_chain: str,
        antigen: str,
        attention_matrix: np.ndarray,
        normalize: bool = True,
    ):
        """Initialize queryable attention matrix.
        
        Args:
            heavy_chain: Heavy chain amino acid sequence
            light_chain: Light chain amino acid sequence
            antigen: Antigen amino acid sequence
            attention_matrix: Cross-attention matrix of shape (Ab_len, Ag_len)
                              where Ab_len = len(heavy) + len(light)
            normalize: Whether to normalize attention rows to sum to 1
        """
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.antigen = antigen
        
        self.heavy_len = len(heavy_chain)
        self.light_len = len(light_chain)
        self.ab_len = self.heavy_len + self.light_len
        self.ag_len = len(antigen)
        
        # Store and optionally normalize attention matrix
        self._attention = attention_matrix.copy()
        
        # Ensure correct shape
        if self._attention.shape != (self.ab_len, self.ag_len):
            logger.warning(f"Reshaping attention matrix from {self._attention.shape} to ({self.ab_len}, {self.ag_len})")
            self._attention = np.resize(self._attention, (self.ab_len, self.ag_len))
        
        if normalize:
            # Normalize each row to sum to 1
            row_sums = self._attention.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1)
            self._attention = self._attention / row_sums
        
        # Pre-compute statistics for percentile calculations
        self._all_scores = self._attention.flatten()
        self._sorted_scores = np.sort(self._all_scores)
        
        # Pre-compute CDR masks
        self._build_region_masks()
    
    def _build_region_masks(self):
        """Build masks for CDR and framework regions."""
        self._ab_regions = {}  # position -> (region_name, region_type)
        
        # Heavy chain
        for pos in range(self.heavy_len):
            region_name = "Framework"
            region_type = "Framework"
            
            for cdr_name, (start, end, full_name) in CDR_REGIONS["H"].items():
                if start <= pos <= end:
                    region_name = full_name
                    region_type = cdr_name
                    break
            
            self._ab_regions[("H", pos)] = (region_name, region_type)
        
        # Light chain
        for pos in range(self.light_len):
            region_name = "Framework"
            region_type = "Framework"
            
            for cdr_name, (start, end, full_name) in CDR_REGIONS["L"].items():
                if start <= pos <= end:
                    region_name = full_name
                    region_type = cdr_name
                    break
            
            self._ab_regions[("L", pos)] = (region_name, region_type)
    
    def _get_ab_index(self, chain: str, position: int) -> int:
        """Convert chain + position to matrix row index."""
        if chain.upper() == "H":
            if position >= self.heavy_len:
                raise ValueError(f"Position {position} out of range for heavy chain (len={self.heavy_len})")
            return position
        elif chain.upper() == "L":
            if position >= self.light_len:
                raise ValueError(f"Position {position} out of range for light chain (len={self.light_len})")
            return self.heavy_len + position
        else:
            raise ValueError(f"Invalid chain: {chain}. Use 'H' or 'L'")
    
    def _get_chain_and_position(self, ab_index: int) -> Tuple[str, int]:
        """Convert matrix row index to chain + position."""
        if ab_index < self.heavy_len:
            return "H", ab_index
        else:
            return "L", ab_index - self.heavy_len
    
    def _get_percentile(self, score: float) -> float:
        """Get percentile rank of a score."""
        return np.searchsorted(self._sorted_scores, score) / len(self._sorted_scores) * 100
    
    # =========================================================================
    # CORE QUERY METHODS
    # =========================================================================
    
    def query_antibody_residue(
        self,
        chain: str,
        position: int,
        top_k: int = 10,
    ) -> AntibodyResidueQuery:
        """Query what antigen positions an antibody residue attends to.
        
        This implements: "Click any antibody residue → see attention targets"
        
        Args:
            chain: 'H' for heavy chain, 'L' for light chain
            position: 0-indexed position in the chain
            top_k: Number of top targets to return
            
        Returns:
            AntibodyResidueQuery with attention information
            
        Example:
            >>> result = qam.query_antibody_residue("H", 56)
            >>> print(f"{result.residue_id} attends to {result.top_target_str}")
            H:G57 attends to AG:H69 (0.798)
        """
        chain = chain.upper()
        ab_index = self._get_ab_index(chain, position)
        
        # Get residue
        if chain == "H":
            residue = self.heavy_chain[position]
        else:
            residue = self.light_chain[position]
        
        # Get attention vector to all antigen positions
        attention_vector = self._attention[ab_index, :]
        
        # Get region info
        region_name, region_type = self._ab_regions.get((chain, position), ("Unknown", "Unknown"))
        
        # Sort by attention
        sorted_indices = np.argsort(attention_vector)[::-1]
        
        # Build targets list
        all_targets = []
        for rank, ag_idx in enumerate(sorted_indices[:top_k], 1):
            all_targets.append(AttentionTarget(
                position=int(ag_idx),
                residue=self.antigen[ag_idx] if ag_idx < len(self.antigen) else "X",
                attention_score=float(attention_vector[ag_idx]),
                rank=rank,
            ))
        
        return AntibodyResidueQuery(
            chain=chain,
            position=position,
            residue=residue,
            region=region_name,
            region_type=region_type,
            top_target=all_targets[0] if all_targets else AttentionTarget(0, "X", 0.0),
            all_targets=all_targets,
            attention_vector=attention_vector,
        )
    
    def query_antigen_residue(
        self,
        position: int,
        top_k: int = 10,
    ) -> AntigenResidueQuery:
        """Query which antibody residues attend to an antigen position.
        
        This implements: "Click any antigen residue → see antibody contacts"
        
        Args:
            position: 0-indexed position in antigen
            top_k: Number of top contacts to return
            
        Returns:
            AntigenResidueQuery with contact information
            
        Example:
            >>> result = qam.query_antigen_residue(68)
            >>> print(f"{result.residue_id} is contacted by:")
            >>> for contact in result.top_contacts[:3]:
            ...     print(f"  {contact['id']} ({contact['score']:.3f})")
        """
        if position >= self.ag_len:
            raise ValueError(f"Position {position} out of range for antigen (len={self.ag_len})")
        
        residue = self.antigen[position]
        
        # Get attention from all antibody positions to this antigen position
        attention_vector = self._attention[:, position]
        
        # Total attention to this position
        total_attention = float(attention_vector.sum())
        
        # Sort by attention
        sorted_indices = np.argsort(attention_vector)[::-1]
        
        # Build contacts list
        top_contacts = []
        for ab_idx in sorted_indices[:top_k]:
            chain, pos = self._get_chain_and_position(ab_idx)
            
            if chain == "H":
                ab_residue = self.heavy_chain[pos] if pos < self.heavy_len else "X"
            else:
                ab_residue = self.light_chain[pos] if pos < self.light_len else "X"
            
            region_name, region_type = self._ab_regions.get((chain, pos), ("Unknown", "Unknown"))
            
            top_contacts.append({
                "id": f"{chain}:{ab_residue}{pos + 1}",
                "chain": chain,
                "position": pos,
                "residue": ab_residue,
                "region": region_name,
                "region_type": region_type,
                "score": float(attention_vector[ab_idx]),
            })
        
        return AntigenResidueQuery(
            position=position,
            residue=residue,
            top_contacts=top_contacts,
            total_attention=total_attention,
            attention_vector=attention_vector,
        )
    
    def get_attention_score(
        self,
        ab_chain: str,
        ab_position: int,
        ag_position: int,
    ) -> float:
        """Get attention score for a specific antibody-antigen residue pair.
        
        This implements: "Get attention score for ANY residue pair on demand"
        
        Args:
            ab_chain: 'H' for heavy, 'L' for light
            ab_position: 0-indexed position in antibody chain
            ag_position: 0-indexed position in antigen
            
        Returns:
            Attention score (float between 0 and 1)
            
        Example:
            >>> score = qam.get_attention_score("H", 56, 68)
            >>> print(f"Attention: {score:.4f}")
            0.7983
        """
        ab_index = self._get_ab_index(ab_chain, ab_position)
        
        if ag_position >= self.ag_len:
            raise ValueError(f"Antigen position {ag_position} out of range (len={self.ag_len})")
        
        return float(self._attention[ab_index, ag_position])
    
    def query_pair(
        self,
        ab_chain: str,
        ab_position: int,
        ag_position: int,
    ) -> ResidueResiduePairQuery:
        """Get detailed information about a specific residue pair.
        
        Args:
            ab_chain: 'H' for heavy, 'L' for light
            ab_position: 0-indexed position in antibody chain
            ag_position: 0-indexed position in antigen
            
        Returns:
            ResidueResiduePairQuery with detailed pair information
            
        Example:
            >>> pair = qam.query_pair("H", 56, 68)
            >>> print(pair.format_display())
        """
        ab_chain = ab_chain.upper()
        ab_index = self._get_ab_index(ab_chain, ab_position)
        
        if ag_position >= self.ag_len:
            raise ValueError(f"Antigen position {ag_position} out of range (len={self.ag_len})")
        
        # Get residues
        if ab_chain == "H":
            ab_residue = self.heavy_chain[ab_position]
        else:
            ab_residue = self.light_chain[ab_position]
        
        ag_residue = self.antigen[ag_position]
        
        # Get attention score
        attention_score = float(self._attention[ab_index, ag_position])
        
        # Get region
        region_name, _ = self._ab_regions.get((ab_chain, ab_position), ("Unknown", "Unknown"))
        
        # Get percentile
        percentile = self._get_percentile(attention_score)
        
        return ResidueResiduePairQuery(
            ab_chain=ab_chain,
            ab_position=ab_position,
            ab_residue=ab_residue,
            ag_position=ag_position,
            ag_residue=ag_residue,
            attention_score=attention_score,
            ab_region=region_name,
            percentile=percentile,
        )
    
    # =========================================================================
    # BULK QUERY METHODS
    # =========================================================================
    
    def get_all_cdr_contacts(
        self,
        cdr_name: str = "CDR3",
        chain: str = "H",
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get all antigen contacts for a specific CDR.
        
        Args:
            cdr_name: 'CDR1', 'CDR2', or 'CDR3'
            chain: 'H' or 'L'
            top_k: Number of top contacts to return
            
        Returns:
            List of contact dictionaries sorted by attention
        """
        chain = chain.upper()
        cdr_key = cdr_name.upper().replace("-", "").replace(" ", "")
        
        if cdr_key not in CDR_REGIONS.get(chain, {}):
            raise ValueError(f"Unknown CDR: {cdr_name} for chain {chain}")
        
        start, end, full_name = CDR_REGIONS[chain][cdr_key]
        
        contacts = []
        for pos in range(start, min(end + 1, self.heavy_len if chain == "H" else self.light_len)):
            ab_index = self._get_ab_index(chain, pos)
            attention_row = self._attention[ab_index, :]
            
            top_ag_idx = np.argmax(attention_row)
            top_score = attention_row[top_ag_idx]
            
            if chain == "H":
                ab_residue = self.heavy_chain[pos] if pos < self.heavy_len else "X"
            else:
                ab_residue = self.light_chain[pos] if pos < self.light_len else "X"
            
            contacts.append({
                "ab_id": f"{chain}:{ab_residue}{pos + 1}",
                "ab_position": pos,
                "ag_id": f"AG:{self.antigen[top_ag_idx]}{top_ag_idx + 1}",
                "ag_position": int(top_ag_idx),
                "attention": float(top_score),
            })
        
        # Sort by attention
        contacts.sort(key=lambda x: -x["attention"])
        
        return contacts[:top_k]
    
    def get_epitope_hotspots(
        self,
        threshold_percentile: float = 90.0,
    ) -> List[Dict[str, Any]]:
        """Get antigen positions with highest total attention (epitope hotspots).
        
        Args:
            threshold_percentile: Return positions above this percentile
            
        Returns:
            List of hotspot positions with attention statistics
        """
        # Sum attention from all antibody positions
        ag_total_attention = self._attention.sum(axis=0)
        
        # Normalize
        ag_total_attention = ag_total_attention / (ag_total_attention.max() + 1e-8)
        
        # Find threshold
        threshold = np.percentile(ag_total_attention, threshold_percentile)
        
        hotspots = []
        for pos in range(self.ag_len):
            if ag_total_attention[pos] >= threshold:
                # Get top antibody contact
                ab_attention = self._attention[:, pos]
                top_ab_idx = np.argmax(ab_attention)
                chain, ab_pos = self._get_chain_and_position(top_ab_idx)
                
                if chain == "H":
                    ab_residue = self.heavy_chain[ab_pos] if ab_pos < self.heavy_len else "X"
                else:
                    ab_residue = self.light_chain[ab_pos] if ab_pos < self.light_len else "X"
                
                region_name, _ = self._ab_regions.get((chain, ab_pos), ("Unknown", "Unknown"))
                
                hotspots.append({
                    "position": pos,
                    "residue": self.antigen[pos],
                    "total_attention": float(ag_total_attention[pos]),
                    "top_contact": f"{chain}:{ab_residue}{ab_pos + 1}",
                    "top_contact_region": region_name,
                    "top_contact_score": float(ab_attention[top_ab_idx]),
                })
        
        # Sort by attention
        hotspots.sort(key=lambda x: -x["total_attention"])
        
        return hotspots
    
    def get_paratope_hotspots(
        self,
        threshold_percentile: float = 90.0,
    ) -> List[Dict[str, Any]]:
        """Get antibody positions with highest total attention (paratope hotspots).
        
        Args:
            threshold_percentile: Return positions above this percentile
            
        Returns:
            List of hotspot positions with attention statistics
        """
        # Sum attention to all antigen positions
        ab_total_attention = self._attention.sum(axis=1)
        
        # Normalize
        ab_total_attention = ab_total_attention / (ab_total_attention.max() + 1e-8)
        
        # Find threshold
        threshold = np.percentile(ab_total_attention, threshold_percentile)
        
        hotspots = []
        for ab_idx in range(self.ab_len):
            if ab_total_attention[ab_idx] >= threshold:
                chain, pos = self._get_chain_and_position(ab_idx)
                
                if chain == "H":
                    residue = self.heavy_chain[pos] if pos < self.heavy_len else "X"
                else:
                    residue = self.light_chain[pos] if pos < self.light_len else "X"
                
                region_name, region_type = self._ab_regions.get((chain, pos), ("Unknown", "Unknown"))
                
                # Get top antigen contact
                ag_attention = self._attention[ab_idx, :]
                top_ag_idx = np.argmax(ag_attention)
                
                hotspots.append({
                    "chain": chain,
                    "position": pos,
                    "residue": residue,
                    "region": region_name,
                    "region_type": region_type,
                    "total_attention": float(ab_total_attention[ab_idx]),
                    "top_target": f"AG:{self.antigen[top_ag_idx]}{top_ag_idx + 1}",
                    "top_target_score": float(ag_attention[top_ag_idx]),
                })
        
        # Sort by attention
        hotspots.sort(key=lambda x: -x["total_attention"])
        
        return hotspots
    
    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for JSON serialization."""
        return {
            "sequences": {
                "heavy_chain": self.heavy_chain,
                "light_chain": self.light_chain,
                "antigen": self.antigen,
            },
            "dimensions": {
                "heavy_len": self.heavy_len,
                "light_len": self.light_len,
                "ab_len": self.ab_len,
                "ag_len": self.ag_len,
            },
            "statistics": {
                "mean_attention": float(self._attention.mean()),
                "max_attention": float(self._attention.max()),
                "std_attention": float(self._attention.std()),
            },
        }
    
    def get_attention_matrix(self) -> np.ndarray:
        """Return a copy of the full attention matrix."""
        return self._attention.copy()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_queryable_attention(
    heavy_chain: str,
    light_chain: str,
    antigen: str,
    mammal_adapter: Optional[Any] = None,
    attention_matrix: Optional[np.ndarray] = None,
) -> QueryableAttentionMatrix:
    """Create a QueryableAttentionMatrix from sequences.
    
    If attention_matrix is not provided, will attempt to get it from
    MAMMAL adapter or generate heuristic attention.
    
    Args:
        heavy_chain: Heavy chain sequence
        light_chain: Light chain sequence
        antigen: Antigen sequence
        mammal_adapter: Optional MAMMALAdapter for getting real attention
        attention_matrix: Optional pre-computed attention matrix
        
    Returns:
        QueryableAttentionMatrix ready for queries
    """
    if attention_matrix is not None:
        return QueryableAttentionMatrix(
            heavy_chain=heavy_chain,
            light_chain=light_chain,
            antigen=antigen,
            attention_matrix=attention_matrix,
        )
    
    # Try to get from MAMMAL
    if mammal_adapter is not None:
        try:
            result = mammal_adapter.predict_binding(
                heavy_chain=heavy_chain,
                light_chain=light_chain,
                ha_sequence=antigen,
                return_attention=True,
            )
            
            if "attention" in result and result["attention"] is not None:
                attn = result["attention"]
                if hasattr(attn, 'cpu'):
                    attn = attn.cpu().numpy()
                
                # Extract cross-attention (may need processing)
                ab_len = len(heavy_chain) + len(light_chain)
                ag_len = len(antigen)
                
                if attn.shape[-2] >= ab_len and attn.shape[-1] >= ag_len:
                    attention_matrix = attn[..., :ab_len, :ag_len]
                    if attention_matrix.ndim > 2:
                        attention_matrix = attention_matrix.mean(axis=tuple(range(attention_matrix.ndim - 2)))
                    
                    return QueryableAttentionMatrix(
                        heavy_chain=heavy_chain,
                        light_chain=light_chain,
                        antigen=antigen,
                        attention_matrix=attention_matrix,
                    )
        except Exception as e:
            logger.warning(f"Could not get attention from MAMMAL: {e}")
    
    # Generate heuristic attention
    logger.info("Generating heuristic attention matrix")
    attention_matrix = _generate_heuristic_attention(heavy_chain, light_chain, antigen)
    
    return QueryableAttentionMatrix(
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        antigen=antigen,
        attention_matrix=attention_matrix,
    )


def _generate_heuristic_attention(
    heavy_chain: str,
    light_chain: str,
    antigen: str,
) -> np.ndarray:
    """Generate heuristic attention based on CDRs and known patterns."""
    ab_len = len(heavy_chain) + len(light_chain)
    ag_len = len(antigen)
    
    # Base attention with low values
    attention = np.random.rand(ab_len, ag_len) * 0.05 + 0.01
    
    # Known antigenic sites
    antigenic_sites = {
        "Sa": list(range(128, 132)) + list(range(156, 160)),
        "Sb": list(range(187, 198)),
        "Ca1": list(range(169, 173)),
        "Ca2": list(range(140, 145)),
        "Cb": list(range(74, 79)),
        "Stem": list(range(18, 21)) + list(range(38, 46)),
    }
    
    all_antigenic_positions = []
    for positions in antigenic_sites.values():
        all_antigenic_positions.extend([p for p in positions if p < ag_len])
    
    # High attention from CDRs to antigenic sites
    for cdr_name, (start, end, _) in CDR_REGIONS["H"].items():
        weight = 0.9 if cdr_name == "CDR3" else 0.6
        for ab_pos in range(start, min(end + 1, len(heavy_chain))):
            for ag_pos in all_antigenic_positions:
                attention[ab_pos, ag_pos] = np.random.uniform(weight * 0.7, weight)
    
    offset = len(heavy_chain)
    for cdr_name, (start, end, _) in CDR_REGIONS["L"].items():
        weight = 0.7 if cdr_name == "CDR3" else 0.5
        for ab_pos in range(start, min(end + 1, len(light_chain))):
            for ag_pos in all_antigenic_positions:
                attention[offset + ab_pos, ag_pos] = np.random.uniform(weight * 0.7, weight)
    
    return attention


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Demonstrate queryable attention functionality."""
    print("=" * 70)
    print("QUERYABLE ATTENTION MATRIX")
    print("=" * 70)
    
    # Example sequences
    heavy_chain = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARGRYTGYYFDYWGQGTTVTVSS"
    light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
    antigen = "MKAKLLVLLCTFTATYADTICIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKLRGVAPLHLGKCNIAGWILGNPECESLSTASSWSYIVETPSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAACPHAGAKSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPSTSADQQSLYQNADAYVFVGSSRYSKKFKPEIAIRPKVRDQEGRMNYYWTLVEPGDKITFEATGNLVVPRYAFAMERNAGSGIIISDTPVHDCNTTCQTPKGAINTSLPFQNIHPITIGKCPKYVKSTKLRLATGLRNIPSIQSR"
    
    print(f"\nSequences:")
    print(f"  Heavy: {len(heavy_chain)} aa")
    print(f"  Light: {len(light_chain)} aa")
    print(f"  Antigen: {len(antigen)} aa")
    
    # Create queryable attention matrix
    print("\nCreating queryable attention matrix...")
    qam = create_queryable_attention(heavy_chain, light_chain, antigen)
    
    # ==========================================================================
    # DEMO: Query antibody residue (Feature A)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FEATURE A: Query Any Antibody Residue")
    print("=" * 70)
    
    # Query G56 in heavy chain (CDR2)
    result = qam.query_antibody_residue("H", 56)
    print(result.format_display())
    
    # Query position in CDR3
    result = qam.query_antibody_residue("H", 98)
    print(result.format_display())
    
    # ==========================================================================
    # DEMO: Query antigen residue
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Query Any Antigen Residue")
    print("=" * 70)
    
    # Query antigen position 68
    result = qam.query_antigen_residue(68)
    print(result.format_display())
    
    # Query position in antigenic site
    result = qam.query_antigen_residue(140)
    print(result.format_display())
    
    # ==========================================================================
    # DEMO: Get attention for specific pair (Feature B)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FEATURE B: Attention Score for ANY Residue Pair")
    print("=" * 70)
    
    # Query specific pairs
    pairs_to_query = [
        ("H", 56, 68),
        ("H", 98, 140),
        ("H", 100, 142),
        ("L", 91, 68),
        ("L", 55, 140),
    ]
    
    print("\nQuerying specific residue pairs:")
    print("-" * 60)
    for chain, ab_pos, ag_pos in pairs_to_query:
        pair = qam.query_pair(chain, ab_pos, ag_pos)
        print(f"  {pair.ab_id:12s} → {pair.ag_id:10s}  "
              f"Attention: {pair.attention_score:.4f}  "
              f"(Top {100 - pair.percentile:.1f}%)")
    
    # ==========================================================================
    # DEMO: Detailed pair query
    # ==========================================================================
    print("\n" + "-" * 60)
    pair = qam.query_pair("H", 98, 140)
    print(pair.format_display())
    
    # ==========================================================================
    # DEMO: Bulk queries
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BULK QUERIES")
    print("=" * 70)
    
    # Epitope hotspots
    print("\nTop Epitope Hotspots (Antigen):")
    hotspots = qam.get_epitope_hotspots(threshold_percentile=85)
    for i, h in enumerate(hotspots[:5], 1):
        print(f"  {i}. AG:{h['residue']}{h['position']+1:3d}  "
              f"Total: {h['total_attention']:.3f}  "
              f"Top contact: {h['top_contact']} ({h['top_contact_region']})")
    
    # CDR3 contacts
    print("\nCDR-H3 Contacts:")
    contacts = qam.get_all_cdr_contacts("CDR3", "H", top_k=5)
    for c in contacts:
        print(f"  {c['ab_id']} → {c['ag_id']}  Attention: {c['attention']:.3f}")
    
    print("\n" + "=" * 70)
    print("✓ All queries complete!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
