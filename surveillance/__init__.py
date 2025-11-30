"""
BEREAN Surveillance Module
==========================

Status: NEW
Source: berean_v2_refactor

Integration with influenza surveillance databases for
real-time monitoring of circulating strains and escape prediction.

Components:
    - GISAID integration for influenza HA sequences
    - Variant tracking and clustering
    - Escape mutation prediction
    - Vaccine strain matching

Note: GISAID access requires credentials and data sharing agreement.
"""

from .gisaid import (
    GISAIDClient,
    InfluenzaSequence,
    download_sequences,
    parse_fasta,
)

__all__ = [
    "GISAIDClient",
    "InfluenzaSequence",
    "download_sequences",
    "parse_fasta",
]
