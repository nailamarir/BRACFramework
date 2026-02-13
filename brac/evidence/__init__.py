"""Evidence processing modules for BRAC.

This package handles evidence ingestion and semantic normalization:
- ingestion: Raw data loading per modality
- semantic: Semantic evidence tuples (SNOMED-CT, ICD-O-3)
- quality: Quality scoring (Q, C, S)
"""

from brac.evidence.quality import compute_quality_scores

__all__ = ["compute_quality_scores"]
