"""Evidence quality scoring for BRAC.

This module computes quality scores (Q, C, S) for evidence:
- Q: Data quality - technical quality of input data
- C: Coverage - fraction of expected evidence present
- S: Consistency - internal coherence of evidence
"""

from typing import Optional
import torch

from brac.types import EvidenceQuality, SemanticEvidence, Modality


def compute_data_quality(
    evidence: list[SemanticEvidence],
    modality: Modality,
) -> float:
    """Compute data quality score Q.

    Based on technical quality of evidence items:
    - Confidence scores
    - Quality scores from evidence
    - Modality-specific criteria

    Args:
        evidence: List of semantic evidence items
        modality: The source modality

    Returns:
        Quality score Q in [0, 1]
    """
    if not evidence:
        return 0.5  # Default when no evidence

    # Average quality scores from evidence items
    quality_scores = [e.quality_score for e in evidence]
    avg_quality = sum(quality_scores) / len(quality_scores)

    # Weight by confidence
    confidence_scores = [e.confidence for e in evidence]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)

    # Combined score
    Q = 0.6 * avg_quality + 0.4 * avg_confidence

    return min(1.0, max(0.0, Q))


def compute_coverage(
    evidence: list[SemanticEvidence],
    modality: Modality,
    expected_findings: Optional[list[str]] = None,
) -> float:
    """Compute coverage score C.

    Measures what fraction of expected evidence types are present.

    Args:
        evidence: List of semantic evidence items
        modality: The source modality
        expected_findings: Optional list of expected finding types

    Returns:
        Coverage score C in [0, 1]
    """
    # Default expected findings by modality
    if expected_findings is None:
        expected_findings = _get_expected_findings(modality)

    if not expected_findings:
        return 1.0  # No expectations, full coverage

    # Count present finding types
    present_types = set(e.finding_type for e in evidence)
    expected_types = set(expected_findings)

    if not expected_types:
        return 1.0

    coverage = len(present_types & expected_types) / len(expected_types)

    return min(1.0, max(0.0, coverage))


def compute_consistency(
    evidence: list[SemanticEvidence],
    belief: Optional[torch.Tensor] = None,
) -> float:
    """Compute consistency score S.

    Measures internal coherence of evidence:
    - Agreement between evidence items
    - Consistency with belief distribution

    Args:
        evidence: List of semantic evidence items
        belief: Optional belief distribution for consistency check

    Returns:
        Consistency score S in [0, 1]
    """
    if not evidence:
        return 0.5

    # Check confidence variance (low variance = more consistent)
    confidences = [e.confidence for e in evidence]
    if len(confidences) > 1:
        variance = torch.tensor(confidences).var().item()
        consistency_from_variance = 1.0 - min(1.0, variance * 4)  # Scale variance
    else:
        consistency_from_variance = 1.0

    # Check if evidence types agree (same provenance should be consistent)
    provenances = [e.provenance for e in evidence]
    if len(set(provenances)) == 1:
        consistency_from_provenance = 1.0
    else:
        consistency_from_provenance = 0.8  # Multiple sources, slightly less consistent

    S = 0.7 * consistency_from_variance + 0.3 * consistency_from_provenance

    return min(1.0, max(0.0, S))


def compute_quality_scores(
    evidence: list[SemanticEvidence],
    modality: Modality,
    belief: Optional[torch.Tensor] = None,
) -> EvidenceQuality:
    """Compute all quality scores (Q, C, S) for evidence.

    Args:
        evidence: List of semantic evidence items
        modality: The source modality
        belief: Optional belief distribution

    Returns:
        EvidenceQuality with Q, C, S scores
    """
    Q = compute_data_quality(evidence, modality)
    C = compute_coverage(evidence, modality)
    S = compute_consistency(evidence, belief)

    return EvidenceQuality(Q=Q, C=C, S=S)


def _get_expected_findings(modality: Modality) -> list[str]:
    """Get expected finding types for a modality.

    Args:
        modality: The diagnostic modality

    Returns:
        List of expected finding type strings
    """
    expected = {
        Modality.PATHOLOGY: [
            "morphology",
            "immunophenotype",
            "molecular",
        ],
        Modality.RADIOLOGY: [
            "pet_avidity",
            "ct_findings",
            "nodal_involvement",
        ],
        Modality.LABORATORY: [
            "flow_cytometry",
            "biochemistry",
            "hematology",
        ],
        Modality.CLINICAL: [
            "demographics",
            "staging",
            "symptoms",
        ],
    }
    return expected.get(modality, [])
