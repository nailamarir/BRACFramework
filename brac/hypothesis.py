"""NHL hypothesis space based on WHO classification.

This module provides the hypothesis space for B-cell Non-Hodgkin Lymphoma
subtyping, including clinical characteristics and diagnostic criteria.
"""

from dataclasses import dataclass
from typing import Optional

from brac.types import NHLSubtype, Modality


@dataclass
class SubtypeCharacteristics:
    """Clinical and diagnostic characteristics of an NHL subtype.

    Attributes:
        subtype: The NHL subtype enum
        full_name: Full WHO classification name
        morphology: Key morphological features
        immunophenotype: Characteristic immunophenotype markers
        molecular: Key molecular/genetic features
        clinical: Typical clinical presentation
        primary_modality: Most diagnostically informative modality
    """
    subtype: NHLSubtype
    full_name: str
    morphology: list[str]
    immunophenotype: list[str]
    molecular: list[str]
    clinical: list[str]
    primary_modality: Modality


# WHO Classification Characteristics for B-cell NHL Subtypes
NHL_CHARACTERISTICS: dict[NHLSubtype, SubtypeCharacteristics] = {
    NHLSubtype.FL: SubtypeCharacteristics(
        subtype=NHLSubtype.FL,
        full_name="Follicular Lymphoma",
        morphology=["Follicular growth pattern", "Centrocytes and centroblasts", "Low mitotic rate"],
        immunophenotype=["CD20+", "CD10+", "BCL6+", "BCL2+", "CD5-", "CD23+/-"],
        molecular=["t(14;18) BCL2-IGH", "Grade 1-3A"],
        clinical=["Indolent course", "Generalized lymphadenopathy", "Bone marrow involvement common"],
        primary_modality=Modality.PATHOLOGY,
    ),
    NHLSubtype.MCL: SubtypeCharacteristics(
        subtype=NHLSubtype.MCL,
        full_name="Mantle Cell Lymphoma",
        morphology=["Small to medium lymphocytes", "Irregular nuclei", "Mantle zone pattern"],
        immunophenotype=["CD20+", "CD5+", "CD23-", "Cyclin D1+", "SOX11+"],
        molecular=["t(11;14) CCND1-IGH"],
        clinical=["Aggressive course", "GI tract involvement", "Leukemic spread"],
        primary_modality=Modality.PATHOLOGY,
    ),
    NHLSubtype.CLL_SLL: SubtypeCharacteristics(
        subtype=NHLSubtype.CLL_SLL,
        full_name="Chronic Lymphocytic Leukemia / Small Lymphocytic Lymphoma",
        morphology=["Small mature lymphocytes", "Proliferation centers", "Smudge cells (blood)"],
        immunophenotype=["CD20+ (dim)", "CD5+", "CD23+", "CD200+", "CD10-"],
        molecular=["del(13q)", "del(11q)", "del(17p)", "Trisomy 12"],
        clinical=["Lymphocytosis >5000/uL", "Generalized lymphadenopathy", "Autoimmune cytopenias"],
        primary_modality=Modality.LABORATORY,
    ),
    NHLSubtype.DLBCL_GCB: SubtypeCharacteristics(
        subtype=NHLSubtype.DLBCL_GCB,
        full_name="Diffuse Large B-Cell Lymphoma, Germinal Center B-cell type",
        morphology=["Large cells", "Diffuse growth", "High mitotic rate"],
        immunophenotype=["CD20+", "CD10+", "BCL6+", "MUM1-", "Hans algorithm: GCB"],
        molecular=["BCL2 rearrangement", "MYC rearrangement possible", "EZH2 mutations"],
        clinical=["Aggressive", "Rapidly enlarging mass", "B symptoms common"],
        primary_modality=Modality.PATHOLOGY,
    ),
    NHLSubtype.DLBCL_ABC: SubtypeCharacteristics(
        subtype=NHLSubtype.DLBCL_ABC,
        full_name="Diffuse Large B-Cell Lymphoma, Activated B-cell type",
        morphology=["Large cells", "Diffuse growth", "High mitotic rate"],
        immunophenotype=["CD20+", "CD10-", "BCL6+/-", "MUM1+", "Hans algorithm: non-GCB"],
        molecular=["MYD88 L265P", "CD79B mutations", "NF-kB activation"],
        clinical=["More aggressive than GCB", "Extranodal involvement", "Worse prognosis"],
        primary_modality=Modality.PATHOLOGY,
    ),
    NHLSubtype.MZL: SubtypeCharacteristics(
        subtype=NHLSubtype.MZL,
        full_name="Marginal Zone Lymphoma",
        morphology=["Small to medium cells", "Marginal zone pattern", "Monocytoid B cells"],
        immunophenotype=["CD20+", "CD5-", "CD10-", "CD23-", "CD43+/-"],
        molecular=["t(11;18) MALT1-API2", "t(14;18) MALT1-IGH", "NOTCH2 mutations"],
        clinical=["MALT lymphoma most common", "Extranodal sites", "Association with chronic inflammation"],
        primary_modality=Modality.PATHOLOGY,
    ),
    NHLSubtype.BL: SubtypeCharacteristics(
        subtype=NHLSubtype.BL,
        full_name="Burkitt Lymphoma",
        morphology=["Medium-sized cells", "Starry sky pattern", "Extremely high Ki-67 (>95%)"],
        immunophenotype=["CD20+", "CD10+", "BCL6+", "BCL2-", "Ki-67 >95%"],
        molecular=["MYC translocation t(8;14)", "Simple karyotype", "ID3/TCF3 mutations"],
        clinical=["Highly aggressive", "Bulky disease", "Tumor lysis syndrome risk"],
        primary_modality=Modality.RADIOLOGY,  # PET avidity characteristic
    ),
    NHLSubtype.LPL: SubtypeCharacteristics(
        subtype=NHLSubtype.LPL,
        full_name="Lymphoplasmacytic Lymphoma",
        morphology=["Small lymphocytes", "Plasmacytoid differentiation", "Dutcher bodies"],
        immunophenotype=["CD20+", "CD5-", "CD10-", "CD23-", "CD138+ (plasma cells)"],
        molecular=["MYD88 L265P (>90%)", "CXCR4 mutations"],
        clinical=["Waldenstrom macroglobulinemia", "IgM paraprotein", "Hyperviscosity syndrome"],
        primary_modality=Modality.LABORATORY,
    ),
    NHLSubtype.HCL: SubtypeCharacteristics(
        subtype=NHLSubtype.HCL,
        full_name="Hairy Cell Leukemia",
        morphology=["Medium cells with hairy projections", "Fried egg appearance", "Bone marrow fibrosis"],
        immunophenotype=["CD20+", "CD11c+", "CD25+", "CD103+", "CD123+", "Annexin A1+"],
        molecular=["BRAF V600E mutation (>95%)"],
        clinical=["Pancytopenia", "Splenomegaly", "Monocytopenia"],
        primary_modality=Modality.LABORATORY,
    ),
}


def get_characteristics(subtype: NHLSubtype) -> SubtypeCharacteristics:
    """Get clinical characteristics for a given NHL subtype.

    Args:
        subtype: The NHL subtype to look up

    Returns:
        SubtypeCharacteristics dataclass with clinical information
    """
    return NHL_CHARACTERISTICS[subtype]


def get_primary_modality(subtype: NHLSubtype) -> Modality:
    """Get the primary diagnostic modality for a subtype.

    Args:
        subtype: The NHL subtype

    Returns:
        The most informative diagnostic modality for this subtype
    """
    return NHL_CHARACTERISTICS[subtype].primary_modality


def get_subtype_names() -> dict[NHLSubtype, str]:
    """Get full names for all subtypes.

    Returns:
        Dictionary mapping subtypes to their WHO classification names
    """
    return {s: c.full_name for s, c in NHL_CHARACTERISTICS.items()}


def get_differential_diagnosis(immunophenotype: dict[str, bool]) -> list[NHLSubtype]:
    """Get differential diagnosis based on immunophenotype.

    A simplified helper that returns possible subtypes based on
    key immunophenotype markers.

    Args:
        immunophenotype: Dictionary of marker name to positive/negative

    Returns:
        List of possible NHL subtypes
    """
    candidates = list(NHLSubtype)

    # CD5+ narrows differential
    if immunophenotype.get("CD5", False):
        cd5_positive = [NHLSubtype.MCL, NHLSubtype.CLL_SLL]
        candidates = [s for s in candidates if s in cd5_positive]

    # CD10+ suggests follicle center origin
    if immunophenotype.get("CD10", False):
        cd10_positive = [NHLSubtype.FL, NHLSubtype.DLBCL_GCB, NHLSubtype.BL]
        candidates = [s for s in candidates if s in cd10_positive]

    # Cyclin D1+ is diagnostic for MCL
    if immunophenotype.get("CyclinD1", False):
        candidates = [NHLSubtype.MCL]

    return candidates
