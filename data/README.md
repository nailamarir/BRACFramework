# BRAC Framework Data Directory

This directory contains data for the Byzantine-Resilient Agentic Consensus (BRAC) framework for Non-Hodgkin Lymphoma subtyping.

## Directory Structure

```
data/
├── pathology/          # Whole slide images (WSI) and pathology features
├── radiology/          # PET/CT scans and radiology features
├── laboratory/         # Flow cytometry and lab results
├── clinical/           # Structured clinical data
├── reference/          # WHO classification and reference data
│   ├── who_nhl_classification.csv           # Core 9 subtypes (BRAC compatible)
│   ├── who_nhl_classification_extended.csv  # Extended WHO-HAEM5 classification
│   └── brac_subtype_mapping.csv             # Mapping between BRAC indices and WHO entities
└── manifest.csv        # Case manifest linking all modalities
```

## WHO 2022 (WHO-HAEM5) B-Cell NHL Classification

The BRAC framework uses 9 primary B-cell NHL subtypes based on the WHO 2022 classification:

| Index | Code | Name | Aggressiveness |
|-------|------|------|----------------|
| 0 | DLBCL_GCB | Diffuse Large B-Cell Lymphoma (GCB) | Aggressive |
| 1 | DLBCL_ABC | Diffuse Large B-Cell Lymphoma (ABC) | Aggressive |
| 2 | FL | Follicular Lymphoma | Indolent |
| 3 | MCL | Mantle Cell Lymphoma | Aggressive |
| 4 | BL | Burkitt Lymphoma | Highly Aggressive |
| 5 | MZL | Marginal Zone Lymphoma | Indolent |
| 6 | CLL_SLL | CLL/Small Lymphocytic Lymphoma | Indolent |
| 7 | LPL | Lymphoplasmacytic Lymphoma | Indolent |
| 8 | PMBL | Primary Mediastinal Large B-Cell Lymphoma | Aggressive |

## Data Requirements by Modality

### Pathology (`pathology/`)
- **Format**: Whole slide images (SVS, TIFF, NDPI) or pre-extracted features
- **Required**: H&E stained slides, IHC panels (CD20, CD10, BCL6, MUM1, Ki67, etc.)
- **File naming**: `{case_id}_HE.svs`, `{case_id}_CD20.svs`, etc.

### Radiology (`radiology/`)
- **Format**: DICOM, NIfTI, or pre-extracted features
- **Required**: PET/CT scans (SUVmax measurements, lesion locations)
- **File naming**: `{case_id}_PET.nii.gz`, `{case_id}_CT.nii.gz`

### Laboratory (`laboratory/`)
- **Format**: CSV, FCS (flow cytometry), or pre-extracted features
- **Required**: Flow cytometry panels, CBC, LDH, beta-2 microglobulin
- **File naming**: `{case_id}_flow.fcs`, `{case_id}_labs.csv`

### Clinical (`clinical/`)
- **Format**: CSV or JSON
- **Required**: Age, sex, B symptoms, stage, ECOG, IPI score, treatment history
- **File naming**: `{case_id}_clinical.csv`

## Manifest File Format

The `manifest.csv` file links all modalities for each case:

```csv
case_id,label,pathology_path,radiology_path,laboratory_path,clinical_path,split
CASE001,2,pathology/CASE001_features.pt,radiology/CASE001_pet.nii.gz,laboratory/CASE001_flow.fcs,clinical/CASE001.csv,train
CASE002,0,pathology/CASE002_features.pt,radiology/CASE002_pet.nii.gz,laboratory/CASE002_flow.fcs,clinical/CASE002.csv,train
...
```

## Data Sources

For real clinical data, consider:

1. **TCGA-DLBC**: The Cancer Genome Atlas Diffuse Large B-Cell Lymphoma
   - https://portal.gdc.cancer.gov/projects/TCGA-DLBC

2. **GEO Datasets**: Gene expression and flow cytometry data
   - https://www.ncbi.nlm.nih.gov/geo/

3. **TCIA**: The Cancer Imaging Archive (for radiology)
   - https://www.cancerimagingarchive.net/

4. **Institutional Data**: Partner with medical institutions for multimodal data

## Privacy and Compliance

- All patient data must be de-identified per HIPAA/GDPR
- Obtain appropriate IRB approval before using clinical data
- Do not commit actual patient data to version control

## References

- WHO Classification of Tumours Editorial Board. Haematolymphoid Tumours, WHO Classification of Tumours, 5th Edition, Volume 11. Lyon: IARC Press; 2022.
- Alaggio R, et al. The 5th edition of the World Health Organization Classification of Haematolymphoid Tumours: Lymphoid Neoplasms. Leukemia. 2022.
