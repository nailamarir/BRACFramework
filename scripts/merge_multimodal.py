#!/usr/bin/env python3
"""
Multi-Modal Data Merge Pipeline for BRAC Framework

Merges all preprocessed modalities into a unified patient-level dataset:
- Pathology (gold standard labels + IHC features)
- Laboratory (CBC features)
- Clinical (demographics + epidemiological priors)
- Radiology (imaging features)
- Bone Marrow (aspirate features)

Output:
- multimodal_full.csv: All patients with any modality
- multimodal_brac.csv: BRAC-compatible patients only
- multimodal_complete.csv: Patients with all 5 modalities
- feature_matrix.csv: Model-ready numeric features

Author: BRAC Framework
Date: 2026-02-13
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "merged"
OUTPUT_DIR.mkdir(exist_ok=True)

# Input files (patient-level)
INPUT_FILES = {
    'pathology': DATA_DIR / "pathology" / "pathology_cleaned_full.csv",
    'laboratory': DATA_DIR / "laboratory" / "laboratory_patient_features.csv",
    'clinical': DATA_DIR / "clinical" / "clinical_cleaned.csv",
    'radiology': DATA_DIR / "radiology" / "radiology_patient_level.csv",
    'bone_marrow': DATA_DIR / "clinical" / "bone_marrow_patient_level.csv",
}

# BRAC label names
BRAC_LABELS = {
    0: 'DLBCL_GCB',
    1: 'DLBCL_ABC',
    2: 'FL',
    3: 'MCL',
    4: 'BL',
    5: 'MZL',
    6: 'CLL_SLL',
    7: 'LPL',
    8: 'PMBL'
}

# ============================================================================
# STEP 1: LOAD ALL MODALITIES
# ============================================================================

def load_modalities():
    """Load all preprocessed modality files."""
    print("=" * 70)
    print("STEP 1: LOAD ALL MODALITIES")
    print("=" * 70)

    modalities = {}

    for name, path in INPUT_FILES.items():
        if path.exists():
            df = pd.read_csv(path)
            modalities[name] = df
            print(f"  {name}: {len(df)} patients, {len(df.columns)} columns")
        else:
            print(f"  {name}: FILE NOT FOUND - {path}")
            modalities[name] = None

    return modalities

# ============================================================================
# STEP 2: PREPARE PATHOLOGY (BASE TABLE)
# ============================================================================

def prepare_pathology(df):
    """Prepare pathology as the base table with BRAC labels."""
    print("\n" + "=" * 70)
    print("STEP 2: PREPARE PATHOLOGY (BASE TABLE)")
    print("=" * 70)

    if df is None:
        print("  ERROR: Pathology data not available")
        return None

    # Deduplicate to patient level (keep first record with BRAC label)
    df_brac = df[df['brac_label'].notna()].drop_duplicates(subset='patient_id', keep='first')
    df_other = df[df['brac_label'].isna()].drop_duplicates(subset='patient_id', keep='first')
    df_dedup = pd.concat([df_brac, df_other[~df_other['patient_id'].isin(df_brac['patient_id'])]])

    # Select key columns
    path_cols = [
        'patient_id',
        # Labels
        'disease_type_final', 'brac_label', 'is_brac_compatible',
        'who_category', 'who_name',
        # IHC markers
        'ihc_cd20', 'ihc_cd3', 'ihc_cd10', 'ihc_cd5', 'ihc_cd23',
        'ihc_bcl2', 'ihc_bcl6', 'ihc_mum1', 'ihc_ki67', 'ihc_cyclin_d1',
        # Numeric IHC
        'ihc_cd20_numeric', 'ihc_cd10_numeric', 'ihc_bcl6_numeric',
        'ihc_mum1_numeric', 'ihc_ki67_numeric', 'ki67_percentage',
        # Morphology
        'morph_large_cells', 'morph_small_cells', 'morph_atypical',
        'morph_diffuse_pattern', 'morph_nodular_pattern',
        # Quality
        'has_ihc', 'has_complete_hans',
        'quality_completeness', 'quality_evidence_strength',
        'label_confidence', 'label_ambiguity',
    ]

    # Filter to existing columns
    path_cols = [c for c in path_cols if c in df_dedup.columns]
    df_path = df_dedup[path_cols].copy()

    # Add prefix to avoid column conflicts (except patient_id and labels)
    rename_cols = {c: f'path_{c}' for c in df_path.columns
                   if c not in ['patient_id', 'disease_type_final', 'brac_label', 'is_brac_compatible', 'who_category']}
    df_path = df_path.rename(columns=rename_cols)

    print(f"  Deduplicated: {len(df)} -> {len(df_path)} patients")
    print(f"  BRAC-compatible: {df_path['is_brac_compatible'].sum()}")
    print(f"  Columns selected: {len(df_path.columns)}")

    return df_path

# ============================================================================
# STEP 3: PREPARE LABORATORY
# ============================================================================

def prepare_laboratory(df):
    """Prepare laboratory features for merge."""
    print("\n" + "=" * 70)
    print("STEP 3: PREPARE LABORATORY")
    print("=" * 70)

    if df is None:
        print("  ERROR: Laboratory data not available")
        return None

    # Drop duplicate BRAC columns (already in pathology)
    drop_cols = ['disease_type_final', 'brac_label', 'is_brac_compatible']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Select key columns
    lab_cols = [
        'patient_id',
        # Baseline values
        'wbc_baseline', 'hgb_baseline', 'plt_baseline', 'lymph_abs_baseline', 'neut_abs_baseline',
        # First/Last values
        'wbc_first', 'wbc_last', 'hgb_first', 'hgb_last', 'plt_first', 'plt_last',
        'lymph_abs_first', 'lymph_abs_last', 'neut_abs_first', 'neut_abs_last',
        # Mean values
        'wbc_mean', 'hgb_mean', 'plt_mean', 'lymph_abs_mean', 'neut_abs_mean',
        'mcv_mean', 'rdw_mean', 'nlr_mean', 'plr_mean', 'lmr_mean',
        # Min/Max
        'wbc_min', 'wbc_max', 'hgb_min', 'hgb_max', 'plt_min', 'plt_max',
        'lymph_abs_min', 'lymph_abs_max',
        # Ever flags
        'ever_anemia', 'ever_severe_anemia', 'ever_thrombocytopenia', 'ever_severe_tcp',
        'ever_leukocytosis', 'ever_leukopenia', 'ever_lymphocytosis',
        'ever_neutropenia', 'ever_severe_neutropenia', 'ever_high_nlr', 'ever_low_lmr',
        # Treatment phases
        'ever_on_treatment', 'ever_relapse',
        'n_phase_baseline', 'n_phase_on_treatment', 'n_phase_recovery', 'n_phase_stable', 'n_phase_relapse',
        # Temporal
        'n_cbc_tests', 'followup_days',
        # Quality
        'quality_score',
    ]

    # Filter to existing columns
    lab_cols = [c for c in lab_cols if c in df.columns]
    df_lab = df[lab_cols].copy()

    # Add prefix
    rename_cols = {c: f'lab_{c}' for c in df_lab.columns if c != 'patient_id'}
    df_lab = df_lab.rename(columns=rename_cols)

    print(f"  Patients: {len(df_lab)}")
    print(f"  Columns selected: {len(df_lab.columns)}")

    return df_lab

# ============================================================================
# STEP 4: PREPARE CLINICAL
# ============================================================================

def prepare_clinical(df):
    """Prepare clinical/demographic features for merge."""
    print("\n" + "=" * 70)
    print("STEP 4: PREPARE CLINICAL")
    print("=" * 70)

    if df is None:
        print("  ERROR: Clinical data not available")
        return None

    # Select key columns
    clin_cols = [
        'patient_id',
        # Age features
        'age_at_dx', 'age_primary', 'age_group', 'age_group_fine', 'age_group_elderly',
        'is_adult', 'is_elderly', 'is_very_elderly',
        # Location
        'region', 'nile_delta', 'is_main_catchment', 'is_rural', 'is_urban',
        # Epidemiological priors
        'prior_dlbcl_age', 'prior_fl_age', 'prior_cll_age', 'prior_mcl_age',
        'prior_bl_age', 'prior_mzl_age', 'prior_hcv_exposure', 'prior_smzl_regional',
        'prior_brac_0', 'prior_brac_1', 'prior_brac_2', 'prior_brac_3', 'prior_brac_4',
        'prior_brac_5', 'prior_brac_6', 'prior_brac_7', 'prior_brac_8',
        # Quality
        'quality_Q', 'coverage_C', 'consistency_S',
    ]

    # Filter to existing columns
    clin_cols = [c for c in clin_cols if c in df.columns]
    df_clin = df[clin_cols].copy()

    # Add prefix (except patient_id and priors which are already prefixed)
    rename_cols = {c: f'clin_{c}' for c in df_clin.columns
                   if c != 'patient_id' and not c.startswith('prior_')}
    df_clin = df_clin.rename(columns=rename_cols)

    print(f"  Patients: {len(df_clin)}")
    print(f"  Columns selected: {len(df_clin.columns)}")

    return df_clin

# ============================================================================
# STEP 5: PREPARE RADIOLOGY
# ============================================================================

def prepare_radiology(df):
    """Prepare radiology features for merge."""
    print("\n" + "=" * 70)
    print("STEP 5: PREPARE RADIOLOGY")
    print("=" * 70)

    if df is None:
        print("  ERROR: Radiology data not available")
        return None

    # Select key columns (matching actual column names)
    rad_cols = [
        'patient_id',
        # Exam counts
        'n_exams_total', 'n_exams_ct', 'n_exams_us', 'n_exams_pet_ct', 'n_exams_mri',
        # Lymph node involvement
        'ever_ln_cervical', 'ever_ln_axillary', 'ever_ln_mediastinal', 'ever_ln_hilar',
        'ever_ln_paraaortic', 'ever_ln_mesenteric', 'ever_ln_inguinal', 'ever_ln_pelvic',
        'max_ln_sites',
        # Staging
        'ever_above_diaphragm', 'ever_below_diaphragm', 'ever_both_sides',
        'ever_extranodal',
        # Organ involvement
        'ever_liver', 'ever_spleen', 'ever_bone', 'ever_lung',
        # Measurements
        'ever_bulky', 'ever_very_bulky', 'max_lesion_ever_mm',
        # PET features
        'ever_pet_avid',
        # Disease status
        'ever_complete_response', 'ever_partial_response', 'ever_stable_disease', 'ever_progressive_disease',
        # Quality
        'quality_Q_mean', 'coverage_C_mean', 'consistency_S_mean',
    ]

    # Filter to existing columns
    rad_cols = [c for c in rad_cols if c in df.columns]
    df_rad = df[rad_cols].copy()

    # Add prefix
    rename_cols = {c: f'rad_{c}' for c in df_rad.columns if c != 'patient_id'}
    df_rad = df_rad.rename(columns=rename_cols)

    print(f"  Patients: {len(df_rad)}")
    print(f"  Columns selected: {len(df_rad.columns)}")

    return df_rad

# ============================================================================
# STEP 6: PREPARE BONE MARROW
# ============================================================================

def prepare_bone_marrow(df):
    """Prepare bone marrow features for merge."""
    print("\n" + "=" * 70)
    print("STEP 6: PREPARE BONE MARROW")
    print("=" * 70)

    if df is None:
        print("  ERROR: Bone marrow data not available")
        return None

    # Select key columns (already prefixed with bm_)
    bm_cols = [
        'patient_id',
        # Exam info
        'bm_exam_count',
        # Baseline features
        'bm_baseline_cellularity', 'bm_baseline_lymphocytes_pct', 'bm_baseline_blasts_pct',
        'bm_baseline_myeloid_status', 'bm_baseline_erythroid_status',
        # Max values
        'bm_max_lymphocytes_pct', 'bm_max_blasts_pct', 'bm_max_plasma_cells_pct',
        # Mean values
        'bm_mean_lymphocytes_pct', 'bm_mean_myeloid_total', 'bm_mean_erythroid_total',
        # Ever flags
        'bm_ever_hypercellular', 'bm_ever_hypocellular',
        'bm_ever_lymph_suspicious', 'bm_ever_lymph_likely', 'bm_ever_lymph_cll_sll',
        'bm_ever_blast_excess', 'bm_ever_diffuse_infiltration',
        'bm_ever_myeloid_suppressed', 'bm_ever_erythroid_suppressed',
        # Quality
        'bm_mean_quality_Q', 'bm_mean_quality_C', 'bm_mean_quality_S', 'bm_quality_composite',
    ]

    # Filter to existing columns
    bm_cols = [c for c in bm_cols if c in df.columns]
    df_bm = df[bm_cols].copy()

    print(f"  Patients: {len(df_bm)}")
    print(f"  Columns selected: {len(df_bm.columns)}")

    return df_bm

# ============================================================================
# STEP 7: MERGE ALL MODALITIES
# ============================================================================

def merge_modalities(path_df, lab_df, clin_df, rad_df, bm_df):
    """Merge all modalities on patient_id."""
    print("\n" + "=" * 70)
    print("STEP 7: MERGE ALL MODALITIES")
    print("=" * 70)

    # Start with pathology as base (has labels)
    if path_df is None:
        print("  ERROR: Cannot merge without pathology data")
        return None

    merged = path_df.copy()
    print(f"  Base (pathology): {len(merged)} patients")

    # Merge each modality
    if lab_df is not None:
        merged = merged.merge(lab_df, on='patient_id', how='outer')
        print(f"  + Laboratory: {len(merged)} patients")

    if clin_df is not None:
        merged = merged.merge(clin_df, on='patient_id', how='outer')
        print(f"  + Clinical: {len(merged)} patients")

    if rad_df is not None:
        merged = merged.merge(rad_df, on='patient_id', how='outer')
        print(f"  + Radiology: {len(merged)} patients")

    if bm_df is not None:
        merged = merged.merge(bm_df, on='patient_id', how='outer')
        print(f"  + Bone Marrow: {len(merged)} patients")

    print(f"\n  Final merged: {len(merged)} patients, {len(merged.columns)} columns")

    return merged

# ============================================================================
# STEP 8: ADD MODALITY FLAGS
# ============================================================================

def add_modality_flags(df, modalities):
    """Add flags indicating which modalities are available per patient."""
    print("\n" + "=" * 70)
    print("STEP 8: ADD MODALITY FLAGS")
    print("=" * 70)

    df = df.copy()

    # Modality presence flags based on key columns
    modality_indicators = {
        'has_pathology': 'disease_type_final',
        'has_laboratory': 'lab_n_cbc_tests',
        'has_clinical': 'clin_age_primary',
        'has_radiology': 'rad_n_exams_total',
        'has_bone_marrow': 'bm_exam_count',
    }

    for flag, indicator_col in modality_indicators.items():
        if indicator_col in df.columns:
            df[flag] = df[indicator_col].notna().astype(int)
        else:
            df[flag] = 0

    # Count modalities per patient
    modality_cols = [c for c in df.columns if c.startswith('has_') and c != 'has_ihc' and c != 'has_complete_hans']
    df['n_modalities'] = df[modality_cols].sum(axis=1)

    # Flag for complete data (all 5 modalities)
    df['has_all_modalities'] = (df['n_modalities'] == 5).astype(int)

    # Print summary
    print("  Modality availability:")
    for flag in modality_cols:
        count = df[flag].sum()
        pct = 100 * count / len(df)
        print(f"    {flag}: {count} ({pct:.1f}%)")

    print(f"\n  Patients with all 5 modalities: {df['has_all_modalities'].sum()}")

    # Modality combination summary
    print("\n  Modality combinations (top 10):")
    combo_cols = ['has_pathology', 'has_laboratory', 'has_clinical', 'has_radiology', 'has_bone_marrow']
    combo_counts = df.groupby(combo_cols).size().sort_values(ascending=False).head(10)
    for combo, count in combo_counts.items():
        combo_str = ''.join(['P' if combo[0] else '-',
                            'L' if combo[1] else '-',
                            'C' if combo[2] else '-',
                            'R' if combo[3] else '-',
                            'B' if combo[4] else '-'])
        print(f"    {combo_str}: {count} patients")

    return df

# ============================================================================
# STEP 9: CREATE FEATURE MATRIX
# ============================================================================

def create_feature_matrix(df):
    """Create model-ready numeric feature matrix."""
    print("\n" + "=" * 70)
    print("STEP 9: CREATE FEATURE MATRIX")
    print("=" * 70)

    # Select numeric columns only (excluding IDs and categorical)
    exclude_patterns = ['patient_id', 'date', 'disease_type', 'who_', 'age_group', 'region',
                        'cellularity', 'status', '_en', '_ar']

    numeric_cols = []
    for col in df.columns:
        if any(pat in col for pat in exclude_patterns):
            continue
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)

    # Always include key identifiers
    key_cols = ['patient_id', 'brac_label', 'is_brac_compatible']
    key_cols = [c for c in key_cols if c in df.columns]

    feature_cols = key_cols + numeric_cols
    feature_df = df[feature_cols].copy()

    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Total columns: {len(feature_cols)}")

    # Missing value summary
    missing_pct = feature_df.isna().sum() / len(feature_df) * 100
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
    if len(high_missing) > 0:
        print(f"\n  Features with >50% missing ({len(high_missing)}):")
        for col, pct in high_missing.head(10).items():
            print(f"    {col}: {pct:.1f}%")

    return feature_df

# ============================================================================
# STEP 10: SAVE OUTPUTS
# ============================================================================

def save_outputs(merged_df, feature_df):
    """Save all output files."""
    print("\n" + "=" * 70)
    print("STEP 10: SAVE OUTPUTS")
    print("=" * 70)

    # 1. Full merged dataset
    full_path = OUTPUT_DIR / "multimodal_full.csv"
    merged_df.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")
    print(f"    Shape: {merged_df.shape}")

    # 2. BRAC-compatible subset
    brac_df = merged_df[merged_df['is_brac_compatible'] == True].copy()
    brac_path = OUTPUT_DIR / "multimodal_brac.csv"
    brac_df.to_csv(brac_path, index=False)
    print(f"  Saved: {brac_path}")
    print(f"    Shape: {brac_df.shape}")

    # 3. Complete modalities subset
    complete_df = merged_df[merged_df['has_all_modalities'] == 1].copy()
    complete_path = OUTPUT_DIR / "multimodal_complete.csv"
    complete_df.to_csv(complete_path, index=False)
    print(f"  Saved: {complete_path}")
    print(f"    Shape: {complete_df.shape}")

    # 4. Feature matrix (BRAC-compatible only)
    feature_brac = feature_df[feature_df['is_brac_compatible'] == True].copy()
    feature_path = OUTPUT_DIR / "feature_matrix_brac.csv"
    feature_brac.to_csv(feature_path, index=False)
    print(f"  Saved: {feature_path}")
    print(f"    Shape: {feature_brac.shape}")

    return full_path, brac_path, complete_path, feature_path

# ============================================================================
# STEP 11: GENERATE REPORT
# ============================================================================

def generate_report(merged_df, feature_df):
    """Generate merge pipeline report."""
    print("\n" + "=" * 70)
    print("STEP 11: GENERATE REPORT")
    print("=" * 70)

    # BRAC-compatible subset
    brac_df = merged_df[merged_df['is_brac_compatible'] == True]

    report = f"""# Multi-Modal Data Merge Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Script:** `scripts/merge_multimodal.py`

---

## 1. Overview

| Metric | Value |
|--------|-------|
| Total patients | {len(merged_df)} |
| BRAC-compatible patients | {len(brac_df)} |
| Total columns | {len(merged_df.columns)} |
| Numeric features | {len([c for c in feature_df.columns if c not in ['patient_id', 'brac_label', 'is_brac_compatible']])} |

---

## 2. Modality Availability

| Modality | Patients | Percentage |
|----------|----------|------------|
| Pathology | {merged_df['has_pathology'].sum()} | {100*merged_df['has_pathology'].mean():.1f}% |
| Laboratory | {merged_df['has_laboratory'].sum()} | {100*merged_df['has_laboratory'].mean():.1f}% |
| Clinical | {merged_df['has_clinical'].sum()} | {100*merged_df['has_clinical'].mean():.1f}% |
| Radiology | {merged_df['has_radiology'].sum()} | {100*merged_df['has_radiology'].mean():.1f}% |
| Bone Marrow | {merged_df['has_bone_marrow'].sum()} | {100*merged_df['has_bone_marrow'].mean():.1f}% |
| **All 5 Modalities** | {merged_df['has_all_modalities'].sum()} | {100*merged_df['has_all_modalities'].mean():.1f}% |

---

## 3. BRAC Label Distribution

| Label | Name | Count | Percentage |
|-------|------|-------|------------|
"""

    for label in sorted(brac_df['brac_label'].dropna().unique()):
        count = (brac_df['brac_label'] == label).sum()
        pct = 100 * count / len(brac_df)
        name = BRAC_LABELS.get(int(label), 'UNKNOWN')
        report += f"| {int(label)} | {name} | {count} | {pct:.1f}% |\n"

    report += f"""
---

## 4. Modality Combinations (BRAC-compatible)

| Combination | Patients | Description |
|-------------|----------|-------------|
"""

    combo_cols = ['has_pathology', 'has_laboratory', 'has_clinical', 'has_radiology', 'has_bone_marrow']
    combo_counts = brac_df.groupby(combo_cols).size().sort_values(ascending=False)

    for combo, count in combo_counts.head(10).items():
        combo_str = ''.join(['P' if combo[0] else '-',
                            'L' if combo[1] else '-',
                            'C' if combo[2] else '-',
                            'R' if combo[3] else '-',
                            'B' if combo[4] else '-'])
        desc = []
        if combo[0]: desc.append('Pathology')
        if combo[1]: desc.append('Lab')
        if combo[2]: desc.append('Clinical')
        if combo[3]: desc.append('Radiology')
        if combo[4]: desc.append('Bone Marrow')
        report += f"| {combo_str} | {count} | {', '.join(desc) if desc else 'None'} |\n"

    report += f"""
---

## 5. Feature Summary by Modality

### Pathology Features
- IHC markers (CD20, CD10, BCL6, MUM1, Ki67, etc.)
- Morphology flags (large cells, diffuse pattern, etc.)
- Quality scores (completeness, evidence strength)

### Laboratory Features
- Baseline CBC values (WBC, HGB, PLT, lymphocytes, neutrophils)
- Temporal features (first, last, mean, min, max values)
- Clinical flags (anemia, thrombocytopenia, lymphocytosis, etc.)
- Treatment phase indicators

### Clinical Features
- Age at diagnosis and age groups
- Geographic features (region, Nile Delta, urban/rural)
- Epidemiological priors for each BRAC label

### Radiology Features
- Lymph node involvement by region
- Staging features (above/below diaphragm, extranodal)
- Organ involvement (liver, spleen, bone, lung)
- Measurement features (bulky disease)
- PET-CT features

### Bone Marrow Features
- Baseline cellularity and differential counts
- Lymphocyte infiltration levels
- Blast excess flags
- Series status (myeloid, erythroid)

---

## 6. Output Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `multimodal_full.csv` | {len(merged_df)} | {len(merged_df.columns)} | All patients with any modality |
| `multimodal_brac.csv` | {len(brac_df)} | {len(merged_df.columns)} | BRAC-compatible patients |
| `multimodal_complete.csv` | {merged_df['has_all_modalities'].sum()} | {len(merged_df.columns)} | Patients with all 5 modalities |
| `feature_matrix_brac.csv` | {len(brac_df)} | {len(feature_df.columns)} | Model-ready numeric features |

---

## 7. Data Quality Notes

### Coverage by Modality (BRAC patients)
"""

    for mod in ['pathology', 'laboratory', 'clinical', 'radiology', 'bone_marrow']:
        flag = f'has_{mod}'
        if flag in brac_df.columns:
            count = brac_df[flag].sum()
            pct = 100 * count / len(brac_df)
            report += f"- **{mod.title()}**: {count}/{len(brac_df)} ({pct:.1f}%)\n"

    report += f"""
### Recommended Training Cohorts

1. **Full multi-modal** ({merged_df['has_all_modalities'].sum()} patients): Patients with all 5 modalities
2. **Core BRAC** ({len(brac_df)} patients): All BRAC-compatible patients
3. **Pathology + Lab** ({brac_df[(brac_df['has_pathology']==1) & (brac_df['has_laboratory']==1)].shape[0]} patients): Essential diagnostic data
"""

    # Save report
    report_path = OUTPUT_DIR / "MERGE_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"  Report saved: {report_path}")

    return report

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete merge pipeline."""
    print("\n" + "=" * 70)
    print("MULTI-MODAL DATA MERGE PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Step 1: Load all modalities
    modalities = load_modalities()

    # Step 2-6: Prepare each modality
    path_df = prepare_pathology(modalities['pathology'])
    lab_df = prepare_laboratory(modalities['laboratory'])
    clin_df = prepare_clinical(modalities['clinical'])
    rad_df = prepare_radiology(modalities['radiology'])
    bm_df = prepare_bone_marrow(modalities['bone_marrow'])

    # Step 7: Merge all modalities
    merged_df = merge_modalities(path_df, lab_df, clin_df, rad_df, bm_df)

    if merged_df is None:
        print("\nERROR: Merge failed")
        return None

    # Step 8: Add modality flags
    merged_df = add_modality_flags(merged_df, modalities)

    # Step 9: Create feature matrix
    feature_df = create_feature_matrix(merged_df)

    # Step 10: Save outputs
    save_outputs(merged_df, feature_df)

    # Step 11: Generate report
    generate_report(merged_df, feature_df)

    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Completed: {datetime.now()}")

    return merged_df, feature_df

if __name__ == "__main__":
    main()
