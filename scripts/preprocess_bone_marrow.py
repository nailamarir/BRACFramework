#!/usr/bin/env python3
"""
Bone Marrow Preprocessing Pipeline for BRAC Framework

Enhanced pipeline based on detailed specifications:
- Step 1: Data cleaning (drop headers, replace None/---, cast types)
- Step 2: Normalize free-text qualitative fields (20+ spelling variants)
- Step 3: Validate differential counts (sum ≈ 100%)
- Step 4: Select representative sample (baseline aspirate)
- Step 5: Feature engineering (numeric, categorical, derived NHL flags)
- Step 6: Quality scores (Q/C/S)
- Step 7: Patient-level aggregation
- Step 8: Link to pathology for BRAC labels

Input: data/clinical/Bone_marrow.xlsx
Output:
  - bone_marrow_cleaned.csv (exam-level)
  - bone_marrow_patient_level.csv (patient-level aggregation)
  - bone_marrow_brac_compatible.csv (BRAC-compatible subset)

Author: BRAC Framework
Date: 2026-02-13
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("data/clinical/Bone_marrow.xlsx")
OUTPUT_DIR = Path("data/clinical")
PATHOLOGY_FILE = Path("data/pathology/pathology_cleaned_full.csv")

# Cell differential columns (percentage values) - 13 total
DIFFERENTIAL_COLS = [
    'promyelocytes', 'myelocytes', 'metamyelocytes', 'polymorphs',
    'eosinophils', 'basophils', 'lymphocytes_pct', 'plasma_cells_pct',
    'monocytes', 'pronormoblast', 'normoblast', 'megakaryocytes_count', 'blast_cells'
]

# Lineage groupings
MYELOID_COLS = ['promyelocytes', 'myelocytes', 'metamyelocytes', 'polymorphs']
ERYTHROID_COLS = ['pronormoblast', 'normoblast']
LYMPHOID_COLS = ['lymphocytes_pct', 'plasma_cells_pct']

# Reference ranges for NHL-relevant abnormalities
LYMPHOCYTE_THRESHOLDS = {
    'suspicious': 20,   # % - suspicious for involvement
    'likely': 40,       # % - likely involvement
    'cll_sll': 60       # % - suggestive of CLL/SLL
}
BLAST_THRESHOLD = 5     # % - elevated blasts (transformation concern)
PLASMA_CELL_THRESHOLD = 10  # % - elevated plasma cells

# Differential sum tolerance
DIFFERENTIAL_SUM_TOLERANCE = 5  # ±5% from 100%

# ============================================================================
# TEXT NORMALIZATION MAPPINGS (20+ spelling variants)
# ============================================================================

# Myeloid series normalization
MYELOID_SERIES_MAP = {
    # Normal variants (10+ spellings)
    'normal in morphology and maturation': 'normal',
    'normal in maturation and morphology': 'normal',
    'normal in morpholog and maturation': 'normal',
    'normal in morphology & maturation': 'normal',
    'normal in maturation andmorphology': 'normal',
    'normal in count and morphology': 'normal',
    'normal in count and maturation': 'normal',
    'normal in count and mortphology': 'normal',
    'normal in shape and count': 'normal',
    'normal in shape and morphology': 'normal',
    'normal': 'normal',
    # Suppressed
    'suppressed': 'suppressed',
    # Disturbed
    'disturbed': 'disturbed',
    'disturbed maturation': 'disturbed',
    # Increased
    'increased': 'increased',
    # Default for ---
    '---': None,
}

# Erythroid series normalization
ERYTHROID_SERIES_MAP = {
    # Normal variants
    'normal in morphology and maturation': 'normal',
    'normal in maturation and morphology': 'normal',
    'normal in morphology & maturation': 'normal',
    'normal in count and morphology': 'normal',
    'normal in count and maturation': 'normal',
    'normal in shape and morpholology': 'normal',
    'normal in shape and morphology': 'normal',
    'normal time n morphology and maturation': 'normal',
    'normal': 'normal',
    # Suppressed
    'suppressed': 'suppressed',
    # Hyperplasia variants
    'hyperplasia': 'increased',
    'hypeplasia': 'increased',
    'hyperplais': 'increased',
    'mild hyperplasia': 'increased',
    'hyperplasia with dyserythropiosis': 'increased_dysplastic',
    # Dysplastic variants (multiple misspellings)
    'dyserythropoiesis': 'dysplastic',
    'dyserythropiosis': 'dysplastic',
    'dyseryhtropiosis': 'dysplastic',
    'show dyserythropiosis': 'dysplastic',
}

# Lymphocyte status normalization
LYMPHOCYTE_STATUS_MAP = {
    # Normal variants
    'normal in morphology and count': 'normal',
    'normal in count and morphology': 'normal',
    'normal in count and maturation': 'normal',
    'normal in morphology and maturation': 'normal',
    'normal in count & morphology': 'normal',
    'normal in morphology & count': 'normal',
    'normal in count and morphologhy': 'normal',
    'small and mature': 'normal',
    'normal': 'normal',
    # Increased
    'increased': 'increased',
    'increased with abnormal lymph': 'increased_atypical',
    'increased lymphocytes (large immature with fine chromatin) (blast-like lymph)': 'increased_atypical',
    'hyperplasia': 'increased',
    # Atypical
    'atypical': 'atypical',
    'normal in count with 6% atypical lymph': 'atypical',
    # Suppressed
    'suppressed': 'suppressed',
    'supprressed': 'suppressed',
}

# Megakaryocyte status normalization
MEGAKARYOCYTE_STATUS_MAP = {
    'present and mature': 'normal',
    'present mature': 'normal',
    'normal': 'normal',
    'adequate': 'normal',
    'increased': 'increased',
    'scarce': 'decreased',
    'decreased': 'decreased',
    'mostly immature': 'immature',
    'immature': 'immature',
}

# ============================================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================================

def load_and_clean_data():
    """Load and clean raw bone marrow data."""
    print("=" * 70)
    print("STEP 1: LOAD AND CLEAN DATA")
    print("=" * 70)

    # Load with proper header row (row 1, after blank row 0)
    df = pd.read_excel(INPUT_FILE, header=1)

    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')

    print(f"  Input file: {INPUT_FILE}")
    print(f"  Raw shape: {df.shape}")

    # Standardize column names
    column_mapping = {
        'Patient ID': 'patient_id',
        'Investigations ID': 'investigation_id',
        'Date': 'exam_date',
        'Site of Aspiration': 'aspiration_site',
        'Cellularity': 'cellularity',
        'Promyelocytes': 'promyelocytes',
        'Myelocytes': 'myelocytes',
        'Metamyelocytes': 'metamyelocytes',
        'Polymorphs': 'polymorphs',
        'Eosinophils ( C )': 'eosinophils',
        'Basophils ( C )': 'basophils',
        'Lymphocytes ( C )': 'lymphocytes_pct',
        'Plasma cells': 'plasma_cells_pct',
        'Monocytes ( C )': 'monocytes',
        'Pronormoblast': 'pronormoblast',
        'Normoblast': 'normoblast',
        'Megakaryocytes': 'megakaryocytes_count',
        'Blast cells': 'blast_cells',
        'Diagnosis': 'diagnosis',
        'Myeloid / Erythroid ratio': 'me_ratio',
        'Myeloid series': 'myeloid_series',
        'Erythroid series': 'erythroid_series',
        'Lymphocytes': 'lymphocytes_assessment',
        'Plasma cells.1': 'plasma_cells_assessment',
        'Megakaryocytes.1': 'megakaryocytes_assessment',
        'Iron stores': 'iron_stores'
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]

    # Replace string "None" with NaN
    df = df.replace('None', np.nan)
    df = df.replace('none', np.nan)

    # Replace "---" in Diagnosis and M:E ratio with NaN
    df.loc[df['diagnosis'] == '---', 'diagnosis'] = np.nan
    df.loc[df['me_ratio'] == '---', 'me_ratio'] = np.nan

    # Drop fully empty rows (all None/NaN except patient_id)
    data_cols = [c for c in df.columns if c not in ['patient_id', 'investigation_id']]
    empty_rows = df[data_cols].isna().all(axis=1)
    n_empty = empty_rows.sum()
    if n_empty > 0:
        df = df[~empty_rows]
        print(f"  Dropped {n_empty} fully empty rows")

    # Parse dates
    df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')

    # Cast differential counts to float
    for col in DIFFERENTIAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Cleaned columns: {len(df.columns)}")
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    print(f"  Date range: {df['exam_date'].min()} to {df['exam_date'].max()}")

    return df

# ============================================================================
# STEP 2: NORMALIZE FREE-TEXT QUALITATIVE FIELDS
# ============================================================================

def normalize_qualitative_fields(df):
    """Normalize free-text fields with 20+ spelling variants."""
    print("\n" + "=" * 70)
    print("STEP 2: NORMALIZE FREE-TEXT QUALITATIVE FIELDS")
    print("=" * 70)

    df = df.copy()

    # --- Cellularity (3 values) ---
    cellularity_map = {
        'hypercellular': 'hyper',
        'normocellular': 'normo',
        'hypocellular': 'hypo'
    }
    df['cellularity_norm'] = df['cellularity'].str.lower().str.strip().map(cellularity_map)

    # Ordinal encoding: hypo=0, normo=1, hyper=2
    cellularity_ordinal = {'hypo': 0, 'normo': 1, 'hyper': 2}
    df['cellularity_ordinal'] = df['cellularity_norm'].map(cellularity_ordinal)

    cell_counts = df['cellularity_norm'].value_counts(dropna=False)
    print(f"  Cellularity: {cell_counts.to_dict()}")

    # --- Myeloid series ---
    def normalize_myeloid(text):
        if pd.isna(text):
            return None
        text = str(text).lower().strip()
        # Try direct match
        if text in MYELOID_SERIES_MAP:
            return MYELOID_SERIES_MAP[text]
        # Fuzzy matching for common patterns
        if 'normal' in text:
            return 'normal'
        if 'suppress' in text:
            return 'suppressed'
        if 'disturb' in text:
            return 'disturbed'
        if 'increas' in text:
            return 'increased'
        return None

    df['myeloid_status_norm'] = df['myeloid_series'].apply(normalize_myeloid)
    myeloid_counts = df['myeloid_status_norm'].value_counts(dropna=False)
    print(f"  Myeloid series: {myeloid_counts.to_dict()}")

    # --- Erythroid series ---
    def normalize_erythroid(text):
        if pd.isna(text):
            return None
        text = str(text).lower().strip()
        if text in ERYTHROID_SERIES_MAP:
            return ERYTHROID_SERIES_MAP[text]
        if 'normal' in text:
            return 'normal'
        if 'suppress' in text:
            return 'suppressed'
        if 'hyperplasia' in text or 'hypeplasia' in text:
            return 'increased'
        if 'dyserythro' in text or 'dyseryth' in text:
            return 'dysplastic'
        return None

    df['erythroid_status_norm'] = df['erythroid_series'].apply(normalize_erythroid)
    erythroid_counts = df['erythroid_status_norm'].value_counts(dropna=False)
    print(f"  Erythroid series: {erythroid_counts.to_dict()}")

    # --- Lymphocyte status ---
    def normalize_lymphocyte(text):
        if pd.isna(text):
            return None
        text = str(text).lower().strip()
        if text in LYMPHOCYTE_STATUS_MAP:
            return LYMPHOCYTE_STATUS_MAP[text]
        if 'normal' in text:
            return 'normal'
        if 'increas' in text:
            if 'atypical' in text or 'abnormal' in text:
                return 'increased_atypical'
            return 'increased'
        if 'atypical' in text:
            return 'atypical'
        if 'suppress' in text:
            return 'suppressed'
        return None

    df['lymphocyte_status_norm'] = df['lymphocytes_assessment'].apply(normalize_lymphocyte)
    lymph_counts = df['lymphocyte_status_norm'].value_counts(dropna=False)
    print(f"  Lymphocyte status: {lymph_counts.to_dict()}")

    # --- Megakaryocyte status ---
    def normalize_megakaryocyte(text):
        if pd.isna(text):
            return None
        text = str(text).lower().strip()
        if text in MEGAKARYOCYTE_STATUS_MAP:
            return MEGAKARYOCYTE_STATUS_MAP[text]
        if 'normal' in text or 'adequate' in text or 'present' in text:
            return 'normal'
        if 'increas' in text:
            return 'increased'
        if 'scarce' in text or 'decreas' in text:
            return 'decreased'
        if 'immature' in text:
            return 'immature'
        return None

    df['megakaryocyte_status_norm'] = df['megakaryocytes_assessment'].apply(normalize_megakaryocyte) if 'megakaryocytes_assessment' in df.columns else None
    if 'megakaryocyte_status_norm' in df.columns:
        mega_counts = df['megakaryocyte_status_norm'].value_counts(dropna=False)
        print(f"  Megakaryocyte status: {mega_counts.to_dict()}")

    # --- M:E Ratio ---
    def normalize_me_ratio(text):
        if pd.isna(text):
            return None
        text = str(text).lower().strip()
        if text in ['normal', 'nl', 'wnl']:
            return 'normal'
        if text in ['increased', 'high', 'elevated']:
            return 'increased'
        if text in ['decreased', 'low', 'reduced', 'reversed', 'reserved']:
            return 'decreased'
        return None

    df['me_ratio_norm'] = df['me_ratio'].apply(normalize_me_ratio)
    me_counts = df['me_ratio_norm'].value_counts(dropna=False)
    print(f"  M:E ratio: {me_counts.to_dict()}")

    return df

# ============================================================================
# STEP 3: VALIDATE DIFFERENTIAL COUNTS
# ============================================================================

def validate_differential_counts(df):
    """Validate differential counts (sum should ≈ 100%)."""
    print("\n" + "=" * 70)
    print("STEP 3: VALIDATE DIFFERENTIAL COUNTS")
    print("=" * 70)

    df = df.copy()

    # Columns that contribute to 100% differential
    # Note: megakaryocytes_count is absolute count, not percentage
    diff_pct_cols = [
        'promyelocytes', 'myelocytes', 'metamyelocytes', 'polymorphs',
        'eosinophils', 'basophils', 'lymphocytes_pct', 'plasma_cells_pct',
        'monocytes', 'pronormoblast', 'normoblast', 'blast_cells'
    ]
    diff_pct_cols = [c for c in diff_pct_cols if c in df.columns]

    # Calculate differential sum
    df['differential_sum'] = df[diff_pct_cols].sum(axis=1, min_count=1)

    # Count tested columns per row
    df['n_tested_counts'] = df[diff_pct_cols].notna().sum(axis=1)

    # Flag rows where sum is outside tolerance
    df['diff_sum_valid'] = (
        (df['differential_sum'] >= 100 - DIFFERENTIAL_SUM_TOLERANCE) &
        (df['differential_sum'] <= 100 + DIFFERENTIAL_SUM_TOLERANCE)
    ).astype(int)

    # Flag quality penalty for invalid sums
    df['flag_diff_sum_low'] = (df['differential_sum'] < 80).astype(int)
    df['flag_diff_sum_high'] = (df['differential_sum'] > 110).astype(int)

    # Validate individual ranges
    range_validation = {
        'blast_cells': (0, 30),
        'lymphocytes_pct': (0, 95),
        'promyelocytes': (0, 15),
        'plasma_cells_pct': (0, 30),
    }

    for col, (low, high) in range_validation.items():
        if col in df.columns:
            out_of_range = (df[col] < low) | (df[col] > high)
            df[f'flag_{col}_out_range'] = out_of_range.astype(int)
            n_flagged = out_of_range.sum()
            if n_flagged > 0:
                print(f"  Warning: {n_flagged} rows with {col} outside [{low}-{high}]")

    # Statistics
    valid_sums = df[df['differential_sum'].notna()]
    print(f"  Rows with differential counts: {len(valid_sums)}")
    print(f"  Differential sum: mean={valid_sums['differential_sum'].mean():.1f}, "
          f"range=[{valid_sums['differential_sum'].min():.0f}-{valid_sums['differential_sum'].max():.0f}]")
    print(f"  Valid sums (within ±{DIFFERENTIAL_SUM_TOLERANCE}% of 100): {df['diff_sum_valid'].sum()}")
    print(f"  Low sums (<80%): {df['flag_diff_sum_low'].sum()}")
    print(f"  High sums (>110%): {df['flag_diff_sum_high'].sum()}")

    return df

# ============================================================================
# STEP 4: SELECT REPRESENTATIVE SAMPLE (BASELINE)
# ============================================================================

def add_baseline_selection(df):
    """Mark baseline aspirate for each patient (earliest dated)."""
    print("\n" + "=" * 70)
    print("STEP 4: SELECT REPRESENTATIVE SAMPLE (BASELINE)")
    print("=" * 70)

    df = df.copy()

    # Sort by patient and date
    df = df.sort_values(['patient_id', 'exam_date']).reset_index(drop=True)

    # Mark investigation order per patient
    df['investigation_order'] = df.groupby('patient_id').cumcount() + 1
    df['total_investigations'] = df.groupby('patient_id')['patient_id'].transform('count')

    # Mark baseline (first aspirate)
    df['is_baseline'] = (df['investigation_order'] == 1).astype(int)

    # Mark follow-up
    df['is_followup'] = (df['investigation_order'] > 1).astype(int)

    # Days since baseline
    df['days_since_baseline'] = df.groupby('patient_id')['exam_date'].transform(
        lambda x: (x - x.min()).dt.days
    )

    # Statistics
    baseline_count = df['is_baseline'].sum()
    followup_count = df['is_followup'].sum()
    multi_exam_patients = (df.groupby('patient_id').size() > 1).sum()

    print(f"  Baseline aspirates: {baseline_count}")
    print(f"  Follow-up aspirates: {followup_count}")
    print(f"  Patients with multiple aspirates: {multi_exam_patients}")
    print(f"  Mean exams per patient: {df.groupby('patient_id').size().mean():.1f}")

    return df

# ============================================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create numeric, categorical, and derived NHL flags."""
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    # =========================================
    # A. Compute lineage totals
    # =========================================
    myeloid_cols = [c for c in MYELOID_COLS if c in df.columns]
    erythroid_cols = [c for c in ERYTHROID_COLS if c in df.columns]

    df['myeloid_total'] = df[myeloid_cols].sum(axis=1, min_count=1)
    df['erythroid_total'] = df[erythroid_cols].sum(axis=1, min_count=1)

    # Computed M:E ratio
    df['me_ratio_computed'] = df['myeloid_total'] / df['erythroid_total'].replace(0, np.nan)

    print(f"  A. Lineage totals computed")
    print(f"     Myeloid total: mean={df['myeloid_total'].mean():.1f}")
    print(f"     Erythroid total: mean={df['erythroid_total'].mean():.1f}")
    print(f"     M:E ratio (computed): mean={df['me_ratio_computed'].mean():.2f}")

    # =========================================
    # B. Categorical feature encoding
    # =========================================
    # Cellularity binary flags
    df['is_hypercellular'] = (df['cellularity_norm'] == 'hyper').astype(int)
    df['is_hypocellular'] = (df['cellularity_norm'] == 'hypo').astype(int)
    df['is_normocellular'] = (df['cellularity_norm'] == 'normo').astype(int)

    # Series status binary flags
    df['myeloid_normal'] = (df['myeloid_status_norm'] == 'normal').astype(int)
    df['myeloid_suppressed'] = (df['myeloid_status_norm'] == 'suppressed').astype(int)
    df['myeloid_disturbed'] = (df['myeloid_status_norm'] == 'disturbed').astype(int)

    df['erythroid_normal'] = (df['erythroid_status_norm'] == 'normal').astype(int)
    df['erythroid_suppressed'] = (df['erythroid_status_norm'] == 'suppressed').astype(int)
    df['erythroid_dysplastic'] = (df['erythroid_status_norm'].str.contains('dysplastic', na=False)).astype(int)

    df['lymphocyte_normal'] = (df['lymphocyte_status_norm'] == 'normal').astype(int)
    df['lymphocyte_increased'] = (df['lymphocyte_status_norm'].str.contains('increased', na=False)).astype(int)
    df['lymphocyte_atypical'] = (df['lymphocyte_status_norm'].str.contains('atypical', na=False)).astype(int)

    print(f"  B. Categorical features encoded")

    # =========================================
    # C. Derived NHL flags
    # =========================================

    # Lymphocyte infiltration levels
    df['lymph_infiltration_suspicious'] = (df['lymphocytes_pct'] >= LYMPHOCYTE_THRESHOLDS['suspicious']).astype(int)
    df['lymph_infiltration_likely'] = (df['lymphocytes_pct'] >= LYMPHOCYTE_THRESHOLDS['likely']).astype(int)
    df['lymph_infiltration_cll_sll'] = (df['lymphocytes_pct'] >= LYMPHOCYTE_THRESHOLDS['cll_sll']).astype(int)

    # Blast excess (transformation concern)
    df['blast_excess'] = (df['blast_cells'] >= BLAST_THRESHOLD).astype(int)
    df['blast_very_high'] = (df['blast_cells'] >= 20).astype(int)

    # Myeloid suppression pattern (myeloid + erythroid both suppressed = displacement)
    df['myeloid_erythroid_suppressed'] = (
        (df['myeloid_status_norm'] == 'suppressed') &
        (df['erythroid_status_norm'] == 'suppressed')
    ).astype(int)

    # M:E ratio inverted (< 1.0 suggests erythroid hyperplasia or myeloid suppression)
    df['me_ratio_inverted'] = (df['me_ratio_computed'] < 1.0).astype(int)

    # M:E ratio category from computed value
    df['me_ratio_computed_cat'] = pd.cut(
        df['me_ratio_computed'],
        bins=[0, 1.5, 4.5, np.inf],
        labels=['decreased', 'normal', 'increased']
    ).astype(str).replace('nan', None)

    # Plasma cell elevation
    df['plasma_cells_elevated'] = (df['plasma_cells_pct'] >= PLASMA_CELL_THRESHOLD).astype(int)

    # Composite NHL involvement pattern
    # High lymphocytes + hypercellular = diffuse infiltration
    df['bm_diffuse_infiltration'] = (
        (df['lymphocytes_pct'] >= 40) &
        (df['is_hypercellular'] == 1)
    ).astype(int)

    # Interstitial/nodular pattern
    df['bm_interstitial_pattern'] = (
        (df['lymphocytes_pct'] >= 20) &
        (df['lymphocytes_pct'] < 50) &
        (df['is_normocellular'] == 1)
    ).astype(int)

    print(f"  C. Derived NHL flags created")
    print(f"     Lymph infiltration suspicious (≥20%): {df['lymph_infiltration_suspicious'].sum()}")
    print(f"     Lymph infiltration likely (≥40%): {df['lymph_infiltration_likely'].sum()}")
    print(f"     Lymph infiltration CLL/SLL (≥60%): {df['lymph_infiltration_cll_sll'].sum()}")
    print(f"     Blast excess (≥5%): {df['blast_excess'].sum()}")
    print(f"     M:E ratio inverted: {df['me_ratio_inverted'].sum()}")
    print(f"     Diffuse BM infiltration: {df['bm_diffuse_infiltration'].sum()}")

    return df

# ============================================================================
# STEP 6: QUALITY SCORES (Q, C, S)
# ============================================================================

def compute_quality_scores(df):
    """Compute Q/C/S quality scores for bone marrow data."""
    print("\n" + "=" * 70)
    print("STEP 6: QUALITY SCORES (Q, C, S)")
    print("=" * 70)

    df = df.copy()

    # =========================================
    # Q = f(site, cellularity present, no None row)
    # =========================================
    def calc_quality(row):
        score = 0.5  # Base score

        # Has aspiration site (+0.1)
        if pd.notna(row.get('aspiration_site')):
            score += 0.1

        # Has cellularity (+0.2)
        if pd.notna(row.get('cellularity_norm')):
            score += 0.2

        # Has differential counts (+0.2)
        if pd.notna(row.get('lymphocytes_pct')):
            score += 0.2

        return min(1.0, score)

    df['quality_Q'] = df.apply(calc_quality, axis=1)

    # =========================================
    # C = tested_counts / 13 (coverage)
    # =========================================
    df['quality_C'] = df['n_tested_counts'] / len(DIFFERENTIAL_COLS)

    # =========================================
    # S = differential sum consistency
    # =========================================
    def calc_consistency(row):
        diff_sum = row.get('differential_sum')
        if pd.isna(diff_sum):
            return np.nan

        # Perfect if sum is within ±5% of 100
        deviation = abs(diff_sum - 100)
        if deviation <= 5:
            return 1.0
        elif deviation <= 10:
            return 0.8
        elif deviation <= 20:
            return 0.5
        else:
            return 0.2

    df['quality_S'] = df.apply(calc_consistency, axis=1)

    # Composite quality score
    df['quality_composite'] = (
        0.4 * df['quality_Q'] +
        0.4 * df['quality_C'].fillna(0) +
        0.2 * df['quality_S'].fillna(0.5)
    )

    print(f"  Quality scores:")
    print(f"    Q (quality): mean={df['quality_Q'].mean():.3f}")
    print(f"    C (coverage): mean={df['quality_C'].mean():.3f}")
    print(f"    S (consistency): mean={df['quality_S'].mean():.3f}")
    print(f"    Composite: mean={df['quality_composite'].mean():.3f}")

    return df

# ============================================================================
# STEP 7: PATIENT-LEVEL AGGREGATION
# ============================================================================

def aggregate_patient_level(df):
    """Aggregate to patient level with baseline and longitudinal features."""
    print("\n" + "=" * 70)
    print("STEP 7: PATIENT-LEVEL AGGREGATION")
    print("=" * 70)

    df_sorted = df.sort_values(['patient_id', 'exam_date'])
    patient_agg = []

    for patient_id, patient_df in df_sorted.groupby('patient_id'):
        record = {'patient_id': patient_id}

        # Exam counts
        record['bm_exam_count'] = len(patient_df)
        record['bm_first_date'] = patient_df['exam_date'].min()
        record['bm_last_date'] = patient_df['exam_date'].max()

        # Baseline features (first/earliest aspirate)
        baseline = patient_df[patient_df['is_baseline'] == 1].iloc[0] if len(patient_df) > 0 else patient_df.iloc[0]

        record['bm_baseline_cellularity'] = baseline.get('cellularity_norm')
        record['bm_baseline_lymphocytes_pct'] = baseline.get('lymphocytes_pct')
        record['bm_baseline_blasts_pct'] = baseline.get('blast_cells')
        record['bm_baseline_myeloid_status'] = baseline.get('myeloid_status_norm')
        record['bm_baseline_erythroid_status'] = baseline.get('erythroid_status_norm')
        record['bm_baseline_me_ratio'] = baseline.get('me_ratio_norm')

        # Latest exam features
        latest = patient_df.iloc[-1]
        record['bm_latest_lymphocytes_pct'] = latest.get('lymphocytes_pct')
        record['bm_latest_blasts_pct'] = latest.get('blast_cells')

        # Maximum values across all exams (worst case)
        record['bm_max_lymphocytes_pct'] = patient_df['lymphocytes_pct'].max()
        record['bm_max_blasts_pct'] = patient_df['blast_cells'].max()
        record['bm_max_plasma_cells_pct'] = patient_df['plasma_cells_pct'].max()

        # Ever flags (NHL-relevant)
        record['bm_ever_hypercellular'] = int(patient_df['is_hypercellular'].max() == 1)
        record['bm_ever_hypocellular'] = int(patient_df['is_hypocellular'].max() == 1)
        record['bm_ever_lymph_suspicious'] = int(patient_df['lymph_infiltration_suspicious'].max() == 1)
        record['bm_ever_lymph_likely'] = int(patient_df['lymph_infiltration_likely'].max() == 1)
        record['bm_ever_lymph_cll_sll'] = int(patient_df['lymph_infiltration_cll_sll'].max() == 1)
        record['bm_ever_blast_excess'] = int(patient_df['blast_excess'].max() == 1)
        record['bm_ever_diffuse_infiltration'] = int(patient_df['bm_diffuse_infiltration'].max() == 1)
        record['bm_ever_myeloid_suppressed'] = int(patient_df['myeloid_suppressed'].max() == 1)
        record['bm_ever_erythroid_suppressed'] = int(patient_df['erythroid_suppressed'].max() == 1)

        # Mean values
        record['bm_mean_lymphocytes_pct'] = patient_df['lymphocytes_pct'].mean()
        record['bm_mean_myeloid_total'] = patient_df['myeloid_total'].mean()
        record['bm_mean_erythroid_total'] = patient_df['erythroid_total'].mean()

        # Quality scores (mean across exams)
        record['bm_mean_quality_Q'] = patient_df['quality_Q'].mean()
        record['bm_mean_quality_C'] = patient_df['quality_C'].mean()
        record['bm_mean_quality_S'] = patient_df['quality_S'].mean()
        record['bm_quality_composite'] = patient_df['quality_composite'].mean()

        patient_agg.append(record)

    patient_df = pd.DataFrame(patient_agg)

    print(f"  Patient-level records: {len(patient_df)}")
    print(f"  Patients with lymph infiltration (suspicious): {patient_df['bm_ever_lymph_suspicious'].sum()}")
    print(f"  Patients with lymph infiltration (likely): {patient_df['bm_ever_lymph_likely'].sum()}")
    print(f"  Patients with CLL/SLL pattern: {patient_df['bm_ever_lymph_cll_sll'].sum()}")
    print(f"  Patients with blast excess: {patient_df['bm_ever_blast_excess'].sum()}")

    return patient_df

# ============================================================================
# STEP 8: LINK TO PATHOLOGY
# ============================================================================

def link_to_pathology(patient_df):
    """Link bone marrow data to pathology for BRAC labels."""
    print("\n" + "=" * 70)
    print("STEP 8: LINK TO PATHOLOGY")
    print("=" * 70)

    if not PATHOLOGY_FILE.exists():
        print(f"  Warning: Pathology file not found: {PATHOLOGY_FILE}")
        patient_df['brac_label'] = np.nan
        patient_df['disease_type_final'] = np.nan
        patient_df['is_brac_compatible'] = False
        return patient_df

    # Load pathology data
    path_df = pd.read_csv(PATHOLOGY_FILE, usecols=['patient_id', 'brac_label', 'disease_type_final'])

    # Deduplicate pathology to one record per patient (keep first with BRAC label)
    path_df_brac = path_df[path_df['brac_label'].notna()].drop_duplicates(subset='patient_id', keep='first')
    path_df_other = path_df[path_df['brac_label'].isna()].drop_duplicates(subset='patient_id', keep='first')
    path_df_dedup = pd.concat([path_df_brac, path_df_other[~path_df_other['patient_id'].isin(path_df_brac['patient_id'])]])

    print(f"  Pathology records: {len(path_df)} -> {len(path_df_dedup)} (deduplicated per patient)")

    # Merge
    patient_df = patient_df.merge(
        path_df_dedup[['patient_id', 'brac_label', 'disease_type_final']],
        on='patient_id',
        how='left'
    )

    # BRAC compatibility
    patient_df['is_brac_compatible'] = patient_df['brac_label'].notna()
    patient_df.loc[patient_df['is_brac_compatible'].isna(), 'is_brac_compatible'] = False
    patient_df['is_brac_compatible'] = patient_df['is_brac_compatible'].astype(bool)

    brac_count = patient_df['is_brac_compatible'].sum()
    print(f"  Patients linked to pathology: {path_df_dedup['patient_id'].nunique()}")
    print(f"  BRAC-compatible patients: {brac_count}")

    if brac_count > 0:
        label_names = {0: 'DLBCL_GCB', 1: 'DLBCL_ABC', 2: 'FL', 3: 'MCL', 4: 'BL', 5: 'MZL', 6: 'CLL_SLL', 7: 'LPL', 8: 'PMBL'}
        print(f"  BRAC label distribution:")
        for label in sorted(patient_df[patient_df['is_brac_compatible']]['brac_label'].dropna().unique()):
            count = (patient_df['brac_label'] == label).sum()
            print(f"    {int(label)} ({label_names.get(int(label), 'Unknown')}): {count}")

    return patient_df

# ============================================================================
# STEP 9: SELECT AND SAVE OUTPUTS
# ============================================================================

def select_and_save(exam_df, patient_df):
    """Select final columns and save outputs."""
    print("\n" + "=" * 70)
    print("STEP 9: SELECT AND SAVE OUTPUTS")
    print("=" * 70)

    # Exam-level output columns
    exam_output_cols = [
        # IDs
        'patient_id', 'investigation_id', 'exam_date',
        'investigation_order', 'total_investigations', 'is_baseline', 'is_followup',
        'days_since_baseline',
        # Aspiration site
        'aspiration_site',
        # Cellularity
        'cellularity_norm', 'cellularity_ordinal',
        'is_hypercellular', 'is_hypocellular', 'is_normocellular',
        # Cell differentials (raw)
        'promyelocytes', 'myelocytes', 'metamyelocytes', 'polymorphs',
        'eosinophils', 'basophils', 'lymphocytes_pct', 'plasma_cells_pct',
        'monocytes', 'pronormoblast', 'normoblast', 'blast_cells',
        'megakaryocytes_count',
        # Computed totals
        'myeloid_total', 'erythroid_total', 'me_ratio_computed', 'me_ratio_computed_cat',
        'differential_sum', 'n_tested_counts', 'diff_sum_valid',
        # Normalized status
        'myeloid_status_norm', 'erythroid_status_norm', 'lymphocyte_status_norm',
        'megakaryocyte_status_norm', 'me_ratio_norm',
        # Status binary flags
        'myeloid_normal', 'myeloid_suppressed', 'myeloid_disturbed',
        'erythroid_normal', 'erythroid_suppressed', 'erythroid_dysplastic',
        'lymphocyte_normal', 'lymphocyte_increased', 'lymphocyte_atypical',
        # NHL flags
        'lymph_infiltration_suspicious', 'lymph_infiltration_likely', 'lymph_infiltration_cll_sll',
        'blast_excess', 'blast_very_high', 'plasma_cells_elevated',
        'myeloid_erythroid_suppressed', 'me_ratio_inverted',
        'bm_diffuse_infiltration', 'bm_interstitial_pattern',
        # Quality scores
        'quality_Q', 'quality_C', 'quality_S', 'quality_composite',
        # Validation flags
        'flag_diff_sum_low', 'flag_diff_sum_high',
    ]

    exam_output_cols = [c for c in exam_output_cols if c in exam_df.columns]
    exam_output = exam_df[exam_output_cols]

    # Save exam-level
    exam_output_path = OUTPUT_DIR / "bone_marrow_cleaned.csv"
    exam_output.to_csv(exam_output_path, index=False)
    print(f"  Saved exam-level: {exam_output_path}")
    print(f"    Shape: {exam_output.shape}")

    # Save patient-level
    patient_output_path = OUTPUT_DIR / "bone_marrow_patient_level.csv"
    patient_df.to_csv(patient_output_path, index=False)
    print(f"  Saved patient-level: {patient_output_path}")
    print(f"    Shape: {patient_df.shape}")

    # Save BRAC-compatible subset
    brac_df = patient_df[patient_df['is_brac_compatible'] == True].copy()
    brac_output_path = OUTPUT_DIR / "bone_marrow_brac_compatible.csv"
    brac_df.to_csv(brac_output_path, index=False)
    print(f"  Saved BRAC-compatible: {brac_output_path}")
    print(f"    Shape: {brac_df.shape}")

    return exam_output, patient_df, brac_df

# ============================================================================
# STEP 10: GENERATE REPORT
# ============================================================================

def generate_report(exam_df, patient_df, brac_df):
    """Generate preprocessing pipeline report."""
    print("\n" + "=" * 70)
    print("STEP 10: GENERATE REPORT")
    print("=" * 70)

    report = f"""# Bone Marrow Preprocessing Pipeline Report

**Generated:** {datetime.now().strftime('%Y-%m-%d')}
**Script:** `scripts/preprocess_bone_marrow.py`

---

## 1. Overview

### Input/Output Summary

| Metric | Value |
|--------|-------|
| Input file | `{INPUT_FILE}` |
| Input exams | {len(exam_df)} |
| Unique patients | {patient_df['patient_id'].nunique()} |
| Date range | {exam_df['exam_date'].min()} to {exam_df['exam_date'].max()} |
| Output exam-level columns | {len(exam_df.columns)} |
| Output patient-level columns | {len(patient_df.columns)} |
| BRAC-compatible patients | {len(brac_df)} |

---

## 2. Cellularity Distribution

| Cellularity | Count | Percentage |
|-------------|-------|------------|
"""
    for val, count in exam_df['cellularity_norm'].value_counts(dropna=False).items():
        pct = 100 * count / len(exam_df)
        report += f"| {val} | {count} | {pct:.1f}% |\n"

    report += f"""
---

## 3. Cell Differential Statistics

| Cell Type | Mean % | Median % | Range |
|-----------|--------|----------|-------|
| Lymphocytes | {exam_df['lymphocytes_pct'].mean():.1f} | {exam_df['lymphocytes_pct'].median():.1f} | {exam_df['lymphocytes_pct'].min():.0f}-{exam_df['lymphocytes_pct'].max():.0f} |
| Blast cells | {exam_df['blast_cells'].mean():.1f} | {exam_df['blast_cells'].median():.1f} | {exam_df['blast_cells'].min():.0f}-{exam_df['blast_cells'].max():.0f} |
| Myeloid total | {exam_df['myeloid_total'].mean():.1f} | {exam_df['myeloid_total'].median():.1f} | {exam_df['myeloid_total'].min():.0f}-{exam_df['myeloid_total'].max():.0f} |
| Erythroid total | {exam_df['erythroid_total'].mean():.1f} | {exam_df['erythroid_total'].median():.1f} | {exam_df['erythroid_total'].min():.0f}-{exam_df['erythroid_total'].max():.0f} |

### Differential Sum Validation

| Metric | Value |
|--------|-------|
| Mean differential sum | {exam_df['differential_sum'].mean():.1f}% |
| Valid sums (95-105%) | {exam_df['diff_sum_valid'].sum()} ({100*exam_df['diff_sum_valid'].mean():.1f}%) |
| Low sums (<80%) | {exam_df['flag_diff_sum_low'].sum()} |
| High sums (>110%) | {exam_df['flag_diff_sum_high'].sum()} |

---

## 4. NHL-Relevant Features

| Feature | Exams | % of Total |
|---------|-------|------------|
| Lymph infiltration suspicious (≥20%) | {exam_df['lymph_infiltration_suspicious'].sum()} | {100*exam_df['lymph_infiltration_suspicious'].mean():.1f}% |
| Lymph infiltration likely (≥40%) | {exam_df['lymph_infiltration_likely'].sum()} | {100*exam_df['lymph_infiltration_likely'].mean():.1f}% |
| Lymph infiltration CLL/SLL (≥60%) | {exam_df['lymph_infiltration_cll_sll'].sum()} | {100*exam_df['lymph_infiltration_cll_sll'].mean():.1f}% |
| Blast excess (≥5%) | {exam_df['blast_excess'].sum()} | {100*exam_df['blast_excess'].mean():.1f}% |
| M:E ratio inverted (<1.0) | {exam_df['me_ratio_inverted'].sum()} | {100*exam_df['me_ratio_inverted'].mean():.1f}% |
| Diffuse BM infiltration | {exam_df['bm_diffuse_infiltration'].sum()} | {100*exam_df['bm_diffuse_infiltration'].mean():.1f}% |

---

## 5. Patient-Level Summary

| Metric | Value |
|--------|-------|
| Total patients | {len(patient_df)} |
| Ever lymph infiltration (suspicious) | {patient_df['bm_ever_lymph_suspicious'].sum()} |
| Ever lymph infiltration (likely) | {patient_df['bm_ever_lymph_likely'].sum()} |
| Ever CLL/SLL pattern | {patient_df['bm_ever_lymph_cll_sll'].sum()} |
| Ever blast excess | {patient_df['bm_ever_blast_excess'].sum()} |
| Mean exams per patient | {patient_df['bm_exam_count'].mean():.1f} |

---

## 6. Quality Scores

| Score | Mean | Min | Max |
|-------|------|-----|-----|
| Q (quality) | {exam_df['quality_Q'].mean():.3f} | {exam_df['quality_Q'].min():.2f} | {exam_df['quality_Q'].max():.2f} |
| C (coverage) | {exam_df['quality_C'].mean():.3f} | {exam_df['quality_C'].min():.2f} | {exam_df['quality_C'].max():.2f} |
| S (consistency) | {exam_df['quality_S'].mean():.3f} | {exam_df['quality_S'].min():.2f} | {exam_df['quality_S'].max():.2f} |
| Composite | {exam_df['quality_composite'].mean():.3f} | {exam_df['quality_composite'].min():.2f} | {exam_df['quality_composite'].max():.2f} |

---

## 7. BRAC Label Distribution (Compatible Cases)

"""
    if len(brac_df) > 0 and 'brac_label' in brac_df.columns:
        brac_names = {0: 'DLBCL_GCB', 1: 'DLBCL_ABC', 2: 'FL', 3: 'MCL', 4: 'BL', 5: 'MZL', 6: 'CLL_SLL', 7: 'LPL', 8: 'PMBL'}
        report += "| Label | Name | Count | Percentage |\n|-------|------|-------|------------|\n"
        for label in sorted(brac_df['brac_label'].dropna().unique()):
            count = (brac_df['brac_label'] == label).sum()
            pct = 100 * count / len(brac_df)
            report += f"| {int(label)} | {brac_names.get(int(label), 'UNKNOWN')} | {count} | {pct:.1f}% |\n"
    else:
        report += "*No BRAC-compatible cases found*\n"

    report += f"""
---

## 8. File Outputs

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `bone_marrow_cleaned.csv` | {len(exam_df)} | {len(exam_df.columns)} | Exam-level preprocessed data |
| `bone_marrow_patient_level.csv` | {len(patient_df)} | {len(patient_df.columns)} | Patient-level aggregation |
| `bone_marrow_brac_compatible.csv` | {len(brac_df)} | {len(brac_df.columns)} | BRAC-compatible subset |

---

## 9. Key Column Descriptions

### Exam-Level

| Column | Type | Description |
|--------|------|-------------|
| `cellularity_norm` | str | Normalized: hyper/normo/hypo |
| `cellularity_ordinal` | int | Ordinal: 0=hypo, 1=normo, 2=hyper |
| `myeloid_status_norm` | str | normal/suppressed/disturbed/increased |
| `erythroid_status_norm` | str | normal/suppressed/increased/dysplastic |
| `lymphocyte_status_norm` | str | normal/increased/atypical/suppressed |
| `lymph_infiltration_suspicious` | int | Lymphocytes ≥20% |
| `lymph_infiltration_likely` | int | Lymphocytes ≥40% |
| `lymph_infiltration_cll_sll` | int | Lymphocytes ≥60% (CLL/SLL pattern) |
| `blast_excess` | int | Blasts ≥5% (transformation concern) |
| `me_ratio_inverted` | int | M:E ratio <1.0 |
| `diff_sum_valid` | int | Differential sum within ±5% of 100 |
| `quality_Q` | float | Quality score (0-1) |
| `quality_C` | float | Coverage score (0-1) |
| `quality_S` | float | Consistency score (0-1) |

### Patient-Level

| Column | Type | Description |
|--------|------|-------------|
| `bm_baseline_*` | various | Features from baseline (first) aspirate |
| `bm_ever_lymph_suspicious` | int | Ever had lymphocytes ≥20% |
| `bm_ever_lymph_cll_sll` | int | Ever had lymphocytes ≥60% |
| `bm_ever_blast_excess` | int | Ever had blasts ≥5% |
| `is_brac_compatible` | bool | Has BRAC label |
| `brac_label` | int | BRAC label (0-8) |
"""

    # Save report
    report_path = OUTPUT_DIR / "BONE_MARROW_PREPROCESSING_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"  Report saved: {report_path}")

    return report

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete bone marrow preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("BONE MARROW PREPROCESSING PIPELINE (ENHANCED)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Step 1: Load and clean data
    df = load_and_clean_data()

    # Step 2: Normalize free-text qualitative fields
    df = normalize_qualitative_fields(df)

    # Step 3: Validate differential counts
    df = validate_differential_counts(df)

    # Step 4: Select representative sample (baseline)
    df = add_baseline_selection(df)

    # Step 5: Feature engineering
    df = engineer_features(df)

    # Step 6: Quality scores
    df = compute_quality_scores(df)

    # Step 7: Patient-level aggregation
    patient_df = aggregate_patient_level(df)

    # Step 8: Link to pathology
    patient_df = link_to_pathology(patient_df)

    # Step 9: Select and save outputs
    exam_output, patient_output, brac_output = select_and_save(df, patient_df)

    # Step 10: Generate report
    generate_report(exam_output, patient_output, brac_output)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Completed: {datetime.now()}")

    return exam_output, patient_output, brac_output

if __name__ == "__main__":
    main()
