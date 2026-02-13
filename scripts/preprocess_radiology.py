"""Preprocess radiology data for BRAC framework.

This script processes radiology reports for NHL patients, extracting:
1. Exam categorization (CT, US, PET, MRI, X-ray)
2. Anatomical region involvement (nodal sites, organs)
3. Lesion measurements and bulky disease markers
4. Disease status indicators (progression, response, stable)
5. Staging-relevant features (above/below diaphragm, extranodal)
6. Temporal patterns and treatment response tracking
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

# === PATHS ===
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "data/radiology/Radiology.xlsx"
PATHOLOGY_PATH = BASE_DIR / "data/pathology/pathology_cleaned_full.csv"
OUTPUT_DIR = BASE_DIR / "data/radiology"


def load_data():
    """Load radiology Excel file."""
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    df = pd.read_excel(INPUT_PATH)
    print(f"Loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"Unique patients: {df['Patient ID'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df


def clean_structure(df):
    """Clean and standardize column names and basic structure."""
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING STRUCTURE")
    print("=" * 60)

    # Rename columns to lowercase with underscores
    df = df.rename(columns={
        'Ray Name': 'ray_name',
        'Date': 'date',
        'Patient ID': 'patient_id',
        'Radiology ID': 'radiology_id',
    })

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    missing_dates = df['date'].isna().sum()
    print(f"  Parsed dates ({missing_dates} missing)")

    # Sort by patient and date
    df = df.sort_values(['patient_id', 'date']).reset_index(drop=True)

    return df


def categorize_exam_types(df):
    """Categorize radiology exams by modality and anatomical region."""
    print("\n" + "=" * 60)
    print("STEP 3: CATEGORIZING EXAM TYPES")
    print("=" * 60)

    ray_name = df['ray_name'].fillna('').str.lower()

    # =========================================
    # A. Primary modality classification
    # =========================================
    df['modality'] = 'OTHER'

    # PET-CT (check first as it contains 'CT')
    pet_mask = ray_name.str.contains('pet', na=False)
    df.loc[pet_mask, 'modality'] = 'PET_CT'

    # CT (excluding PET-CT)
    ct_mask = ray_name.str.contains('ct', na=False) & ~pet_mask
    df.loc[ct_mask, 'modality'] = 'CT'

    # MRI
    mri_mask = ray_name.str.contains('mri|mr ', na=False)
    df.loc[mri_mask, 'modality'] = 'MRI'

    # Ultrasound
    us_mask = ray_name.str.contains('us |us$|ultrasound|doppler', na=False, regex=True)
    df.loc[us_mask, 'modality'] = 'US'

    # X-ray (chest films)
    xray_mask = ray_name.str.contains('chest|x-ray|view|p\\.a\\.|radiograph', na=False, regex=True) & ~ct_mask & ~pet_mask
    df.loc[xray_mask, 'modality'] = 'XRAY'

    # Echocardiography
    echo_mask = ray_name.str.contains('echo', na=False)
    df.loc[echo_mask, 'modality'] = 'ECHO'

    # Biopsy-guided
    biopsy_mask = ray_name.str.contains('biopsy', na=False)
    df.loc[biopsy_mask, 'modality'] = 'BIOPSY_GUIDED'

    # Print modality distribution
    print("  Modality distribution:")
    for modality, count in df['modality'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {modality}: {count} ({pct:.1f}%)")

    # =========================================
    # B. Anatomical region flags (for CT/PET scans)
    # =========================================
    print("\n  Extracting anatomical regions...")

    df['region_brain'] = ray_name.str.contains('brain', na=False).astype(int)
    df['region_neck'] = ray_name.str.contains('neck', na=False).astype(int)
    df['region_chest'] = ray_name.str.contains('chest|thorax|hrct', na=False, regex=True).astype(int)
    df['region_abdomen'] = ray_name.str.contains('abd|abdominal', na=False, regex=True).astype(int)
    df['region_pelvis'] = ray_name.str.contains('pelvis|pelvic', na=False, regex=True).astype(int)

    # US-specific regions
    df['region_axilla'] = ray_name.str.contains('axill', na=False).astype(int)
    df['region_inguinal'] = ray_name.str.contains('inguin', na=False).astype(int)

    # Whole body (PET or multi-region CT)
    df['is_whole_body'] = (
        ray_name.str.contains('whole body', na=False) |
        (df['region_neck'] & df['region_chest'] & df['region_abdomen'] & df['region_pelvis'])
    ).astype(int)

    # One-hot encode modality
    modality_dummies = pd.get_dummies(df['modality'], prefix='mod', dtype=int)
    df = pd.concat([df, modality_dummies], axis=1)

    print(f"  Created {len([c for c in df.columns if c.startswith('region_')])} region flags")
    print(f"  Whole body scans: {df['is_whole_body'].sum()}")

    return df


def extract_findings_features(df):
    """Extract structured features from free-text findings."""
    print("\n" + "=" * 60)
    print("STEP 4: EXTRACTING FINDINGS FEATURES")
    print("=" * 60)

    findings = df['combined_findings_normalized'].fillna('').str.lower()

    # =========================================
    # A. Lymph node involvement by location
    # =========================================
    print("  Extracting lymph node locations...")

    ln_patterns = {
        'ln_cervical': r'cervical',
        'ln_submandibular': r'submandibular',
        'ln_supraclavicular': r'supraclavicular|supra-clavicular',
        'ln_axillary': r'axillary|axilla',
        'ln_mediastinal': r'mediastinal|mediastinum',
        'ln_hilar': r'hilar|hilum',
        'ln_paraaortic': r'para-aortic|paraaortic|retroperitoneal|retro-peritoneal',
        'ln_mesenteric': r'mesenteric',
        'ln_inguinal': r'inguinal',
        'ln_pelvic': r'pelvic|iliac',
    }

    for col, pattern in ln_patterns.items():
        df[col] = findings.str.contains(pattern, na=False, regex=True).astype(int)

    # Count total LN sites involved (per exam)
    ln_cols = [c for c in df.columns if c.startswith('ln_')]
    df['n_ln_sites'] = df[ln_cols].sum(axis=1)

    print(f"  Created {len(ln_cols)} lymph node location flags")

    # =========================================
    # B. Ann Arbor staging features
    # =========================================
    print("  Extracting staging features...")

    # Above diaphragm (cervical, axillary, mediastinal, supraclavicular)
    df['involvement_above_diaphragm'] = (
        df['ln_cervical'] | df['ln_submandibular'] | df['ln_supraclavicular'] |
        df['ln_axillary'] | df['ln_mediastinal'] | df['ln_hilar']
    ).astype(int)

    # Below diaphragm (para-aortic, mesenteric, inguinal, pelvic)
    df['involvement_below_diaphragm'] = (
        df['ln_paraaortic'] | df['ln_mesenteric'] | df['ln_inguinal'] | df['ln_pelvic']
    ).astype(int)

    # Both sides of diaphragm (Stage III indicator)
    df['involvement_both_sides'] = (
        df['involvement_above_diaphragm'] & df['involvement_below_diaphragm']
    ).astype(int)

    # Bilateral involvement
    df['is_bilateral'] = findings.str.contains(r'bilateral', na=False, regex=True).astype(int)

    print(f"    Above diaphragm: {df['involvement_above_diaphragm'].sum()}")
    print(f"    Below diaphragm: {df['involvement_below_diaphragm'].sum()}")
    print(f"    Both sides (Stage III+): {df['involvement_both_sides'].sum()}")

    # =========================================
    # C. Organ/extranodal involvement
    # =========================================
    print("  Extracting organ involvement...")

    organ_patterns = {
        'organ_spleen': r'spleen|splenic|splenomegaly',
        'organ_liver': r'liver|hepatic|hepatomegaly',
        'organ_bone': r'bone|vertebr|skeletal|osseous',
        'organ_lung': r'lung|pulmonary nodule|pulmonary mass',
        'organ_kidney': r'kidney|renal',
        'organ_gi': r'stomach|gastric|intestin|bowel',
        'organ_cns': r'brain|cns|central nervous',
        'organ_skin': r'skin|cutaneous|subcutaneous',
    }

    for col, pattern in organ_patterns.items():
        df[col] = findings.str.contains(pattern, na=False, regex=True).astype(int)

    # Extranodal involvement (Stage IV indicator)
    organ_cols = [c for c in df.columns if c.startswith('organ_')]
    df['has_extranodal'] = df[organ_cols].max(axis=1)
    df['n_organs_involved'] = df[organ_cols].sum(axis=1)

    print(f"  Created {len(organ_cols)} organ involvement flags")
    print(f"    Extranodal involvement: {df['has_extranodal'].sum()}")

    # =========================================
    # D. Effusions and fluid
    # =========================================
    df['has_pleural_effusion'] = findings.str.contains(r'pleural effusion', na=False, regex=True).astype(int)
    df['has_ascites'] = findings.str.contains(r'ascites|peritoneal fluid|abdominal fluid', na=False, regex=True).astype(int)
    df['has_pericardial_effusion'] = findings.str.contains(r'pericardial effusion', na=False, regex=True).astype(int)

    print(f"    Pleural effusion: {df['has_pleural_effusion'].sum()}")
    print(f"    Ascites: {df['has_ascites'].sum()}")

    # =========================================
    # E. Disease status indicators
    # =========================================
    print("  Extracting disease status...")

    df['status_progressive'] = findings.str.contains(
        r'progressive|progression|increased|worsening|enlarging',
        na=False, regex=True
    ).astype(int)

    df['status_regressive'] = findings.str.contains(
        r'regressive|regression|decreased|improved|smaller|reduction',
        na=False, regex=True
    ).astype(int)

    df['status_stable'] = findings.str.contains(
        r'stable|stationary|unchanged|no change|similar',
        na=False, regex=True
    ).astype(int)

    df['status_new_findings'] = findings.str.contains(
        r'new|newly developed|new finding',
        na=False, regex=True
    ).astype(int)

    df['status_complete_response'] = findings.str.contains(
        r'complete response|complete remission|no evidence of disease|resolved|disappeared',
        na=False, regex=True
    ).astype(int)

    df['status_residual'] = findings.str.contains(
        r'residual',
        na=False, regex=True
    ).astype(int)

    df['status_recurrence'] = findings.str.contains(
        r'recurrence|relapse|recurrent',
        na=False, regex=True
    ).astype(int)

    # Overall response category
    df['response_category'] = 'UNKNOWN'
    df.loc[df['status_complete_response'] == 1, 'response_category'] = 'CR'
    df.loc[df['status_regressive'] == 1, 'response_category'] = 'PR'
    df.loc[df['status_stable'] == 1, 'response_category'] = 'SD'
    df.loc[df['status_progressive'] == 1, 'response_category'] = 'PD'

    print("  Response category distribution:")
    for cat, count in df['response_category'].value_counts().items():
        print(f"    {cat}: {count}")

    # =========================================
    # F. Findings quality flag
    # =========================================
    df['has_findings'] = df['combined_findings_normalized'].notna().astype(int)

    return df


def process_measurements(df):
    """Process lesion measurements and extract size features."""
    print("\n" + "=" * 60)
    print("STEP 5: PROCESSING MEASUREMENTS")
    print("=" * 60)

    # =========================================
    # A. Clean measurement_value_mm
    # =========================================
    # This is already the primary/largest measurement
    df['max_lesion_mm'] = df['measurement_value_mm'].copy()

    # Flag outliers (>500mm is likely an error or sum of multiple lesions)
    outlier_mask = df['max_lesion_mm'] > 500
    if outlier_mask.any():
        print(f"  Flagged {outlier_mask.sum()} measurements > 500mm as potential outliers")
        df.loc[outlier_mask, 'flag_measurement_outlier'] = 1
    else:
        df['flag_measurement_outlier'] = 0

    # =========================================
    # B. Parse multi-dimensional measurements
    # =========================================
    print("  Parsing multi-dimensional measurements...")

    def parse_all_measurements(dims_str):
        """Parse measurement_all_values_mm to get all individual values."""
        if pd.isna(dims_str):
            return []
        try:
            values = []
            parts = str(dims_str).split(';')
            for part in parts:
                part = part.strip()
                if part:
                    val = float(part)
                    values.append(val)
            return values
        except:
            return []

    def parse_dims(dims_str):
        """Parse measurement_dims_mm format like '12.0x13.0;28.0'."""
        if pd.isna(dims_str):
            return []
        try:
            all_dims = []
            parts = str(dims_str).split(';')
            for part in parts:
                part = part.strip()
                if 'x' in part.lower():
                    # 2D measurement: take max dimension
                    dims = [float(d) for d in re.split(r'x', part, flags=re.IGNORECASE)]
                    all_dims.append(max(dims))
                elif part:
                    all_dims.append(float(part))
            return all_dims
        except:
            return []

    # Parse all measurements
    df['all_measurements'] = df['measurement_all_values_mm'].apply(parse_all_measurements)
    df['all_dims'] = df['measurement_dims_mm'].apply(parse_dims)

    # Number of measured lesions
    df['n_lesions_measured'] = df['all_dims'].apply(len)

    # Sum of all measurements (total tumor burden proxy)
    df['sum_lesions_mm'] = df['all_dims'].apply(lambda x: sum(x) if x else np.nan)

    # Mean lesion size
    df['mean_lesion_mm'] = df['all_dims'].apply(lambda x: np.mean(x) if x else np.nan)

    # Second largest lesion
    df['second_largest_mm'] = df['all_dims'].apply(
        lambda x: sorted(x, reverse=True)[1] if len(x) > 1 else np.nan
    )

    print(f"  Measurements processed:")
    print(f"    Records with measurements: {df['max_lesion_mm'].notna().sum()}")
    print(f"    Multi-lesion records: {(df['n_lesions_measured'] > 1).sum()}")

    # =========================================
    # C. Size categories and bulky disease
    # =========================================
    print("  Creating size categories...")

    # Size categories
    def size_category(mm):
        if pd.isna(mm):
            return 'UNKNOWN'
        if mm < 10:
            return 'SMALL'      # <1cm
        elif mm < 20:
            return 'MODERATE'   # 1-2cm
        elif mm < 50:
            return 'LARGE'      # 2-5cm
        elif mm < 75:
            return 'VERY_LARGE' # 5-7.5cm
        else:
            return 'BULKY'      # >7.5cm (bulky disease threshold)

    df['size_category'] = df['max_lesion_mm'].apply(size_category)

    # Bulky disease flag (>=75mm = 7.5cm, standard definition)
    df['is_bulky'] = (df['max_lesion_mm'] >= 75).astype(int)

    # Very bulky (>=100mm = 10cm)
    df['is_very_bulky'] = (df['max_lesion_mm'] >= 100).astype(int)

    print(f"    Bulky disease (>=75mm): {df['is_bulky'].sum()}")
    print(f"    Very bulky (>=100mm): {df['is_very_bulky'].sum()}")

    # Size category distribution
    print("  Size category distribution:")
    for cat, count in df['size_category'].value_counts().items():
        print(f"    {cat}: {count}")

    # Clean up temporary columns
    df = df.drop(columns=['all_measurements', 'all_dims'], errors='ignore')

    return df


def add_temporal_features(df):
    """Add temporal features for tracking disease over time."""
    print("\n" + "=" * 60)
    print("STEP 6: ADDING TEMPORAL FEATURES")
    print("=" * 60)

    # Sort by patient and date
    df = df.sort_values(['patient_id', 'date']).reset_index(drop=True)

    # Exam order per patient
    df['exam_order'] = df.groupby('patient_id').cumcount() + 1

    # Days since first exam
    first_exam = df.groupby('patient_id')['date'].transform('min')
    df['days_since_first_exam'] = (df['date'] - first_exam).dt.days

    # Days since previous exam
    df['days_since_prev_exam'] = df.groupby('patient_id')['date'].diff().dt.days

    # Total exams per patient
    df['total_exams'] = df.groupby('patient_id')['exam_number'].transform('nunique')

    # Year/month
    df['exam_year'] = df['date'].dt.year
    df['exam_month'] = df['date'].dt.month

    print(f"  Exams per patient: mean={df['total_exams'].mean():.1f}, max={df['total_exams'].max()}")

    # =========================================
    # Change in lesion size over time
    # =========================================
    print("  Calculating lesion size changes...")

    # Previous max lesion size
    df['prev_max_lesion_mm'] = df.groupby('patient_id')['max_lesion_mm'].shift(1)

    # Change in max lesion
    df['lesion_change_mm'] = df['max_lesion_mm'] - df['prev_max_lesion_mm']
    df['lesion_change_pct'] = (df['lesion_change_mm'] / df['prev_max_lesion_mm'].replace(0, np.nan)) * 100

    # Classify response based on size change
    # RECIST-like criteria: >20% increase = progression, >30% decrease = response
    df['size_response'] = 'STABLE'
    df.loc[df['lesion_change_pct'] > 20, 'size_response'] = 'PROGRESSION'
    df.loc[df['lesion_change_pct'] < -30, 'size_response'] = 'RESPONSE'
    df.loc[df['max_lesion_mm'].isna() | df['prev_max_lesion_mm'].isna(), 'size_response'] = 'UNKNOWN'

    return df


def create_pet_features(df):
    """Extract PET-CT specific features."""
    print("\n" + "=" * 60)
    print("STEP 7: EXTRACTING PET-CT FEATURES")
    print("=" * 60)

    findings = df['combined_findings_normalized'].fillna('').str.lower()

    # SUV mentions (metabolic activity)
    df['has_suv_data'] = findings.str.contains(r'suv|metabolic|fdg|uptake|avid', na=False, regex=True).astype(int)

    # Metabolically active disease
    df['is_metabolically_active'] = findings.str.contains(
        r'metabolically active|fdg avid|increased uptake|hypermetabolic',
        na=False, regex=True
    ).astype(int)

    # Metabolic complete response
    df['is_metabolic_cr'] = findings.str.contains(
        r'metabolic complete|no fdg|no metabolic|resolved',
        na=False, regex=True
    ).astype(int)

    pet_df = df[df['modality'] == 'PET_CT']
    print(f"  PET-CT scans: {len(pet_df)}")
    print(f"  With SUV data: {df['has_suv_data'].sum()}")
    print(f"  Metabolically active: {df['is_metabolically_active'].sum()}")
    print(f"  Metabolic CR: {df['is_metabolic_cr'].sum()}")

    return df


def create_quality_scores(df):
    """Create data quality scores for radiology."""
    print("\n" + "=" * 60)
    print("STEP 8: CREATING QUALITY SCORES")
    print("=" * 60)

    # =========================================
    # Q = Report completeness (has findings text)
    # =========================================
    df['quality_Q'] = df['has_findings'].astype(float)

    # =========================================
    # C = Coverage (key fields present)
    # =========================================
    key_fields = ['date', 'ray_name', 'combined_findings_normalized', 'measurement_value_mm']
    df['coverage_C'] = df[key_fields].notna().sum(axis=1) / len(key_fields)

    # =========================================
    # S = 1.0 for radiology (no internal consistency check applicable)
    # =========================================
    df['consistency_S'] = 1.0

    print(f"  Quality scores:")
    print(f"    Q (report completeness): mean={df['quality_Q'].mean():.2f}")
    print(f"    C (field coverage): mean={df['coverage_C'].mean():.2f}")
    print(f"    S (consistency): 1.0 (N/A for radiology)")

    return df


def aggregate_patient_features(df):
    """Aggregate exam-level data to patient-level features."""
    print("\n" + "=" * 60)
    print("STEP 9: AGGREGATING PATIENT-LEVEL FEATURES")
    print("=" * 60)

    results = []

    for patient_id, patient_data in df.groupby('patient_id'):
        features = {'patient_id': patient_id}

        # =========================================
        # A. Exam counts by modality
        # =========================================
        features['n_exams_total'] = len(patient_data)
        for mod in ['CT', 'US', 'PET_CT', 'MRI', 'XRAY', 'ECHO']:
            features[f'n_exams_{mod.lower()}'] = (patient_data['modality'] == mod).sum()

        # =========================================
        # B. Lesion size features
        # =========================================
        sizes = patient_data['max_lesion_mm'].dropna()
        if len(sizes) > 0:
            features['max_lesion_ever_mm'] = sizes.max()
            features['first_lesion_mm'] = sizes.iloc[0]
            features['last_lesion_mm'] = sizes.iloc[-1]
            features['mean_lesion_mm'] = sizes.mean()
            if len(sizes) > 1:
                features['lesion_trend_mm'] = sizes.iloc[-1] - sizes.iloc[0]
            else:
                features['lesion_trend_mm'] = 0

        # Bulky disease ever
        features['ever_bulky'] = patient_data['is_bulky'].max()
        features['ever_very_bulky'] = patient_data['is_very_bulky'].max()

        # =========================================
        # C. Lymph node involvement (ever)
        # =========================================
        ln_cols = [c for c in patient_data.columns if c.startswith('ln_')]
        for col in ln_cols:
            features[f'ever_{col}'] = patient_data[col].max()

        # Max LN sites in any single exam
        features['max_ln_sites'] = patient_data['n_ln_sites'].max()

        # =========================================
        # D. Staging features (ever)
        # =========================================
        features['ever_above_diaphragm'] = patient_data['involvement_above_diaphragm'].max()
        features['ever_below_diaphragm'] = patient_data['involvement_below_diaphragm'].max()
        features['ever_both_sides'] = patient_data['involvement_both_sides'].max()
        features['ever_bilateral'] = patient_data['is_bilateral'].max()
        features['ever_extranodal'] = patient_data['has_extranodal'].max()

        # Max organs involved
        features['max_organs_involved'] = patient_data['n_organs_involved'].max()

        # Specific organ involvement (ever)
        organ_cols = [c for c in patient_data.columns if c.startswith('organ_')]
        for col in organ_cols:
            features[f'ever_{col}'] = patient_data[col].max()

        # =========================================
        # E. Effusions (ever)
        # =========================================
        features['ever_pleural_effusion'] = patient_data['has_pleural_effusion'].max()
        features['ever_ascites'] = patient_data['has_ascites'].max()

        # =========================================
        # F. Disease status summary
        # =========================================
        features['ever_progressive'] = patient_data['status_progressive'].max()
        features['ever_regressive'] = patient_data['status_regressive'].max()
        features['ever_stable'] = patient_data['status_stable'].max()
        features['ever_complete_response'] = patient_data['status_complete_response'].max()
        features['ever_recurrence'] = patient_data['status_recurrence'].max()

        # Best response achieved
        response_order = {'CR': 4, 'PR': 3, 'SD': 2, 'PD': 1, 'UNKNOWN': 0}
        responses = patient_data['response_category'].map(response_order)
        best_response_val = responses.max()
        best_response_map = {v: k for k, v in response_order.items()}
        features['best_response'] = best_response_map.get(best_response_val, 'UNKNOWN')

        # =========================================
        # G. PET features
        # =========================================
        pet_data = patient_data[patient_data['modality'] == 'PET_CT']
        features['has_pet'] = int(len(pet_data) > 0)
        features['n_pet_scans'] = len(pet_data)
        if len(pet_data) > 0:
            features['ever_metabolically_active'] = pet_data['is_metabolically_active'].max()
            features['ever_metabolic_cr'] = pet_data['is_metabolic_cr'].max()
        else:
            features['ever_metabolically_active'] = 0
            features['ever_metabolic_cr'] = 0

        # =========================================
        # H. Temporal features
        # =========================================
        valid_dates = patient_data['date'].dropna()
        if len(valid_dates) > 0:
            features['first_exam_date'] = valid_dates.min()
            features['last_exam_date'] = valid_dates.max()
            features['followup_days'] = (valid_dates.max() - valid_dates.min()).days

        results.append(features)

    patient_df = pd.DataFrame(results)
    print(f"  Aggregated {len(patient_df)} patients with {len(patient_df.columns)} features")

    return patient_df


def link_pathology(patient_df):
    """Link to pathology data for BRAC labels."""
    print("\n" + "=" * 60)
    print("STEP 10: LINKING TO PATHOLOGY DATA")
    print("=" * 60)

    # Load pathology data
    path_df = pd.read_csv(PATHOLOGY_PATH)

    # Get first investigation per patient for labels
    path_first = path_df.sort_values(['patient_id', 'date']).groupby('patient_id').first().reset_index()

    # Select relevant columns
    path_cols = ['patient_id', 'disease_type_final', 'brac_label', 'is_brac_compatible']
    path_subset = path_first[path_cols]

    # Merge
    patient_df = patient_df.merge(path_subset, on='patient_id', how='left')

    # Fill missing values (avoid FutureWarning)
    patient_df.loc[patient_df['is_brac_compatible'].isna(), 'is_brac_compatible'] = False
    patient_df['is_brac_compatible'] = patient_df['is_brac_compatible'].astype(bool)

    print(f"  Patients with pathology: {patient_df['disease_type_final'].notna().sum()}")
    print(f"  BRAC-compatible: {patient_df['is_brac_compatible'].sum()}")

    if patient_df['is_brac_compatible'].sum() > 0:
        print("\n  BRAC label distribution:")
        label_names = {
            0: "DLBCL_GCB", 1: "DLBCL_ABC", 2: "FL", 3: "MCL", 4: "BL",
            5: "MZL", 6: "CLL_SLL", 7: "LPL", 8: "PMBL"
        }
        brac_counts = patient_df[patient_df['is_brac_compatible']]['brac_label'].value_counts().sort_index()
        for label, count in brac_counts.items():
            name = label_names.get(int(label), "Unknown")
            print(f"    {int(label)} ({name}): {count}")

    return patient_df


def save_outputs(df, patient_df):
    """Save processed radiology data."""
    print("\n" + "=" * 60)
    print("STEP 11: SAVING OUTPUTS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save exam-level data
    exam_path = OUTPUT_DIR / "radiology_cleaned.csv"
    df.to_csv(exam_path, index=False)
    print(f"  Saved exam-level: {exam_path}")
    print(f"    {len(df)} rows × {len(df.columns)} columns")

    # Save patient-level data
    patient_path = OUTPUT_DIR / "radiology_patient_level.csv"
    patient_df.to_csv(patient_path, index=False)
    print(f"\n  Saved patient-level: {patient_path}")
    print(f"    {len(patient_df)} rows × {len(patient_df.columns)} columns")

    # Save BRAC-compatible subset
    brac_df = patient_df[patient_df['is_brac_compatible'] == True].copy()
    brac_path = OUTPUT_DIR / "radiology_brac_compatible.csv"
    brac_df.to_csv(brac_path, index=False)
    print(f"\n  Saved BRAC-compatible: {brac_path}")
    print(f"    {len(brac_df)} rows × {len(brac_df.columns)} columns")

    return df, patient_df


def generate_report(df, patient_df):
    """Generate preprocessing report."""
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    report_lines = [
        "# Radiology Data Preprocessing Pipeline Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Script:** `scripts/preprocess_radiology.py`",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "### Input/Output Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Input file | `data/radiology/Radiology.xlsx` |",
        f"| Input rows | {len(df)} |",
        f"| Unique patients | {df['patient_id'].nunique()} |",
        f"| Date range | {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')} |",
        f"| Output exam-level columns | {len(df.columns)} |",
        f"| Output patient-level columns | {len(patient_df.columns)} |",
        f"| BRAC-compatible patients | {patient_df['is_brac_compatible'].sum()} |",
        "",
        "---",
        "",
        "## 2. Modality Distribution",
        "",
        "| Modality | Count | Percentage |",
        "|----------|-------|------------|",
    ]

    for mod, count in df['modality'].value_counts().items():
        pct = count / len(df) * 100
        report_lines.append(f"| {mod} | {count} | {pct:.1f}% |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Key Features Extracted",
        "",
        "### Lymph Node Locations",
        "| Location | Exams with involvement |",
        "|----------|----------------------|",
    ])

    ln_cols = [c for c in df.columns if c.startswith('ln_')]
    for col in ln_cols:
        count = df[col].sum()
        report_lines.append(f"| {col.replace('ln_', '').title()} | {count} |")

    report_lines.extend([
        "",
        "### Staging Features (Exam-Level)",
        "",
        f"- Above diaphragm involvement: {df['involvement_above_diaphragm'].sum()} exams",
        f"- Below diaphragm involvement: {df['involvement_below_diaphragm'].sum()} exams",
        f"- Both sides (Stage III+): {df['involvement_both_sides'].sum()} exams",
        f"- Extranodal involvement: {df['has_extranodal'].sum()} exams",
        "",
        "### Measurements",
        "",
        f"- Exams with measurements: {df['max_lesion_mm'].notna().sum()}",
        f"- Bulky disease (>=75mm): {df['is_bulky'].sum()} exams",
        f"- Very bulky (>=100mm): {df['is_very_bulky'].sum()} exams",
        "",
        "---",
        "",
        "## 4. Patient-Level Summary",
        "",
        f"- Total patients: {len(patient_df)}",
        f"- Patients with PET-CT: {patient_df['has_pet'].sum()}",
        f"- Ever bulky disease: {patient_df['ever_bulky'].sum()}",
        f"- Ever extranodal: {patient_df['ever_extranodal'].sum()}",
        f"- Ever complete response: {patient_df['ever_complete_response'].sum()}",
        "",
    ])

    report = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "PREPROCESSING_PIPELINE_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"  Saved report: {report_path}")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("RADIOLOGY DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # Execute pipeline
    df = load_data()
    df = clean_structure(df)
    df = categorize_exam_types(df)
    df = extract_findings_features(df)
    df = process_measurements(df)
    df = add_temporal_features(df)
    df = create_pet_features(df)
    df = create_quality_scores(df)

    # Patient-level aggregation
    patient_df = aggregate_patient_features(df)
    patient_df = link_pathology(patient_df)

    # Save outputs
    df, patient_df = save_outputs(df, patient_df)
    generate_report(df, patient_df)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
