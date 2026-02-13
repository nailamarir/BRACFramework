"""Pathology Data Preprocessing Script.

This script cleans and processes the raw pathology Excel data,
extracts features, and saves cleaned data for review.

Usage:
    python scripts/preprocess_pathology.py

Output:
    - data/pathology/pathology_cleaned.csv (cleaned text data)
    - data/pathology/pathology_with_features.csv (with extracted features)
    - data/pathology/preprocessing_summary.txt (statistics report)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path("data/pathology/pathology.xlsx")
OUTPUT_DIR = Path("data/pathology")

# BRAC label mapping
BRAC_LABELS = {
    'DLBCL_GCB': 0,
    'DLBCL_ABC': 1,
    'FL': 2,
    'MCL': 3,
    'BL': 4,
    'MZL': 5,
    'CLL_SLL': 6,
    'LPL': 7,
    'PMBL': 8,
}


# =============================================================================
# STEP 1: LOAD AND BASIC CLEANING
# =============================================================================

def load_and_clean_basic(filepath):
    """Load Excel and perform basic cleaning."""
    print("=" * 60)
    print("STEP 1: Loading and Basic Cleaning")
    print("=" * 60)

    df = pd.read_excel(filepath)
    print(f"Loaded {len(df)} records")

    # Standardize column names
    df = df.rename(columns={
        'Patient ID': 'patient_id',
        'Invistigation ID': 'investigation_id',  # Note: typo in original
        'Date': 'date',
        'Microscopic Examination': 'microscopic_exam',
        'Diagnosis': 'diagnosis'
    })

    # Remove "NoneNone" artifacts
    for col in ['microscopic_exam', 'diagnosis']:
        if col in df.columns:
            before_none = df[col].str.contains('NoneNone', na=False).sum()
            df[col] = df[col].str.replace('NoneNone', '', regex=False).str.strip()
            print(f"Removed 'NoneNone' from {before_none} records in {col}")

    # Handle missing values
    missing_diag = df['diagnosis'].isna().sum()
    missing_micro = df['microscopic_exam'].isna().sum()
    print(f"Missing diagnosis: {missing_diag}")
    print(f"Missing microscopic_exam: {missing_micro}")

    # Create lowercase versions for processing
    df['diagnosis_lower'] = df['diagnosis'].str.lower().fillna('')
    df['micro_lower'] = df['microscopic_exam'].str.lower().fillna('')

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df


# =============================================================================
# STEP 2: IHC MARKER EXTRACTION
# =============================================================================

def extract_marker_status(text, marker_pattern):
    """Extract positive/negative/weak status for a marker."""
    if pd.isna(text) or text == '':
        return None

    text = text.lower()

    # Positive patterns
    pos_patterns = [
        rf'{marker_pattern}[:\s]*positive',
        rf'{marker_pattern}[:\s]*\+',
        rf'{marker_pattern}[:\s]*diffuse\s*positive',
        rf'{marker_pattern}[:\s]*strong',
        rf'positive[^.{{0,30}}]*{marker_pattern}',
        rf'{marker_pattern}[:\s]*>\s*\d+\s*%',
        rf'{marker_pattern}[:\s]*\d{{2,3}}\s*%',  # High percentage like 70%, 90%
    ]

    # Negative patterns
    neg_patterns = [
        rf'{marker_pattern}[:\s]*negative',
        rf'{marker_pattern}[:\s]*\-',
        rf'negative[^.{{0,30}}]*{marker_pattern}',
        rf'{marker_pattern}[:\s]*<\s*[1-9]\s*%',
    ]

    # Weak patterns
    weak_patterns = [
        rf'{marker_pattern}[:\s]*weak',
        rf'{marker_pattern}[:\s]*focal',
        rf'{marker_pattern}[:\s]*dim',
    ]

    for pat in pos_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return 'positive'

    for pat in neg_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return 'negative'

    for pat in weak_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return 'weak'

    # Check if marker is mentioned at all
    if re.search(marker_pattern, text, re.IGNORECASE):
        return 'mentioned'

    return None


def extract_ki67_percentage(text):
    """Extract Ki-67 proliferation index as percentage."""
    if pd.isna(text) or text == '':
        return None

    # Patterns: "ki-67: >70%", "ki67 80%", "ki-67 positive in 90%"
    patterns = [
        r'ki[\-\s]?67[:\s]*[>]?\s*(\d+)\s*%',
        r'ki[\-\s]?67[:\s]*positive\s*in\s*[>]?\s*(\d+)\s*%',
        r'(\d+)\s*%\s*of\s*[^.]*ki[\-\s]?67',
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def extract_all_ihc_markers(df):
    """Extract all IHC markers from microscopic examination."""
    print("\n" + "=" * 60)
    print("STEP 2: Extracting IHC Markers")
    print("=" * 60)

    markers = {
        'ihc_cd20': r'cd\s*20',
        'ihc_cd3': r'cd\s*3(?!\d)',
        'ihc_cd10': r'cd\s*10',
        'ihc_cd5': r'cd\s*5(?!\d)',
        'ihc_cd23': r'cd\s*23',
        'ihc_cd30': r'cd\s*30',
        'ihc_bcl2': r'bcl[\-\s]?2(?!\d)',
        'ihc_bcl6': r'bcl[\-\s]?6',
        'ihc_mum1': r'mum[\-\s]?1',
        'ihc_ki67': r'ki[\-\s]?67',
        'ihc_cyclin_d1': r'cyclin\s*d\s*1|ccnd1',
        'ihc_sox11': r'sox\s*11',
        'ihc_cd138': r'cd\s*138',
        'ihc_pax5': r'pax\s*5',
        'ihc_cd79a': r'cd\s*79\s*a',
        'ihc_tdt': r'tdt|terminal\s*deoxynucleotidyl',
    }

    for col_name, pattern in markers.items():
        df[col_name] = df['micro_lower'].apply(lambda x: extract_marker_status(x, pattern))
        found = df[col_name].notna().sum()
        print(f"  {col_name}: {found} records with data")

    # Extract Ki-67 percentage separately
    df['ki67_percentage'] = df['micro_lower'].apply(extract_ki67_percentage)
    ki67_found = df['ki67_percentage'].notna().sum()
    print(f"  ki67_percentage: {ki67_found} numeric values extracted")

    return df


# =============================================================================
# STEP 3: MORPHOLOGICAL FEATURE EXTRACTION
# =============================================================================

def extract_morphological_features(df):
    """Extract morphological features from text."""
    print("\n" + "=" * 60)
    print("STEP 3: Extracting Morphological Features")
    print("=" * 60)

    morph_patterns = {
        'morph_large_cells': r'large\s+(lymphoid\s+)?cells|large\s+b[\-\s]?cell',
        'morph_small_cells': r'small\s+(lymphoid\s+)?cells|small\s+b[\-\s]?cell',
        'morph_atypical': r'atypia|atypical',
        'morph_necrosis': r'necrosis|necrotic',
        'morph_mitoses': r'mitosis|mitoses|mitotic',
        'morph_fibrosis': r'fibrosis|fibrotic|sclerosis',
        'morph_effacement': r'effacement|effaced',
        'morph_nodular_pattern': r'nodular\s+pattern|nodular\s+growth',
        'morph_diffuse_pattern': r'diffuse\s+pattern|diffuse\s+infiltrat|diffuse\s+growth',
        'morph_starry_sky': r'starry[\-\s]?sky',
        'morph_tingible_body': r'tingible\s+body',
        'morph_plasmacytoid': r'plasmacytoid|plasmablastic',
        'morph_centroblast': r'centroblast',
        'morph_centrocyte': r'centrocyte',
        'morph_mantle_zone': r'mantle\s+zone',
    }

    for col_name, pattern in morph_patterns.items():
        df[col_name] = df['micro_lower'].str.contains(pattern, regex=True, na=False).astype(int)
        found = df[col_name].sum()
        if found > 0:
            print(f"  {col_name}: {found} records")

    return df


# =============================================================================
# STEP 4: SPECIMEN TYPE EXTRACTION
# =============================================================================

def extract_specimen_type(df):
    """Extract specimen/tissue type."""
    print("\n" + "=" * 60)
    print("STEP 4: Extracting Specimen Type")
    print("=" * 60)

    def get_specimen_type(text):
        if pd.isna(text) or text == '':
            return 'unknown'

        text = text.lower()

        if re.search(r'bone\s*marrow|b\.?m\.?b?|bmb|trephine', text):
            return 'bone_marrow'
        elif re.search(r'lymph\s*node|l\.?\s*n\.?|excision.*node', text):
            return 'lymph_node'
        elif re.search(r'spleen|splenic|splenectomy', text):
            return 'spleen'
        elif re.search(r'skin|cutaneous|dermis', text):
            return 'skin'
        elif re.search(r'gastric|stomach|intestin|colon|gi\s+tract|bowel', text):
            return 'gi_tract'
        elif re.search(r'liver|hepatic', text):
            return 'liver'
        elif re.search(r'tonsil', text):
            return 'tonsil'
        elif re.search(r'thyroid', text):
            return 'thyroid'
        elif re.search(r'mediastin', text):
            return 'mediastinum'
        elif re.search(r'aspirate|smear|fna', text):
            return 'aspirate'
        else:
            return 'other'

    df['specimen_type'] = df['micro_lower'].apply(get_specimen_type)

    print("Specimen type distribution:")
    for spec_type, count in df['specimen_type'].value_counts().items():
        print(f"  {spec_type}: {count}")

    return df


# =============================================================================
# STEP 5: NHL SUBTYPE CLASSIFICATION
# =============================================================================

def classify_disease_type(df):
    """Classify into disease categories."""
    print("\n" + "=" * 60)
    print("STEP 5: Classifying Disease Type")
    print("=" * 60)

    def get_disease_type(diag, micro):
        diag = diag if isinstance(diag, str) else ''
        micro = micro if isinstance(micro, str) else ''
        combined = diag + ' ' + micro

        # Check for non-lymphoma first
        if re.search(r'carcinoma|adenocarcinoma|metastatic|squamous', diag):
            return 'NON_LYMPHOID'
        if re.search(r'plasma\s*cell|myeloma|plasmacytoma', diag):
            return 'PLASMA_CELL'
        if re.search(r'no\s+malignancy|no\s+evidence\s+of\s+malignancy|negative\s+for\s+malignancy', diag):
            return 'REACTIVE_BENIGN'
        if re.search(r'reactive\s+lymph|reactive\s+node|reactive\s+hyperplasia|benign', diag):
            return 'REACTIVE_BENIGN'
        if re.search(r'inconclusive|inadequate|non[\-\s]representative|insufficient', diag):
            return 'INCONCLUSIVE'

        # Hodgkin vs Non-Hodgkin
        if re.search(r'hodgkin', diag):
            # Check if it's "non-Hodgkin"
            if re.search(r'non[\-\s]?hodgkin', diag):
                pass  # Continue to NHL classification
            else:
                return 'HODGKIN'

        # T-cell lymphoma
        if re.search(r't[\-\s]?cell\s+lymphoma|peripheral\s+t[\-\s]?cell|anaplastic\s+large\s+cell', diag):
            return 'T_CELL_LYMPHOMA'

        # Specific B-cell NHL subtypes
        if re.search(r'burkitt', diag):
            return 'BL'
        if re.search(r'mantle\s+cell|mcl', diag):
            return 'MCL'
        if re.search(r'primary\s+mediastinal', diag):
            return 'PMBL'
        if re.search(r'lymphoplasmacytic|waldenstrom', diag):
            return 'LPL'
        if re.search(r'marginal\s+zone|malt', diag):
            return 'MZL'
        if re.search(r'chronic\s+lymphocytic|cll|small\s+lymphocytic|sll', diag):
            return 'CLL_SLL'
        if re.search(r'follicular\s+lymphoma|follicular\s+nhl', diag):
            if re.search(r'high[\-\s]?grade|grade\s*(3|iii)', diag):
                return 'FL_HIGH_GRADE'
            return 'FL'

        # DLBCL with subtyping
        if re.search(r'diffuse\s+large\s+b[\-\s]?cell|dlbcl', diag):
            if re.search(r'germinal\s+center|gcb', combined):
                return 'DLBCL_GCB'
            elif re.search(r'activated\s+b[\-\s]?cell|non[\-\s]?gcb|non[\-\s]?germinal|abc|post[\-\s]?germinal', combined):
                return 'DLBCL_ABC'
            else:
                return 'DLBCL_NOS'

        # Generic B-cell
        if re.search(r'b[\-\s]?cell\s+lymphoma|b[\-\s]?cell\s+nhl', diag):
            return 'B_CELL_NOS'

        # Generic NHL
        if re.search(r'non[\-\s]?hodgkin|nhl|lymphoma', diag):
            return 'NHL_NOS'

        # Leukemia
        if re.search(r'leukemia|leukaemia', diag):
            return 'LEUKEMIA_OTHER'

        return 'OTHER'

    df['disease_type'] = df.apply(
        lambda row: get_disease_type(row['diagnosis_lower'], row['micro_lower']),
        axis=1
    )

    print("Disease type distribution:")
    for dtype, count in df['disease_type'].value_counts().items():
        print(f"  {dtype}: {count}")

    return df


# =============================================================================
# STEP 6: HANS ALGORITHM FOR DLBCL SUBTYPING
# =============================================================================

def apply_hans_algorithm(df):
    """Apply Hans algorithm to subtype DLBCL_NOS cases."""
    print("\n" + "=" * 60)
    print("STEP 6: Applying Hans Algorithm to DLBCL_NOS")
    print("=" * 60)

    def hans_classify(row):
        if row['disease_type'] != 'DLBCL_NOS':
            return row['disease_type']

        cd10 = row.get('ihc_cd10')
        bcl6 = row.get('ihc_bcl6')
        mum1 = row.get('ihc_mum1')

        # Hans algorithm
        if cd10 == 'positive':
            return 'DLBCL_GCB'
        elif cd10 == 'negative':
            if bcl6 == 'negative':
                return 'DLBCL_ABC'
            elif bcl6 == 'positive':
                if mum1 == 'negative':
                    return 'DLBCL_GCB'
                elif mum1 == 'positive':
                    return 'DLBCL_ABC'

        # Can't determine
        return 'DLBCL_NOS'

    # Count DLBCL_NOS before
    nos_before = (df['disease_type'] == 'DLBCL_NOS').sum()

    df['disease_type_final'] = df.apply(hans_classify, axis=1)

    # Count after
    nos_after = (df['disease_type_final'] == 'DLBCL_NOS').sum()
    gcb_gained = (df['disease_type_final'] == 'DLBCL_GCB').sum() - (df['disease_type'] == 'DLBCL_GCB').sum()
    abc_gained = (df['disease_type_final'] == 'DLBCL_ABC').sum() - (df['disease_type'] == 'DLBCL_ABC').sum()

    print(f"DLBCL_NOS before Hans: {nos_before}")
    print(f"DLBCL_NOS after Hans: {nos_after}")
    print(f"Reclassified to GCB: {gcb_gained}")
    print(f"Reclassified to ABC: {abc_gained}")

    return df


# =============================================================================
# STEP 7: ASSIGN BRAC LABELS
# =============================================================================

def assign_brac_labels(df):
    """Assign BRAC labels (0-8) to compatible cases."""
    print("\n" + "=" * 60)
    print("STEP 7: Assigning BRAC Labels")
    print("=" * 60)

    # Map disease types to BRAC labels
    label_map = {
        'DLBCL_GCB': 0,
        'DLBCL_ABC': 1,
        'FL': 2,
        'FL_HIGH_GRADE': 2,  # Group with FL
        'MCL': 3,
        'BL': 4,
        'MZL': 5,
        'CLL_SLL': 6,
        'LPL': 7,
        'PMBL': 8,
    }

    df['brac_label'] = df['disease_type_final'].map(label_map)
    df['is_brac_compatible'] = df['brac_label'].notna()

    brac_count = df['is_brac_compatible'].sum()
    print(f"BRAC-compatible cases: {brac_count} / {len(df)} ({brac_count/len(df)*100:.1f}%)")

    print("\nBRAC label distribution:")
    for dtype in label_map.keys():
        count = (df['disease_type_final'] == dtype).sum()
        if count > 0:
            label = label_map[dtype]
            print(f"  {dtype} (label={label}): {count}")

    return df


# =============================================================================
# STEP 8: ADD QUALITY FLAGS
# =============================================================================

def add_quality_flags(df):
    """Add quality flags for data review."""
    print("\n" + "=" * 60)
    print("STEP 8: Adding Quality Flags")
    print("=" * 60)

    # Flag: Has IHC data
    ihc_cols = [c for c in df.columns if c.startswith('ihc_')]
    df['has_ihc'] = df[ihc_cols].notna().any(axis=1)

    # Flag: Has complete Hans markers
    df['has_complete_hans'] = (
        df['ihc_cd10'].notna() &
        df['ihc_bcl6'].notna() &
        df['ihc_mum1'].notna()
    )

    # Flag: Short diagnosis (potential issue)
    df['flag_short_diagnosis'] = df['diagnosis'].str.len() < 30

    # Flag: Missing microscopic exam
    df['flag_missing_micro'] = df['microscopic_exam'].isna() | (df['microscopic_exam'].str.len() < 10)

    # Flag: Transformation case
    df['flag_transformation'] = df['diagnosis_lower'].str.contains(r'transform|richter', regex=True, na=False)

    # Flag: Relapse/recurrent
    df['flag_relapse'] = df['diagnosis_lower'].str.contains(r'relaps|recurr', regex=True, na=False)

    print(f"Cases with IHC data: {df['has_ihc'].sum()}")
    print(f"Cases with complete Hans markers: {df['has_complete_hans'].sum()}")
    print(f"Short diagnosis (<30 chars): {df['flag_short_diagnosis'].sum()}")
    print(f"Transformation cases: {df['flag_transformation'].sum()}")
    print(f"Relapse cases: {df['flag_relapse'].sum()}")

    return df


# =============================================================================
# STEP 9: SAVE OUTPUTS
# =============================================================================

def save_outputs(df, output_dir):
    """Save cleaned data and summary."""
    print("\n" + "=" * 60)
    print("STEP 9: Saving Outputs")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full cleaned data with all features
    full_path = output_dir / "pathology_cleaned_full.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

    # 2. Save simplified version for easy review
    review_cols = [
        'patient_id', 'investigation_id', 'date',
        'diagnosis', 'microscopic_exam',
        'disease_type', 'disease_type_final', 'brac_label', 'is_brac_compatible',
        'specimen_type',
        'ihc_cd20', 'ihc_cd10', 'ihc_bcl6', 'ihc_mum1', 'ihc_ki67', 'ki67_percentage',
        'ihc_cd5', 'ihc_cd23', 'ihc_cyclin_d1',
        'has_ihc', 'has_complete_hans',
        'flag_short_diagnosis', 'flag_transformation', 'flag_relapse'
    ]
    review_cols = [c for c in review_cols if c in df.columns]
    review_df = df[review_cols].copy()

    review_path = output_dir / "pathology_cleaned_review.csv"
    review_df.to_csv(review_path, index=False)
    print(f"Saved: {review_path}")

    # 3. Save BRAC-compatible only
    brac_df = df[df['is_brac_compatible']].copy()
    brac_path = output_dir / "pathology_brac_compatible.csv"
    brac_df.to_csv(brac_path, index=False)
    print(f"Saved: {brac_path} ({len(brac_df)} cases)")

    # 4. Save summary report
    summary_path = output_dir / "preprocessing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PATHOLOGY DATA PREPROCESSING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("INPUT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Unique patients: {df['patient_id'].nunique()}\n")
        f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n\n")

        f.write("DISEASE TYPE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for dtype, count in df['disease_type_final'].value_counts().items():
            f.write(f"  {dtype}: {count}\n")

        f.write("\nBRAC COMPATIBILITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"BRAC-compatible: {df['is_brac_compatible'].sum()}\n")
        f.write(f"Non-compatible: {(~df['is_brac_compatible']).sum()}\n\n")

        f.write("BRAC LABEL DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        brac_labels = df[df['is_brac_compatible']]['brac_label'].value_counts().sort_index()
        label_names = {v: k for k, v in BRAC_LABELS.items()}
        for label, count in brac_labels.items():
            name = label_names.get(int(label), 'Unknown')
            f.write(f"  {int(label)} ({name}): {count}\n")

        f.write("\nIHC DATA AVAILABILITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Cases with any IHC: {df['has_ihc'].sum()}\n")
        f.write(f"Cases with complete Hans: {df['has_complete_hans'].sum()}\n\n")

        f.write("QUALITY FLAGS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Short diagnosis: {df['flag_short_diagnosis'].sum()}\n")
        f.write(f"Transformation: {df['flag_transformation'].sum()}\n")
        f.write(f"Relapse: {df['flag_relapse'].sum()}\n")

    print(f"Saved: {summary_path}")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("PATHOLOGY DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Load and clean
    df = load_and_clean_basic(INPUT_FILE)

    # Step 2: Extract IHC markers
    df = extract_all_ihc_markers(df)

    # Step 3: Extract morphological features
    df = extract_morphological_features(df)

    # Step 4: Extract specimen type
    df = extract_specimen_type(df)

    # Step 5: Classify disease type
    df = classify_disease_type(df)

    # Step 6: Apply Hans algorithm
    df = apply_hans_algorithm(df)

    # Step 7: Assign BRAC labels
    df = assign_brac_labels(df)

    # Step 8: Add quality flags
    df = add_quality_flags(df)

    # Step 9: Save outputs
    df = save_outputs(df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutput files:")
    print("  1. pathology_cleaned_full.csv - All data with all features")
    print("  2. pathology_cleaned_review.csv - Simplified for review")
    print("  3. pathology_brac_compatible.csv - BRAC-compatible cases only")
    print("  4. preprocessing_summary.txt - Summary statistics")
    print("\nPlease review the data and report any issues!")


if __name__ == "__main__":
    main()
