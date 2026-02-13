"""Preprocess CBC Laboratory Data for BRAC Framework.

This script processes Complete Blood Count (CBC) laboratory data:
1. Cleans and standardizes column names
2. Handles missing values and outliers
3. Creates temporal features
4. Derives clinically meaningful ratios (NLR, PLR, LMR)
5. Creates binary clinical flags
6. Aggregates to patient-level features
7. Links to pathology data for BRAC labels
8. Calculates quality scores

Input: data/laboratory/CBClaboratory.xlsx
Output:
    - data/laboratory/laboratory_cleaned.csv (investigation-level)
    - data/laboratory/laboratory_patient_features.csv (patient-level)
    - data/laboratory/laboratory_brac_compatible.csv (BRAC subset)
    - data/laboratory/preprocessing_summary.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
INPUT_PATH = DATA_DIR / "laboratory" / "CBClaboratory.xlsx"
OUTPUT_DIR = DATA_DIR / "laboratory"
PATHOLOGY_PATH = DATA_DIR / "pathology" / "pathology_training_ready.csv"

# Column name mapping
COLUMN_MAPPING = {
    'Patient ID': 'patient_id',
    'Investigations ID': 'investigation_id',
    'Date': 'date',
    'WBC': 'wbc',
    'RBC': 'rbc',
    'HGB': 'hgb',
    'HCT': 'hct',
    'MCV': 'mcv',
    'MCH': 'mch',
    'MCHC': 'mchc',
    'PLT': 'plt',
    'LYMPH%': 'lymph_pct',
    'NEUT%': 'neut_pct',
    'LYMPH No': 'lymph_abs',
    'NEUT No': 'neut_abs',
    'PDW': 'pdw',
    'MPV': 'mpv',
    'RDW': 'rdw',
    'Mono count': 'mono_abs',
    'Eos count': 'eos_abs',
    'Baso count': 'baso_abs',
    'Mono Per': 'mono_pct',
    'Eos Per': 'eos_pct',
    'Baso Per': 'baso_pct',
}

# Columns to drop (near-empty) - Note: MXD will be unified with Mono/Eos/Baso first
DROP_COLUMNS = ['P-LCR']  # MXD handled separately in unify_mxd_columns()

# Numeric columns for conversion
NUMERIC_COLUMNS = [
    'wbc', 'rbc', 'hgb', 'hct', 'mcv', 'mch', 'mchc', 'plt',
    'lymph_pct', 'neut_pct', 'lymph_abs', 'neut_abs',
    'pdw', 'mpv', 'rdw', 'mono_abs', 'eos_abs', 'baso_abs',
    'mono_pct', 'eos_pct', 'baso_pct'
]

# Columns where zero is biologically impossible
ZERO_IMPOSSIBLE = ['wbc', 'rbc', 'hgb', 'hct', 'plt']

# Clinical reference ranges (for flagging, not filtering)
CLINICAL_RANGES = {
    'wbc': (4.0, 11.0),
    'rbc': (4.0, 6.0),
    'hgb': (12.0, 17.0),
    'hct': (36.0, 50.0),
    'mcv': (80.0, 100.0),
    'mch': (27.0, 33.0),
    'mchc': (32.0, 36.0),
    'plt': (150.0, 400.0),
    'lymph_pct': (20.0, 40.0),
    'neut_pct': (40.0, 70.0),
    'rdw': (11.5, 14.5),
}


def load_data():
    """Load and perform initial cleaning."""
    print("Loading data...")
    df = pd.read_excel(INPUT_PATH, header=1)
    print(f"  Loaded {len(df)} records")
    return df


def clean_structure(df):
    """Clean column names and structure."""
    print("\nCleaning structure...")

    # Drop near-empty columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"  Dropped {len(cols_to_drop)} near-empty columns: {cols_to_drop}")

    # Rename columns
    df = df.rename(columns=COLUMN_MAPPING)
    print(f"  Renamed columns to lowercase")

    return df


def unify_mxd_columns(df):
    """Unify MXD (mixed cells) with Mono/Eos/Baso components.

    Older analyzers report MXD% and MXD No (mixed cells = mono + eos + baso).
    Newer analyzers report individual Mono, Eos, Baso counts.

    Strategy:
    - If Mono+Eos+Baso are present: compute MXD from sum
    - If only MXD is present: keep MXD, leave components as NaN
    - Create unified mxd_abs and mxd_pct columns
    """
    print("\nUnifying MXD with Mono/Eos/Baso components...")

    # First, clean any instrument flags in MXD columns
    for col in ['MXD%', 'MXD No']:
        if col in df.columns:
            str_col = df[col].astype(str)
            # Clean X suffix
            x_mask = str_col.str.match(r'^\.?\d+\.?\d*X$', case=False, na=False)
            if x_mask.any():
                df.loc[x_mask, col] = str_col[x_mask].str.replace('X', '', case=False, regex=False)
            # Clean dashes
            dash_mask = str_col.str.match(r'^-{2,}$', na=False)
            if dash_mask.any():
                df.loc[dash_mask, col] = np.nan
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename MXD columns to lowercase
    if 'MXD%' in df.columns:
        df = df.rename(columns={'MXD%': 'mxd_pct_original'})
    if 'MXD No' in df.columns:
        df = df.rename(columns={'MXD No': 'mxd_abs_original'})

    # Check what data we have
    has_mxd = df.get('mxd_abs_original', pd.Series([np.nan]*len(df))).notna()
    has_mono = df.get('mono_abs', pd.Series([np.nan]*len(df))).notna()
    has_eos = df.get('eos_abs', pd.Series([np.nan]*len(df))).notna()
    has_baso = df.get('baso_abs', pd.Series([np.nan]*len(df))).notna()
    has_all_components = has_mono & has_eos & has_baso

    # Create unified MXD columns
    df['mxd_abs'] = np.nan
    df['mxd_pct'] = np.nan

    # Case 1: Has all components - compute MXD
    if has_all_components.any():
        # Need to convert components to numeric first
        for col in ['mono_abs', 'eos_abs', 'baso_abs']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        mask = has_all_components
        df.loc[mask, 'mxd_abs'] = (
            df.loc[mask, 'mono_abs'].fillna(0) +
            df.loc[mask, 'eos_abs'].fillna(0) +
            df.loc[mask, 'baso_abs'].fillna(0)
        )
        computed_from_components = mask.sum()
    else:
        computed_from_components = 0

    # Case 2: Has only MXD (no components) - use original MXD
    only_mxd = has_mxd & ~has_all_components
    if only_mxd.any():
        df.loc[only_mxd, 'mxd_abs'] = df.loc[only_mxd, 'mxd_abs_original']
        if 'mxd_pct_original' in df.columns:
            df.loc[only_mxd, 'mxd_pct'] = df.loc[only_mxd, 'mxd_pct_original']
        used_original = only_mxd.sum()
    else:
        used_original = 0

    # Also compute MXD percentage if we have components percentages
    for col in ['mono_pct', 'eos_pct', 'baso_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    has_pct_components = (
        df.get('mono_pct', pd.Series([np.nan]*len(df))).notna() &
        df.get('eos_pct', pd.Series([np.nan]*len(df))).notna() &
        df.get('baso_pct', pd.Series([np.nan]*len(df))).notna()
    )

    if has_pct_components.any():
        mask = has_pct_components & df['mxd_pct'].isna()
        df.loc[mask, 'mxd_pct'] = (
            df.loc[mask, 'mono_pct'].fillna(0) +
            df.loc[mask, 'eos_pct'].fillna(0) +
            df.loc[mask, 'baso_pct'].fillna(0)
        )

    # Drop original MXD columns (now unified)
    cols_to_drop = ['mxd_abs_original', 'mxd_pct_original']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Report
    total_mxd = df['mxd_abs'].notna().sum()
    print(f"  Computed MXD from Mono+Eos+Baso: {computed_from_components}")
    print(f"  Used original MXD (no components): {used_original}")
    print(f"  Total unified MXD values: {total_mxd} ({total_mxd/len(df)*100:.1f}%)")

    return df


def clean_instrument_flags(df):
    """Clean instrument flags and special values from raw data."""
    print("\nCleaning instrument flags...")

    stats = {
        'x_suffix': 0,
        'dashes': 0,
        'over_range': 0,
        'under_range': 0,
    }

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        # Convert to string for pattern matching
        str_col = df[col].astype(str)

        # 1. Handle "X" suffix (instrument flag - value is still valid)
        # "4.20X" → 4.20, ".246X" → 0.246
        x_mask = str_col.str.match(r'^\.?\d+\.?\d*X$', case=False, na=False)
        if x_mask.any():
            stats['x_suffix'] += x_mask.sum()
            # Extract numeric part by removing X
            df.loc[x_mask, col] = str_col[x_mask].str.replace('X', '', case=False, regex=False)

        # 2. Handle "-----" or "----" (no result / measurement failed)
        dash_mask = str_col.str.match(r'^-{2,}$', na=False)
        if dash_mask.any():
            stats['dashes'] += dash_mask.sum()
            df.loc[dash_mask, col] = np.nan

        # 3. Handle ">>>>>" (over measurement range)
        over_mask = str_col.str.match(r'^>{2,}$', na=False)
        if over_mask.any():
            stats['over_range'] += over_mask.sum()
            # Flag these separately - they indicate very high values
            df.loc[over_mask, f'{col}_over_range'] = 1
            df.loc[over_mask, col] = np.nan

        # 4. Handle "<<<<<" (under measurement range)
        under_mask = str_col.str.match(r'^<{2,}$', na=False)
        if under_mask.any():
            stats['under_range'] += under_mask.sum()
            # Flag these separately - they indicate very low values
            df.loc[under_mask, f'{col}_under_range'] = 1
            df.loc[under_mask, col] = np.nan

        # 5. Handle single "." (empty/error)
        dot_mask = str_col == '.'
        if dot_mask.any():
            df.loc[dot_mask, col] = np.nan

    print(f"  Cleaned {stats['x_suffix']} 'X' suffix values (instrument flags)")
    print(f"  Replaced {stats['dashes']} '-----' values with NaN")
    print(f"  Flagged {stats['over_range']} '>>>>>' (over-range) values")
    print(f"  Flagged {stats['under_range']} '<<<<<' (under-range) values")

    return df


def convert_types(df):
    """Convert data types and handle invalid values."""
    print("\nConverting data types...")

    # Convert numeric columns (after cleaning instrument flags)
    converted = 0
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_na = df[col].isna().sum()
            converted += (after_na - before_na)

    if converted > 0:
        print(f"  Note: {converted} additional values couldn't be converted to numeric")

    # Handle impossible zero values
    zeros_fixed = 0
    for col in ZERO_IMPOSSIBLE:
        if col in df.columns:
            mask = df[col] == 0
            zeros_fixed += mask.sum()
            df.loc[mask, col] = np.nan
    print(f"  Replaced {zeros_fixed} impossible zero values with NaN")

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    missing_dates = df['date'].isna().sum()
    print(f"  Parsed dates ({missing_dates} missing)")

    return df


def handle_outliers(df):
    """Handle outliers with physiological range validation.

    Strategy:
    - Values outside physiological bounds → flag and set to NaN
    - Values outside normal but physiologically possible → flag only (keep value)
    - Log all flagged values for transparency
    """
    print("\nHandling outliers with physiological validation...")

    # Physiological bounds (values outside these are measurement errors)
    PHYSIOLOGICAL_BOUNDS = {
        'wbc': (0.1, 500),      # Can be very high in leukemia
        'rbc': (0.5, 10),       # Rare to exceed 8
        'hgb': (1.0, 25),       # Can't survive <1, max ~25 in polycythemia
        'hct': (5, 75),         # Extreme but possible
        'mcv': (40, 150),       # Micro to macro
        'mch': (10, 50),        # Derived value
        'mchc': (20, 40),       # Hemoglobin concentration has hard limits
        'plt': (1, 2500),       # Can be very high in reactive thrombocytosis
        'lymph_abs': (0, 500),  # Can be extreme in CLL
        'neut_abs': (0, 100),   # Can be high in infection/leukemia
        'mono_abs': (0, 60),    # Monocytosis
        'eos_abs': (0, 50),     # Eosinophilia
        'baso_abs': (0, 10),    # Basophilia rare
        'rdw': (5, 40),         # Distribution width
        'mpv': (3, 20),         # Platelet volume
        'pdw': (5, 30),         # Platelet distribution
    }

    # Clinical alert thresholds (flag but keep - clinically significant)
    CLINICAL_ALERTS = {
        'wbc': {'high': 100, 'low': 1.0},      # Extreme leukocytosis/leukopenia
        'hgb': {'high': 20, 'low': 6},          # Polycythemia/critical anemia
        'plt': {'high': 1000, 'low': 20},       # Extreme TCP/thrombocytosis
        'mch': {'high': 35, 'low': 20},         # Possible lipemia artifact
        'mchc': {'high': 36, 'low': 28},        # Possible lipemia/cold agglutinins
        'lymph_abs': {'high': 50, 'low': None}, # Extreme lymphocytosis (CLL)
    }

    stats = {
        'impossible_replaced': 0,
        'clinically_flagged': 0,
    }

    outlier_log = []

    # Step 1: Replace physiologically impossible values with NaN
    for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
        if col not in df.columns:
            continue

        # Below minimum (impossible)
        below_mask = df[col] < low
        if below_mask.any():
            count = below_mask.sum()
            examples = df.loc[below_mask, col].head(3).tolist()
            outlier_log.append(f"  {col} < {low}: {count} values replaced (examples: {examples})")
            df.loc[below_mask, col] = np.nan
            stats['impossible_replaced'] += count

        # Above maximum (impossible)
        above_mask = df[col] > high
        if above_mask.any():
            count = above_mask.sum()
            examples = df.loc[above_mask, col].head(3).tolist()
            outlier_log.append(f"  {col} > {high}: {count} values replaced (examples: {examples})")
            df.loc[above_mask, col] = np.nan
            stats['impossible_replaced'] += count

    print(f"  Replaced {stats['impossible_replaced']} physiologically impossible values")

    if outlier_log:
        print("\n  Outlier details:")
        for line in outlier_log:
            print(line)

    # Step 2: Flag clinically significant extremes (keep values, add flags)
    print("\n  Creating clinical alert flags...")

    for col, thresholds in CLINICAL_ALERTS.items():
        if col not in df.columns:
            continue

        if thresholds.get('high') is not None:
            high_mask = df[col] > thresholds['high']
            if high_mask.any():
                df[f'flag_{col}_high'] = high_mask.astype(int)
                stats['clinically_flagged'] += high_mask.sum()
                print(f"    {col} > {thresholds['high']}: {high_mask.sum()} flagged")

        if thresholds.get('low') is not None:
            low_mask = df[col] < thresholds['low']
            if low_mask.any():
                df[f'flag_{col}_low'] = low_mask.astype(int)
                stats['clinically_flagged'] += low_mask.sum()
                print(f"    {col} < {thresholds['low']}: {low_mask.sum()} flagged")

    # Step 3: Check for lipemia/interference artifacts
    # High MCHC (>36) with high MCH (>35) often indicates lipemia
    if 'mch' in df.columns and 'mchc' in df.columns:
        lipemia_suspect = (df['mchc'] > 36) & (df['mch'] > 35)
        if lipemia_suspect.any():
            df['flag_lipemia_suspect'] = lipemia_suspect.astype(int)
            print(f"    Lipemia suspect (MCH>35 & MCHC>36): {lipemia_suspect.sum()} flagged")

    print(f"\n  Total clinically flagged: {stats['clinically_flagged']}")

    return df


def add_temporal_features(df):
    """Add temporal features."""
    print("\nAdding temporal features...")

    # Sort by patient and date
    df = df.sort_values(['patient_id', 'date']).reset_index(drop=True)

    # Investigation order per patient
    df['investigation_order'] = df.groupby('patient_id').cumcount() + 1
    df['total_investigations'] = df.groupby('patient_id')['patient_id'].transform('count')
    df['is_first_visit'] = (df['investigation_order'] == 1).astype(int)

    # Time-based features
    df['days_since_first'] = df.groupby('patient_id')['date'].transform(
        lambda x: (x - x.min()).dt.days
    )
    df['days_since_prev'] = df.groupby('patient_id')['date'].diff().dt.days
    df['visit_year'] = df['date'].dt.year

    print(f"  Added investigation order, days since first/prev visit")

    return df


def add_clinical_phase_tagging(df):
    """Tag each CBC by clinical phase based on temporal patterns.

    Phases:
    - BASELINE: First CBC or pre-treatment (stable counts, early in timeline)
    - ON_TREATMENT: Active chemotherapy (dropping WBC/PLT, nadir period)
    - RECOVERY: Counts rising after nadir
    - STABLE: Stable counts during follow-up
    - RELAPSE: New cytopenias or abnormalities after recovery

    Uses:
    - Investigation order
    - Delta (change) in WBC and PLT from previous visit
    - Absolute values and clinical flags
    """
    print("\nAdding clinical phase tagging...")

    # Calculate deltas (change from previous visit)
    df = df.sort_values(['patient_id', 'date']).reset_index(drop=True)

    # Calculate changes from previous visit
    for col in ['wbc', 'plt', 'hgb', 'lymph_abs', 'neut_abs']:
        if col in df.columns:
            df[f'{col}_prev'] = df.groupby('patient_id')[col].shift(1)
            df[f'{col}_delta'] = df[col] - df[f'{col}_prev']
            df[f'{col}_pct_change'] = (df[f'{col}_delta'] / df[f'{col}_prev'].replace(0, np.nan)) * 100

    # Initialize phase column
    df['clinical_phase'] = 'UNKNOWN'

    # Process each patient
    phase_counts = {'BASELINE': 0, 'ON_TREATMENT': 0, 'RECOVERY': 0, 'STABLE': 0, 'RELAPSE': 0, 'UNKNOWN': 0}

    for patient_id, patient_data in df.groupby('patient_id'):
        indices = patient_data.index.tolist()

        if len(indices) == 1:
            # Single visit - mark as BASELINE
            df.loc[indices[0], 'clinical_phase'] = 'BASELINE'
            phase_counts['BASELINE'] += 1
            continue

        # Track patient state
        had_nadir = False
        had_recovery = False
        prev_phase = None

        for i, idx in enumerate(indices):
            row = df.loc[idx]

            # First visit is always BASELINE
            if i == 0:
                df.loc[idx, 'clinical_phase'] = 'BASELINE'
                phase_counts['BASELINE'] += 1
                prev_phase = 'BASELINE'
                continue

            # Get deltas
            wbc_delta = row.get('wbc_delta', np.nan)
            plt_delta = row.get('plt_delta', np.nan)
            wbc = row.get('wbc', np.nan)
            plt = row.get('plt', np.nan)
            wbc_pct = row.get('wbc_pct_change', np.nan)
            plt_pct = row.get('plt_pct_change', np.nan)

            # Determine phase based on patterns
            phase = 'STABLE'  # Default

            # ON_TREATMENT: Significant drop in WBC or PLT (>30% drop or absolute low)
            is_dropping = False
            if pd.notna(wbc_pct) and wbc_pct < -30:
                is_dropping = True
            if pd.notna(plt_pct) and plt_pct < -30:
                is_dropping = True
            if pd.notna(wbc) and wbc < 2:  # Severe leukopenia
                is_dropping = True
            if pd.notna(plt) and plt < 50:  # Severe thrombocytopenia
                is_dropping = True

            if is_dropping and not had_recovery:
                phase = 'ON_TREATMENT'
                had_nadir = True

            # RECOVERY: Rising after nadir
            is_recovering = False
            if had_nadir and not had_recovery:
                if pd.notna(wbc_pct) and wbc_pct > 20:
                    is_recovering = True
                if pd.notna(plt_pct) and plt_pct > 20:
                    is_recovering = True
                if pd.notna(wbc) and wbc > 4 and prev_phase == 'ON_TREATMENT':
                    is_recovering = True

            if is_recovering:
                phase = 'RECOVERY'
                had_recovery = True

            # RELAPSE: New cytopenias after recovery period
            if had_recovery:
                # Check for new abnormalities after being stable
                days_since = row.get('days_since_first', 0)
                if days_since > 180:  # At least 6 months out
                    if is_dropping:
                        phase = 'RELAPSE'

            # STABLE: No significant changes, counts in acceptable range
            if phase == 'STABLE':
                if pd.notna(wbc) and 4 <= wbc <= 11:
                    if pd.notna(plt) and 150 <= plt <= 400:
                        phase = 'STABLE'

            df.loc[idx, 'clinical_phase'] = phase
            phase_counts[phase] += 1
            prev_phase = phase

    # Report phase distribution
    print(f"  Phase distribution:")
    for phase, count in sorted(phase_counts.items()):
        pct = count / len(df) * 100
        print(f"    {phase}: {count} ({pct:.1f}%)")

    # Create binary flags for each phase
    for phase in ['BASELINE', 'ON_TREATMENT', 'RECOVERY', 'STABLE', 'RELAPSE']:
        df[f'phase_{phase.lower()}'] = (df['clinical_phase'] == phase).astype(int)

    # Clean up temporary columns
    temp_cols = [c for c in df.columns if c.endswith('_prev')]
    df = df.drop(columns=temp_cols, errors='ignore')

    return df


def create_derived_features(df):
    """Create clinically meaningful derived features."""
    print("\nCreating derived features...")

    # =========================================
    # A. Prognostic ratios
    # =========================================
    df['nlr'] = df['neut_abs'] / df['lymph_abs'].replace(0, np.nan)
    df['plr'] = df['plt'] / df['lymph_abs'].replace(0, np.nan)
    df['lmr'] = df['lymph_abs'] / df['mono_abs'].replace(0, np.nan)

    # Cap extreme ratios (likely due to very low denominators)
    df.loc[df['nlr'] > 100, 'nlr'] = np.nan
    df.loc[df['plr'] > 1000, 'plr'] = np.nan
    df.loc[df['lmr'] > 100, 'lmr'] = np.nan

    print(f"  Created NLR, PLR, LMR ratios")

    # =========================================
    # B. Binary clinical flags
    # =========================================
    df['has_anemia'] = (df['hgb'] < 12).astype(int)
    df['has_severe_anemia'] = (df['hgb'] < 8).astype(int)
    df['has_thrombocytopenia'] = (df['plt'] < 150).astype(int)
    df['has_severe_tcp'] = (df['plt'] < 50).astype(int)
    df['has_leukocytosis'] = (df['wbc'] > 11).astype(int)
    df['has_leukopenia'] = (df['wbc'] < 4).astype(int)
    df['has_lymphocytosis'] = (df['lymph_abs'] > 5).astype(int)
    df['has_neutropenia'] = (df['neut_abs'] < 1.5).astype(int)
    df['has_severe_neutropenia'] = (df['neut_abs'] < 0.5).astype(int)
    df['has_high_nlr'] = (df['nlr'] > 3).astype(int)
    df['has_low_lmr'] = (df['lmr'] < 2.6).astype(int)

    # CLL signature: WBC > 10 AND LYMPH% > 60 (or absolute lymphocytes > 5)
    # This is a key diagnostic criterion for CLL/SLL
    cll_signature = (
        (df['wbc'] > 10) &
        ((df['lymph_pct'] > 60) | (df['lymph_abs'] > 5))
    )
    df['has_cll_signature'] = cll_signature.astype(int)
    cll_count = cll_signature.sum()
    print(f"  CLL signature (WBC>10 & LYMPH%>60): {cll_count} records")

    # Bone marrow failure pattern (pancytopenia)
    pancytopenia = (
        (df['wbc'] < 4) &
        (df['hgb'] < 10) &
        (df['plt'] < 100)
    )
    df['has_pancytopenia'] = pancytopenia.astype(int)

    # High tumor burden pattern (all elevated)
    high_burden = (
        (df['wbc'] > 15) &
        ((df['hgb'] < 10) | (df['plt'] < 100))
    )
    df['has_high_tumor_burden'] = high_burden.astype(int)

    flag_cols = [c for c in df.columns if c.startswith('has_')]
    print(f"  Created {len(flag_cols)} binary clinical flags")

    # =========================================
    # C. Quality scores (Q, C, S)
    # =========================================
    print("\n  Calculating quality scores...")

    # Core markers expected in every CBC
    core_markers = ['wbc', 'rbc', 'hgb', 'hct', 'mcv', 'mch', 'mchc', 'plt']
    extended_markers = ['lymph_abs', 'neut_abs', 'mono_abs', 'rdw', 'mpv', 'pdw']

    # Q score: 1 - missing fraction (of core markers)
    df['quality_q'] = df[core_markers].notna().sum(axis=1) / len(core_markers)

    # C score: completeness - tested columns / expected
    all_expected = core_markers + extended_markers
    available_cols = [c for c in all_expected if c in df.columns]
    df['quality_c'] = df[available_cols].notna().sum(axis=1) / len(available_cols)

    # S score: internal consistency
    # Check: HCT ≈ RBC × MCV / 10 (should be within 10%)
    # Check: MCHC ≈ HGB / HCT × 100 (should be within 10%)
    def calc_consistency(row):
        scores = []

        # HCT consistency: HCT ≈ RBC × MCV / 10
        if pd.notna(row['hct']) and pd.notna(row['rbc']) and pd.notna(row['mcv']):
            expected_hct = row['rbc'] * row['mcv'] / 10
            if expected_hct > 0:
                hct_error = abs(row['hct'] - expected_hct) / expected_hct
                scores.append(1 - min(hct_error, 1))  # Cap at 1

        # MCHC consistency: MCHC ≈ HGB / HCT × 100
        if pd.notna(row['mchc']) and pd.notna(row['hgb']) and pd.notna(row['hct']) and row['hct'] > 0:
            expected_mchc = row['hgb'] / row['hct'] * 100
            if expected_mchc > 0:
                mchc_error = abs(row['mchc'] - expected_mchc) / expected_mchc
                scores.append(1 - min(mchc_error, 1))

        # MCH consistency: MCH ≈ HGB / RBC × 10
        if pd.notna(row['mch']) and pd.notna(row['hgb']) and pd.notna(row['rbc']) and row['rbc'] > 0:
            expected_mch = row['hgb'] / row['rbc'] * 10
            if expected_mch > 0:
                mch_error = abs(row['mch'] - expected_mch) / expected_mch
                scores.append(1 - min(mch_error, 1))

        if scores:
            return np.mean(scores)
        return np.nan

    df['quality_s'] = df.apply(calc_consistency, axis=1)

    # Combined quality score
    df['quality_overall'] = (
        df['quality_q'] * 0.4 +
        df['quality_c'] * 0.3 +
        df['quality_s'].fillna(0.5) * 0.3
    )

    # Flag low quality records
    df['flag_low_quality'] = (df['quality_overall'] < 0.5).astype(int)

    print(f"    Q score (completeness): mean={df['quality_q'].mean():.3f}")
    print(f"    C score (coverage): mean={df['quality_c'].mean():.3f}")
    print(f"    S score (consistency): mean={df['quality_s'].mean():.3f}")
    print(f"    Low quality records: {df['flag_low_quality'].sum()}")

    return df


def aggregate_patient_features(df):
    """Aggregate to patient-level features."""
    print("\nAggregating patient-level features...")

    # Markers to aggregate
    agg_markers = [
        'wbc', 'rbc', 'hgb', 'hct', 'plt', 'mcv', 'rdw',
        'lymph_abs', 'neut_abs', 'mono_abs', 'eos_abs', 'baso_abs', 'mxd_abs',
        'nlr', 'plr', 'lmr'
    ]

    # Binary ever-flags
    binary_flags = [
        'has_anemia', 'has_severe_anemia', 'has_thrombocytopenia',
        'has_severe_tcp', 'has_leukocytosis', 'has_leukopenia',
        'has_lymphocytosis', 'has_neutropenia', 'has_severe_neutropenia',
        'has_high_nlr', 'has_low_lmr'
    ]

    results = []

    for patient_id, patient_data in df.groupby('patient_id'):
        features = {'patient_id': patient_id}

        # Numeric aggregations
        for col in agg_markers:
            if col in patient_data.columns:
                values = patient_data[col].dropna()
                if len(values) > 0:
                    features[f'{col}_first'] = values.iloc[0]
                    features[f'{col}_last'] = values.iloc[-1]
                    features[f'{col}_mean'] = values.mean()
                    features[f'{col}_min'] = values.min()
                    features[f'{col}_max'] = values.max()
                    features[f'{col}_std'] = values.std() if len(values) > 1 else 0
                    features[f'{col}_range'] = values.max() - values.min()
                    if len(values) > 1:
                        features[f'{col}_trend'] = values.iloc[-1] - values.iloc[0]
                    else:
                        features[f'{col}_trend'] = 0

        # Binary ever-flags
        for flag in binary_flags:
            if flag in patient_data.columns:
                features[f'ever_{flag.replace("has_", "")}'] = patient_data[flag].max()

        # Temporal features
        features['n_cbc_tests'] = len(patient_data)
        features['followup_days'] = patient_data['days_since_first'].max()

        # Date range
        valid_dates = patient_data['date'].dropna()
        if len(valid_dates) > 0:
            features['first_cbc_date'] = valid_dates.min()
            features['last_cbc_date'] = valid_dates.max()

        # Clinical phase counts
        if 'clinical_phase' in patient_data.columns:
            phase_counts = patient_data['clinical_phase'].value_counts()
            for phase in ['BASELINE', 'ON_TREATMENT', 'RECOVERY', 'STABLE', 'RELAPSE']:
                features[f'n_phase_{phase.lower()}'] = phase_counts.get(phase, 0)

            # Ever had treatment/relapse
            features['ever_on_treatment'] = int('ON_TREATMENT' in phase_counts.index)
            features['ever_relapse'] = int('RELAPSE' in phase_counts.index)

        # Baseline values (from first BASELINE phase visit)
        if 'clinical_phase' in patient_data.columns:
            baseline_data = patient_data[patient_data['clinical_phase'] == 'BASELINE']
            if len(baseline_data) > 0:
                for col in ['wbc', 'hgb', 'plt', 'lymph_abs', 'neut_abs']:
                    if col in baseline_data.columns:
                        val = baseline_data[col].iloc[0]
                        if pd.notna(val):
                            features[f'{col}_baseline'] = val

        results.append(features)

    patient_df = pd.DataFrame(results)
    print(f"  Aggregated {len(patient_df)} patients with {len(patient_df.columns)} features")

    return patient_df


def link_pathology(patient_df):
    """Link to pathology data for BRAC labels."""
    print("\nLinking to pathology data...")

    if not PATHOLOGY_PATH.exists():
        print(f"  WARNING: Pathology file not found at {PATHOLOGY_PATH}")
        return patient_df

    pathology = pd.read_csv(PATHOLOGY_PATH)

    # Get first diagnosis per patient
    patient_diagnosis = pathology.groupby('patient_id').agg({
        'disease_type_final': 'first',
        'brac_label': 'first',
        'is_brac_compatible': 'first',
    }).reset_index()

    # Merge
    original_len = len(patient_df)
    patient_df = patient_df.merge(patient_diagnosis, on='patient_id', how='left')

    linked = patient_df['disease_type_final'].notna().sum()
    brac_compat = patient_df['is_brac_compatible'].sum() if 'is_brac_compatible' in patient_df.columns else 0

    print(f"  Linked {linked} / {original_len} patients to pathology")
    print(f"  BRAC-compatible patients with labs: {int(brac_compat)}")

    return patient_df


def calculate_quality_scores(patient_df):
    """Calculate data quality scores."""
    print("\nCalculating quality scores...")

    def quality_score(row):
        # Core markers availability
        core_markers = ['wbc', 'rbc', 'hgb', 'plt', 'lymph_abs', 'neut_abs']
        core_complete = sum(pd.notna(row.get(f'{m}_first')) for m in core_markers) / len(core_markers)

        # Extended markers
        extended = ['mono_abs', 'rdw', 'mcv']
        extended_complete = sum(pd.notna(row.get(f'{m}_first')) for m in extended) / len(extended)

        # Longitudinal richness
        n_tests = row.get('n_cbc_tests', 1)
        temporal_score = min(n_tests / 10, 1.0)

        # Overall
        return round(0.5 * core_complete + 0.3 * extended_complete + 0.2 * temporal_score, 3)

    patient_df['quality_score'] = patient_df.apply(quality_score, axis=1)
    print(f"  Mean quality score: {patient_df['quality_score'].mean():.3f}")

    return patient_df


def generate_summary(df, patient_df):
    """Generate preprocessing summary."""
    summary_lines = [
        "LABORATORY DATA PREPROCESSING SUMMARY",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "INPUT",
        "-" * 40,
        f"Total records: {len(df)}",
        f"Unique patients: {df['patient_id'].nunique()}",
    ]

    # Date range
    valid_dates = df['date'].dropna()
    if len(valid_dates) > 0:
        summary_lines.append(f"Date range: {valid_dates.min()} to {valid_dates.max()}")

    # Temporal stats
    summary_lines.extend([
        "",
        "TEMPORAL STATISTICS",
        "-" * 40,
        f"Mean investigations per patient: {df.groupby('patient_id').size().mean():.1f}",
        f"Max investigations per patient: {df.groupby('patient_id').size().max()}",
        f"Mean follow-up duration: {patient_df['followup_days'].mean():.0f} days",
    ])

    # Clinical flags prevalence
    summary_lines.extend([
        "",
        "CLINICAL FLAGS PREVALENCE",
        "-" * 40,
    ])

    flag_cols = [c for c in df.columns if c.startswith('has_')]
    for flag in sorted(flag_cols):
        pct = df[flag].mean() * 100
        summary_lines.append(f"  {flag}: {pct:.1f}%")

    # Derived ratios
    summary_lines.extend([
        "",
        "DERIVED RATIOS (mean values)",
        "-" * 40,
        f"  NLR: {df['nlr'].mean():.2f} (median: {df['nlr'].median():.2f})",
        f"  PLR: {df['plr'].mean():.1f} (median: {df['plr'].median():.1f})",
        f"  LMR: {df['lmr'].mean():.2f} (median: {df['lmr'].median():.2f})",
    ])

    # Patient-level stats
    summary_lines.extend([
        "",
        "PATIENT-LEVEL AGGREGATION",
        "-" * 40,
        f"Total patients: {len(patient_df)}",
        f"Features per patient: {len(patient_df.columns)}",
        f"Mean quality score: {patient_df['quality_score'].mean():.3f}",
    ])

    # Pathology linkage
    if 'disease_type_final' in patient_df.columns:
        linked = patient_df['disease_type_final'].notna().sum()
        brac = patient_df['is_brac_compatible'].sum() if 'is_brac_compatible' in patient_df.columns else 0
        summary_lines.extend([
            "",
            "PATHOLOGY LINKAGE",
            "-" * 40,
            f"Linked to pathology: {linked}",
            f"BRAC-compatible: {int(brac)}",
        ])

        if 'brac_label' in patient_df.columns:
            summary_lines.append("")
            summary_lines.append("BRAC Label Distribution (patients with labs):")
            label_names = {
                0: 'DLBCL_GCB', 1: 'DLBCL_ABC', 2: 'FL', 3: 'MCL', 4: 'BL',
                5: 'MZL', 6: 'CLL_SLL', 7: 'LPL', 8: 'PMBL'
            }
            brac_subset = patient_df[patient_df['is_brac_compatible'] == True]
            if len(brac_subset) > 0:
                label_counts = brac_subset['brac_label'].value_counts().sort_index()
                for label, count in label_counts.items():
                    name = label_names.get(int(label), 'Unknown')
                    summary_lines.append(f"  {int(label)} ({name}): {count}")

    # Clinical phase distribution
    if 'clinical_phase' in df.columns:
        summary_lines.extend([
            "",
            "CLINICAL PHASE DISTRIBUTION",
            "-" * 40,
        ])
        phase_counts = df['clinical_phase'].value_counts()
        for phase in ['BASELINE', 'ON_TREATMENT', 'RECOVERY', 'STABLE', 'RELAPSE', 'UNKNOWN']:
            if phase in phase_counts.index:
                count = phase_counts[phase]
                pct = count / len(df) * 100
                summary_lines.append(f"  {phase}: {count} ({pct:.1f}%)")

    # Over/under range flags
    over_range_cols = [c for c in df.columns if c.endswith('_over_range')]
    under_range_cols = [c for c in df.columns if c.endswith('_under_range')]
    if over_range_cols or under_range_cols:
        summary_lines.extend([
            "",
            "INSTRUMENT FLAGS (over/under range)",
            "-" * 40,
        ])
        for col in over_range_cols:
            count = df[col].sum() if col in df.columns else 0
            if count > 0:
                summary_lines.append(f"  {col}: {int(count)} records")
        for col in under_range_cols:
            count = df[col].sum() if col in df.columns else 0
            if count > 0:
                summary_lines.append(f"  {col}: {int(count)} records")

    # Missing data
    summary_lines.extend([
        "",
        "MISSING DATA (investigation-level)",
        "-" * 40,
    ])
    for col in NUMERIC_COLUMNS[:10]:
        if col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            summary_lines.append(f"  {col}: {missing_pct:.1f}%")

    return "\n".join(summary_lines)


def main():
    print("=" * 60)
    print("LABORATORY DATA PREPROCESSING")
    print("=" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Clean structure
    df = clean_structure(df)

    # Step 3: Unify MXD with Mono/Eos/Baso components
    df = unify_mxd_columns(df)

    # Step 4: Clean instrument flags (X suffix, -----, >>>>>, etc.)
    df = clean_instrument_flags(df)

    # Step 5: Convert types
    df = convert_types(df)

    # Step 4: Handle outliers
    df = handle_outliers(df)

    # Step 6: Temporal features
    df = add_temporal_features(df)

    # Step 7: Clinical phase tagging
    df = add_clinical_phase_tagging(df)

    # Step 8: Derived features
    df = create_derived_features(df)

    # Step 7: Patient-level aggregation
    patient_df = aggregate_patient_features(df)

    # Step 8: Link to pathology
    patient_df = link_pathology(patient_df)

    # Step 9: Quality scores
    patient_df = calculate_quality_scores(patient_df)

    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Investigation-level
    output_inv = OUTPUT_DIR / "laboratory_cleaned.csv"
    df.to_csv(output_inv, index=False)
    print(f"  Saved investigation-level: {output_inv}")
    print(f"    {len(df)} rows × {len(df.columns)} columns")

    # Patient-level
    output_patient = OUTPUT_DIR / "laboratory_patient_features.csv"
    patient_df.to_csv(output_patient, index=False)
    print(f"  Saved patient-level: {output_patient}")
    print(f"    {len(patient_df)} rows × {len(patient_df.columns)} columns")

    # BRAC-compatible subset
    if 'is_brac_compatible' in patient_df.columns:
        brac_df = patient_df[patient_df['is_brac_compatible'] == True].copy()
        output_brac = OUTPUT_DIR / "laboratory_brac_compatible.csv"
        brac_df.to_csv(output_brac, index=False)
        print(f"  Saved BRAC-compatible: {output_brac}")
        print(f"    {len(brac_df)} rows × {len(brac_df.columns)} columns")

    # Summary
    summary = generate_summary(df, patient_df)
    summary_path = OUTPUT_DIR / "preprocessing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Saved summary: {summary_path}")

    print("\n" + summary)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
