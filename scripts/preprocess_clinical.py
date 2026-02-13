"""Preprocess clinical demographic data for BRAC framework."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === PATHS ===
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "data/clinical/Clinical.xlsx"
PATHOLOGY_PATH = BASE_DIR / "data/pathology/pathology_cleaned_full.csv"
OUTPUT_DIR = BASE_DIR / "data/clinical"

# === REFERENCE DATE ===
REFERENCE_DATE = datetime(2026, 2, 13)


def load_data():
    """Load clinical Excel file."""
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    df = pd.read_excel(INPUT_PATH)
    print(f"Loaded: {len(df)} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    return df


def clean_text_fields(df):
    """Clean and normalize text fields in clinical data.

    Handles:
    - Missing Country (row 8: blank → "Egypt")
    - Trailing pipes in village ("بهوت|" → "بهوت")
    - Empty strings → NaN
    - Whitespace normalization
    """
    print("\n" + "=" * 60)
    print("STEP 2: TEXT CLEANING")
    print("=" * 60)

    stats = {'country_filled': 0, 'trailing_pipes': 0, 'empty_to_nan': 0}

    # Handle missing Country (fill with "مصر|Egypt")
    country_missing = df['Country'].isna() | (df['Country'].astype(str).str.strip() == '')
    if country_missing.any():
        stats['country_filled'] = country_missing.sum()
        df.loc[country_missing, 'Country'] = 'مصر|Egypt'
        print(f"  Filled {stats['country_filled']} missing Country values with 'مصر|Egypt'")

    # Clean trailing pipes in village field (e.g., "بهوت|" → "بهوت")
    if 'village' in df.columns:
        # Check for trailing pipe pattern
        has_trailing_pipe = df['village'].astype(str).str.match(r'^[^|]+\|$', na=False)
        if has_trailing_pipe.any():
            stats['trailing_pipes'] = has_trailing_pipe.sum()
            df.loc[has_trailing_pipe, 'village'] = df.loc[has_trailing_pipe, 'village'].str.rstrip('|')
            print(f"  Removed {stats['trailing_pipes']} trailing pipes from village names")

    # Convert empty strings to NaN for all text columns
    text_cols = ['Country', 'Governorate', 'city', 'village']
    for col in text_cols:
        if col in df.columns:
            empty_mask = df[col].astype(str).str.strip() == ''
            if empty_mask.any():
                stats['empty_to_nan'] += empty_mask.sum()
                df.loc[empty_mask, col] = np.nan

    if stats['empty_to_nan'] > 0:
        print(f"  Converted {stats['empty_to_nan']} empty strings to NaN")

    # Strip whitespace from all text fields
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)

    print(f"  Normalized whitespace in text fields")

    return df


def parse_bilingual_columns(df):
    """Extract Arabic and English components from bilingual columns."""
    print("\n" + "=" * 60)
    print("STEP 3: PARSING BILINGUAL COLUMNS")
    print("=" * 60)

    bilingual_cols = ['Country', 'Governorate', 'city', 'village']

    stats = {'split': 0, 'arabic_only': 0}

    for col in bilingual_cols:
        if col in df.columns:
            # Split on | to get Arabic and English
            def extract_arabic(x):
                if pd.isna(x):
                    return np.nan
                s = str(x).strip()
                if '|' in s:
                    return s.split('|')[0].strip()
                return s.strip() if s else np.nan

            def extract_english(x):
                if pd.isna(x):
                    return np.nan
                s = str(x).strip()
                if '|' in s:
                    parts = s.split('|')
                    if len(parts) > 1 and parts[1].strip():
                        return parts[1].strip()
                return np.nan

            df[f'{col}_ar'] = df[col].apply(extract_arabic)
            df[f'{col}_en'] = df[col].apply(extract_english)

            # Count how many had both vs Arabic only
            has_english = df[f'{col}_en'].notna().sum()
            has_arabic = df[f'{col}_ar'].notna().sum()
            print(f"  {col}: {has_english} with English, {has_arabic - has_english} Arabic-only")

    # Create clean English governorate for easier analysis
    df['governorate_clean'] = df['Governorate_en'].str.replace(' ', '_').str.upper()

    # Print governorate distribution
    print("\n  Governorate distribution:")
    gov_counts = df['Governorate_en'].value_counts()
    for gov, count in gov_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"    {gov}: {count} ({pct:.1f}%)")

    return df


def parse_birth_dates_and_compute_age(df):
    """Parse birth dates and compute age at diagnosis.

    Steps:
    - Parse birth_date (DD/MM/YYYY format)
    - Join with pathology date to compute age_at_dx = pathology_date - birth_date
    - Validate: age ∈ [0, 110] (flag outliers)
    - Bin into age groups including elderly bins: 65-74 / 75-84 / 85+
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATE PARSING & AGE COMPUTATION")
    print("=" * 60)

    # Parse birth_date (format: DD/MM/YYYY)
    df['birth_date_parsed'] = pd.to_datetime(df['birth_date'], format='%d/%m/%Y', errors='coerce')

    # Handle alternative format (D/M/YYYY)
    mask = df['birth_date_parsed'].isna()
    df.loc[mask, 'birth_date_parsed'] = pd.to_datetime(
        df.loc[mask, 'birth_date'], format='mixed', dayfirst=True, errors='coerce'
    )

    parsed_count = df['birth_date_parsed'].notna().sum()
    print(f"  Parsed birth dates: {parsed_count}/{len(df)}")

    # Load pathology data to get diagnosis dates
    print("\n  Loading pathology dates for age_at_dx computation...")
    path_df = pd.read_csv(PATHOLOGY_PATH)
    path_df['date'] = pd.to_datetime(path_df['date'])

    # Get first diagnosis date per patient
    first_diag = path_df.groupby('patient_id')['date'].min().reset_index()
    first_diag.columns = ['patient_id', 'first_diagnosis_date']

    # Merge diagnosis dates
    df = df.merge(first_diag, on='patient_id', how='left')

    # Compute age_at_dx = pathology_date - birth_date
    df['age_at_dx'] = (df['first_diagnosis_date'] - df['birth_date_parsed']).dt.days / 365.25

    # For patients without pathology, compute age at reference date
    df['age_at_reference'] = (REFERENCE_DATE - df['birth_date_parsed']).dt.days / 365.25

    # Use age_at_dx where available, otherwise age_at_reference
    df['age_primary'] = df['age_at_dx'].fillna(df['age_at_reference'])

    # Validate: age ∈ [0, 110]
    invalid_age_low = df['age_primary'] < 0
    invalid_age_high = df['age_primary'] > 110
    invalid_count = (invalid_age_low | invalid_age_high).sum()

    if invalid_count > 0:
        print(f"\n  WARNING: {invalid_count} patients with invalid ages:")
        if invalid_age_low.any():
            print(f"    Age < 0: {invalid_age_low.sum()} (negative - data error)")
        if invalid_age_high.any():
            print(f"    Age > 110: {invalid_age_high.sum()} (exceeds max)")
        # Flag but don't remove
        df['flag_invalid_age'] = (invalid_age_low | invalid_age_high).astype(int)
    else:
        df['flag_invalid_age'] = 0

    # Age statistics (for valid ages only)
    valid_ages = df.loc[df['flag_invalid_age'] == 0, 'age_primary']
    print(f"\n  Age statistics (valid only):")
    print(f"    Min: {valid_ages.min():.1f} years")
    print(f"    Max: {valid_ages.max():.1f} years")
    print(f"    Mean: {valid_ages.mean():.1f} years")
    print(f"    Median: {valid_ages.median():.1f} years")

    # Patients with age_at_dx vs reference
    has_dx_age = df['age_at_dx'].notna().sum()
    print(f"\n  Patients with age_at_dx: {has_dx_age}")
    print(f"  Patients using age_at_reference: {len(df) - has_dx_age}")

    # Extract birth year
    df['birth_year'] = df['birth_date_parsed'].dt.year

    return df


def create_age_groups(df):
    """Create categorical age groups with elderly-specific bins."""
    print("\n" + "=" * 60)
    print("STEP 5: CREATING AGE GROUPS")
    print("=" * 60)

    # Use age_primary (age_at_dx if available, else age_at_reference)
    age_col = 'age_primary'

    # Standard clinical age groups
    bins = [0, 18, 40, 60, 80, 150]
    labels = ['PEDIATRIC', 'YOUNG_ADULT', 'MIDDLE_AGE', 'SENIOR', 'ELDERLY']
    df['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels)

    # Fine-grained age groups (clinical standard)
    bins_fine = [0, 18, 30, 40, 50, 60, 70, 80, 150]
    labels_fine = ['0-18', '18-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
    df['age_group_fine'] = pd.cut(df[age_col], bins=bins_fine, labels=labels_fine)

    # Elderly-specific bins (65-74 / 75-84 / 85+) for geriatric analysis
    bins_elderly = [0, 18, 45, 55, 65, 75, 85, 150]
    labels_elderly = ['<18', '18-44', '45-54', '55-64', '65-74', '75-84', '85+']
    df['age_group_elderly'] = pd.cut(df[age_col], bins=bins_elderly, labels=labels_elderly)

    # Binary flags
    df['is_adult'] = (df[age_col] >= 18).astype(int)
    df['is_elderly'] = (df[age_col] >= 65).astype(int)
    df['is_very_elderly'] = (df[age_col] >= 75).astype(int)

    # Print standard age group distribution
    print("  Age group distribution:")
    for group, count in df['age_group'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"    {group}: {count} ({pct:.1f}%)")

    # Print elderly-specific distribution
    print("\n  Elderly-specific distribution (65+):")
    elderly_df = df[df['is_elderly'] == 1]
    if len(elderly_df) > 0:
        for group in ['65-74', '75-84', '85+']:
            count = (df['age_group_elderly'] == group).sum()
            pct = count / len(df) * 100
            print(f"    {group}: {count} ({pct:.1f}%)")
    else:
        print("    No elderly patients")

    return df


def create_location_features(df):
    """Create location-based features with geographic encoding.

    Uses Governorate as primary geo feature (city too sparse).
    Encodes: El Dakahlia, Gharbia, Kafr El-Sheikh, Damietta, Cairo, Sharqia, etc.
    Derives urban_vs_rural flag (Mansoura/Cairo = urban; village present = rural).
    """
    print("\n" + "=" * 60)
    print("STEP 6: GEOGRAPHIC ENCODING")
    print("=" * 60)

    # Primary governorates for encoding (based on data distribution)
    # El Dakahlia, Gharbia, Kafr El-Sheikh, Damietta, Cairo, Sharqia, Assuit, Matrooh, Swees
    PRIMARY_GOVS = [
        'EL_DAKAHLIA', 'GHARBIA', 'KAFR_EL-SHEIKH', 'DAMIETTA',
        'CAIRO', 'SHARQIA', 'ASSUIT', 'MATROOH', 'SWEES'
    ]

    # Governorate one-hot encoding (primary governorates only, rest grouped as OTHER)
    print("  Using Governorate as primary geo feature (city too sparse)")

    # Create governorate one-hot columns
    gov_dummies = pd.get_dummies(df['governorate_clean'], prefix='gov', dtype=int)
    print(f"  Created {len(gov_dummies.columns)} governorate one-hot columns")

    # List encoded governorates
    encoded_govs = [g.replace('gov_', '') for g in gov_dummies.columns]
    print(f"  Encoded: {', '.join(encoded_govs[:5])}...")

    # Main regions (simplified)
    delta_govs = ['EL_DAKAHLIA', 'GHARBIA', 'KAFR_EL-SHEIKH', 'DAMIETTA', 'SHARQIA', 'KALUBIA']
    df['region'] = df['governorate_clean'].apply(
        lambda x: 'DELTA' if x in delta_govs else ('CAIRO_GIZA' if x in ['CAIRO', 'GIZA'] else 'OTHER')
    )

    # Is from main catchment area (El Dakahlia)
    df['is_main_catchment'] = (df['governorate_clean'] == 'EL_DAKAHLIA').astype(int)

    # Urban vs Rural classification
    # Urban: Mansoura, Cairo, Giza, or other major cities
    # Rural: village field is present
    print("\n  Deriving urban_vs_rural flag...")

    # Define urban cities (normalize to uppercase for matching)
    urban_cities = ['MANSOURA', 'CAIRO', 'GIZA', 'ALEXANDRIA', 'TANTA', 'ZAGAZIG', 'DAMIETTA']

    # Check city_en for urban indicators
    df['city_upper'] = df['city_en'].fillna('').str.upper().str.strip()

    # Urban: city is in urban_cities OR governorate is Cairo/Giza
    is_urban_city = df['city_upper'].isin(urban_cities)
    is_urban_gov = df['governorate_clean'].isin(['CAIRO', 'GIZA'])

    # Rural: has village data
    has_village = df['village'].notna()

    # Classification logic:
    # - If has village → Rural
    # - If urban city or urban governorate → Urban
    # - Otherwise → Unknown (default to Urban if in major city, Rural if in small town)
    df['is_rural'] = has_village.astype(int)
    df['is_urban'] = ((is_urban_city | is_urban_gov) & ~has_village).astype(int)

    # Create combined urban_vs_rural flag
    # 0 = Unknown, 1 = Urban, 2 = Rural
    df['urban_rural'] = 0  # Unknown
    df.loc[df['is_urban'] == 1, 'urban_rural'] = 1  # Urban
    df.loc[df['is_rural'] == 1, 'urban_rural'] = 2  # Rural

    # Clean up temp column
    df = df.drop(columns=['city_upper'], errors='ignore')

    # Print distribution
    print("\n  Region distribution:")
    for region, count in df['region'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {region}: {count} ({pct:.1f}%)")

    print(f"\n  Urban/Rural classification:")
    print(f"    Rural (has village): {df['is_rural'].sum()} ({df['is_rural'].mean()*100:.1f}%)")
    print(f"    Urban (major city): {df['is_urban'].sum()} ({df['is_urban'].mean()*100:.1f}%)")
    unknown = ((df['is_rural'] == 0) & (df['is_urban'] == 0)).sum()
    print(f"    Unclassified: {unknown} ({unknown/len(df)*100:.1f}%)")
    print(f"\n  Main catchment (Dakahlia): {df['is_main_catchment'].sum()} ({df['is_main_catchment'].mean()*100:.1f}%)")

    # Merge one-hot columns
    df = pd.concat([df, gov_dummies], axis=1)

    return df


def link_to_pathology(df):
    """Link clinical data to pathology for BRAC labels."""
    print("\n" + "=" * 60)
    print("STEP 7: LINKING TO PATHOLOGY DATA")
    print("=" * 60)

    # Load pathology data
    path_df = pd.read_csv(PATHOLOGY_PATH)

    # Get first investigation per patient for labels
    path_first = path_df.sort_values(['patient_id', 'date']).groupby('patient_id').first().reset_index()

    # Select relevant columns
    path_cols = ['patient_id', 'disease_type_final', 'brac_label', 'is_brac_compatible',
                 'who_category', 'has_ihc', 'has_complete_hans']
    path_subset = path_first[path_cols]

    # Merge
    df = df.merge(path_subset, on='patient_id', how='left')

    # Fill missing values (avoid deprecation warning)
    df.loc[df['is_brac_compatible'].isna(), 'is_brac_compatible'] = False
    df.loc[df['has_ihc'].isna(), 'has_ihc'] = False
    df.loc[df['has_complete_hans'].isna(), 'has_complete_hans'] = False
    df['is_brac_compatible'] = df['is_brac_compatible'].astype(bool)
    df['has_ihc'] = df['has_ihc'].astype(bool)
    df['has_complete_hans'] = df['has_complete_hans'].astype(bool)

    print(f"  Patients with pathology data: {df['disease_type_final'].notna().sum()}")
    print(f"  BRAC-compatible: {df['is_brac_compatible'].sum()}")

    if df['is_brac_compatible'].sum() > 0:
        print("\n  BRAC label distribution (compatible cases):")
        label_names = {
            0: "DLBCL_GCB", 1: "DLBCL_ABC", 2: "FL", 3: "MCL", 4: "BL",
            5: "MZL", 6: "CLL_SLL", 7: "LPL", 8: "PMBL"
        }
        brac_counts = df[df['is_brac_compatible']]['brac_label'].value_counts().sort_index()
        for label, count in brac_counts.items():
            name = label_names.get(int(label), "Unknown")
            print(f"    {int(label)} ({name}): {count}")

    return df


def add_epidemiological_priors(df):
    """Add epidemiological prior weights based on age and regional factors.

    These are SOFT priors (weights), NOT hard rules. They encode:
    - NHL subtype incidence by age group (DLBCL peaks 60-80, FL peaks 50-70, CLL peaks >65)
    - Regional factors: Nile Delta → higher HCV exposure → splenic MZL risk

    The priors can be used as auxiliary features or Bayesian priors in downstream models.
    """
    print("\n" + "=" * 60)
    print("STEP 8: EPIDEMIOLOGICAL PRIORS (OPTIONAL)")
    print("=" * 60)

    # Use age_primary for prior computation
    age_col = 'age_primary'

    # =========================================
    # A. Age-based NHL subtype priors
    # =========================================
    # Based on epidemiological data for NHL subtypes
    # Values are relative incidence weights (normalized 0-1)

    print("  Computing age-based NHL subtype priors...")

    # DLBCL: Most common NHL, peaks 60-80 years
    # Higher incidence in elderly, lower in young adults
    def dlbcl_prior(age):
        if pd.isna(age):
            return 0.5  # neutral prior
        if age < 40:
            return 0.3
        elif age < 60:
            return 0.6
        elif age < 80:
            return 0.9  # peak incidence
        else:
            return 0.7

    # FL: Second most common, peaks 50-70 years
    def fl_prior(age):
        if pd.isna(age):
            return 0.5
        if age < 40:
            return 0.2
        elif age < 50:
            return 0.5
        elif age < 70:
            return 0.8  # peak incidence
        else:
            return 0.5

    # CLL/SLL: Primarily disease of elderly, peaks >65
    def cll_prior(age):
        if pd.isna(age):
            return 0.5
        if age < 50:
            return 0.1  # rare in young
        elif age < 65:
            return 0.4
        elif age < 80:
            return 0.9  # peak incidence
        else:
            return 0.8

    # MCL: Older adults, peak 60-70
    def mcl_prior(age):
        if pd.isna(age):
            return 0.5
        if age < 50:
            return 0.2
        elif age < 60:
            return 0.5
        elif age < 75:
            return 0.8  # peak incidence
        else:
            return 0.6

    # BL: Bimodal - endemic (children) and sporadic (young adults)
    def bl_prior(age):
        if pd.isna(age):
            return 0.5
        if age < 20:
            return 0.7  # endemic BL peak
        elif age < 40:
            return 0.5  # sporadic BL
        else:
            return 0.2  # rare in elderly

    # MZL (including splenic MZL): Middle-aged to elderly
    def mzl_prior(age):
        if pd.isna(age):
            return 0.5
        if age < 40:
            return 0.2
        elif age < 60:
            return 0.6
        else:
            return 0.7

    # Apply age priors
    df['prior_dlbcl_age'] = df[age_col].apply(dlbcl_prior)
    df['prior_fl_age'] = df[age_col].apply(fl_prior)
    df['prior_cll_age'] = df[age_col].apply(cll_prior)
    df['prior_mcl_age'] = df[age_col].apply(mcl_prior)
    df['prior_bl_age'] = df[age_col].apply(bl_prior)
    df['prior_mzl_age'] = df[age_col].apply(mzl_prior)

    # =========================================
    # B. Regional factors
    # =========================================
    # Nile Delta region has higher HCV prevalence
    # HCV is associated with splenic MZL and some B-cell lymphomas

    print("  Computing regional epidemiological factors...")

    # HCV exposure risk proxy (based on region)
    # Higher in Nile Delta (historical endemic region)
    def hcv_risk_proxy(region):
        if region == 'DELTA':
            return 0.7  # Higher HCV prevalence historically
        elif region == 'CAIRO_GIZA':
            return 0.5  # Urban, moderate
        else:
            return 0.4  # Other regions

    df['prior_hcv_exposure'] = df['region'].apply(hcv_risk_proxy)

    # Splenic MZL prior: elevated in HCV-endemic regions
    df['prior_smzl_regional'] = df['prior_hcv_exposure'] * df['prior_mzl_age']

    # =========================================
    # C. Combined priors (age + region)
    # =========================================
    # These can be used as soft weights in Bayesian inference

    # For each BRAC label, compute a combined prior
    # BRAC labels: 0=DLBCL_GCB, 1=DLBCL_ABC, 2=FL, 3=MCL, 4=BL, 5=MZL, 6=CLL_SLL, 7=LPL, 8=PMBL

    # DLBCL (both subtypes) - primarily age-driven
    df['prior_brac_0'] = df['prior_dlbcl_age']  # DLBCL_GCB
    df['prior_brac_1'] = df['prior_dlbcl_age']  # DLBCL_ABC

    # FL - age-driven
    df['prior_brac_2'] = df['prior_fl_age']

    # MCL - age-driven
    df['prior_brac_3'] = df['prior_mcl_age']

    # BL - age-driven (bimodal)
    df['prior_brac_4'] = df['prior_bl_age']

    # MZL - age + regional (HCV)
    df['prior_brac_5'] = (df['prior_mzl_age'] * 0.6 + df['prior_smzl_regional'] * 0.4)

    # CLL/SLL - primarily age-driven
    df['prior_brac_6'] = df['prior_cll_age']

    # LPL - similar to CLL age pattern
    df['prior_brac_7'] = df['prior_cll_age'] * 0.8

    # PMBL - young adults (20-40)
    def pmbl_prior(age):
        if pd.isna(age):
            return 0.5
        if age < 20:
            return 0.3
        elif age < 40:
            return 0.8  # peak incidence
        elif age < 50:
            return 0.4
        else:
            return 0.1  # rare in elderly
    df['prior_brac_8'] = df[age_col].apply(pmbl_prior)

    # Print summary statistics
    prior_cols = [c for c in df.columns if c.startswith('prior_')]
    print(f"\n  Created {len(prior_cols)} epidemiological prior columns")

    # Print mean priors by region
    print("\n  Mean regional priors:")
    for region in ['DELTA', 'CAIRO_GIZA', 'OTHER']:
        region_df = df[df['region'] == region]
        if len(region_df) > 0:
            hcv = region_df['prior_hcv_exposure'].mean()
            print(f"    {region}: HCV exposure proxy = {hcv:.2f}")

    # Print mean priors by age group
    print("\n  Mean age-based priors by age group:")
    for group in ['PEDIATRIC', 'YOUNG_ADULT', 'MIDDLE_AGE', 'SENIOR', 'ELDERLY']:
        group_df = df[df['age_group'] == group]
        if len(group_df) > 0:
            dlbcl = group_df['prior_dlbcl_age'].mean()
            cll = group_df['prior_cll_age'].mean()
            bl = group_df['prior_bl_age'].mean()
            print(f"    {group}: DLBCL={dlbcl:.2f}, CLL={cll:.2f}, BL={bl:.2f}")

    return df


def create_quality_scores(df):
    """Create data quality scores for clinical demographics.

    For clinical demographic data (structured fields):
    - Q (quality) = 1.0: Structured data is always complete when present
    - C (coverage) = 1 - missing_frac: Based on missing fraction of key fields
    - S (self-consistency) = 1.0: No internal conflict possible in demographics

    These are simpler than laboratory Q/C/S because demographic data doesn't have
    the same measurement uncertainty or internal consistency checks.
    """
    print("\n" + "=" * 60)
    print("STEP 9: CREATING QUALITY SCORES")
    print("=" * 60)

    # =========================================
    # Q = 1.0 (structured data - always complete when present)
    # =========================================
    # Clinical demographics are structured fields (dates, locations, IDs)
    # When present, they have no measurement uncertainty
    df['quality_Q'] = 1.0

    # =========================================
    # C = 1 - missing_frac (coverage score)
    # =========================================
    # Key clinical features for coverage calculation
    key_cols = [
        'birth_date_parsed',  # Required for age
        'Governorate',        # Primary location
        'city',               # Secondary location
        'age_at_dx',          # Age at diagnosis (if available)
    ]

    # Calculate missing fraction per patient
    missing_frac = df[key_cols].isna().sum(axis=1) / len(key_cols)
    df['coverage_C'] = 1 - missing_frac

    # =========================================
    # S = 1.0 (no internal conflict in demographics)
    # =========================================
    # Demographic data has no internal consistency checks
    # (unlike lab values which can have conflicting measurements)
    df['consistency_S'] = 1.0

    # =========================================
    # Legacy columns (for backward compatibility)
    # =========================================
    # Location completeness score
    location_cols = ['Country', 'Governorate', 'city', 'village']
    df['location_completeness'] = df[location_cols].notna().sum(axis=1) / len(location_cols)

    # Overall data completeness (alias for coverage_C)
    df['data_completeness'] = df['coverage_C']

    # =========================================
    # Nile Delta binary flag
    # =========================================
    # Important epidemiological feature (HCV endemic region)
    df['nile_delta'] = (df['region'] == 'DELTA').astype(int)

    # Print summary
    print(f"  Quality scores (Q, C, S) for clinical demographics:")
    print(f"    Q (quality): {df['quality_Q'].mean():.2f} (structured data = always 1.0)")
    print(f"    C (coverage): {df['coverage_C'].mean():.2f} (1 - missing_frac)")
    print(f"    S (consistency): {df['consistency_S'].mean():.2f} (no conflict possible = always 1.0)")
    print(f"\n  Legacy scores:")
    print(f"    Mean location completeness: {df['location_completeness'].mean():.2f}")
    print(f"    Mean data completeness: {df['data_completeness'].mean():.2f}")
    print(f"\n  Nile Delta flag: {df['nile_delta'].sum()} patients ({df['nile_delta'].mean()*100:.1f}%)")

    return df


def select_final_columns(df):
    """Select and order final columns for output."""
    print("\n" + "=" * 60)
    print("STEP 10: SELECTING FINAL COLUMNS")
    print("=" * 60)

    # Define column groups
    id_cols = ['patient_id']

    age_cols = [
        'birth_date_parsed', 'birth_year',
        'age_at_reference', 'age_at_dx', 'age_primary',
        'age_group', 'age_group_fine', 'age_group_elderly',
        'is_adult', 'is_elderly', 'is_very_elderly',
        'flag_invalid_age'
    ]

    location_cols = [
        'Governorate_en', 'city_en', 'village_en',
        'governorate_clean', 'region', 'nile_delta', 'is_main_catchment',
        'is_rural', 'is_urban', 'urban_rural'
    ]

    # Get governorate one-hot columns
    gov_onehot_cols = [c for c in df.columns if c.startswith('gov_')]

    pathology_cols = [
        'disease_type_final', 'brac_label', 'is_brac_compatible',
        'who_category', 'has_ihc', 'has_complete_hans',
        'first_diagnosis_date'
    ]

    # Epidemiological prior columns
    prior_cols = [c for c in df.columns if c.startswith('prior_')]

    quality_cols = ['quality_Q', 'coverage_C', 'consistency_S', 'location_completeness', 'data_completeness']

    # Combine all columns
    final_cols = id_cols + age_cols + location_cols + gov_onehot_cols + pathology_cols + prior_cols + quality_cols

    # Filter to existing columns
    final_cols = [c for c in final_cols if c in df.columns]

    df_final = df[final_cols].copy()

    print(f"  Final columns: {len(final_cols)}")
    print(f"\n  Column groups:")
    print(f"    ID: {len(id_cols)}")
    print(f"    Age: {len([c for c in age_cols if c in df.columns])}")
    print(f"    Location: {len([c for c in location_cols if c in df.columns])}")
    print(f"    Governorate one-hot: {len(gov_onehot_cols)}")
    print(f"    Pathology link: {len([c for c in pathology_cols if c in df.columns])}")
    print(f"    Epidemiological priors: {len(prior_cols)}")
    print(f"    Quality: {len([c for c in quality_cols if c in df.columns])}")

    return df_final


def save_outputs(df):
    """Save processed clinical data."""
    print("\n" + "=" * 60)
    print("STEP 11: SAVING OUTPUTS")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    output_path = OUTPUT_DIR / "clinical_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"    {len(df)} rows × {len(df.columns)} columns")

    # Save BRAC-compatible subset
    brac_df = df[df['is_brac_compatible'] == True].copy()
    brac_path = OUTPUT_DIR / "clinical_brac_compatible.csv"
    brac_df.to_csv(brac_path, index=False)
    print(f"\n  Saved: {brac_path}")
    print(f"    {len(brac_df)} rows × {len(brac_df.columns)} columns")

    return df


def generate_report(df):
    """Generate preprocessing summary report."""
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    # Use age_primary (age_at_dx if available, else age_at_reference)
    age_col = 'age_primary' if 'age_primary' in df.columns else 'age_at_reference'

    report_lines = [
        "CLINICAL DATA PREPROCESSING SUMMARY",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "INPUT",
        "-" * 40,
        f"Total patients: {len(df)}",
        "",
        "AGE STATISTICS",
        "-" * 40,
        f"Age range: {df[age_col].min():.1f} - {df[age_col].max():.1f} years",
        f"Mean age: {df[age_col].mean():.1f} years",
        f"Median age: {df[age_col].median():.1f} years",
        "",
        "AGE GROUP DISTRIBUTION",
        "-" * 40
    ]

    for group in ['PEDIATRIC', 'YOUNG_ADULT', 'MIDDLE_AGE', 'SENIOR', 'ELDERLY']:
        count = (df['age_group'] == group).sum()
        report_lines.append(f"  {group}: {count}")

    report_lines.extend([
        "",
        "LOCATION DISTRIBUTION",
        "-" * 40
    ])

    for gov, count in df['Governorate_en'].value_counts().head(10).items():
        report_lines.append(f"  {gov}: {count}")

    report_lines.extend([
        "",
        "PATHOLOGY LINKAGE",
        "-" * 40,
        f"Patients with pathology: {df['disease_type_final'].notna().sum()}",
        f"BRAC-compatible: {df['is_brac_compatible'].sum()}",
        "",
        "BRAC LABEL DISTRIBUTION",
        "-" * 40
    ])

    label_names = {
        0: "DLBCL_GCB", 1: "DLBCL_ABC", 2: "FL", 3: "MCL", 4: "BL",
        5: "MZL", 6: "CLL_SLL", 7: "LPL", 8: "PMBL"
    }

    brac_counts = df[df['is_brac_compatible']]['brac_label'].value_counts().sort_index()
    for label, count in brac_counts.items():
        name = label_names.get(int(label), "Unknown")
        report_lines.append(f"  {int(label)} ({name}): {count}")

    report_lines.extend([
        "",
        "OUTPUT COLUMNS",
        "-" * 40,
        f"Total columns: {len(df.columns)}"
    ])

    report = "\n".join(report_lines)

    # Save report
    report_path = OUTPUT_DIR / "preprocessing_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"  Saved report: {report_path}")
    print(report)


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("CLINICAL DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # Execute pipeline
    df = load_data()
    df = clean_text_fields(df)
    df = parse_bilingual_columns(df)
    df = parse_birth_dates_and_compute_age(df)  # Now includes pathology date join for age_at_dx
    df = create_age_groups(df)
    df = create_location_features(df)
    df = link_to_pathology(df)
    df = add_epidemiological_priors(df)  # NHL subtype priors based on age + regional factors
    df = create_quality_scores(df)
    df = select_final_columns(df)
    df = save_outputs(df)
    generate_report(df)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
