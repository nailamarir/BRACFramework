"""Create optimized training-ready dataset by removing redundant columns."""

import pandas as pd
from pathlib import Path

# Paths
INPUT_PATH = Path(__file__).parent.parent / "data/pathology/pathology_cleaned_full.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data/pathology/pathology_training_ready.csv"

# Columns to KEEP (60 columns organized by category)
COLUMNS_TO_KEEP = [
    # === ID & TIME (9 columns) ===
    "patient_id",
    "date",
    "investigation_order",
    "total_investigations",
    "is_first_visit",
    "days_since_first_visit",
    "disease_duration_days",
    "days_since_prev_visit",
    "visit_year",

    # === TEXT (2 columns) ===
    "microscopic_exam",
    "diagnosis",

    # === LABELS & CLASSIFICATION (6 columns) ===
    "disease_type_final",
    "brac_label",
    "is_brac_compatible",
    "who_icd_o",
    "who_name",
    "who_category",

    # === QUALITY & CONFIDENCE (7 columns) ===
    "has_ihc",
    "has_complete_hans",
    "flag_transformation",
    "flag_relapse",
    "label_confidence",
    "label_ambiguity",
    "quality_completeness",
    "quality_evidence_strength",

    # === IHC NUMERIC (17 columns) ===
    "ihc_cd20_numeric",
    "ihc_cd10_numeric",
    "ihc_bcl6_numeric",
    "ihc_mum1_numeric",
    "ihc_ki67_numeric",
    "ihc_cd5_numeric",
    "ihc_cd23_numeric",
    "ihc_cd3",           # Keep categorical for non-numeric markers
    "ihc_cd30",
    "ihc_bcl2",
    "ihc_cyclin_d1",
    "ihc_sox11",
    "ihc_cd138",
    "ihc_pax5",
    "ihc_cd79a",
    "ihc_tdt",
    "ki67_percentage",
    "ihc_vector_completeness",

    # === MORPHOLOGY (16 columns) ===
    "morph_large_cells",
    "morph_small_cells",
    "morph_atypical",
    "morph_necrosis",
    "morph_mitoses",
    "morph_fibrosis",
    "morph_effacement",
    "morph_nodular_pattern",
    "morph_diffuse_pattern",
    "morph_starry_sky",
    "morph_tingible_body",
    "morph_plasmacytoid",
    "morph_centroblast",
    "morph_centrocyte",
    "morph_mantle_zone",
    "morph_feature_count",

    # === OTHER (3 columns) ===
    "specimen_type",
]


def main():
    print("=" * 60)
    print("CREATING TRAINING-READY DATASET")
    print("=" * 60)

    # Load full dataset
    print(f"\nLoading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Original: {len(df)} rows × {len(df.columns)} columns")

    # Get available columns (some might not exist)
    available_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    missing_cols = [c for c in COLUMNS_TO_KEEP if c not in df.columns]

    if missing_cols:
        print(f"\n  Warning: {len(missing_cols)} columns not found:")
        for col in missing_cols:
            print(f"    - {col}")

    # Select columns
    df_optimized = df[available_cols].copy()

    # Report removed columns
    removed_cols = [c for c in df.columns if c not in available_cols]
    print(f"\n  Removed {len(removed_cols)} redundant/unnecessary columns:")
    for col in removed_cols:
        print(f"    - {col}")

    # Save optimized dataset
    print(f"\nSaving: {OUTPUT_PATH}")
    df_optimized.to_csv(OUTPUT_PATH, index=False)
    print(f"  Final: {len(df_optimized)} rows × {len(df_optimized.columns)} columns")

    # Print column summary by category
    print("\n" + "=" * 60)
    print("COLUMN SUMMARY")
    print("=" * 60)

    categories = {
        "ID & Time": ["patient_id", "date", "investigation_order", "total_investigations",
                      "is_first_visit", "days_since_first_visit", "disease_duration_days",
                      "days_since_prev_visit", "visit_year"],
        "Text": ["microscopic_exam", "diagnosis"],
        "Labels": ["disease_type_final", "brac_label", "is_brac_compatible",
                   "who_icd_o", "who_name", "who_category"],
        "Quality": ["has_ihc", "has_complete_hans", "flag_transformation", "flag_relapse",
                    "label_confidence", "label_ambiguity", "quality_completeness",
                    "quality_evidence_strength"],
        "IHC": [c for c in available_cols if c.startswith("ihc_") or c == "ki67_percentage"],
        "Morphology": [c for c in available_cols if c.startswith("morph_")],
        "Other": ["specimen_type"],
    }

    for cat_name, cat_cols in categories.items():
        actual = [c for c in cat_cols if c in df_optimized.columns]
        print(f"\n{cat_name}: {len(actual)} columns")
        for col in actual:
            print(f"  • {col}")

    # Print data statistics
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)

    brac_compat = df_optimized["is_brac_compatible"].sum()
    print(f"\nTotal cases: {len(df_optimized)}")
    print(f"BRAC-compatible: {brac_compat}")

    if "brac_label" in df_optimized.columns:
        print("\nBRAC Label Distribution:")
        label_counts = df_optimized[df_optimized["is_brac_compatible"]]["brac_label"].value_counts().sort_index()
        label_names = {
            0: "DLBCL_GCB", 1: "DLBCL_ABC", 2: "FL", 3: "MCL", 4: "BL",
            5: "MZL", 6: "CLL_SLL", 7: "LPL", 8: "PMBL"
        }
        for label, count in label_counts.items():
            name = label_names.get(int(label), "Unknown")
            print(f"  {int(label)} ({name}): {count}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
