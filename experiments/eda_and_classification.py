"""
Exploratory Data Analysis and Multi-class NHL Classification
=============================================================

This script performs:
1. EDA - Feature distributions, correlations, class separability
2. Multi-class NHL classifier training with class imbalance handling
3. Model evaluation and feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "eda_classification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# BRAC label names
BRAC_LABELS = {
    0: "DLBCL_GCB",
    1: "DLBCL_ABC",
    2: "FL",
    3: "MCL",
    4: "BL",
    5: "MZL",
    6: "CLL_SLL",
    7: "LPL",
    8: "PMBL"
}


def load_data():
    """Load the merged feature matrix."""
    df = pd.read_csv(DATA_DIR / "merged" / "feature_matrix_brac.csv")
    print(f"Loaded data: {df.shape[0]} patients, {df.shape[1]} features")
    return df


def perform_eda(df):
    """Perform comprehensive exploratory data analysis."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # 1. Class distribution
    print("\n1. CLASS DISTRIBUTION")
    print("-" * 40)
    label_counts = df['brac_label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {int(label)}: {BRAC_LABELS[int(label)]:12s} - {count:3d} ({pct:5.1f}%)")

    # Plot class distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    bars = ax.bar([BRAC_LABELS[int(i)] for i in label_counts.index],
                   label_counts.values, color=colors)
    ax.set_xlabel("NHL Subtype", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("BRAC Label Distribution (174 patients)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    for bar, count in zip(bars, label_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: class_distribution.png")

    # 2. Feature groups
    print("\n2. FEATURE GROUPS")
    print("-" * 40)
    feature_groups = {
        'pathology': [c for c in df.columns if c.startswith('path_')],
        'laboratory': [c for c in df.columns if c.startswith('lab_')],
        'clinical': [c for c in df.columns if c.startswith('clin_')],
        'radiology': [c for c in df.columns if c.startswith('rad_')],
        'bone_marrow': [c for c in df.columns if c.startswith('bm_')],
        'priors': [c for c in df.columns if c.startswith('prior_')]
    }
    for group, cols in feature_groups.items():
        print(f"  {group:12s}: {len(cols):3d} features")

    # 3. Missing data analysis
    print("\n3. MISSING DATA ANALYSIS")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]
    print(f"  Features with >50% missing: {len(high_missing)}")
    print(f"  Features with no missing: {(missing_pct == 0).sum()}")

    # Missing data heatmap by feature group
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (group, cols) in enumerate(feature_groups.items()):
        if cols:
            missing_data = df[cols].isnull().mean() * 100
            ax = axes[idx]
            missing_data.plot(kind='bar', ax=ax, color='coral')
            ax.set_title(f"{group.upper()} Missing %", fontsize=11)
            ax.set_ylabel("% Missing")
            ax.set_xticklabels([])
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "missing_data_by_group.png", dpi=150)
    plt.close()
    print(f"  Saved: missing_data_by_group.png")

    # 4. Feature distributions for key features
    print("\n4. KEY FEATURE DISTRIBUTIONS")
    print("-" * 40)

    key_numeric_features = [
        'lab_wbc_mean', 'lab_hgb_mean', 'lab_plt_mean', 'lab_lymph_abs_mean',
        'lab_nlr_mean', 'clin_age_at_dx', 'path_quality_completeness'
    ]
    existing_features = [f for f in key_numeric_features if f in df.columns]

    if existing_features:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for idx, feat in enumerate(existing_features[:8]):
            ax = axes[idx]
            for label in sorted(df['brac_label'].dropna().unique()):
                subset = df[df['brac_label'] == label][feat].dropna()
                if len(subset) > 2:
                    ax.hist(subset, bins=15, alpha=0.5, label=BRAC_LABELS[int(label)])
            ax.set_title(feat.replace('_', ' ').title(), fontsize=10)
            ax.set_xlabel("")
        axes[0].legend(fontsize=7, loc='upper right')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_distributions.png", dpi=150)
        plt.close()
        print(f"  Saved: feature_distributions.png")

    # 5. Correlation matrix for laboratory features
    print("\n5. CORRELATION ANALYSIS")
    print("-" * 40)

    lab_features = [c for c in df.columns if c.startswith('lab_') and
                    df[c].dtype in ['float64', 'int64'] and df[c].notna().sum() > 50][:20]

    if lab_features:
        corr_matrix = df[lab_features].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                    annot=False, square=True, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Laboratory Feature Correlations", fontsize=14)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=150)
        plt.close()
        print(f"  Saved: correlation_matrix.png")

    # 6. Class separability - mean feature values by class
    print("\n6. CLASS SEPARABILITY ANALYSIS")
    print("-" * 40)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['patient_id', 'brac_label', 'is_brac_compatible']]

    # Calculate mean by class for features with low missing
    low_missing_features = [c for c in numeric_cols if df[c].notna().sum() > 100][:30]

    if low_missing_features:
        class_means = df.groupby('brac_label')[low_missing_features].mean()

        # Standardize for visualization
        scaler = StandardScaler()
        class_means_scaled = pd.DataFrame(
            scaler.fit_transform(class_means.T).T,
            index=class_means.index,
            columns=class_means.columns
        )

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(class_means_scaled, cmap='RdBu_r', center=0, ax=ax,
                    yticklabels=[BRAC_LABELS[int(i)] for i in class_means_scaled.index])
        ax.set_title("Standardized Feature Means by NHL Subtype", fontsize=14)
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("NHL Subtype", fontsize=12)
        plt.xticks(rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "class_separability_heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved: class_separability_heatmap.png")

    return feature_groups


def prepare_features(df, feature_groups):
    """Prepare feature matrix for classification."""
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)

    # Select features (exclude IDs, labels, and metadata)
    exclude_cols = ['patient_id', 'brac_label', 'is_brac_compatible', 'has_pathology',
                    'has_laboratory', 'has_clinical', 'has_radiology', 'has_bone_marrow',
                    'n_modalities', 'has_all_modalities']

    # Also exclude second brac_label if present (and any duplicates)
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in exclude_cols and
                    df[c].dtype in ['float64', 'int64', 'bool'] and
                    'brac_label' not in c]  # Exclude any brac_label variants

    # Remove duplicate columns
    feature_cols = list(dict.fromkeys(feature_cols))

    # Remove features with >70% missing
    feature_cols = [c for c in feature_cols if df[c].notna().mean() > 0.3]

    print(f"Selected {len(feature_cols)} features after filtering")

    X = df[feature_cols].copy()
    y = df['brac_label'].copy()

    # Drop rows with missing labels
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)

    # Handle rare classes: Merge classes with < 3 samples into "OTHER_RARE" (label 9)
    # Or exclude them for more robust training
    print("\nHandling rare classes:")
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()
    if rare_classes:
        print(f"  Rare classes (n<3): {[BRAC_LABELS[c] for c in rare_classes]}")
        # Option: Exclude rare classes for model training
        mask = ~y.isin(rare_classes)
        X = X[mask]
        y = y[mask]
        print(f"  Excluded {(~mask).sum()} samples from rare classes")

    print(f"Samples with valid labels: {len(y)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Classes retained: {sorted(y.unique())}")

    return X, y, feature_cols


def train_classifiers(X, y, feature_cols):
    """Train and evaluate multiple classifiers."""
    print("\n" + "="*60)
    print("CLASSIFICATION MODEL TRAINING")
    print("="*60)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE for class imbalance...")
    # Use k_neighbors based on minimum class size
    min_class_size = y_train.value_counts().min()
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1

    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_train_resampled)} samples")
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Using original imbalanced data")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=3,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42,
            solver='lbfgs'
        ),
        'SVM': SVC(
            kernel='rbf', class_weight='balanced', random_state=42,
            probability=True
        )
    }

    results = {}
    best_model = None
    best_score = 0

    print("\n" + "-"*60)
    print("MODEL EVALUATION")
    print("-"*60)

    for name, clf in classifiers.items():
        print(f"\n{name}:")

        # Train
        clf.fit(X_train_resampled, y_train_resampled)

        # Predict
        y_pred = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        train_acc = accuracy_score(y_train, y_pred_train)

        results[name] = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'train_accuracy': train_acc
        }

        print(f"  Train Accuracy:    {train_acc:.3f}")
        print(f"  Test Accuracy:     {acc:.3f}")
        print(f"  Balanced Accuracy: {bal_acc:.3f}")
        print(f"  F1 (macro):        {f1_macro:.3f}")
        print(f"  F1 (weighted):     {f1_weighted:.3f}")

        if bal_acc > best_score:
            best_score = bal_acc
            best_model = (name, clf)

    # Best model detailed analysis
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model[0]} (Balanced Accuracy: {best_score:.3f})")
    print("="*60)

    clf = best_model[1]
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels_present = sorted(y.unique())
    label_names = [BRAC_LABELS[int(i)] for i in labels_present]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix - {best_model[0]}", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\nSaved: confusion_matrix.png")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred,
                                   target_names=label_names,
                                   zero_division=0)
    print(report)

    # Feature importance (for tree-based models)
    if hasattr(clf, 'feature_importances_'):
        importance = pd.Series(clf.feature_importances_, index=feature_cols)
        importance = importance.sort_values(ascending=False)

        print("\nTop 20 Most Important Features:")
        for i, (feat, imp) in enumerate(importance.head(20).items()):
            print(f"  {i+1:2d}. {feat:40s} {imp:.4f}")

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        importance.head(25).plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Top 25 Feature Importances", fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
        plt.close()
        print(f"\nSaved: feature_importance.png")

    # Cross-validation
    print("\n" + "-"*60)
    print("CROSS-VALIDATION (5-fold Stratified)")
    print("-"*60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='balanced_accuracy')
    print(f"CV Balanced Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"Individual folds: {cv_scores}")

    # Save results summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"\nSaved: model_comparison.csv")

    # Save trained model components for BRAC integration
    model_artifacts = {
        'imputer': imputer,
        'scaler': scaler,
        'classifier': clf,
        'feature_cols': feature_cols,
        'label_encoder': {i: BRAC_LABELS[i] for i in range(9)}
    }

    return model_artifacts, results


def create_modality_agents(df, model_artifacts):
    """Create per-modality classifiers for BRAC consensus integration."""
    print("\n" + "="*60)
    print("MODALITY-SPECIFIC AGENT TRAINING")
    print("="*60)

    # Define modality feature groups
    modality_features = {
        'pathology': [c for c in df.columns if c.startswith('path_')],
        'laboratory': [c for c in df.columns if c.startswith('lab_')],
        'clinical': [c for c in df.columns if c.startswith('clin_') or c.startswith('prior_')],
        'radiology': [c for c in df.columns if c.startswith('rad_') or c.startswith('bm_')]
    }

    y = df['brac_label'].dropna().astype(int)
    valid_idx = df['brac_label'].notna()

    # Handle rare classes consistently
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()
    if rare_classes:
        mask = ~y.isin(rare_classes)
        valid_idx = valid_idx & mask.reindex(valid_idx.index, fill_value=False)
        y = y[mask]

    modality_models = {}

    for modality, features in modality_features.items():
        # Filter to numeric features with enough data
        features = [f for f in features if f in df.columns and
                    df[f].dtype in ['float64', 'int64'] and
                    df[f].notna().mean() > 0.3]

        if len(features) < 3:
            print(f"\n{modality.upper()}: Insufficient features ({len(features)}), skipping")
            continue

        print(f"\n{modality.upper()} Agent:")
        print(f"  Features: {len(features)}")

        X_mod = df.loc[valid_idx, features].copy()
        y_mod = y.loc[valid_idx].copy()

        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_mod)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_mod, test_size=0.2, random_state=42, stratify=y_mod
        )

        # Handle class imbalance
        min_class = min(np.bincount(y_train))
        k_neighbors = min(3, min_class - 1) if min_class > 1 else 1

        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except:
            X_train_res, y_train_res = X_train, y_train

        # Train Random Forest for probability outputs
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(X_train_res, y_train_res)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        print(f"  Test Accuracy: {acc:.3f}")
        print(f"  Balanced Accuracy: {bal_acc:.3f}")

        # Store model
        modality_models[modality] = {
            'classifier': clf,
            'imputer': imputer,
            'scaler': scaler,
            'features': features,
            'accuracy': acc,
            'balanced_accuracy': bal_acc
        }

    return modality_models


def main():
    """Main execution."""
    print("\n" + "#"*60)
    print("# BRAC FRAMEWORK - EDA & CLASSIFICATION PIPELINE")
    print("#"*60)

    # Load data
    df = load_data()

    # EDA
    feature_groups = perform_eda(df)

    # Prepare features
    X, y, feature_cols = prepare_features(df, feature_groups)

    # Train classifiers
    model_artifacts, results = train_classifiers(X, y, feature_cols)

    # Create modality-specific agents
    modality_models = create_modality_agents(df, model_artifacts)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset: 174 patients, {len(feature_cols)} features")
    print(f"Classes: {len(y.unique())} NHL subtypes")
    print(f"Best overall model: Random Forest or Gradient Boosting")
    print(f"Modality agents trained: {list(modality_models.keys())}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nNext step: Integrate modality_models into BRAC consensus pipeline")

    return model_artifacts, modality_models


if __name__ == "__main__":
    main()
