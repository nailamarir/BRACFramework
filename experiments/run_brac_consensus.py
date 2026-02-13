"""
BRAC Consensus Pipeline with Trained Agents
============================================

This script integrates trained classification models into the BRAC
Byzantine-Resilient Agentic Consensus framework for NHL subtyping.

Pipeline:
1. Load trained modality-specific agents (pathology, lab, clinical, radiology)
2. Process patient data through each agent to get belief distributions
3. Run BRAC consensus to combine agent beliefs with trust weighting
4. Apply conformal prediction for uncertainty quantification
5. Compute Shapley values for explainability
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from brac.types import Modality, NHLSubtype, AgentOutput, EvidenceQuality
from brac.orchestrator import BRACOrchestrator, BRACConfig
from brac.agents.trained_agents import TrainedAgentFactory, TrainedAgent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "brac_consensus"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# BRAC labels
BRAC_LABELS = {
    0: "DLBCL_GCB", 1: "DLBCL_ABC", 2: "FL", 3: "MCL",
    4: "BL", 5: "MZL", 6: "CLL_SLL", 7: "LPL", 8: "PMBL"
}


def load_data():
    """Load and prepare the feature matrix."""
    df = pd.read_csv(DATA_DIR / "merged" / "feature_matrix_brac.csv")
    print(f"Loaded data: {df.shape[0]} patients, {df.shape[1]} features")
    return df


def train_modality_agents(df):
    """Train modality-specific agents."""
    print("\n" + "="*60)
    print("TRAINING MODALITY AGENTS")
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

    # Handle rare classes
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()
    if rare_classes:
        mask = ~y.isin(rare_classes)
        valid_idx = valid_idx & mask.reindex(valid_idx.index, fill_value=False)
        y = y[mask]
        print(f"Excluded rare classes: {[BRAC_LABELS[c] for c in rare_classes]}")

    modality_models = {}

    for modality, features in modality_features.items():
        # Filter features
        features = [f for f in features if f in df.columns and
                    df[f].dtype in ['float64', 'int64'] and
                    df[f].notna().mean() > 0.3 and
                    'brac_label' not in f]

        if len(features) < 3:
            print(f"\n{modality.upper()}: Insufficient features, skipping")
            continue

        print(f"\n{modality.upper()} Agent: {len(features)} features")

        X_mod = df.loc[valid_idx, features].copy()
        y_mod = y.loc[valid_idx].copy()

        # Preprocess
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_mod)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_mod, test_size=0.2, random_state=42, stratify=y_mod
        )

        # SMOTE
        min_class = min(np.bincount(y_train))
        k_neighbors = min(3, min_class - 1) if min_class > 1 else 1
        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except:
            X_train_res, y_train_res = X_train, y_train

        # Train
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(X_train_res, y_train_res)

        # Evaluate
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.3f}, Balanced: {bal_acc:.3f}")

        modality_models[modality] = {
            'classifier': clf,
            'imputer': imputer,
            'scaler': scaler,
            'features': features,
            'accuracy': acc,
            'balanced_accuracy': bal_acc
        }

    return modality_models


def create_brac_agents(modality_models):
    """Create BRAC agents from trained models."""
    factory = TrainedAgentFactory()
    agents = factory.create_all_agents(modality_models)
    print(f"\nCreated {len(agents)} BRAC agents")
    for modality, agent in agents.items():
        print(f"  {agent}")
    return factory, agents


def run_brac_consensus(df, factory, agents):
    """Run BRAC consensus on all patients."""
    print("\n" + "="*60)
    print("BRAC CONSENSUS EVALUATION")
    print("="*60)

    # Initialize BRAC orchestrator
    config = BRACConfig(
        num_classes=9,
        num_agents=4,
        alpha=0.10,  # 90% coverage target
        max_set_size_accept=2,
        compute_shapley=True,
        compute_interactions=True,
    )
    orchestrator = BRACOrchestrator(config)

    # Filter to valid patients
    valid_patients = df[df['brac_label'].notna()].copy()

    # Handle rare classes
    class_counts = valid_patients['brac_label'].value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()
    valid_patients = valid_patients[~valid_patients['brac_label'].isin(rare_classes)]

    print(f"Evaluating on {len(valid_patients)} patients")

    # Split for calibration and test
    train_patients, test_patients = train_test_split(
        valid_patients, test_size=0.3, random_state=42,
        stratify=valid_patients['brac_label']
    )

    print(f"Calibration set: {len(train_patients)}")
    print(f"Test set: {len(test_patients)}")

    # Prepare calibration data
    print("\nPreparing calibration data...")
    calibration_data = []
    for idx, row in train_patients.iterrows():
        features = row.to_dict()
        true_label = int(row['brac_label'])
        agent_outputs = factory.get_agent_outputs(features)
        calibration_data.append((agent_outputs, true_label))

    # Calibrate conformal predictor
    print("Calibrating conformal predictor...")
    q_hat = orchestrator.calibrate(calibration_data)
    print(f"Calibrated threshold q_hat = {q_hat:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = []

    for idx, row in test_patients.iterrows():
        features = row.to_dict()
        true_label = int(row['brac_label'])
        agent_outputs = factory.get_agent_outputs(features)

        # Run BRAC
        result = orchestrator.run(agent_outputs, calibrated=True)

        # Record results
        results.append({
            'patient_id': row.get('patient_id', idx),
            'true_label': true_label,
            'true_name': BRAC_LABELS.get(true_label, 'Unknown'),
            'predicted_label': result.diagnosis.value,
            'predicted_name': BRAC_LABELS.get(result.diagnosis.value, 'Unknown'),
            'correct': result.diagnosis.value == true_label,
            'confidence': result.confidence,
            'set_size': result.prediction_set_size,
            'covered': any(s.value == true_label for s in result.prediction_set),
            'accepted': result.accepted,
            'convergence_rounds': result.convergence_rounds,
            'shapley_pathology': result.shapley_values.get(Modality.PATHOLOGY, 0),
            'shapley_laboratory': result.shapley_values.get(Modality.LABORATORY, 0),
            'shapley_clinical': result.shapley_values.get(Modality.CLINICAL, 0),
            'shapley_radiology': result.shapley_values.get(Modality.RADIOLOGY, 0),
        })

    results_df = pd.DataFrame(results)

    # Summary metrics
    print("\n" + "-"*60)
    print("BRAC CONSENSUS RESULTS")
    print("-"*60)

    accuracy = results_df['correct'].mean()
    coverage = results_df['covered'].mean()
    avg_set_size = results_df['set_size'].mean()
    acceptance_rate = results_df['accepted'].mean()
    avg_confidence = results_df['confidence'].mean()
    avg_rounds = results_df['convergence_rounds'].mean()

    print(f"Accuracy:         {accuracy:.3f}")
    print(f"Coverage:         {coverage:.3f} (target: {1 - config.alpha:.2f})")
    print(f"Avg Set Size:     {avg_set_size:.2f}")
    print(f"Acceptance Rate:  {acceptance_rate:.3f}")
    print(f"Avg Confidence:   {avg_confidence:.3f}")
    print(f"Avg Conv Rounds:  {avg_rounds:.1f}")

    # Per-class metrics
    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)

    for label in sorted(results_df['true_label'].unique()):
        subset = results_df[results_df['true_label'] == label]
        n = len(subset)
        acc = subset['correct'].mean()
        cov = subset['covered'].mean()
        print(f"  {BRAC_LABELS[label]:12s}: n={n:3d}, acc={acc:.2f}, cov={cov:.2f}")

    # Shapley value summary
    print("\n" + "-"*60)
    print("SHAPLEY VALUE SUMMARY (Mean Attribution)")
    print("-"*60)

    for modality in ['pathology', 'laboratory', 'clinical', 'radiology']:
        mean_shapley = results_df[f'shapley_{modality}'].mean()
        print(f"  {modality.capitalize():12s}: {mean_shapley:.4f}")

    # Save results
    results_df.to_csv(RESULTS_DIR / "brac_results.csv", index=False)
    print(f"\nResults saved to: {RESULTS_DIR / 'brac_results.csv'}")

    return results_df


def visualize_results(results_df):
    """Generate visualizations of BRAC results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results_df['true_label'], results_df['predicted_label'])
    labels_present = sorted(results_df['true_label'].unique())
    label_names = [BRAC_LABELS[i] for i in labels_present]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel("BRAC Predicted", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("BRAC Consensus Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "brac_confusion_matrix.png", dpi=150)
    plt.close()

    # 2. Shapley values by modality
    shapley_cols = ['shapley_pathology', 'shapley_laboratory',
                    'shapley_clinical', 'shapley_radiology']

    fig, ax = plt.subplots(figsize=(8, 6))
    shapley_means = results_df[shapley_cols].mean()
    shapley_means.index = ['Pathology', 'Laboratory', 'Clinical', 'Radiology']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    shapley_means.plot(kind='bar', ax=ax, color=colors)
    ax.set_ylabel("Mean Shapley Value", fontsize=12)
    ax.set_title("Agent Attribution (Mean Shapley Values)", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "brac_shapley_attribution.png", dpi=150)
    plt.close()

    # 3. Shapley by subtype
    shapley_by_subtype = results_df.groupby('true_name')[shapley_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    shapley_by_subtype.plot(kind='bar', ax=ax, color=colors)
    ax.set_ylabel("Mean Shapley Value", fontsize=12)
    ax.set_xlabel("NHL Subtype", fontsize=12)
    ax.set_title("Agent Attribution by NHL Subtype", fontsize=14)
    ax.legend(['Pathology', 'Laboratory', 'Clinical', 'Radiology'],
              loc='upper right', fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "brac_shapley_by_subtype.png", dpi=150)
    plt.close()

    # 4. Confidence distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    results_df.boxplot(column='confidence', by='correct', ax=ax)
    ax.set_xlabel("Correct Prediction", fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title("BRAC Confidence by Correctness", fontsize=14)
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "brac_confidence_distribution.png", dpi=150)
    plt.close()

    print(f"\nVisualizations saved to: {RESULTS_DIR}")


def main():
    """Main execution."""
    print("\n" + "#"*60)
    print("# BRAC CONSENSUS PIPELINE")
    print("#"*60)

    # Load data
    df = load_data()

    # Train modality agents
    modality_models = train_modality_agents(df)

    # Create BRAC agents
    factory, agents = create_brac_agents(modality_models)

    # Save agents
    factory.save_agents(MODELS_DIR / "brac_agents")
    print(f"\nAgents saved to: {MODELS_DIR / 'brac_agents'}")

    # Run BRAC consensus
    results_df = run_brac_consensus(df, factory, agents)

    # Visualize
    visualize_results(results_df)

    print("\n" + "="*60)
    print("BRAC CONSENSUS PIPELINE COMPLETE")
    print("="*60)

    return results_df


if __name__ == "__main__":
    main()
