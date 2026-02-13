"""
Comparison: Single Agent vs Agentic System vs BRAC Consensus
=============================================================

This script compares three approaches:
1. Single Agent: Each modality classifier independently
2. Agentic System: Simple ensemble (average/voting) of all agents
3. BRAC Consensus: Byzantine-resilient geometric median consensus
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import torch
import warnings
warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from brac.types import Modality, EvidenceQuality
from brac.orchestrator import BRACOrchestrator, BRACConfig
from brac.agents.trained_agents import TrainedAgentFactory, TrainedAgentConfig, TrainedAgent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BRAC_LABELS = {
    0: "DLBCL_GCB", 1: "DLBCL_ABC", 2: "FL", 3: "MCL",
    4: "BL", 5: "MZL", 6: "CLL_SLL", 7: "LPL", 8: "PMBL"
}


def load_and_prepare_data():
    """Load data and prepare train/test splits."""
    df = pd.read_csv(DATA_DIR / "merged" / "feature_matrix_brac.csv")

    # Filter valid labels and exclude rare classes
    valid_mask = df['brac_label'].notna()
    df_valid = df[valid_mask].copy()

    class_counts = df_valid['brac_label'].value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()
    df_valid = df_valid[~df_valid['brac_label'].isin(rare_classes)]

    # Split
    train_df, test_df = train_test_split(
        df_valid, test_size=0.3, random_state=42,
        stratify=df_valid['brac_label']
    )

    print(f"Training set: {len(train_df)} patients")
    print(f"Test set: {len(test_df)} patients")
    print(f"Classes: {sorted(df_valid['brac_label'].unique())}")

    return train_df, test_df


def train_modality_agent(train_df, modality_name, feature_prefix_list):
    """Train a single modality agent."""
    # Get features
    features = []
    for prefix in feature_prefix_list:
        features.extend([c for c in train_df.columns if c.startswith(prefix)])

    # Filter
    features = [f for f in features if
                train_df[f].dtype in ['float64', 'int64'] and
                train_df[f].notna().mean() > 0.3 and
                'brac_label' not in f]

    if len(features) < 3:
        return None, None, None, None

    X = train_df[features].copy()
    y = train_df['brac_label'].astype(int)

    # Preprocess
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # SMOTE
    min_class = min(np.bincount(y))
    k = min(3, min_class - 1) if min_class > 1 else 1
    try:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_scaled, y)
    except:
        X_res, y_res = X_scaled, y

    # Train
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(X_res, y_res)

    return clf, imputer, scaler, features


def evaluate_single_agent(clf, imputer, scaler, features, test_df, agent_name):
    """Evaluate a single agent."""
    X_test = test_df[features].copy()
    y_test = test_df['brac_label'].astype(int)

    X_imp = imputer.transform(X_test)
    X_scaled = scaler.transform(X_imp)

    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)

    metrics = {
        'agent': agent_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
    }

    return metrics, y_pred, y_proba


def evaluate_ensemble(all_proba, y_test, method='average'):
    """Evaluate simple ensemble methods."""
    # Stack probabilities: (n_agents, n_samples, n_classes)
    proba_stack = np.stack(list(all_proba.values()))

    if method == 'average':
        # Simple average
        ensemble_proba = proba_stack.mean(axis=0)
    elif method == 'voting':
        # Majority voting
        votes = np.argmax(proba_stack, axis=2)  # (n_agents, n_samples)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=9).argmax(),
            axis=0, arr=votes
        )
        return ensemble_pred
    elif method == 'max':
        # Max confidence
        ensemble_proba = proba_stack.max(axis=0)

    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    return ensemble_pred


def evaluate_brac_consensus(train_df, test_df, agents_dict):
    """Evaluate BRAC consensus system."""
    # Create BRAC agents
    factory = TrainedAgentFactory()

    modality_map = {
        'pathology': Modality.PATHOLOGY,
        'laboratory': Modality.LABORATORY,
        'clinical': Modality.CLINICAL,
        'radiology': Modality.RADIOLOGY,
    }

    for name, (clf, imputer, scaler, features) in agents_dict.items():
        if clf is not None and name in modality_map:
            factory.create_agent(
                modality=modality_map[name],
                classifier=clf,
                imputer=imputer,
                scaler=scaler,
                feature_cols=features,
                accuracy=0.5,
                balanced_accuracy=0.3
            )

    # Initialize BRAC
    config = BRACConfig(
        num_classes=9,
        num_agents=4,
        alpha=0.10,
        max_set_size_accept=2,
        compute_shapley=True,
    )
    orchestrator = BRACOrchestrator(config)

    # Calibrate on training data
    calibration_data = []
    for idx, row in train_df.iterrows():
        features = row.to_dict()
        true_label = int(row['brac_label'])
        agent_outputs = factory.get_agent_outputs(features)
        calibration_data.append((agent_outputs, true_label))

    orchestrator.calibrate(calibration_data)

    # Evaluate on test
    y_test = test_df['brac_label'].astype(int).values
    y_pred = []
    coverages = []
    set_sizes = []
    confidences = []

    for idx, row in test_df.iterrows():
        features = row.to_dict()
        agent_outputs = factory.get_agent_outputs(features)
        result = orchestrator.run(agent_outputs, calibrated=True)

        y_pred.append(result.diagnosis.value)
        set_sizes.append(result.prediction_set_size)
        confidences.append(result.confidence)

        # Check coverage
        true_label = int(row['brac_label'])
        covered = any(s.value == true_label for s in result.prediction_set)
        coverages.append(covered)

    y_pred = np.array(y_pred)

    metrics = {
        'agent': 'BRAC Consensus',
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'coverage': np.mean(coverages),
        'avg_set_size': np.mean(set_sizes),
        'avg_confidence': np.mean(confidences),
    }

    return metrics, y_pred


def main():
    print("\n" + "="*70)
    print(" COMPARISON: Single Agent vs Agentic System vs BRAC Consensus")
    print("="*70)

    # Load data
    train_df, test_df = load_and_prepare_data()
    y_test = test_df['brac_label'].astype(int).values

    # Define modality features
    modality_config = {
        'pathology': ['path_'],
        'laboratory': ['lab_'],
        'clinical': ['clin_', 'prior_'],
        'radiology': ['rad_', 'bm_'],
    }

    # Train all agents
    print("\n" + "-"*70)
    print("TRAINING AGENTS")
    print("-"*70)

    agents_dict = {}
    all_proba = {}
    single_agent_results = []

    for name, prefixes in modality_config.items():
        print(f"\nTraining {name.upper()} agent...")
        clf, imputer, scaler, features = train_modality_agent(train_df, name, prefixes)

        if clf is not None:
            agents_dict[name] = (clf, imputer, scaler, features)

            # Evaluate single agent
            metrics, y_pred, y_proba = evaluate_single_agent(
                clf, imputer, scaler, features, test_df, name.capitalize()
            )
            single_agent_results.append(metrics)

            # Expand proba to 9 classes
            full_proba = np.zeros((len(y_test), 9))
            for i, cls in enumerate(clf.classes_):
                if cls < 9:
                    full_proba[:, int(cls)] = y_proba[:, i]
            full_proba = full_proba / (full_proba.sum(axis=1, keepdims=True) + 1e-10)
            all_proba[name] = full_proba

            print(f"  Accuracy: {metrics['accuracy']:.3f}, Balanced: {metrics['balanced_accuracy']:.3f}")

    # Evaluate ensemble methods
    print("\n" + "-"*70)
    print("EVALUATING ENSEMBLE METHODS")
    print("-"*70)

    ensemble_results = []

    # Average ensemble
    y_pred_avg = evaluate_ensemble(all_proba, y_test, method='average')
    ensemble_results.append({
        'agent': 'Ensemble (Average)',
        'accuracy': accuracy_score(y_test, y_pred_avg),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_avg),
        'f1_macro': f1_score(y_test, y_pred_avg, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred_avg, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred_avg, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred_avg, average='macro', zero_division=0),
    })

    # Voting ensemble
    y_pred_vote = evaluate_ensemble(all_proba, y_test, method='voting')
    ensemble_results.append({
        'agent': 'Ensemble (Voting)',
        'accuracy': accuracy_score(y_test, y_pred_vote),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_vote),
        'f1_macro': f1_score(y_test, y_pred_vote, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred_vote, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred_vote, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred_vote, average='macro', zero_division=0),
    })

    # Max confidence ensemble
    y_pred_max = evaluate_ensemble(all_proba, y_test, method='max')
    ensemble_results.append({
        'agent': 'Ensemble (Max Conf)',
        'accuracy': accuracy_score(y_test, y_pred_max),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_max),
        'f1_macro': f1_score(y_test, y_pred_max, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred_max, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred_max, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred_max, average='macro', zero_division=0),
    })

    # BRAC Consensus
    print("\n" + "-"*70)
    print("EVALUATING BRAC CONSENSUS")
    print("-"*70)

    brac_metrics, y_pred_brac = evaluate_brac_consensus(train_df, test_df, agents_dict)

    # Compile all results
    all_results = single_agent_results + ensemble_results + [brac_metrics]
    results_df = pd.DataFrame(all_results)

    # Fill missing columns
    for col in ['coverage', 'avg_set_size', 'avg_confidence']:
        if col not in results_df.columns:
            results_df[col] = np.nan

    # Print results
    print("\n" + "="*70)
    print(" RESULTS COMPARISON")
    print("="*70)

    print("\n" + "-"*70)
    print("1. SINGLE AGENT PERFORMANCE")
    print("-"*70)
    print(f"{'Agent':<15} {'Accuracy':>10} {'Bal.Acc':>10} {'F1-Macro':>10} {'F1-Wgt':>10} {'Prec':>10} {'Recall':>10}")
    print("-"*70)
    for r in single_agent_results:
        print(f"{r['agent']:<15} {r['accuracy']:>10.3f} {r['balanced_accuracy']:>10.3f} "
              f"{r['f1_macro']:>10.3f} {r['f1_weighted']:>10.3f} "
              f"{r['precision_macro']:>10.3f} {r['recall_macro']:>10.3f}")

    print("\n" + "-"*70)
    print("2. AGENTIC SYSTEM (Simple Ensemble)")
    print("-"*70)
    print(f"{'Method':<20} {'Accuracy':>10} {'Bal.Acc':>10} {'F1-Macro':>10} {'F1-Wgt':>10} {'Prec':>10} {'Recall':>10}")
    print("-"*70)
    for r in ensemble_results:
        print(f"{r['agent']:<20} {r['accuracy']:>10.3f} {r['balanced_accuracy']:>10.3f} "
              f"{r['f1_macro']:>10.3f} {r['f1_weighted']:>10.3f} "
              f"{r['precision_macro']:>10.3f} {r['recall_macro']:>10.3f}")

    print("\n" + "-"*70)
    print("3. BRAC CONSENSUS (Byzantine-Resilient)")
    print("-"*70)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-"*70)
    print(f"{'Accuracy':<25} {brac_metrics['accuracy']:>15.3f}")
    print(f"{'Balanced Accuracy':<25} {brac_metrics['balanced_accuracy']:>15.3f}")
    print(f"{'F1 (Macro)':<25} {brac_metrics['f1_macro']:>15.3f}")
    print(f"{'F1 (Weighted)':<25} {brac_metrics['f1_weighted']:>15.3f}")
    print(f"{'Precision (Macro)':<25} {brac_metrics['precision_macro']:>15.3f}")
    print(f"{'Recall (Macro)':<25} {brac_metrics['recall_macro']:>15.3f}")
    print(f"{'Coverage':<25} {brac_metrics['coverage']:>15.3f}")
    print(f"{'Avg Set Size':<25} {brac_metrics['avg_set_size']:>15.2f}")
    print(f"{'Avg Confidence':<25} {brac_metrics['avg_confidence']:>15.3f}")

    # Summary comparison
    print("\n" + "="*70)
    print(" SUMMARY: BEST ACCURACY BY APPROACH")
    print("="*70)

    best_single = max(single_agent_results, key=lambda x: x['accuracy'])
    best_ensemble = max(ensemble_results, key=lambda x: x['accuracy'])

    print(f"\n{'Approach':<25} {'Best Method':<20} {'Accuracy':>10} {'Bal.Acc':>10}")
    print("-"*70)
    print(f"{'Single Agent':<25} {best_single['agent']:<20} {best_single['accuracy']:>10.3f} {best_single['balanced_accuracy']:>10.3f}")
    print(f"{'Agentic System':<25} {best_ensemble['agent']:<20} {best_ensemble['accuracy']:>10.3f} {best_ensemble['balanced_accuracy']:>10.3f}")
    print(f"{'BRAC Consensus':<25} {'Geometric Median':<20} {brac_metrics['accuracy']:>10.3f} {brac_metrics['balanced_accuracy']:>10.3f}")

    # Improvement calculation
    print("\n" + "-"*70)
    print("IMPROVEMENT ANALYSIS")
    print("-"*70)

    baseline_acc = best_single['accuracy']
    ensemble_improvement = (best_ensemble['accuracy'] - baseline_acc) / baseline_acc * 100
    brac_improvement = (brac_metrics['accuracy'] - baseline_acc) / baseline_acc * 100

    print(f"Baseline (Best Single Agent): {baseline_acc:.3f}")
    print(f"Agentic System improvement:   {ensemble_improvement:+.1f}%")
    print(f"BRAC Consensus improvement:   {brac_improvement:+.1f}%")

    # Additional BRAC benefits
    print("\n" + "-"*70)
    print("BRAC ADDITIONAL BENEFITS")
    print("-"*70)
    print(f"Coverage Guarantee:    {brac_metrics['coverage']:.1%} (target: 90%)")
    print(f"Prediction Set Size:   {brac_metrics['avg_set_size']:.2f} subtypes")
    print(f"Byzantine Resilience:  Tolerates up to 1 faulty agent")
    print(f"Explainability:        Shapley attribution per agent")

    # Save results
    results_df.to_csv(RESULTS_DIR / "comparison_results.csv", index=False)
    print(f"\nResults saved to: {RESULTS_DIR / 'comparison_results.csv'}")

    return results_df


if __name__ == "__main__":
    main()
