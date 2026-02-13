"""Trained classifier agents for BRAC consensus integration.

This module provides agents that use trained scikit-learn classifiers
to generate belief distributions over NHL subtypes from real patient data.

Each agent corresponds to a diagnostic modality:
- PathologyAgent: Uses histopathology and IHC features
- LaboratoryAgent: Uses CBC and flow cytometry features
- ClinicalAgent: Uses demographics and epidemiological priors
- RadiologyAgent: Uses imaging and bone marrow features
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from brac.agents.base_agent import BaseAgent
from brac.types import AgentOutput, EvidenceQuality, Modality, SemanticEvidence


@dataclass
class TrainedAgentConfig:
    """Configuration for a trained classifier agent."""
    classifier: RandomForestClassifier
    imputer: SimpleImputer
    scaler: StandardScaler
    feature_cols: list[str]
    modality: Modality
    accuracy: float = 0.5
    balanced_accuracy: float = 0.3


class TrainedAgent(BaseAgent):
    """Agent that uses a trained classifier to generate beliefs.

    The agent takes raw feature values for its modality, preprocesses them,
    and outputs a probability distribution over NHL subtypes using the
    classifier's predict_proba method.
    """

    def __init__(self, config: TrainedAgentConfig):
        """Initialize trained agent.

        Args:
            config: TrainedAgentConfig with classifier and preprocessing
        """
        self.config = config
        self.classifier = config.classifier
        self.imputer = config.imputer
        self.scaler = config.scaler
        self.feature_cols = config.feature_cols
        self.modality = config.modality

        # Get number of classes from classifier
        if hasattr(self.classifier, 'n_classes_'):
            self.num_classes = self.classifier.n_classes_
        else:
            self.num_classes = 9  # Default BRAC classes

    def forward(self, features: dict[str, float]) -> AgentOutput:
        """Generate belief distribution from features.

        Args:
            features: Dictionary mapping feature names to values

        Returns:
            AgentOutput with belief and quality scores
        """
        # Extract features in correct order
        feature_values = []
        missing_count = 0

        for col in self.feature_cols:
            if col in features and not pd.isna(features[col]):
                feature_values.append(features[col])
            else:
                feature_values.append(np.nan)
                missing_count += 1

        # Reshape for sklearn
        X = np.array(feature_values).reshape(1, -1)

        # Impute and scale
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        # Get probability distribution
        proba = self.classifier.predict_proba(X_scaled)[0]

        # Handle case where classifier doesn't predict all classes
        # Expand to full 9-class distribution
        full_proba = np.zeros(9)
        for i, cls in enumerate(self.classifier.classes_):
            if cls < 9:
                full_proba[int(cls)] = proba[i]

        # Normalize
        if full_proba.sum() > 0:
            full_proba = full_proba / full_proba.sum()
        else:
            full_proba = np.ones(9) / 9  # Uniform if all zero

        # Add small epsilon for numerical stability
        eps = 1e-6
        full_proba = np.clip(full_proba, eps, 1 - eps)
        full_proba = full_proba / full_proba.sum()

        belief = torch.tensor(full_proba, dtype=torch.float32)

        # Compute quality scores
        coverage = 1.0 - (missing_count / len(self.feature_cols))
        quality = EvidenceQuality(
            Q=min(0.9, self.config.accuracy),  # Data quality
            C=coverage,                         # Coverage
            S=0.85                              # Consistency
        )

        # Create semantic evidence
        evidence = self._create_evidence(features)

        return AgentOutput(
            belief=belief,
            quality=quality,
            evidence=evidence,
            modality=self.modality,
        )

    def _create_evidence(self, features: dict[str, float]) -> SemanticEvidence:
        """Create semantic evidence description from features.

        Args:
            features: Raw feature values

        Returns:
            SemanticEvidence object
        """
        findings = []

        # Extract meaningful findings based on modality
        if self.modality == Modality.PATHOLOGY:
            if features.get('path_ihc_cd20_numeric', 0) > 0.5:
                findings.append("CD20+")
            if features.get('path_ihc_cd10_numeric', 0) > 0.5:
                findings.append("CD10+")
            if features.get('path_ihc_bcl6_numeric', 0) > 0.5:
                findings.append("BCL6+")

        elif self.modality == Modality.LABORATORY:
            wbc = features.get('lab_wbc_mean', 0)
            if wbc and wbc > 10:
                findings.append(f"WBC:{wbc:.1f}")

        elif self.modality == Modality.CLINICAL:
            age = features.get('clin_age_at_dx', 0)
            if age and age > 0:
                findings.append(f"age:{int(age)}")

        elif self.modality == Modality.RADIOLOGY:
            if features.get('rad_ever_bulky', 0) > 0.5:
                findings.append("bulky")

        value_str = ", ".join(findings) if findings else "none"
        return SemanticEvidence(
            finding_type=self.modality.name.lower(),
            code=f"BRAC-{self.modality.name}",
            value=value_str,
            confidence=0.8,
            provenance=self.modality.name,
            quality_score=self.config.accuracy
        )

    def __repr__(self) -> str:
        return f"TrainedAgent({self.modality.name}, features={len(self.feature_cols)}, acc={self.config.accuracy:.2f})"


class TrainedAgentFactory:
    """Factory for creating trained agents from saved models."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize factory.

        Args:
            models_dir: Directory containing saved model artifacts
        """
        self.models_dir = models_dir or Path(__file__).parent.parent.parent / "models"
        self.agents: dict[Modality, TrainedAgent] = {}

    def create_agent(
        self,
        modality: Modality,
        classifier: RandomForestClassifier,
        imputer: SimpleImputer,
        scaler: StandardScaler,
        feature_cols: list[str],
        accuracy: float = 0.5,
        balanced_accuracy: float = 0.3,
    ) -> TrainedAgent:
        """Create a trained agent for a modality.

        Args:
            modality: Diagnostic modality
            classifier: Trained sklearn classifier
            imputer: Fitted imputer
            scaler: Fitted scaler
            feature_cols: List of feature column names
            accuracy: Test accuracy
            balanced_accuracy: Balanced test accuracy

        Returns:
            TrainedAgent instance
        """
        config = TrainedAgentConfig(
            classifier=classifier,
            imputer=imputer,
            scaler=scaler,
            feature_cols=feature_cols,
            modality=modality,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
        )
        agent = TrainedAgent(config)
        self.agents[modality] = agent
        return agent

    def create_all_agents(
        self,
        modality_models: dict[str, dict],
    ) -> dict[Modality, TrainedAgent]:
        """Create agents for all modalities from model dictionary.

        Args:
            modality_models: Dictionary from create_modality_agents()

        Returns:
            Dictionary mapping Modality -> TrainedAgent
        """
        modality_map = {
            'pathology': Modality.PATHOLOGY,
            'laboratory': Modality.LABORATORY,
            'clinical': Modality.CLINICAL,
            'radiology': Modality.RADIOLOGY,
        }

        agents = {}
        for name, model_dict in modality_models.items():
            if name in modality_map:
                modality = modality_map[name]
                agent = self.create_agent(
                    modality=modality,
                    classifier=model_dict['classifier'],
                    imputer=model_dict['imputer'],
                    scaler=model_dict['scaler'],
                    feature_cols=model_dict['features'],
                    accuracy=model_dict.get('accuracy', 0.5),
                    balanced_accuracy=model_dict.get('balanced_accuracy', 0.3),
                )
                agents[modality] = agent

        self.agents = agents
        return agents

    def get_agent_outputs(
        self,
        patient_features: dict[str, float],
    ) -> dict[Modality, AgentOutput]:
        """Get outputs from all agents for a patient.

        Args:
            patient_features: Dictionary with all patient features

        Returns:
            Dictionary mapping Modality -> AgentOutput
        """
        outputs = {}
        for modality, agent in self.agents.items():
            outputs[modality] = agent.forward(patient_features)
        return outputs

    def save_agents(self, path: Path):
        """Save all agents to disk.

        Args:
            path: Path to save directory
        """
        path.mkdir(parents=True, exist_ok=True)

        for modality, agent in self.agents.items():
            agent_path = path / f"{modality.name.lower()}_agent.pkl"
            joblib.dump({
                'classifier': agent.classifier,
                'imputer': agent.imputer,
                'scaler': agent.scaler,
                'feature_cols': agent.feature_cols,
                'accuracy': agent.config.accuracy,
                'balanced_accuracy': agent.config.balanced_accuracy,
            }, agent_path)

    def load_agents(self, path: Path) -> dict[Modality, TrainedAgent]:
        """Load all agents from disk.

        Args:
            path: Path to saved agents directory

        Returns:
            Dictionary of loaded agents
        """
        agents = {}
        modality_map = {
            'pathology': Modality.PATHOLOGY,
            'laboratory': Modality.LABORATORY,
            'clinical': Modality.CLINICAL,
            'radiology': Modality.RADIOLOGY,
        }

        for name, modality in modality_map.items():
            agent_path = path / f"{name}_agent.pkl"
            if agent_path.exists():
                data = joblib.load(agent_path)
                agent = self.create_agent(
                    modality=modality,
                    classifier=data['classifier'],
                    imputer=data['imputer'],
                    scaler=data['scaler'],
                    feature_cols=data['feature_cols'],
                    accuracy=data.get('accuracy', 0.5),
                    balanced_accuracy=data.get('balanced_accuracy', 0.3),
                )
                agents[modality] = agent

        self.agents = agents
        return agents
