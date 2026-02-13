"""PyTorch Dataset for multimodal NHL data.

Combines all modalities into a single dataset for training and inference.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import logging

from .loaders import (
    ManifestLoader,
    PathologyLoader,
    RadiologyLoader,
    LaboratoryLoader,
    ClinicalLoader,
    LoadedSample,
)
from brac.types import Modality

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSample:
    """A single multimodal sample with all modalities."""
    case_id: str
    label: int
    pathology: LoadedSample
    radiology: LoadedSample
    laboratory: LoadedSample
    clinical: LoadedSample

    def get_valid_modalities(self) -> list[Modality]:
        """Get list of modalities with valid data."""
        valid = []
        if self.pathology.valid:
            valid.append(Modality.PATHOLOGY)
        if self.radiology.valid:
            valid.append(Modality.RADIOLOGY)
        if self.laboratory.valid:
            valid.append(Modality.LABORATORY)
        if self.clinical.valid:
            valid.append(Modality.CLINICAL)
        return valid

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "case_id": self.case_id,
            "label": self.label,
            "pathology": self.pathology.data,
            "radiology": self.radiology.data,
            "laboratory": self.laboratory.data,
            "clinical": self.clinical.data,
            "valid_modalities": [m.value for m in self.get_valid_modalities()],
        }


class NHLMultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal NHL classification.

    Loads data from all four modalities (pathology, radiology, laboratory, clinical)
    based on a manifest CSV file.

    Example:
        >>> dataset = NHLMultimodalDataset("data/manifest.csv")
        >>> sample = dataset[0]
        >>> print(sample.case_id, sample.label)
        >>> print(sample.pathology.data.shape)
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        pathology_feature_dim: int = 768,
        radiology_feature_dim: int = 512,
        laboratory_feature_dim: int = 64,
        require_all_modalities: bool = False,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to manifest CSV
            data_root: Root directory for data files (default: manifest parent dir)
            split: Filter by split (train/validation/test). None = all.
            pathology_feature_dim: Expected pathology feature dimension
            radiology_feature_dim: Expected radiology feature dimension
            laboratory_feature_dim: Expected laboratory feature dimension
            require_all_modalities: If True, skip cases missing any modality
        """
        self.manifest = ManifestLoader(manifest_path, data_root)
        self.split = split
        self.require_all_modalities = require_all_modalities

        # Initialize modality loaders
        self.pathology_loader = PathologyLoader(feature_dim=pathology_feature_dim)
        self.radiology_loader = RadiologyLoader(feature_dim=radiology_feature_dim)
        self.laboratory_loader = LaboratoryLoader(feature_dim=laboratory_feature_dim)
        self.clinical_loader = ClinicalLoader()

        # Get cases for this split
        if split:
            self.cases_df = self.manifest.get_split(split)
        else:
            self.cases_df = self.manifest.df

        # Filter cases if require_all_modalities
        if require_all_modalities:
            self._filter_complete_cases()

        logger.info(f"Dataset initialized with {len(self.cases_df)} cases")

    def _filter_complete_cases(self):
        """Filter to only cases with all modalities available."""
        modality_cols = ["pathology_path", "radiology_path", "laboratory_path", "clinical_path"]
        available_cols = [c for c in modality_cols if c in self.cases_df.columns]

        if not available_cols:
            logger.warning("No modality path columns found in manifest")
            return

        # Keep rows where all modality paths are non-null
        mask = self.cases_df[available_cols].notna().all(axis=1)
        original_count = len(self.cases_df)
        self.cases_df = self.cases_df[mask].reset_index(drop=True)
        filtered_count = original_count - len(self.cases_df)

        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} incomplete cases, {len(self.cases_df)} remaining")

    def __len__(self) -> int:
        return len(self.cases_df)

    def __getitem__(self, idx: int) -> MultimodalSample:
        """Get a multimodal sample by index.

        Args:
            idx: Sample index

        Returns:
            MultimodalSample with data from all modalities
        """
        row = self.cases_df.iloc[idx]
        case_id = row["case_id"]
        label = int(row["label"])

        # Load each modality
        pathology_path = self.manifest.resolve_path(row.get("pathology_path", ""))
        radiology_path = self.manifest.resolve_path(row.get("radiology_path", ""))
        laboratory_path = self.manifest.resolve_path(row.get("laboratory_path", ""))
        clinical_path = self.manifest.resolve_path(row.get("clinical_path", ""))

        pathology = self.pathology_loader.load(pathology_path)
        radiology = self.radiology_loader.load(radiology_path)
        laboratory = self.laboratory_loader.load(laboratory_path)
        clinical = self.clinical_loader.load(clinical_path)

        return MultimodalSample(
            case_id=case_id,
            label=label,
            pathology=pathology,
            radiology=radiology,
            laboratory=laboratory,
            clinical=clinical,
        )

    def get_labels(self) -> list[int]:
        """Get all labels in dataset."""
        return self.cases_df["label"].tolist()

    def get_class_counts(self) -> dict[int, int]:
        """Get count of samples per class."""
        return self.cases_df["label"].value_counts().to_dict()

    def get_modality_availability(self) -> dict[str, float]:
        """Get fraction of cases with each modality available."""
        modality_cols = {
            "pathology": "pathology_path",
            "radiology": "radiology_path",
            "laboratory": "laboratory_path",
            "clinical": "clinical_path",
        }

        availability = {}
        for name, col in modality_cols.items():
            if col in self.cases_df.columns:
                available = self.cases_df[col].notna().sum()
                availability[name] = available / len(self.cases_df)
            else:
                availability[name] = 0.0

        return availability


def collate_multimodal(samples: list[MultimodalSample]) -> dict:
    """Collate function for DataLoader.

    Batches multimodal samples into tensors.

    Args:
        samples: List of MultimodalSample objects

    Returns:
        Dictionary with batched tensors
    """
    case_ids = [s.case_id for s in samples]
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)

    # Stack features (assumes all have same shape after loading)
    pathology = torch.stack([s.pathology.data for s in samples])
    radiology = torch.stack([s.radiology.data for s in samples])
    laboratory = torch.stack([s.laboratory.data for s in samples])
    clinical = torch.stack([s.clinical.data for s in samples])

    # Track validity
    pathology_valid = torch.tensor([s.pathology.valid for s in samples])
    radiology_valid = torch.tensor([s.radiology.valid for s in samples])
    laboratory_valid = torch.tensor([s.laboratory.valid for s in samples])
    clinical_valid = torch.tensor([s.clinical.valid for s in samples])

    return {
        "case_ids": case_ids,
        "labels": labels,
        "pathology": pathology,
        "radiology": radiology,
        "laboratory": laboratory,
        "clinical": clinical,
        "pathology_valid": pathology_valid,
        "radiology_valid": radiology_valid,
        "laboratory_valid": laboratory_valid,
        "clinical_valid": clinical_valid,
    }
