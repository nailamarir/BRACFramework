"""Data loaders for each modality in the BRAC framework.

This module provides loaders for:
- Manifest: CSV file linking all modalities per case
- Pathology: WSI features (pre-extracted .pt tensors)
- Radiology: PET/CT features (pre-extracted .pt or .nii.gz)
- Laboratory: Flow cytometry and lab values
- Clinical: Structured clinical data
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Any
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class LoadedSample:
    """Container for a loaded sample from one modality."""
    data: torch.Tensor
    metadata: dict
    valid: bool = True
    error_message: str = ""


class ManifestLoader:
    """Load and manage the case manifest CSV.

    The manifest links all modalities for each case:
    case_id, label, pathology_path, radiology_path, laboratory_path, clinical_path, split
    """

    def __init__(self, manifest_path: Union[str, Path], data_root: Optional[Path] = None):
        """Initialize manifest loader.

        Args:
            manifest_path: Path to manifest CSV file
            data_root: Root directory for relative paths in manifest
        """
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else self.manifest_path.parent

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self._validate_manifest()

        logger.info(f"Loaded manifest with {len(self.df)} cases")

    def _validate_manifest(self):
        """Validate manifest has required columns."""
        required = ["case_id", "label"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Get case info by index."""
        row = self.df.iloc[idx]
        return row.to_dict()

    def get_by_case_id(self, case_id: str) -> dict:
        """Get case info by case ID."""
        matches = self.df[self.df["case_id"] == case_id]
        if len(matches) == 0:
            raise KeyError(f"Case not found: {case_id}")
        return matches.iloc[0].to_dict()

    def get_split(self, split: str) -> pd.DataFrame:
        """Get all cases for a given split (train/validation/test)."""
        if "split" not in self.df.columns:
            logger.warning("No 'split' column in manifest, returning all cases")
            return self.df
        return self.df[self.df["split"] == split]

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path from manifest to absolute path."""
        if pd.isna(relative_path) or relative_path == "":
            return None
        return self.data_root / relative_path

    @property
    def case_ids(self) -> list[str]:
        return self.df["case_id"].tolist()

    @property
    def labels(self) -> list[int]:
        return self.df["label"].tolist()


class PathologyLoader:
    """Load pathology data (WSI features).

    Expects pre-extracted features as PyTorch tensors (.pt files).
    For raw WSI processing, use external tools like CLAM or RetCCL first.
    """

    def __init__(self, feature_dim: int = 768):
        """Initialize pathology loader.

        Args:
            feature_dim: Expected feature dimension (e.g., 768 for ViT-B)
        """
        self.feature_dim = feature_dim

    def load(self, path: Union[str, Path]) -> LoadedSample:
        """Load pathology features from file.

        Args:
            path: Path to .pt file containing extracted features

        Returns:
            LoadedSample with tensor of shape (num_patches, feature_dim) or (feature_dim,)
        """
        if path is None:
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": "missing"},
                valid=False,
                error_message="No pathology path provided"
            )

        path = Path(path)
        if not path.exists():
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message=f"File not found: {path}"
            )

        try:
            data = torch.load(path, map_location="cpu")

            # Handle different formats
            if isinstance(data, dict):
                # Format: {"features": tensor, "coords": tensor, ...}
                features = data.get("features", data.get("embeddings", None))
                if features is None:
                    raise ValueError("No 'features' or 'embeddings' key in dict")
                metadata = {k: v for k, v in data.items() if k != "features"}
            else:
                features = data
                metadata = {}

            # Ensure tensor
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            metadata["source"] = str(path)
            metadata["shape"] = list(features.shape)

            return LoadedSample(data=features, metadata=metadata, valid=True)

        except Exception as e:
            logger.error(f"Error loading pathology from {path}: {e}")
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message=str(e)
            )

    def aggregate_patches(self, features: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """Aggregate patch-level features to slide-level.

        Args:
            features: Shape (num_patches, feature_dim)
            method: Aggregation method ("mean", "max", "attention")

        Returns:
            Aggregated features, shape (feature_dim,)
        """
        if features.dim() == 1:
            return features

        if method == "mean":
            return features.mean(dim=0)
        elif method == "max":
            return features.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class RadiologyLoader:
    """Load radiology data (PET/CT features).

    Supports:
    - Pre-extracted features (.pt files)
    - NIfTI volumes (.nii, .nii.gz) - requires nibabel
    """

    def __init__(self, feature_dim: int = 512):
        """Initialize radiology loader.

        Args:
            feature_dim: Expected feature dimension for pre-extracted features
        """
        self.feature_dim = feature_dim

    def load(self, path: Union[str, Path]) -> LoadedSample:
        """Load radiology data from file.

        Args:
            path: Path to .pt or .nii.gz file

        Returns:
            LoadedSample with tensor
        """
        if path is None:
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": "missing"},
                valid=False,
                error_message="No radiology path provided"
            )

        path = Path(path)
        if not path.exists():
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message=f"File not found: {path}"
            )

        try:
            suffix = "".join(path.suffixes)  # Handle .nii.gz

            if suffix in [".pt", ".pth"]:
                return self._load_features(path)
            elif suffix in [".nii", ".nii.gz"]:
                return self._load_nifti(path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        except Exception as e:
            logger.error(f"Error loading radiology from {path}: {e}")
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message=str(e)
            )

    def _load_features(self, path: Path) -> LoadedSample:
        """Load pre-extracted features."""
        data = torch.load(path, map_location="cpu")

        if isinstance(data, dict):
            features = data.get("features", data.get("embeddings", None))
            if features is None:
                raise ValueError("No 'features' key in dict")
            metadata = {k: v for k, v in data.items() if k != "features"}
        else:
            features = data
            metadata = {}

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        metadata["source"] = str(path)
        metadata["shape"] = list(features.shape)

        return LoadedSample(data=features, metadata=metadata, valid=True)

    def _load_nifti(self, path: Path) -> LoadedSample:
        """Load NIfTI volume (requires nibabel)."""
        try:
            import nibabel as nib
        except ImportError:
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message="nibabel not installed. Run: pip install nibabel"
            )

        img = nib.load(path)
        data = img.get_fdata()

        # Convert to tensor
        tensor = torch.tensor(data, dtype=torch.float32)

        metadata = {
            "source": str(path),
            "shape": list(tensor.shape),
            "affine": img.affine.tolist(),
        }

        return LoadedSample(data=tensor, metadata=metadata, valid=True)


class LaboratoryLoader:
    """Load laboratory data (flow cytometry, CBC, etc.).

    Supports:
    - Pre-extracted features (.pt files)
    - CSV files with marker values
    - FCS files (flow cytometry) - requires fcsparser
    """

    # Standard flow cytometry markers for B-cell NHL
    STANDARD_MARKERS = [
        "CD19", "CD20", "CD5", "CD10", "CD23", "CD200",
        "CD38", "CD138", "kappa", "lambda", "FMC7",
        "CD43", "CD79b", "CD81", "CD103", "CD11c"
    ]

    def __init__(self, feature_dim: int = 64, markers: Optional[list[str]] = None):
        """Initialize laboratory loader.

        Args:
            feature_dim: Expected feature dimension
            markers: List of expected markers (for CSV/FCS validation)
        """
        self.feature_dim = feature_dim
        self.markers = markers or self.STANDARD_MARKERS

    def load(self, path: Union[str, Path]) -> LoadedSample:
        """Load laboratory data from file.

        Args:
            path: Path to .pt, .csv, or .fcs file

        Returns:
            LoadedSample with tensor
        """
        if path is None:
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": "missing"},
                valid=False,
                error_message="No laboratory path provided"
            )

        path = Path(path)
        if not path.exists():
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message=f"File not found: {path}"
            )

        try:
            suffix = path.suffix.lower()

            if suffix in [".pt", ".pth"]:
                return self._load_features(path)
            elif suffix == ".csv":
                return self._load_csv(path)
            elif suffix == ".fcs":
                return self._load_fcs(path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        except Exception as e:
            logger.error(f"Error loading laboratory from {path}: {e}")
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message=str(e)
            )

    def _load_features(self, path: Path) -> LoadedSample:
        """Load pre-extracted features."""
        data = torch.load(path, map_location="cpu")

        if isinstance(data, dict):
            features = data.get("features", data.get("embeddings", None))
            if features is None:
                raise ValueError("No 'features' key in dict")
            metadata = {k: v for k, v in data.items() if k != "features"}
        else:
            features = data
            metadata = {}

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        metadata["source"] = str(path)

        return LoadedSample(data=features, metadata=metadata, valid=True)

    def _load_csv(self, path: Path) -> LoadedSample:
        """Load lab values from CSV."""
        df = pd.read_csv(path)

        # Try to extract marker columns
        available_markers = [m for m in self.markers if m in df.columns]

        if available_markers:
            values = df[available_markers].values
        else:
            # Use all numeric columns
            values = df.select_dtypes(include=[np.number]).values

        if values.size == 0:
            raise ValueError("No numeric data found in CSV")

        # Flatten if single row, otherwise keep as matrix
        if values.shape[0] == 1:
            values = values.flatten()

        tensor = torch.tensor(values, dtype=torch.float32)

        metadata = {
            "source": str(path),
            "markers": available_markers,
            "shape": list(tensor.shape),
        }

        return LoadedSample(data=tensor, metadata=metadata, valid=True)

    def _load_fcs(self, path: Path) -> LoadedSample:
        """Load flow cytometry FCS file (requires fcsparser)."""
        try:
            import fcsparser
        except ImportError:
            return LoadedSample(
                data=torch.zeros(self.feature_dim),
                metadata={"source": str(path)},
                valid=False,
                error_message="fcsparser not installed. Run: pip install fcsparser"
            )

        meta, data = fcsparser.parse(path, reformat_meta=True)

        # Extract marker columns
        available_markers = [m for m in self.markers if m in data.columns]

        if available_markers:
            values = data[available_markers].values
        else:
            values = data.values

        tensor = torch.tensor(values, dtype=torch.float32)

        metadata = {
            "source": str(path),
            "markers": available_markers,
            "shape": list(tensor.shape),
            "num_events": len(data),
        }

        return LoadedSample(data=tensor, metadata=metadata, valid=True)


class ClinicalLoader:
    """Load structured clinical data.

    Expects CSV or JSON with clinical variables:
    - Demographics: age, sex
    - Symptoms: B symptoms, performance status
    - Staging: Ann Arbor stage, IPI score
    - Labs: LDH, beta-2 microglobulin
    """

    # Standard clinical features for NHL
    STANDARD_FEATURES = [
        "age", "sex", "b_symptoms", "ecog_ps", "ann_arbor_stage",
        "ipi_score", "ldh_elevated", "extranodal_sites", "bulky_disease"
    ]

    def __init__(self, features: Optional[list[str]] = None):
        """Initialize clinical loader.

        Args:
            features: List of expected clinical features
        """
        self.features = features or self.STANDARD_FEATURES

    def load(self, path: Union[str, Path]) -> LoadedSample:
        """Load clinical data from file.

        Args:
            path: Path to .csv or .json file

        Returns:
            LoadedSample with tensor
        """
        if path is None:
            return LoadedSample(
                data=torch.zeros(len(self.features)),
                metadata={"source": "missing"},
                valid=False,
                error_message="No clinical path provided"
            )

        path = Path(path)
        if not path.exists():
            return LoadedSample(
                data=torch.zeros(len(self.features)),
                metadata={"source": str(path)},
                valid=False,
                error_message=f"File not found: {path}"
            )

        try:
            suffix = path.suffix.lower()

            if suffix == ".csv":
                return self._load_csv(path)
            elif suffix == ".json":
                return self._load_json(path)
            elif suffix in [".pt", ".pth"]:
                return self._load_features(path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        except Exception as e:
            logger.error(f"Error loading clinical from {path}: {e}")
            return LoadedSample(
                data=torch.zeros(len(self.features)),
                metadata={"source": str(path)},
                valid=False,
                error_message=str(e)
            )

    def _load_csv(self, path: Path) -> LoadedSample:
        """Load clinical data from CSV."""
        df = pd.read_csv(path)

        # Extract expected features
        values = []
        found_features = []

        for feat in self.features:
            if feat in df.columns:
                val = df[feat].iloc[0]
                values.append(self._encode_value(feat, val))
                found_features.append(feat)
            else:
                values.append(0.0)  # Missing value

        tensor = torch.tensor(values, dtype=torch.float32)

        metadata = {
            "source": str(path),
            "features_found": found_features,
            "features_missing": [f for f in self.features if f not in found_features],
        }

        return LoadedSample(data=tensor, metadata=metadata, valid=True)

    def _load_json(self, path: Path) -> LoadedSample:
        """Load clinical data from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        values = []
        found_features = []

        for feat in self.features:
            if feat in data:
                values.append(self._encode_value(feat, data[feat]))
                found_features.append(feat)
            else:
                values.append(0.0)

        tensor = torch.tensor(values, dtype=torch.float32)

        metadata = {
            "source": str(path),
            "features_found": found_features,
            "raw_data": data,
        }

        return LoadedSample(data=tensor, metadata=metadata, valid=True)

    def _load_features(self, path: Path) -> LoadedSample:
        """Load pre-extracted features."""
        data = torch.load(path, map_location="cpu")

        if isinstance(data, dict):
            features = data.get("features", None)
            if features is None:
                raise ValueError("No 'features' key in dict")
            metadata = {k: v for k, v in data.items() if k != "features"}
        else:
            features = data
            metadata = {}

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        metadata["source"] = str(path)

        return LoadedSample(data=features, metadata=metadata, valid=True)

    def _encode_value(self, feature: str, value: Any) -> float:
        """Encode a clinical value to numeric."""
        if pd.isna(value):
            return 0.0

        # Handle categorical encoding
        if feature == "sex":
            return 1.0 if str(value).lower() in ["m", "male", "1"] else 0.0
        elif feature == "b_symptoms":
            return 1.0 if str(value).lower() in ["yes", "true", "1", "b"] else 0.0
        elif feature == "ldh_elevated":
            return 1.0 if str(value).lower() in ["yes", "true", "1", "elevated"] else 0.0
        elif feature == "bulky_disease":
            return 1.0 if str(value).lower() in ["yes", "true", "1"] else 0.0
        elif feature == "ann_arbor_stage":
            # Map I, II, III, IV to 1, 2, 3, 4
            stage_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
            return float(stage_map.get(str(value).lower().strip(), 0))
        else:
            # Numeric value
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
