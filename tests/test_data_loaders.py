"""Unit tests for data loaders."""

import pytest
import torch
import tempfile
import pandas as pd
import json
from pathlib import Path

from brac.data.loaders import (
    ManifestLoader,
    PathologyLoader,
    RadiologyLoader,
    LaboratoryLoader,
    ClinicalLoader,
    LoadedSample,
)
from brac.data.dataset import NHLMultimodalDataset, MultimodalSample


class TestManifestLoader:
    """Tests for ManifestLoader."""

    def test_load_manifest(self, tmp_path):
        """Should load a valid manifest CSV."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "case_id,label,pathology_path,split\n"
            "CASE001,0,path/features.pt,train\n"
            "CASE002,1,path/features2.pt,test\n"
        )

        loader = ManifestLoader(manifest_path)

        assert len(loader) == 2
        assert loader.case_ids == ["CASE001", "CASE002"]
        assert loader.labels == [0, 1]

    def test_get_split(self, tmp_path):
        """Should filter by split."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "case_id,label,split\n"
            "CASE001,0,train\n"
            "CASE002,1,train\n"
            "CASE003,2,test\n"
        )

        loader = ManifestLoader(manifest_path)
        train_df = loader.get_split("train")

        assert len(train_df) == 2

    def test_missing_required_columns(self, tmp_path):
        """Should raise error for missing required columns."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text("some_column,other\na,b\n")

        with pytest.raises(ValueError, match="missing required columns"):
            ManifestLoader(manifest_path)


class TestPathologyLoader:
    """Tests for PathologyLoader."""

    def test_load_tensor(self, tmp_path):
        """Should load PyTorch tensor file."""
        features = torch.randn(100, 768)
        pt_path = tmp_path / "features.pt"
        torch.save(features, pt_path)

        loader = PathologyLoader(feature_dim=768)
        result = loader.load(pt_path)

        assert result.valid
        assert result.data.shape == (100, 768)

    def test_load_dict_format(self, tmp_path):
        """Should load dict with 'features' key."""
        features = torch.randn(50, 768)
        data = {"features": features, "coords": torch.randn(50, 2)}
        pt_path = tmp_path / "features.pt"
        torch.save(data, pt_path)

        loader = PathologyLoader()
        result = loader.load(pt_path)

        assert result.valid
        assert result.data.shape == (50, 768)
        assert "coords" in result.metadata

    def test_missing_file(self, tmp_path):
        """Should handle missing file gracefully."""
        loader = PathologyLoader(feature_dim=768)
        result = loader.load(tmp_path / "nonexistent.pt")

        assert not result.valid
        assert result.data.shape == (768,)
        assert "not found" in result.error_message.lower()

    def test_none_path(self):
        """Should handle None path."""
        loader = PathologyLoader(feature_dim=768)
        result = loader.load(None)

        assert not result.valid

    def test_aggregate_patches(self):
        """Should aggregate patch features."""
        loader = PathologyLoader()
        features = torch.randn(100, 768)

        mean_agg = loader.aggregate_patches(features, method="mean")
        max_agg = loader.aggregate_patches(features, method="max")

        assert mean_agg.shape == (768,)
        assert max_agg.shape == (768,)


class TestRadiologyLoader:
    """Tests for RadiologyLoader."""

    def test_load_tensor(self, tmp_path):
        """Should load PyTorch tensor file."""
        features = torch.randn(512)
        pt_path = tmp_path / "features.pt"
        torch.save(features, pt_path)

        loader = RadiologyLoader(feature_dim=512)
        result = loader.load(pt_path)

        assert result.valid
        assert result.data.shape == (512,)

    def test_missing_file(self, tmp_path):
        """Should handle missing file gracefully."""
        loader = RadiologyLoader(feature_dim=512)
        result = loader.load(tmp_path / "nonexistent.pt")

        assert not result.valid


class TestLaboratoryLoader:
    """Tests for LaboratoryLoader."""

    def test_load_csv(self, tmp_path):
        """Should load CSV with marker values."""
        csv_path = tmp_path / "flow.csv"
        csv_path.write_text("CD19,CD20,CD5,CD10\n0.95,0.89,0.12,0.78\n")

        loader = LaboratoryLoader()
        result = loader.load(csv_path)

        assert result.valid
        assert len(result.data) == 4
        assert "CD19" in result.metadata.get("markers", [])

    def test_load_tensor(self, tmp_path):
        """Should load PyTorch tensor file."""
        features = torch.randn(64)
        pt_path = tmp_path / "features.pt"
        torch.save(features, pt_path)

        loader = LaboratoryLoader(feature_dim=64)
        result = loader.load(pt_path)

        assert result.valid
        assert result.data.shape == (64,)


class TestClinicalLoader:
    """Tests for ClinicalLoader."""

    def test_load_csv(self, tmp_path):
        """Should load clinical CSV."""
        csv_path = tmp_path / "clinical.csv"
        csv_path.write_text("age,sex,b_symptoms,ecog_ps,ann_arbor_stage\n65,M,yes,1,III\n")

        loader = ClinicalLoader()
        result = loader.load(csv_path)

        assert result.valid
        assert result.data[0] == 65.0  # age
        assert result.data[1] == 1.0   # sex (male)
        assert result.data[2] == 1.0   # b_symptoms (yes)
        assert result.data[4] == 3.0   # ann_arbor_stage (III)

    def test_load_json(self, tmp_path):
        """Should load clinical JSON."""
        json_path = tmp_path / "clinical.json"
        data = {
            "age": 58,
            "sex": "female",
            "b_symptoms": "no",
            "ipi_score": 2,
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        loader = ClinicalLoader()
        result = loader.load(json_path)

        assert result.valid
        assert result.data[0] == 58.0  # age
        assert result.data[1] == 0.0   # sex (female)
        assert result.data[2] == 0.0   # b_symptoms (no)

    def test_encoding_values(self):
        """Should correctly encode categorical values."""
        loader = ClinicalLoader()

        assert loader._encode_value("sex", "M") == 1.0
        assert loader._encode_value("sex", "female") == 0.0
        assert loader._encode_value("b_symptoms", "yes") == 1.0
        assert loader._encode_value("b_symptoms", "no") == 0.0
        assert loader._encode_value("ann_arbor_stage", "IV") == 4.0
        assert loader._encode_value("ann_arbor_stage", "II") == 2.0


class TestNHLMultimodalDataset:
    """Tests for NHLMultimodalDataset."""

    def test_dataset_initialization(self, tmp_path):
        """Should initialize dataset from manifest."""
        # Create manifest
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "case_id,label,pathology_path,radiology_path,laboratory_path,clinical_path,split\n"
            "CASE001,0,path/p.pt,path/r.pt,path/l.pt,path/c.csv,train\n"
            "CASE002,1,path/p2.pt,path/r2.pt,path/l2.pt,path/c2.csv,test\n"
        )

        dataset = NHLMultimodalDataset(manifest_path)

        assert len(dataset) == 2
        assert dataset.get_labels() == [0, 1]

    def test_dataset_split_filter(self, tmp_path):
        """Should filter by split."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "case_id,label,split\n"
            "CASE001,0,train\n"
            "CASE002,1,train\n"
            "CASE003,2,test\n"
        )

        train_dataset = NHLMultimodalDataset(manifest_path, split="train")
        test_dataset = NHLMultimodalDataset(manifest_path, split="test")

        assert len(train_dataset) == 2
        assert len(test_dataset) == 1

    def test_getitem_returns_multimodal_sample(self, tmp_path):
        """Should return MultimodalSample."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text("case_id,label\nCASE001,0\n")

        dataset = NHLMultimodalDataset(manifest_path)
        sample = dataset[0]

        assert isinstance(sample, MultimodalSample)
        assert sample.case_id == "CASE001"
        assert sample.label == 0

    def test_get_class_counts(self, tmp_path):
        """Should return class distribution."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "case_id,label\n"
            "C1,0\nC2,0\nC3,1\nC4,2\nC5,2\nC6,2\n"
        )

        dataset = NHLMultimodalDataset(manifest_path)
        counts = dataset.get_class_counts()

        assert counts[0] == 2
        assert counts[1] == 1
        assert counts[2] == 3
