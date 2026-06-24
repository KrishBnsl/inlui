"""
Tests for artifact_loader.py.

Uses the actual artifacts in models/model_artifacts/ — these tests require the
repo's model files to be present and are intended to run in CI with the full
model directory mounted.

Run with:
    cd inference_service && pytest tests/test_artifact_loader.py -v
"""

import sys
from pathlib import Path

import pytest

# Allow importing from the inference_service package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.artifact_loader import ArtifactStore, load_all_artifacts
from app.config import settings


@pytest.fixture(scope="module")
def artifacts():
    """Load artifacts once for all tests in this module."""
    return load_all_artifacts()


class TestArtifactStore:
    def test_loads_without_error(self, artifacts):
        """All artifacts should load without raising exceptions."""
        assert artifacts is not None

    def test_model_is_not_none(self, artifacts):
        """The regression model must be present and callable."""
        assert artifacts.model is not None
        assert hasattr(artifacts.model, "predict"), "Model must have a .predict() method"

    def test_prep_pipeline_is_not_none(self, artifacts):
        """The preprocessing pipeline must be present and have .transform()."""
        assert artifacts.prep_pipeline is not None
        assert hasattr(artifacts.prep_pipeline, "transform")

    def test_label_encoders_have_required_keys(self, artifacts):
        """Label encoders must contain at least the four training-time keys."""
        required = {"institute_type", "category", "quota", "gender"}
        missing = required - set(artifacts.label_encoders.keys())
        assert not missing, f"Missing encoder keys: {missing}"

    def test_category_encoder_values(self, artifacts):
        """Known categories must be in the encoder mapping."""
        cat_enc = artifacts.label_encoders["category"]
        for cat in ("OPEN", "OBC-NCL", "EWS", "SC", "ST"):
            assert cat in cat_enc, f"Category '{cat}' missing from encoder"

    def test_universe_not_empty(self, artifacts):
        """Universe must contain at least one row."""
        assert artifacts.universe_size > 0

    def test_universe_has_required_columns(self, artifacts):
        """The universe DataFrame must have the columns needed for filtering."""
        required = {"institute", "program", "category", "gender", "quota", "institute_type"}
        missing = required - set(artifacts.universe.columns)
        assert not missing, f"Universe missing columns: {missing}"

    def test_universe_only_2024_round5(self, artifacts):
        """The universe should be filtered to 2024, Round 5."""
        df = artifacts.universe
        if "year" in df.columns:
            assert (df["year"] == 2024).all(), "Universe contains rows outside 2024"
        if "round" in df.columns:
            assert (df["round"] == 5).all(), "Universe contains rows outside Round 5"

    def test_std_map_not_empty(self, artifacts):
        """The std_map must have at least one row."""
        assert len(artifacts.std_map) > 0

    def test_std_map_has_required_columns(self, artifacts):
        """std_map must have institute_type_str and std_dev columns."""
        required = {"institute_type_str", "std_dev"}
        missing = required - set(artifacts.std_map.columns)
        assert not missing, f"std_map missing columns: {missing}"

    def test_std_dev_positive(self, artifacts):
        """All std_dev values must be positive (floor applied during loading)."""
        assert (artifacts.std_map["std_dev"] >= 100.0).all()
