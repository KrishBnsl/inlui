"""
Tests for artifact_loader.py.

Uses the actual artifacts in research/artifacts/. CI runs this marked suite
only when the manifest-backed local bundle is available.

Run with:
    cd apps/inference-api && pytest tests/test_artifact_loader.py -v
"""

import pytest

from inference_api.services.artifact_loader import load_all_artifacts


pytestmark = pytest.mark.local_assets


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

    def test_selected_estimator_is_self_contained(self, artifacts):
        """The selected baseline consumes the schema DataFrame directly."""
        assert artifacts.prep_pipeline is None
        assert artifacts.artifact_status["preprocessing_pipeline"] == "bundled_with_model"

    def test_selected_baseline_does_not_require_label_encoders(self, artifacts):
        assert artifacts.label_encoders == {}
        assert artifacts.artifact_status["label_encoders"] == "not_required"

    def test_universe_not_empty(self, artifacts):
        """Universe must contain at least one row."""
        assert artifacts.universe_size > 0

    def test_universe_has_required_columns(self, artifacts):
        """The universe DataFrame must have the columns needed for filtering."""
        required = {"institute", "program", "category", "gender", "quota", "institute_type"}
        missing = required - set(artifacts.universe.columns)
        assert not missing, f"Universe missing columns: {missing}"

    def test_universe_is_past_only_prediction_frame(self, artifacts):
        """Every row is constructed for the explicit target year, not copied observed rows."""
        df = artifacts.universe
        assert (df["year"] == artifacts.metadata["prediction_year"]).all()
        assert artifacts.metadata["feature_mode"] == "pre_counselling"
        unavailable = {
            "closing_rank", "opening_rank", "prev_round_closing_rank", "applicants"
        }
        assert not (unavailable & set(df.columns))
        assert df["round"].between(1, 7).all()

    def test_std_map_not_empty(self, artifacts):
        """The std_map must have at least one row."""
        assert len(artifacts.std_map) > 0

    def test_std_map_has_required_columns(self, artifacts):
        """Uncertainty map exposes both empirical PI radius and MC scale."""
        required = {"institute_type_str", "std_dev", "uncertainty_radius", "count"}
        missing = required - set(artifacts.std_map.columns)
        assert not missing, f"std_map missing columns: {missing}"

    def test_std_dev_positive(self, artifacts):
        assert (artifacts.std_map["std_dev"] > 0).all()
        assert (artifacts.std_map["uncertainty_radius"] > 0).all()

    def test_uncertainty_is_rolling_origin_oof_not_final_test(self, artifacts):
        calibration = artifacts.metadata["uncertainty"]
        assert calibration["source_split"] == "rolling_origin_oof"
        assert calibration["method"] == "rolling_origin_oof_absolute_residual"
        assert max(calibration["calibration_years"]) < calibration["prediction_year"]

    def test_feature_schema_is_explicitly_pre_counselling(self, artifacts):
        assert artifacts.feature_schema == artifacts.metadata["features"]
        assert artifacts.feature_schema == ["last_closing_rank"]
