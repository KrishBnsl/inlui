import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from shared.josaa_core.feature_schema import FORBIDDEN_TARGET_DERIVED_FEATURES, MODEL_FEATURES


def test_model_feature_schema_artifact_has_no_target_derived_features():
    schema_path = Path(__file__).parents[2] / "models" / "model_artifacts" / "feature_schema.json"
    assert schema_path.exists(), "feature_schema.json must be saved with model artifacts"

    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    features = set(payload["features"])

    assert not (features & FORBIDDEN_TARGET_DERIVED_FEATURES)


def test_training_feature_list_has_no_forbidden_target_derived_features():
    assert not (set(MODEL_FEATURES) & FORBIDDEN_TARGET_DERIVED_FEATURES)


def test_training_pipeline_uses_shared_feature_schema():
    train_models_path = Path(__file__).parents[2] / "models" / "train_models.py"
    source = train_models_path.read_text(encoding="utf-8")

    assert "FEATURE_COLS = MODEL_FEATURES" in source
    assert "feature_schema_payload" in source
