import json

from inference_api.config import REPO_ROOT
from josaa_core.feature_schema import FORBIDDEN_TARGET_DERIVED_FEATURES, MODEL_FEATURES


def test_model_feature_schema_artifact_has_no_target_derived_features():
    schema_path = REPO_ROOT / "research" / "artifacts" / "pre_counselling_feature_schema.json"
    assert schema_path.exists(), "pre_counselling_feature_schema.json must accompany the model"

    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    features = set(payload["features"])

    assert not (features & FORBIDDEN_TARGET_DERIVED_FEATURES)


def test_training_feature_list_has_no_forbidden_target_derived_features():
    assert not (set(MODEL_FEATURES) & FORBIDDEN_TARGET_DERIVED_FEATURES)


def test_training_pipeline_uses_shared_feature_schema():
    train_models_path = REPO_ROOT / "research" / "pipeline" / "train_models.py"
    source = train_models_path.read_text(encoding="utf-8")

    assert "FEATURE_COLS = MODEL_FEATURES" in source
    assert "feature_schema_payload" in source
