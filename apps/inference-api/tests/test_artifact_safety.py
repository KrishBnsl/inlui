import hashlib
import json
from pathlib import Path

import pytest

from inference_api.config import settings
from inference_api.services.artifact_loader import (
    ARTIFACT_RECOVERY_COMMAND,
    ArtifactLoadError,
    _load_std_map,
    load_all_artifacts,
)


def _calibration_payload():
    return {
        "schema_version": 1,
        "method": "rolling_origin_oof_absolute_residual",
        "source_split": "rolling_origin_oof",
        "target_coverage": 0.9,
        "calibration_years": [2023, 2024, 2025],
        "model_training_years": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "prediction_year": 2026,
        "dataset_sha256": "a" * 64,
        "seed": 42,
        "global": {"count": 100, "abs_residual_quantile": 1234.0},
        "groups": [],
    }


def _write_calibration(tmp_path: Path, payload: dict) -> dict:
    path = tmp_path / "uncertainty_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "prediction_year": 2026,
        "dataset_sha256": "a" * 64,
        "seed": 42,
        "artifacts": {
            path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        },
    }


def test_final_test_residual_source_is_rejected(tmp_path):
    payload = _calibration_payload()
    payload["source_split"] = "test"
    metadata = _write_calibration(tmp_path, payload)
    with pytest.raises(ArtifactLoadError, match="Final-test residuals"):
        _load_std_map(tmp_path, metadata)


def test_calibration_year_must_precede_prediction_year(tmp_path):
    payload = _calibration_payload()
    payload["calibration_years"].append(2026)
    metadata = _write_calibration(tmp_path, payload)
    with pytest.raises(ArtifactLoadError, match="precede prediction_year"):
        _load_std_map(tmp_path, metadata)


def test_nonpositive_uncertainty_quantile_is_rejected(tmp_path):
    payload = _calibration_payload()
    payload["global"]["abs_residual_quantile"] = 0
    metadata = _write_calibration(tmp_path, payload)
    with pytest.raises(ArtifactLoadError, match="finite and positive"):
        _load_std_map(tmp_path, metadata)


def test_missing_bundle_fails_with_exact_recovery_command(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    with pytest.raises(ArtifactLoadError) as error:
        load_all_artifacts()
    assert ARTIFACT_RECOVERY_COMMAND in str(error.value)
    assert "model_metadata.json" in str(error.value)
