import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.routes import router


def _client_with_artifacts(artifacts):
    app = FastAPI()
    app.state.artifacts = artifacts
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_health_reports_each_required_artifact_missing():
    client = _client_with_artifacts(None)

    response = client.get("/api/v1/health")
    body = response.json()

    assert response.status_code == 200
    assert body["artifacts_loaded"] is False
    assert body["artifact_status"]["model"] == "missing"
    assert body["artifact_status"]["preprocessing_pipeline"] == "missing"
    assert body["artifact_status"]["label_encoders"] == "missing"
    assert body["artifact_status"]["feature_schema"] == "missing"
    assert body["artifact_status"]["residual_uncertainty"] == "missing"
    assert body["artifact_status"]["program_universe"] == "missing"


def test_predict_fails_clearly_when_artifacts_are_missing():
    client = _client_with_artifacts(None)

    response = client.post(
        "/api/v1/predict",
        json={
            "main_rank": 25_000,
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "home_state": "Delhi",
        },
    )

    assert response.status_code == 503
    assert "artifacts" in response.json()["detail"].lower()
