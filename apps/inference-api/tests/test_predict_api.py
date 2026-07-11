import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_api.routes import _stable_choice_id, router
from inference_api.services.artifact_loader import load_all_artifacts


pytestmark = pytest.mark.local_assets


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.state.artifacts = load_all_artifacts()
    app.state.artifact_error = None
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_main_3000_advanced_6700_top_100_routes_rank_per_row(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "main_rank": 3000,
            "advanced_rank": 6700,
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "round": 6,
            "top_n": 100,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 < len(body["results"]) <= 100
    assert body["total_options"] == len(body["results"])

    iit_rows = [row for row in body["results"] if row["institute_type"] == "IIT"]
    main_rows = [
        row
        for row in body["results"]
        if row["institute_type"] in {"NIT", "IIIT", "GFTI"}
    ]
    assert iit_rows, "Generated candidate universe supports IIT rows for this profile"
    assert main_rows, "Generated candidate universe supports Main-rank institute rows"
    assert all(row["rank_used"] == 6700 for row in iit_rows)
    assert all(row["rank_type_used"] == "JEE_ADVANCED" for row in iit_rows)
    assert all(row["rank_used"] == 3000 for row in main_rows)
    assert all(row["rank_type_used"] == "JEE_MAIN" for row in main_rows)

    available_reach = body["bucket_counts"]["ambitious_reach"]
    returned_reach = body["returned_bucket_counts"]["ambitious_reach"]
    assert available_reach > 0
    assert 0 < returned_reach <= available_reach
    assert returned_reach == body["ambitious_count"]


def test_response_has_honest_metadata_counts_and_interval_semantics(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "main_rank": 25_000,
            "category": "OBC-NCL",
            "gender": "Gender-Neutral",
            "round": 6,
            "top_n": 20,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data_cutoff"] == "2025"
    assert body["prediction_year"] == 2026
    assert body["model_version"]
    assert body["interval_label"] == "90% empirical prediction interval"
    probability_method = body["probability_method"].casefold()
    assert "uncalibrated" in probability_method
    assert "not an individual admission probability" in probability_method
    assert "not an admission guarantee" in body["decision_support_disclaimer"]

    counts = body["candidate_counts"]
    assert counts["universe"] >= counts["after_profile_filters"]
    assert counts["after_profile_filters"] >= counts["after_rank_routing"]
    assert counts["scored"] >= counts["eligible_after_reach_filter"]
    assert counts["eligible_after_reach_filter"] >= counts["returned"]
    assert counts["returned"] == len(body["results"])
    assert sum(body["returned_bucket_counts"].values()) == len(body["results"])

    for row in body["results"]:
        assert row["uncertainty_lower"] <= row["projected_closing_rank"]
        assert row["projected_closing_rank"] <= row["uncertainty_upper"]
        assert row["calibrated_probability_percent"] is None
        assert 0 <= row["admission_probability"] <= 1
        assert row["recommendation_score"] >= 0


def test_api_rejects_malformed_and_extreme_inputs_before_inference(client):
    for payload in (
        {"main_rank": True, "category": "OPEN", "gender": "Gender-Neutral"},
        {"main_rank": 1_200_001, "category": "OPEN", "gender": "Gender-Neutral"},
        {"main_rank": 3000, "category": "GENERAL", "gender": "Gender-Neutral"},
        {"main_rank": 3000, "category": "OPEN", "gender": "Gender-Neutral", "unknown": 1},
        {"main_rank": 3000, "category": "OPEN", "gender": "Gender-Neutral", "pref_quotas": ["LOCAL"]},
        {"main_rank": 3000, "category": "OPEN", "gender": "Gender-Neutral", "mc_enabled": False},
    ):
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422


def test_choice_id_is_stable_across_row_order_but_changes_with_identity():
    first = pd.Series(
        {
            "institute": "NIT Example",
            "institute_type": "NIT",
            "program": "Computer Science",
            "quota": "AI",
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "is_pwd": False,
            "round": 5,
        }
    )
    reordered = first.reindex(list(reversed(first.index)))
    changed = first.copy()
    changed["program"] = "Civil Engineering"

    assert _stable_choice_id(first) == _stable_choice_id(reordered)
    assert _stable_choice_id(first) != _stable_choice_id(changed)
