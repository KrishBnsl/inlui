from inference_api.schemas.request_models import PredictRequest
from inference_api.schemas.response_models import RecommendationItem, RecommendResponse


def test_predict_request_matches_frontend_contract():
    expected = {
        "main_rank",
        "advanced_rank",
        "category",
        "home_state",
        "gender",
        "is_pwd",
        "round",
        "top_n",
        "pref_inst_types",
        "pref_branch_keywords",
        "pref_quotas",
        "sort_mode",
    }
    assert expected <= set(PredictRequest.model_fields)


def test_predict_request_defaults_to_hundred_recommendations():
    assert PredictRequest.model_fields["top_n"].default == 100


def test_recommendation_item_matches_typescript_contract():
    expected = {
        "id",
        "institute_name",
        "institute_type",
        "program_name",
        "quota_applied",
        "category",
        "projected_closing_rank",
        "probability_percent",
        "admission_probability",
        "model_probability_percent",
        "calibrated_probability_percent",
        "recommendation_score",
        "fit_score",
        "competitiveness_score",
        "institute_score",
        "branch_score",
        "safety_score",
        "safety_margin",
        "rank_ratio",
        "confidence_label",
        "recommendation_bucket",
        "uncertainty_lower",
        "uncertainty_upper",
        "explanation",
        "rank_used",
        "rank_type_used",
    }
    assert expected <= set(RecommendationItem.model_fields)


def test_legacy_probability_fields_describe_uncalibrated_likelihood():
    for field_name in (
        "probability_percent",
        "admission_probability",
        "model_probability_percent",
    ):
        description = RecommendationItem.model_fields[field_name].description
        assert description is not None
        assert "uncalibrated admission-likelihood estimate" in description.lower()

    compatibility_description = RecommendationItem.model_fields[
        "admission_probability"
    ].description
    assert compatibility_description is not None
    assert "retained for API compatibility" in compatibility_description
    assert "not an individual admission probability" in compatibility_description


def test_recommend_response_includes_bucket_counts():
    assert "bucket_counts" in RecommendResponse.model_fields
    assert "institute_type_counts" in RecommendResponse.model_fields
    assert "rank_window_debug" in RecommendResponse.model_fields
    assert "candidate_counts" in RecommendResponse.model_fields
    assert "returned_bucket_counts" in RecommendResponse.model_fields
    assert "data_cutoff" in RecommendResponse.model_fields
    assert "model_version" in RecommendResponse.model_fields
    assert "prediction_year" in RecommendResponse.model_fields
    assert "interval_label" in RecommendResponse.model_fields


def test_recommend_response_defaults_do_not_claim_outcome_calibration():
    method = RecommendResponse.model_fields["probability_method"].default
    disclaimer = RecommendResponse.model_fields["decision_support_disclaimer"].default

    assert "Uncalibrated admission-likelihood estimate" in method
    assert "not an individual admission probability" in method
    assert "uncalibrated admission-likelihood estimate" in disclaimer
    assert "not an individual admission probability" in disclaimer
