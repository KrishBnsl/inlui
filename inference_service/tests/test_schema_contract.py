import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.request_models import PredictRequest
from app.schemas.response_models import RecommendationItem


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
        "sort_mode",
    }
    assert expected <= set(PredictRequest.model_fields)


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
        "recommendation_score",
        "fit_score",
        "competitiveness_score",
        "institute_score",
        "branch_score",
        "safety_score",
        "safety_margin",
        "confidence_label",
        "recommendation_bucket",
        "uncertainty_lower",
        "uncertainty_upper",
        "explanation",
        "rank_used",
        "rank_type_used",
    }
    assert expected <= set(RecommendationItem.model_fields)
