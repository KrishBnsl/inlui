import pandas as pd
import pytest

from inference_api.services.explanation_service import (
    attach_explanations,
    generate_explanation,
)
from inference_api.services.recommendation_service import build_recommendations


@pytest.mark.parametrize(
    ("confidence_label", "predicted_closing_rank"),
    [
        ("Safe", 12_000),
        ("Moderate", 10_500),
        ("Moderate", 9_500),
        ("Ambitious", 8_000),
    ],
)
def test_explanations_label_estimate_as_uncalibrated(
    confidence_label: str, predicted_closing_rank: int
):
    explanation = generate_explanation(
        pd.Series(
            {
                "institute": "Test Institute",
                "program": "Test Programme",
                "predicted_closing_rank": predicted_closing_rank,
                "admission_probability": 0.625,
                "confidence_label": confidence_label,
            }
        ),
        student_rank=10_000,
    )

    assert "62.5%" in explanation
    assert "uncalibrated admission-likelihood estimate" in explanation
    assert "simulated closing cutoffs" in explanation
    assert "chance of admission" not in explanation
    assert "giving a" not in explanation
    assert "yielding a" not in explanation


def test_outside_cutoff_explanation_uses_unambiguous_rank_language():
    explanation = generate_explanation(
        pd.Series(
            {
                "institute": "Test Institute",
                "program": "Test Programme",
                "predicted_closing_rank": 9_500,
                "admission_probability": 0.4,
                "confidence_label": "Moderate",
            }
        ),
        student_rank=10_000,
    )

    assert "500 places outside the predicted cutoff" in explanation
    assert "below the expected cutoff" not in explanation


def test_mixed_rank_rows_use_their_routed_rank_in_explanations():
    recommendations = pd.DataFrame(
        [
            {
                "institute": "Test NIT",
                "program": "Civil Engineering",
                "predicted_closing_rank": 3_500,
                "admission_probability": 0.8,
                "confidence_label": "Safe",
                "rank_used": 3_000,
            },
            {
                "institute": "Test IIT",
                "program": "Mechanical Engineering",
                "predicted_closing_rank": 6_500,
                "admission_probability": 0.2,
                "confidence_label": "Ambitious",
                "rank_used": 6_700,
            },
        ]
    )

    explained = attach_explanations(recommendations, student_rank=4_850)

    assert "500 places beyond your rank" in explained.loc[0, "explanation"]
    assert "200 ranks tighter than your rank" in explained.loc[1, "explanation"]
    assert "1,350" not in explained.loc[0, "explanation"]
    assert "1,650" not in explained.loc[1, "explanation"]


def test_attach_explanations_falls_back_for_legacy_rows_without_rank_used():
    recommendations = pd.DataFrame(
        [
            {
                "institute": "Test NIT",
                "program": "Civil Engineering",
                "predicted_closing_rank": 3_500,
                "admission_probability": 0.8,
                "confidence_label": "Safe",
            }
        ]
    )

    explained = attach_explanations(recommendations, student_rank=3_000)

    assert "500 places beyond your rank" in explained.loc[0, "explanation"]


def test_build_recommendations_explains_mixed_main_and_advanced_ranks():
    choices = pd.DataFrame(
        [
            {
                "institute": "Test NIT",
                "institute_type": "NIT",
                "program": "Civil Engineering",
                "predicted_closing_rank": 3_500,
                "rank_used": 3_000,
                "rank_type_used": "JEE_MAIN",
            },
            {
                "institute": "Test IIT",
                "institute_type": "IIT",
                "program": "Mechanical Engineering",
                "predicted_closing_rank": 6_500,
                "rank_used": 6_700,
                "rank_type_used": "JEE_ADVANCED",
            },
        ]
    )
    uncertainty = pd.DataFrame(
        [{"institute_type_str": "__default__", "round": -1, "std_dev": 500.0}]
    )

    ranked = build_recommendations(
        choices,
        student_rank=3_000,
        std_map=uncertainty,
        round_=6,
        mc_enabled=False,
        top_n=2,
    ).set_index("institute")

    assert "500 places beyond your rank" in ranked.loc["Test NIT", "explanation"]
    assert "200 ranks tighter than your rank" in ranked.loc["Test IIT", "explanation"]
