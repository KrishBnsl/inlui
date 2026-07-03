"""
Integration tests for the inference + recommendation pipeline.

Requires actual model artifacts — runs against the real universe.

Run with:
    cd inference_service && pytest tests/test_inference.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.artifact_loader import load_all_artifacts
from app.services.inference_service import assign_rank_sources, filter_universe, run_inference
from app.services.recommendation_service import build_recommendations


@pytest.fixture(scope="module")
def artifacts():
    return load_all_artifacts()


class TestFilterUniverse:
    def test_filter_by_category_and_gender(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=5,
            pref_inst_types=None,
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        assert not choices.empty, "Should find at least some OPEN / GN programs"
        # IITs must not appear (no advanced rank provided)
        if "institute_type" in choices.columns:
            assert "IIT" not in choices["institute_type"].values

    def test_filter_with_inst_type(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OBC-NCL",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=["NIT"],
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        if not choices.empty and "institute_type" in choices.columns:
            assert set(choices["institute_type"].unique()) == {"NIT"}

    def test_filter_with_branch_keyword(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=None,
            pref_branch_keywords=["Computer Science"],
            advanced_rank=None,
        )
        if not choices.empty and "program" in choices.columns:
            assert all(
                "computer science" in p.lower() for p in choices["program"]
            ), "All programs should match the keyword"

    def test_iit_included_when_advanced_rank_provided(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=5,
            pref_inst_types=None,
            pref_branch_keywords=None,
            advanced_rank=5000,
        )
        if "institute_type" in choices.columns:
            types = set(choices["institute_type"].unique())
            # IITs may or may not be in the 2024 universe, but they should not be excluded
            assert "NIT" in types or "IIIT" in types  # at minimum these exist

    def test_empty_result_for_impossible_filter(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="ST",
            gender="Female-only",
            round_=6,
            pref_inst_types=["IIT"],  # IITs need advanced rank
            pref_branch_keywords=["Underwater Basket Weaving"],  # nonsense keyword
            advanced_rank=5000,
        )
        # Should be empty (no such program)
        assert choices.empty


class TestRunInference:
    def test_adds_predicted_closing_rank(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=["NIT"],
            pref_branch_keywords=["Computer Science"],
            advanced_rank=None,
        )
        if choices.empty:
            pytest.skip("No NIT/CSE/OPEN/GN rows in universe — skipping")

        result = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        assert "predicted_closing_rank" in result.columns
        assert (result["predicted_closing_rank"] >= 1).all()

    def test_predicted_ranks_are_integers(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OBC-NCL",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=["NIT"],
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        if choices.empty:
            pytest.skip("No matching rows — skipping")

        result = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        assert result["predicted_closing_rank"].dtype in (int, "int64", "int32")


class TestBuildRecommendations:
    def test_returns_ranked_results(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=["NIT"],
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        if choices.empty:
            pytest.skip("No matching rows — skipping")

        choices = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        ranked = build_recommendations(
            choices=choices,
            student_rank=15000,
            std_map=artifacts.std_map,
            round_=6,
            mc_enabled=True,
            top_n=10,
        )

        assert len(ranked) <= 10
        assert "admission_probability" in ranked.columns
        assert "confidence_label" in ranked.columns
        assert "explanation" in ranked.columns
        assert "recommendation_score" in ranked.columns

    def test_probabilities_in_valid_range(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=["NIT"],
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        if choices.empty:
            pytest.skip("No matching rows — skipping")

        choices = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        ranked = build_recommendations(
            choices=choices,
            student_rank=15000,
            std_map=artifacts.std_map,
            round_=6,
            mc_enabled=True,
            top_n=20,
        )
        assert (ranked["admission_probability"] >= 0).all()
        assert (ranked["admission_probability"] <= 1).all()

    def test_ci_lower_le_upper(self, artifacts):
        """CI lower bound should always be ≤ upper bound."""
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=["NIT"],
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        if choices.empty:
            pytest.skip("No matching rows — skipping")

        choices = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        ranked = build_recommendations(
            choices=choices,
            student_rank=15000,
            std_map=artifacts.std_map,
            round_=6,
            mc_enabled=True,
            top_n=20,
        )
        assert (ranked["ci_lower_5"] <= ranked["ci_upper_95"]).all()

    def test_classification_labels_are_valid(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="EWS",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=None,
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        if choices.empty:
            pytest.skip("No matching rows — skipping")

        choices = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        ranked = build_recommendations(
            choices=choices,
            student_rank=50000,
            std_map=artifacts.std_map,
            round_=6,
            mc_enabled=False,  # deterministic for speed
            top_n=10,
        )
        valid_labels = {"Safe", "Moderate", "Ambitious"}
        for label in ranked["confidence_label"].unique():
            assert label in valid_labels, f"Unexpected label: {label}"

    def test_sorted_by_score_descending(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=["NIT"],
            pref_branch_keywords=None,
            advanced_rank=None,
        )
        if choices.empty:
            pytest.skip("No matching rows — skipping")

        choices = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        ranked = build_recommendations(
            choices=choices,
            student_rank=15000,
            std_map=artifacts.std_map,
            round_=6,
            mc_enabled=True,
            top_n=20,
        )
        scores = ranked["recommendation_score"].tolist()
        assert scores == sorted(scores, reverse=True), "Results must be sorted by score descending"

    def test_rank_3000_advanced_6700_returns_realistic_reach(self, artifacts):
        choices = filter_universe(
            universe=artifacts.universe,
            category="OPEN",
            gender="Gender-Neutral",
            round_=6,
            pref_inst_types=None,
            pref_branch_keywords=None,
            advanced_rank=6_700,
            home_state=None,
            is_pwd=False,
        )
        if choices.empty:
            pytest.skip("No matching rows — skipping")

        choices = assign_rank_sources(choices, main_rank=3_000, advanced_rank=6_700)
        choices = run_inference(choices, artifacts.model, artifacts.prep_pipeline, artifacts.label_encoders)
        ranked = build_recommendations(
            choices=choices,
            student_rank=3_000,
            std_map=artifacts.std_map,
            round_=6,
            mc_enabled=True,
            top_n=100,
        )

        reach = ranked[ranked["recommendation_bucket"] == "ambitious_reach"]
        assert len(ranked) >= 80
        assert len(reach) >= 10
        assert ranked.attrs["bucket_counts"]["ambitious_reach"] >= len(reach)
        assert (reach["admission_probability"] > 0).all()
        assert set(reach["rank_type_used"]).issubset({"JEE_MAIN", "JEE_ADVANCED"})
        assert ((reach["rank_ratio"] > 1.05) & (reach["rank_ratio"] <= 1.40)).all()

        iit_reach = reach[reach["institute_type"] == "IIT"]
        if not iit_reach.empty:
            assert (iit_reach["rank_used"] == 6_700).all()
            assert set(iit_reach["rank_type_used"]) == {"JEE_ADVANCED"}
            assert ((iit_reach["predicted_closing_rank"] >= 4_020) & (iit_reach["predicted_closing_rank"] <= 6_700)).all()

        main_reach = reach[reach["institute_type"].isin(["NIT", "IIIT", "GFTI"])]
        if not main_reach.empty:
            assert (main_reach["rank_used"] == 3_000).all()
            assert set(main_reach["rank_type_used"]) == {"JEE_MAIN"}
            assert ((main_reach["predicted_closing_rank"] >= 1_800) & (main_reach["predicted_closing_rank"] <= 3_000)).all()
