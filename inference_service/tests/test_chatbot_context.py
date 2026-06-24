"""
Tests for chatbot_context_service.py — pure unit tests, no ML artifacts needed.

Run with:
    cd inference_service && pytest tests/test_chatbot_context.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.chatbot_context_service import build_context_block, MODEL_NOTES


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _minimal_output():
    """Minimal recommendation output dict."""
    return {
        "total_options": 5,
        "safest_choice": "NIT Trichy — Computer Science and Engineering",
        "top_upgrade": "NIT Surathkal — Computer Science and Engineering",
        "most_ambitious": "NIT Warangal — Computer Science and Engineering",
        "safe_count": 2,
        "moderate_count": 2,
        "ambitious_count": 1,
        "results": [
            {
                "id": "0",
                "institute_name": "National Institute of Technology Trichy",
                "institute_type": "NIT",
                "program_name": "Computer Science and Engineering",
                "quota_applied": "AI",
                "category": "OPEN",
                "probability_percent": 87.3,
                "projected_closing_rank": 1265,
                "uncertainty_lower": 1100,
                "uncertainty_upper": 1450,
                "safety_margin": 265,
                "confidence_label": "Safe",
                "explanation": "NIT Trichy — CSE is a Safe choice...",
                "recommendation_score": 94.2,
                "historical_data": [],
            },
            {
                "id": "1",
                "institute_name": "National Institute of Technology Surathkal",
                "institute_type": "NIT",
                "program_name": "Computer Science and Engineering",
                "quota_applied": "AI",
                "category": "OPEN",
                "probability_percent": 58.1,
                "projected_closing_rank": 2880,
                "uncertainty_lower": 2500,
                "uncertainty_upper": 3300,
                "safety_margin": -120,
                "confidence_label": "Moderate",
                "explanation": "NIT Surathkal — CSE is a Moderate choice...",
                "recommendation_score": 71.3,
                "historical_data": [],
            },
        ],
    }


# ── Tests ────────────────────────────────────────────────────────────────────────

class TestBuildContextBlock:
    def test_no_output_returns_no_results_block(self):
        block, facts = build_context_block(None)
        assert "No recommendation results" in block
        assert facts == {}

    def test_no_output_includes_model_notes(self):
        block, _ = build_context_block(None)
        assert "Ridge regression" in block

    def test_with_output_returns_non_empty_string(self):
        block, facts = build_context_block(_minimal_output())
        assert len(block) > 100

    def test_with_output_contains_institute_name(self):
        block, _ = build_context_block(_minimal_output())
        assert "National Institute of Technology Trichy" in block

    def test_with_output_contains_probability(self):
        block, _ = build_context_block(_minimal_output())
        assert "87.3" in block

    def test_with_output_contains_confidence_label(self):
        block, _ = build_context_block(_minimal_output())
        assert "Safe" in block

    def test_with_output_contains_model_notes(self):
        block, _ = build_context_block(_minimal_output())
        assert "Ridge regression" in block

    def test_key_facts_have_model_type(self):
        _, facts = build_context_block(_minimal_output())
        assert "model_type" in facts
        assert "Ridge" in facts["model_type"]

    def test_key_facts_have_uncertainty_method(self):
        _, facts = build_context_block(_minimal_output())
        assert "uncertainty_method" in facts
        assert "Monte Carlo" in facts["uncertainty_method"]

    def test_key_facts_counts_match(self):
        _, facts = build_context_block(_minimal_output())
        assert facts["safe_count"] == 2
        assert facts["moderate_count"] == 2
        assert facts["ambitious_count"] == 1

    def test_empty_results_list_doesnt_crash(self):
        output = {
            "total_options": 0,
            "safest_choice": "—",
            "top_upgrade": "—",
            "most_ambitious": "—",
            "safe_count": 0,
            "moderate_count": 0,
            "ambitious_count": 0,
            "results": [],
        }
        block, facts = build_context_block(output)
        assert isinstance(block, str)

    def test_partial_output_doesnt_crash(self):
        """Even a dict with only partial keys should not crash."""
        block, facts = build_context_block({"results": []})
        assert isinstance(block, str)

    def test_context_block_mentions_uncertainty_ci(self):
        """The block must explain what CI means to help the chatbot answer questions."""
        block, _ = build_context_block(_minimal_output())
        assert "90%" in block or "confidence" in block.lower()

    def test_model_notes_constant_contains_all_labels(self):
        """Static MODEL_NOTES must mention all three classification labels."""
        assert "Safe" in MODEL_NOTES
        assert "Moderate" in MODEL_NOTES
        assert "Ambitious" in MODEL_NOTES

    def test_model_notes_mentions_limitations(self):
        assert "Limitations" in MODEL_NOTES or "cannot" in MODEL_NOTES
