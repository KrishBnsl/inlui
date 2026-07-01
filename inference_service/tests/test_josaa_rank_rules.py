import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.inference_service import assign_rank_sources, filter_universe


def _universe():
    return pd.DataFrame(
        [
            {
                "institute": "Indian Institute of Technology A",
                "institute_type": "IIT",
                "program": "Computer Science and Engineering",
                "category": "OPEN",
                "gender": "Gender-Neutral",
                "quota": "AI",
                "round": 6,
            },
            {
                "institute": "National Institute B",
                "institute_type": "NIT",
                "program": "Computer Science and Engineering",
                "category": "OPEN",
                "gender": "Gender-Neutral",
                "quota": "AI",
                "round": 6,
            },
            {
                "institute": "Indian Institute of Information Technology C",
                "institute_type": "IIIT",
                "program": "Data Science",
                "category": "OPEN",
                "gender": "Gender-Neutral",
                "quota": "AI",
                "round": 6,
            },
            {
                "institute": "Government Funded Technical Institute D",
                "institute_type": "GFTI",
                "program": "Mechanical Engineering",
                "category": "OPEN",
                "gender": "Gender-Neutral",
                "quota": "AI",
                "round": 6,
            },
        ]
    )


def test_iits_are_excluded_without_advanced_rank():
    choices = filter_universe(_universe(), "OPEN", "Gender-Neutral", 6, None, None, None)
    assert "IIT" not in set(choices["institute_type"])


def test_iit_uses_advanced_rank_and_non_iits_use_main_rank_when_both_exist():
    choices = filter_universe(_universe(), "OPEN", "Gender-Neutral", 6, None, None, 1_000)
    ranked_source = assign_rank_sources(choices, main_rank=25_000, advanced_rank=1_000)

    by_type = ranked_source.set_index("institute_type")
    assert by_type.loc["IIT", "rank_used"] == 1_000
    assert by_type.loc["IIT", "rank_type_used"] == "advanced"
    assert by_type.loc["NIT", "rank_used"] == 25_000
    assert by_type.loc["IIIT", "rank_used"] == 25_000
    assert by_type.loc["GFTI", "rank_used"] == 25_000
    assert set(by_type.loc[["NIT", "IIIT", "GFTI"], "rank_type_used"]) == {"main"}


def test_main_rank_is_required_for_non_iit_recommendations():
    with pytest.raises(ValueError, match="main_rank"):
        assign_rank_sources(_universe(), main_rank=None, advanced_rank=1_000)
