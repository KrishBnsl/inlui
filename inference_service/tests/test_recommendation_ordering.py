import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.recommendation_service import _bucket, build_recommendations


def _candidate(name, inst_type, program, cutoff):
    return {
        "institute": name,
        "institute_type": inst_type,
        "program": program,
        "category": "OPEN",
        "gender": "Gender-Neutral",
        "quota": "AI",
        "round": 6,
        "predicted_closing_rank": cutoff,
    }


def test_best_realistic_fit_beats_very_safe_low_desirability_backup():
    choices = pd.DataFrame(
        [
            _candidate("GFTI Backup", "GFTI", "Civil Engineering", 120_000),
            _candidate("NIT Target", "NIT", "Computer Science and Engineering", 28_000),
            _candidate("IIT Reach", "IIT", "Computer Science and Engineering", 8_000),
        ]
    )
    std_map = pd.DataFrame(
        [{"institute_type_str": "__default__", "round": -1, "std_dev": 8_000.0}]
    )

    ranked = build_recommendations(choices, 25_000, std_map, 6, mc_enabled=True, top_n=3)

    assert ranked.iloc[0]["institute"] == "NIT Target"
    assert ranked.iloc[0]["recommendation_bucket"] == "best_realistic"
    assert "safe_backup" in set(ranked["recommendation_bucket"])
    assert "ambitious_reach" in set(ranked["recommendation_bucket"])


def test_explicit_probability_sort_can_still_rank_safest_first():
    choices = pd.DataFrame(
        [
            _candidate("GFTI Backup", "GFTI", "Civil Engineering", 120_000),
            _candidate("NIT Target", "NIT", "Computer Science and Engineering", 28_000),
        ]
    )
    std_map = pd.DataFrame(
        [{"institute_type_str": "__default__", "round": -1, "std_dev": 8_000.0}]
    )

    ranked = build_recommendations(
        choices,
        25_000,
        std_map,
        6,
        mc_enabled=True,
        top_n=2,
        sort_mode="highest_probability",
    )

    assert ranked.iloc[0]["institute"] == "GFTI Backup"


def test_bucket_rules_keep_reach_realistic_and_exclude_impossible():
    assert _bucket(0.95, 35_000, 25_000, 60_000) == "safe_backup"
    assert _bucket(0.65, 3_000, 25_000, 28_000) == "best_realistic"
    assert _bucket(0.30, -5_000, 25_000, 20_000) == "ambitious_reach"
    assert _bucket(0.02, -22_000, 25_000, 3_000) == "unlikely_reach"


def test_realistic_reach_candidates_are_returned_but_impossible_ones_are_not():
    choices = pd.DataFrame(
        [
            _candidate("Safe option", "GFTI", "Civil Engineering", 60_000),
            _candidate("Target option", "NIT", "Electrical Engineering", 28_000),
            _candidate("Reach option", "NIT", "Computer Science and Engineering", 20_000),
            _candidate("Impossible option", "IIT", "Computer Science and Engineering", 3_000),
        ]
    )
    std_map = pd.DataFrame(
        [{"institute_type_str": "__default__", "round": -1, "std_dev": 10_000.0}]
    )

    ranked = build_recommendations(choices, 25_000, std_map, 6, mc_enabled=True, top_n=10)
    by_name = ranked.set_index("institute")["recommendation_bucket"].to_dict()

    assert by_name["Safe option"] == "safe_backup"
    assert by_name["Target option"] == "best_realistic"
    assert by_name["Reach option"] == "ambitious_reach"
    assert "Impossible option" not in by_name
    assert "ambitious_reach" in set(ranked["recommendation_bucket"])
