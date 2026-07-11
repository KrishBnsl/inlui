import pandas as pd

from inference_api.services.prediction_service import filter_universe
from inference_api.services.recommendation_service import build_recommendations


def _choice(institute, inst_type, program, cutoff):
    return {
        "institute": institute,
        "institute_type": inst_type,
        "program": program,
        "category": "OPEN",
        "gender": "Gender-Neutral",
        "quota": "AI",
        "round": 6,
        "predicted_closing_rank": cutoff,
        "opening_rank": max(1, int(cutoff * 0.65)),
        "closing_rank": cutoff,
        "year": 2024,
    }


def _ranked(student_rank, top_n=10):
    choices = pd.DataFrame(
        [
            _choice("National Institute A", "NIT", "Computer Science and Engineering", 2_800),
            _choice("National Institute B", "NIT", "Artificial Intelligence", 5_500),
            _choice("Indian Institute of Information Technology C", "IIIT", "Data Science", 12_000),
            _choice("National Institute D", "NIT", "Electronics and Communication Engineering", 24_000),
            _choice("Indian Institute of Information Technology E", "IIIT", "Information Technology", 32_000),
            _choice("National Institute F", "NIT", "Mechanical Engineering", 52_000),
            _choice("Government Funded Technical Institute G", "GFTI", "Computer Science and Engineering", 76_000),
            _choice("Government Funded Technical Institute H", "GFTI", "Electrical Engineering", 110_000),
            _choice("Government Funded Technical Institute I", "GFTI", "Civil Engineering", 145_000),
            _choice("Government Funded Technical Institute J", "GFTI", "Mechanical Engineering", 185_000),
            _choice("Government Funded Technical Institute K", "GFTI", "Civil Engineering", 240_000),
            _choice("Government Funded Technical Institute L", "GFTI", "Mining Engineering", 320_000),
        ]
    )
    std_map = pd.DataFrame(
        [
            {"institute_type_str": "NIT", "round": 6, "std_dev": 12_000.0},
            {"institute_type_str": "IIIT", "round": 6, "std_dev": 12_000.0},
            {"institute_type_str": "GFTI", "round": 6, "std_dev": 18_000.0},
        ]
    )
    return build_recommendations(choices, student_rank, std_map, 6, mc_enabled=True, top_n=top_n)


def test_top_recommendations_change_by_rank_profile():
    strong = _ranked(2_000)
    mid = _ranked(25_000)
    lower = _ranked(100_000)

    strong_top = list(zip(strong.head(10)["institute"], strong.head(10)["program"]))
    mid_top = list(zip(mid.head(10)["institute"], mid.head(10)["program"]))
    lower_top = list(zip(lower.head(10)["institute"], lower.head(10)["program"]))

    assert strong_top != mid_top
    assert mid_top != lower_top


def test_stronger_rank_gets_more_competitive_cutoffs_on_average():
    strong = _ranked(2_000)
    mid = _ranked(25_000)
    lower = _ranked(100_000)

    assert strong.head(5)["predicted_closing_rank"].mean() < mid.head(5)["predicted_closing_rank"].mean()
    assert mid.head(5)["predicted_closing_rank"].mean() < lower.head(5)["predicted_closing_rank"].mean()


def test_weaker_rank_gets_safer_options_on_average():
    mid = _ranked(25_000)
    lower = _ranked(100_000)

    assert lower.head(5)["admission_probability"].mean() >= mid.head(5)["admission_probability"].mean()


def test_backend_sort_uses_recommendation_score_not_raw_probability():
    ranked = _ranked(25_000)
    scores = ranked["recommendation_score"].tolist()
    probabilities = ranked["admission_probability"].tolist()

    assert scores == sorted(scores, reverse=True)
    assert probabilities != sorted(probabilities, reverse=True)


def test_iit_excluded_without_advanced_rank():
    universe = pd.DataFrame(
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
        ]
    )

    no_advanced = filter_universe(universe, "OPEN", "Gender-Neutral", 6, None, None, None)
    with_advanced = filter_universe(universe, "OPEN", "Gender-Neutral", 6, None, None, 1_000)

    assert "IIT" not in set(no_advanced["institute_type"])
    assert "IIT" in set(with_advanced["institute_type"])
