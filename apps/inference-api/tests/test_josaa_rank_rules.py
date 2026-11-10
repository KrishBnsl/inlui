import pandas as pd
import pytest

from inference_api.services.prediction_service import assign_rank_sources, filter_universe


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
    assert by_type.loc["IIT", "rank_type_used"] == "JEE_ADVANCED"
    assert by_type.loc["NIT", "rank_used"] == 25_000
    assert by_type.loc["IIIT", "rank_used"] == 25_000
    assert by_type.loc["GFTI", "rank_used"] == 25_000
    assert set(by_type.loc[["NIT", "IIIT", "GFTI"], "rank_type_used"]) == {"JEE_MAIN"}


def test_main_rank_is_required_for_non_iit_recommendations():
    with pytest.raises(ValueError, match="main_rank"):
        assign_rank_sources(_universe(), main_rank=None, advanced_rank=1_000)


def test_requested_round_is_exact_and_never_silently_substituted():
    choices = filter_universe(_universe(), "OPEN", "Gender-Neutral", 5, None, None, None)
    assert choices.empty


def test_pwd_and_quota_filters_are_applied():
    universe = _universe().copy()
    universe["is_pwd"] = [False, True, False, True]
    choices = filter_universe(
        universe,
        "OPEN",
        "Gender-Neutral",
        6,
        None,
        None,
        1000,
        is_pwd=True,
        pref_quotas=["AI"],
    )
    assert set(choices["institute_type"]) == {"NIT", "GFTI"}
    assert choices["is_pwd"].all()
    assert set(choices["quota"]) == {"AI"}


def test_pwd_request_is_conservative_when_universe_lacks_pwd_metadata():
    choices = filter_universe(
        _universe(),
        "OPEN",
        "Gender-Neutral",
        6,
        None,
        None,
        1000,
        is_pwd=True,
    )
    assert choices.empty


def test_home_state_quota_is_conservative_without_state_metadata():
    universe = _universe().copy()
    universe.loc[universe.index[1], "quota"] = "HS"
    choices = filter_universe(
        universe,
        "OPEN",
        "Gender-Neutral",
        6,
        None,
        None,
        1000,
        home_state="Delhi",
    )
    assert "HS" not in set(choices["quota"])


def test_home_state_quota_requires_matching_institute_state():
    universe = _universe().copy()
    universe["institute_state"] = ""
    universe.loc[universe.index[1], ["quota", "institute_state"]] = ["HS", "Delhi"]
    universe.loc[universe.index[2], ["quota", "institute_state"]] = ["HS", "Karnataka"]
    choices = filter_universe(
        universe,
        "OPEN",
        "Gender-Neutral",
        6,
        None,
        None,
        1000,
        home_state="delhi",
        pref_quotas=["HS"],
    )
    assert list(choices["institute_type"]) == ["NIT"]


def test_branch_keywords_are_literal_not_regular_expressions():
    choices = filter_universe(
        _universe(), "OPEN", "Gender-Neutral", 6, None, ["["], 1000
    )
    assert choices.empty
