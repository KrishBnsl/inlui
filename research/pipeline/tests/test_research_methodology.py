from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.pipeline import research_evaluation
from research.pipeline.research_evaluation import (
    ResearchAssetError,
    dataset_quality_payload,
    evaluate_ablations,
    evaluate_in_round_diagnostic,
    evaluate_with_isolated_final_holdout,
    generate_strict_figures,
    load_dataset,
    ridge_pipeline,
    select_model_before_final_test,
    uncertainty_payload,
)
from josaa_core.baselines import LastObservationRegressor
from josaa_core.feature_schema import (
    FeatureAvailabilityMode,
    HISTORICAL_STAT_FEATURES,
    PRE_COUNSELLING_FEATURES,
    PAST_ONLY_FREQUENCY_FEATURES,
    assert_features_available,
    feature_schema_payload,
    features_for_mode,
)
from josaa_core.research_features import (
    OBSERVED_ONLY_COLUMNS,
    build_longitudinal_features,
    build_target_year_frame,
    match_target_outcomes,
)


@pytest.fixture()
def observed_cutoffs() -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(2018, 2023):
        for institute, institute_type in (("Institute A", "IIT"), ("Institute B", "NIT")):
            for round_number in (1, 2):
                rows.append(
                    {
                        "year": year,
                        "round": round_number,
                        "institute": institute,
                        "institute_type": institute_type,
                        "program": "Computer Science",
                        "category": "OPEN",
                        "quota": "AI",
                        "gender": "Gender-Neutral",
                        "is_pwd": False,
                        "duration_years": 4,
                        "degree_type": "Bachelor of Technology",
                        "opening_rank": 100 + (year - 2018) * 10 + round_number,
                        "closing_rank": (
                            200
                            + (year - 2018) * 20
                            + round_number
                            + (50 if institute_type == "NIT" else 0)
                        ),
                        "source_file": f"{year}_round_{round_number}.csv",
                        "source_row_id": 0,
                    }
                )
    return pd.DataFrame(rows)


def test_pre_counselling_policy_rejects_observed_and_same_season_features():
    for unavailable in (
        "closing_rank",
        "opening_rank",
        "log_opening_rank",
        "opening_percentile",
        "prev_round_closing_rank",
        "is_final_round",
        "applicants",
    ):
        with pytest.raises(ValueError, match="unavailable"):
            assert_features_available(
                [*PRE_COUNSELLING_FEATURES, unavailable],
                FeatureAvailabilityMode.PRE_COUNSELLING,
            )


def test_in_round_policy_allows_only_completed_previous_round_features():
    features = features_for_mode(FeatureAvailabilityMode.IN_ROUND_UPDATE)
    assert "prev_round_closing_rank" in features
    assert_features_available(features, FeatureAvailabilityMode.IN_ROUND_UPDATE)
    with pytest.raises(ValueError, match="opening_rank"):
        assert_features_available(
            [*features, "opening_rank"],
            FeatureAvailabilityMode.IN_ROUND_UPDATE,
        )


def test_feature_schema_artifact_declares_availability_mode():
    payload = feature_schema_payload(
        mode=FeatureAvailabilityMode.PRE_COUNSELLING,
    )
    assert payload["availability_mode"] == "pre_counselling"
    assert payload["features"] == PRE_COUNSELLING_FEATURES
    assert "closing_rank" not in payload["features"]


def test_past_only_frequency_features_are_available_in_strict_policy():
    features = features_for_mode(FeatureAvailabilityMode.PRE_COUNSELLING)
    assert PAST_ONLY_FREQUENCY_FEATURES <= set(features)
    assert_features_available(features, FeatureAvailabilityMode.PRE_COUNSELLING)


def test_longitudinal_statistics_are_shifted_to_strictly_prior_years(observed_cutoffs):
    featured = build_longitudinal_features(observed_cutoffs)
    group = featured.loc[
        (featured["institute"] == "Institute A") & (featured["round"] == 1)
    ].sort_values("year")
    first, second, final = group.iloc[0], group.iloc[1], group.iloc[-1]

    assert first["hist_year_count"] == 0
    assert pd.isna(first["last_closing_rank"])
    assert second["hist_year_count"] == 1
    assert second["last_closing_rank"] == first["closing_rank"]
    assert final["hist_year_count"] == len(group) - 1
    assert final["last_closing_rank"] == group.iloc[-2]["closing_rank"]
    assert final["hist_mean_closing_rank"] == pytest.approx(
        group.iloc[:-1]["closing_rank"].mean()
    )


def test_frequency_features_exclude_every_row_from_the_current_year(observed_cutoffs):
    featured = build_longitudinal_features(observed_cutoffs)

    first_year = featured.loc[featured["year"] == 2018]
    assert set(first_year["institute_freq"]) == {0}
    assert set(first_year["program_freq"]) == {0}

    second_year = featured.loc[featured["year"] == 2019]
    assert set(second_year["institute_freq"]) == {2}
    assert set(second_year["program_freq"]) == {4}

    # Every row for an identity/year gets one shifted annual count; row order
    # inside 2019 cannot make a later row observe an earlier 2019 outcome.
    assert second_year.groupby("institute")["institute_freq"].nunique().eq(1).all()
    assert second_year.groupby("program")["program_freq"].nunique().eq(1).all()


def test_target_frame_uses_prior_snapshot_and_contains_no_observed_rank_columns(observed_cutoffs):
    history = observed_cutoffs.loc[observed_cutoffs["year"] < 2022]
    frame = build_target_year_frame(history, 2022, snapshot_year=2021)

    assert (frame["year"] == 2022).all()
    assert (frame["snapshot_year"] == 2021).all()
    assert not (OBSERVED_ONLY_COLUMNS & set(frame.columns))
    institute_a_round_1 = frame.loc[
        (frame["institute"] == "Institute A") & (frame["round"] == 1)
    ].iloc[0]
    expected = history.loc[
        (history["institute"] == "Institute A") & (history["round"] == 1)
    ].sort_values("year")
    assert institute_a_round_1["last_closing_rank"] == expected.iloc[-1]["closing_rank"]
    assert institute_a_round_1["hist_mean_closing_rank"] == pytest.approx(
        expected["closing_rank"].mean()
    )
    assert institute_a_round_1["institute_freq"] == 8
    assert institute_a_round_1["program_freq"] == 16


def test_target_frame_rejects_target_or_future_rows(observed_cutoffs):
    with pytest.raises(ValueError, match="strictly past-only"):
        build_target_year_frame(observed_cutoffs, 2022, snapshot_year=2021)


def test_target_matching_reports_new_target_programs_as_outside_prior_universe(observed_cutoffs):
    history = observed_cutoffs.loc[observed_cutoffs["year"] < 2022]
    frame = build_target_year_frame(history, 2022, snapshot_year=2021)
    new_program = observed_cutoffs.loc[observed_cutoffs["year"] == 2022].iloc[0].copy()
    new_program["program"] = "New Target-Year Program"
    observed_with_new = pd.concat(
        [observed_cutoffs, pd.DataFrame([new_program])],
        ignore_index=True,
    )

    _matched, counts = match_target_outcomes(frame, observed_with_new, 2022)
    assert counts["observed_rows_outside_prior_universe"] == 1
    assert counts["observed_universe_coverage"] < 1.0


def test_in_round_frame_requires_and_uses_only_earlier_completed_round(observed_cutoffs):
    history = observed_cutoffs.loc[observed_cutoffs["year"] < 2022]
    with pytest.raises(ValueError, match="requires completed_rounds"):
        build_target_year_frame(
            history,
            2022,
            mode=FeatureAvailabilityMode.IN_ROUND_UPDATE,
            snapshot_year=2021,
            target_rounds=[2],
        )

    completed = observed_cutoffs.loc[
        (observed_cutoffs["year"] == 2022) & (observed_cutoffs["round"] == 1)
    ]
    frame = build_target_year_frame(
        history,
        2022,
        mode=FeatureAvailabilityMode.IN_ROUND_UPDATE,
        snapshot_year=2021,
        target_rounds=[2],
        completed_rounds=completed,
    )
    assert (frame["prev_round_number"] == 1).all()
    assert frame["prev_round_closing_rank"].notna().all()

    same_round = observed_cutoffs.loc[
        (observed_cutoffs["year"] == 2022) & (observed_cutoffs["round"] == 2)
    ]
    with pytest.raises(ValueError, match="target or a later round"):
        build_target_year_frame(
            history,
            2022,
            mode=FeatureAvailabilityMode.IN_ROUND_UPDATE,
            snapshot_year=2021,
            target_rounds=[2],
            completed_rounds=same_round,
        )


def test_in_round_diagnostic_is_real_but_explicitly_non_primary(observed_cutoffs):
    diagnostic = evaluate_in_round_diagnostic(
        observed_cutoffs,
        target_year=2022,
        target_round=2,
        seed=42,
    )
    assert set(diagnostic["variant"]) == {
        "ridge_in_round_with_previous_round",
        "ridge_in_round_without_previous_round",
    }
    assert set(diagnostic["observed_rows"]) == {
        len(
            observed_cutoffs.loc[
                (observed_cutoffs["year"] == 2022)
                & (observed_cutoffs["round"] == 2)
            ]
        )
    }
    assert diagnostic["n"].gt(0).all()
    assert diagnostic["mae"].notna().all()
    assert not diagnostic["eligible_for_primary_claim"].any()
    assert not diagnostic["comparable_to_primary_pre_counselling"].any()
    comparator = diagnostic.set_index("variant").loc[
        "ridge_in_round_without_previous_round"
    ]
    assert set(comparator["removed_features"]) == {
        "prev_round_closing_rank",
        "prev_round_number",
    }


def test_generated_ablations_include_real_refits_and_ineligible_leakage_diagnostic(
    observed_cutoffs,
):
    longitudinal = build_longitudinal_features(observed_cutoffs)
    primary = pd.DataFrame(
        [
            {
                "year": 2022,
                "model": "ridge",
                "n": 4,
                "mae": 10.0,
                "rmse": 12.0,
                "median_absolute_error": 9.0,
                "r2": 0.5,
            }
        ]
    )
    ablations = evaluate_ablations(
        observed_cutoffs,
        longitudinal,
        2022,
        primary,
        seed=42,
    ).set_index("variant")

    assert ablations.loc["ridge_full_pre_counselling", "status"] == (
        "evaluated_strict_reference"
    )
    for policy_control in (
        "ridge_without_opening_rank_features",
        "ridge_without_previous_round_features",
    ):
        assert ablations.loc[policy_control, "mae"] == 10.0
        assert ablations.loc[policy_control, "policy_equivalent_to"] == (
            "ridge_full_pre_counselling"
        )
    assert set(
        ablations.loc["ridge_without_historical_statistics", "removed_features"]
    ) == HISTORICAL_STAT_FEATURES
    assert set(
        ablations.loc["ridge_without_frequency_encodings", "removed_features"]
    ) == PAST_ONLY_FREQUENCY_FEATURES
    leaky = ablations.loc["legacy_same_round_opening_diagnostic"]
    assert not leaky["policy_compliant"]
    assert not leaky["eligible_for_primary_claim"]
    assert not leaky["eligible_for_strict_comparison"]
    assert not leaky["comparable_to_full_pre_counselling"]


def test_uncertainty_artifact_uses_only_prior_oof_residuals():
    calibration = pd.DataFrame(
        {
            "year": [2021, 2022, 2023, 2024],
            "institute_type": ["IIT", "IIT", "NIT", "NIT"],
            "round": [1, 2, 1, 2],
            "abs_residual": [100.0, 150.0, 200.0, 250.0],
        }
    )
    payload = uncertainty_payload(
        calibration,
        prediction_year=2025,
        model_training_years=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
        dataset_hash="a" * 64,
        seed=42,
    )
    assert payload["source_split"] == "rolling_origin_oof"
    assert payload["formal_coverage_guarantee"] is False
    assert max(payload["calibration_years"]) < payload["prediction_year"]

    contaminated = pd.concat(
        [
            calibration,
            pd.DataFrame(
                [{"year": 2025, "institute_type": "IIT", "round": 1, "abs_residual": 1.0}]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="target-year or future"):
        uncertainty_payload(
            contaminated,
            prediction_year=2025,
            model_training_years=[2018, 2019],
            dataset_hash="a" * 64,
            seed=42,
        )


def test_ridge_pipeline_is_deterministic_for_fixed_inputs(observed_cutoffs):
    featured = build_longitudinal_features(observed_cutoffs)
    train = featured.loc[featured["year"] < 2022]
    target = build_target_year_frame(
        observed_cutoffs.loc[observed_cutoffs["year"] < 2022],
        2022,
        snapshot_year=2021,
    )
    features = features_for_mode(FeatureAvailabilityMode.PRE_COUNSELLING)
    first = ridge_pipeline(features).fit(train[features], train["closing_rank"])
    second = ridge_pipeline(features).fit(train[features], train["closing_rank"])
    np.testing.assert_allclose(first.predict(target[features]), second.predict(target[features]))


def test_model_selection_uses_only_pre_final_origins():
    metrics = pd.DataFrame(
        [
            {"year": 2023, "model": "ridge", "mae": 50.0, "rmse": 60.0, "median_absolute_error": 40.0},
            {"year": 2023, "model": "naive_last_year", "mae": 10.0, "rmse": 20.0, "median_absolute_error": 5.0},
            {"year": 2024, "model": "ridge", "mae": 50.0, "rmse": 60.0, "median_absolute_error": 40.0},
            {"year": 2024, "model": "naive_last_year", "mae": 10.0, "rmse": 20.0, "median_absolute_error": 5.0},
            # Reversing the final-test ordering must not change selection.
            {"year": 2025, "model": "ridge", "mae": 1.0, "rmse": 1.0, "median_absolute_error": 1.0},
            {"year": 2025, "model": "naive_last_year", "mae": 1000.0, "rmse": 1000.0, "median_absolute_error": 1000.0},
        ]
    )
    selected, summary = select_model_before_final_test(metrics, final_test_year=2025)
    assert selected == "naive_last_year"
    assert set(summary["validation_origin_count"]) == {2}


def test_final_holdout_is_not_evaluated_until_after_selection(monkeypatch):
    events: list[tuple[str, tuple[int, ...] | None]] = []

    def fake_evaluate(_observed, _longitudinal, years, *, seed, origin_start_year=None):
        events.append(("evaluate", tuple(years)))
        if years == [2025]:
            assert events[-2][0] == "select"
        metrics = pd.DataFrame(
            [
                {
                    "year": year,
                    "model": model,
                    "mae": mae,
                    "rmse": mae + 1,
                    "median_absolute_error": mae - 1,
                }
                for year in years
                for model, mae in (("ridge", 20.0), ("naive_last_year", 10.0))
            ]
        )
        errors = {
            model: pd.DataFrame({"year": years, "abs_residual": [1.0] * len(years)})
            for model in ("ridge", "naive_last_year")
        }
        return metrics, errors, errors

    original_select = research_evaluation.select_model_before_final_test

    def tracked_select(metrics, final_test_year):
        events.append(("select", None))
        return original_select(metrics, final_test_year)

    monkeypatch.setattr(research_evaluation, "evaluate_rolling_origins", fake_evaluate)
    monkeypatch.setattr(research_evaluation, "select_model_before_final_test", tracked_select)
    observed = pd.DataFrame({"year": range(2018, 2026)})

    result = evaluate_with_isolated_final_holdout(
        observed,
        pd.DataFrame(),
        [2023, 2024, 2025],
        2025,
        seed=42,
    )

    assert events == [("evaluate", (2023, 2024)), ("select", None), ("evaluate", (2025,))]
    assert result[3] == "naive_last_year"


def test_last_observation_estimator_has_no_learned_target_dependency():
    features = pd.DataFrame({"last_closing_rank": [100.0, 200.0, 300.0]})
    first = LastObservationRegressor().fit(features, [1.0, 1.0, 1.0])
    second = LastObservationRegressor().fit(features, [999.0, 999.0, 999.0])
    np.testing.assert_array_equal(first.predict(features), [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(first.predict(features), second.predict(features))


def test_dataset_quality_payload_separates_post_cutoff_local_rows(observed_cutoffs):
    quality = dataset_quality_payload(observed_cutoffs, data_cutoff_year=2021)
    assert quality["full_local_row_count"] == 20
    assert quality["full_local_years"] == [2018, 2019, 2020, 2021, 2022]
    assert quality["selected_row_count"] == 16
    assert quality["selected_period"] == {"start_year": 2018, "end_year": 2021}
    assert quality["post_cutoff_row_count"] == 4
    assert quality["post_cutoff_years"] == [2022]
    assert quality["missing_required_field_cell_count"] == 0
    assert quality["duplicate_outcome_key_row_count"] == 0
    assert quality["opening_rank_greater_than_closing_rank_count"] == 0
    assert len(quality["per_year_round_coverage"]) == 5
    final = quality["per_year_round_coverage"][-1]
    assert final["rounds"] == [1, 2]
    assert final["rows_by_round"] == {"1": 2, "2": 2}
    assert final["experiment_status"] == "post_cutoff_excluded"


def test_strict_figures_have_canonical_names_and_source_date_epoch_bytes(
    tmp_path: Path,
    monkeypatch,
):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    errors = pd.DataFrame(
        {
            "year": [2021, 2021, 2022, 2022],
            "institute_type": ["IIT", "NIT", "IIT", "NIT"],
            "actual_closing_rank": [100.0, 200.0, 120.0, 240.0],
            "predicted_closing_rank": [110.0, 180.0, 125.0, 220.0],
            "residual": [-10.0, 20.0, -5.0, 20.0],
            "abs_residual": [10.0, 20.0, 5.0, 20.0],
        }
    )
    folds = pd.DataFrame(
        {
            "year": [2021, 2022],
            "model": ["naive_last_year", "naive_last_year"],
            "prediction_interval_coverage": [0.75, 1.0],
        }
    )
    ablations = pd.DataFrame(
        {
            "variant": [
                "ridge_full_pre_counselling",
                "ridge_without_frequency_encodings",
                "legacy_same_round_opening_diagnostic",
            ],
            "eligible_for_strict_comparison": [True, True, False],
            "mae": [15.0, 18.0, 2.0],
        }
    )

    first = generate_strict_figures(
        errors=errors,
        fold_metrics=folds,
        ablations=ablations,
        output_dir=tmp_path / "first",
        selected_model="naive_last_year",
    )
    second = generate_strict_figures(
        errors=errors,
        fold_metrics=folds,
        ablations=ablations,
        output_dir=tmp_path / "second",
        selected_model="naive_last_year",
    )
    assert {path.name for path in first} == {
        "strict_pred_vs_actual.png",
        "strict_residual_distribution.png",
        "strict_error_by_year.png",
        "strict_error_by_institute_type.png",
        "strict_interval_coverage.png",
        "strict_ablation.png",
    }
    assert all(path.stat().st_size > 0 for path in first)
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]


def test_missing_dataset_failure_is_precise(tmp_path: Path):
    missing = tmp_path / "josaa_cleaned.csv"
    with pytest.raises(ResearchAssetError) as exc:
        load_dataset(missing, data_cutoff_year=2025)
    message = str(exc.value)
    assert "DATA ASSET BLOCKED" in message
    assert "research.pipeline.preprocessing --data-dir research/data" in message
    assert "No verified public source URL" in message
