"""Canonical model feature schema.

The inference service and offline training pipeline both import this module so
feature order does not drift between model generation and serving.
"""

FORBIDDEN_TARGET_DERIVED_FEATURES = {
    "closing_rank",
    "log_closing_rank",
    "rank_spread",
    "round_delta",
    "closing_rank_vs_hist_mean",
    "closing_rank_ratio_hist",
}

FEATURE_SCHEMA_VERSION = 1

MODEL_FEATURES = [
    "year",
    "round",
    "institute_type_encoded",
    "category_encoded",
    "quota_encoded",
    "gender_encoded",
    "degree_type_encoded",
    "opening_rank",
    "log_opening_rank",
    "duration_years",
    "institute_freq",
    "program_freq",
    "category_freq",
    "institute_type_freq",
    "hist_mean_closing_rank",
    "hist_std_closing_rank",
    "hist_min_closing_rank",
    "hist_max_closing_rank",
    "hist_year_count",
    "prev_round_closing_rank",
    "is_final_round",
    "years_since_ews",
    "is_pwd",
    "opening_percentile",
    "applicants",
]


def assert_no_forbidden_features(features: list[str]) -> None:
    """Raise if a target-derived feature is present in the model input schema."""
    leaked = FORBIDDEN_TARGET_DERIVED_FEATURES.intersection(features)
    if leaked:
        raise ValueError(f"Forbidden target-derived features in model schema: {sorted(leaked)}")


def feature_schema_payload(features: list[str] | None = None) -> dict:
    """Return the JSON-serializable feature schema artifact payload."""
    selected = features or MODEL_FEATURES
    assert_no_forbidden_features(selected)
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "features": selected,
        "forbidden_target_derived_features": sorted(FORBIDDEN_TARGET_DERIVED_FEATURES),
    }
