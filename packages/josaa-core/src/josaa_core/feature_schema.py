"""Canonical feature schemas and feature-availability policy.

There are two deliberately separate contracts in this module:

``LEGACY_SERVING_FEATURES``
    The columns expected by the historical model artifact.  It is retained so
    old artifacts fail compatibly, but it is **not** valid evidence for a
    pre-counselling forecast because it contains target-season observations.

``PRE_COUNSELLING_FEATURES`` / ``IN_ROUND_UPDATE_FEATURES``
    Research-grade policies whose inputs are tied to when information becomes
    available.  New experiments and artifacts must declare one of these modes.

Keeping the legacy list explicit prevents an old pickle from being silently
relabelled as an availability-correct model.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class FeatureAvailabilityMode(str, Enum):
    """Supported decision-time feature policies."""

    PRE_COUNSELLING = "pre_counselling"
    IN_ROUND_UPDATE = "in_round_update"

FORBIDDEN_TARGET_DERIVED_FEATURES = {
    "closing_rank",
    "log_closing_rank",
    "rank_spread",
    "round_delta",
    "closing_rank_vs_hist_mean",
    "closing_rank_ratio_hist",
}

FEATURE_SCHEMA_VERSION = 3

# Compatibility-only schema for the historical deployed artifact.  Do not use
# this list in professor-facing experiments.  ``MODEL_FEATURES`` remains an
# alias because the old training/inference modules import that public name.
LEGACY_SERVING_FEATURES = [
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

MODEL_FEATURES = LEGACY_SERVING_FEATURES


# These columns are either immutable program/profile metadata, a declared
# target identifier, or statistics computed exclusively from earlier years.
PRE_COUNSELLING_FEATURES = [
    "year",
    "round",
    "duration_years",
    "is_pwd",
    "years_since_ews",
    "hist_mean_closing_rank",
    "hist_std_closing_rank",
    "hist_year_count",
    "last_closing_rank",
    "years_since_observed",
    "institute_freq",
    "program_freq",
    "institute_type",
    "institute",
    "program",
    "category",
    "quota",
    "gender",
    "degree_type",
]

# A completed earlier round is observable during counselling.  The same-round
# opening or closing rank remains unavailable and is never allowed here.
IN_ROUND_UPDATE_FEATURES = PRE_COUNSELLING_FEATURES + [
    "prev_round_closing_rank",
    "prev_round_number",
]

HISTORICAL_STAT_FEATURES = {
    "hist_mean_closing_rank",
    "hist_std_closing_rank",
    "hist_year_count",
    "last_closing_rank",
    "years_since_observed",
}

# Counts of observed cutoff rows for the identity in strictly earlier years.
# These names are retained for compatibility with the legacy vocabulary, but
# their research semantics are deliberately different from whole-table
# ``value_counts`` encodings, which would reveal target-year prevalence.
PAST_ONLY_FREQUENCY_FEATURES = {
    "institute_freq",
    "program_freq",
}

# Target-year applicant counts could be admitted only with a separately
# versioned official pre-season source.  The current repository has no such
# provenance, so they stay forbidden in the strict policies.
UNAVAILABLE_PRE_COUNSELLING_FEATURES = {
    "opening_rank",
    "log_opening_rank",
    "opening_percentile",
    "prev_round_closing_rank",
    "prev_round_number",
    "is_final_round",
    "applicants",
    "category_freq",
    "institute_type_freq",
}

UNAVAILABLE_IN_ROUND_FEATURES = (
    UNAVAILABLE_PRE_COUNSELLING_FEATURES
    - {"prev_round_closing_rank", "prev_round_number"}
)

FEATURE_AVAILABILITY = {
    "opening_rank": {
        "raw_columns": ["opening_rank"],
        "pre_counselling": False,
        "in_round_update": False,
        "reason": "Same-round opening rank is observed only after that round is published.",
    },
    "log_opening_rank": {
        "raw_columns": ["opening_rank"],
        "pre_counselling": False,
        "in_round_update": False,
        "reason": "Derived from unavailable same-round opening_rank.",
    },
    "opening_percentile": {
        "raw_columns": ["opening_rank", "applicants"],
        "pre_counselling": False,
        "in_round_update": False,
        "reason": "Uses the unavailable same-round opening rank.",
    },
    "prev_round_closing_rank": {
        "raw_columns": ["closing_rank", "round", "year"],
        "pre_counselling": False,
        "in_round_update": True,
        "reason": "Available only after an earlier target-year round is complete.",
    },
    "is_final_round": {
        "raw_columns": ["round", "year"],
        "pre_counselling": False,
        "in_round_update": False,
        "reason": "Legacy value was computed from the observed target-year maximum round.",
    },
    "hist_mean_closing_rank": {
        "raw_columns": ["closing_rank", "year"],
        "pre_counselling": True,
        "in_round_update": True,
        "reason": "Allowed only when every contributing year is strictly before the target year.",
    },
    "last_closing_rank": {
        "raw_columns": ["closing_rank", "year"],
        "pre_counselling": True,
        "in_round_update": True,
        "reason": "Most recent observation from a strictly earlier year.",
    },
    "institute_freq": {
        "raw_columns": ["institute", "year"],
        "pre_counselling": True,
        "in_round_update": True,
        "reason": (
            "Count of rows for the institute in strictly earlier years; "
            "the complete target year is excluded."
        ),
    },
    "program_freq": {
        "raw_columns": ["program", "year"],
        "pre_counselling": True,
        "in_round_update": True,
        "reason": (
            "Count of rows for the program label in strictly earlier years; "
            "the complete target year is excluded."
        ),
    },
    "applicants": {
        "raw_columns": ["applicants", "year"],
        "pre_counselling": False,
        "in_round_update": False,
        "reason": "No provenance-verified target-year value is available in this repository.",
    },
}


def _coerce_mode(mode: FeatureAvailabilityMode | str) -> FeatureAvailabilityMode:
    try:
        return FeatureAvailabilityMode(mode)
    except ValueError as exc:
        supported = ", ".join(item.value for item in FeatureAvailabilityMode)
        raise ValueError(f"Unknown feature availability mode {mode!r}; expected one of: {supported}") from exc


def features_for_mode(mode: FeatureAvailabilityMode | str) -> list[str]:
    """Return a copy of the ordered, availability-correct feature list."""

    resolved = _coerce_mode(mode)
    if resolved is FeatureAvailabilityMode.PRE_COUNSELLING:
        return list(PRE_COUNSELLING_FEATURES)
    return list(IN_ROUND_UPDATE_FEATURES)


def assert_features_available(
    features: Iterable[str],
    mode: FeatureAvailabilityMode | str,
) -> None:
    """Reject features that are unavailable at the declared decision time."""

    resolved = _coerce_mode(mode)
    selected = set(features)
    allowed = set(features_for_mode(resolved))
    unavailable = (
        UNAVAILABLE_PRE_COUNSELLING_FEATURES
        if resolved is FeatureAvailabilityMode.PRE_COUNSELLING
        else UNAVAILABLE_IN_ROUND_FEATURES
    )
    explicitly_forbidden = selected.intersection(unavailable | FORBIDDEN_TARGET_DERIVED_FEATURES)
    unknown = selected - allowed
    invalid = explicitly_forbidden | unknown
    if invalid:
        raise ValueError(
            f"Features unavailable in {resolved.value} mode: {sorted(invalid)}"
        )


def assert_no_forbidden_features(features: Iterable[str]) -> None:
    """Raise if a target-derived feature is present in the model input schema."""
    leaked = FORBIDDEN_TARGET_DERIVED_FEATURES.intersection(features)
    if leaked:
        raise ValueError(f"Forbidden target-derived features in model schema: {sorted(leaked)}")


def feature_schema_payload(
    features: list[str] | None = None,
    mode: FeatureAvailabilityMode | str | None = None,
) -> dict:
    """Return the JSON-serializable feature schema artifact payload."""
    selected = features or (features_for_mode(mode) if mode else MODEL_FEATURES)
    assert_no_forbidden_features(selected)
    if mode is not None:
        assert_features_available(selected, mode)
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "availability_mode": _coerce_mode(mode).value if mode is not None else "legacy_compatibility",
        "features": selected,
        "forbidden_target_derived_features": sorted(FORBIDDEN_TARGET_DERIVED_FEATURES),
    }
