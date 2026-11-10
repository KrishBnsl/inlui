"""Past-only feature construction for JoSAA cutoff forecasting.

The functions in this module intentionally operate on raw, human-readable
columns.  Encoding belongs inside a fitted sklearn pipeline so target-year
values can never influence an encoder or frequency table.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .feature_schema import (
    FeatureAvailabilityMode,
    assert_features_available,
    features_for_mode,
)


TARGET_COLUMN = "closing_rank"

# A row is a cutoff outcome for one choice/profile/round combination.
CHOICE_KEY = [
    "institute",
    "program",
    "category",
    "quota",
    "gender",
    "is_pwd",
]
OUTCOME_KEY = CHOICE_KEY + ["round"]

STATIC_COLUMNS = [
    "institute_type",
    "duration_years",
    "degree_type",
]

REQUIRED_OBSERVED_COLUMNS = [
    "year",
    *OUTCOME_KEY,
    *STATIC_COLUMNS,
    TARGET_COLUMN,
]

OBSERVED_ONLY_COLUMNS = {
    "opening_rank",
    "closing_rank",
    "log_opening_rank",
    "log_closing_rank",
    "opening_percentile",
    "closing_percentile",
    "rank_spread",
    "competitiveness_ratio",
    "source_file",
    "source_row_id",
    "is_pwd_rank",
    "applicants",
}


def _as_mode(mode: FeatureAvailabilityMode | str) -> FeatureAvailabilityMode:
    return FeatureAvailabilityMode(mode)


def validate_observed_cutoffs(df: pd.DataFrame) -> None:
    """Validate the minimum schema and uniqueness needed for temporal features."""

    missing = sorted(set(REQUIRED_OBSERVED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Observed cutoff data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Observed cutoff data is empty")
    if df["year"].isna().any() or df["round"].isna().any():
        raise ValueError("Observed cutoff data contains null year or round values")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Observed cutoff data contains null closing_rank targets")
    duplicate = df.duplicated(["year", *OUTCOME_KEY], keep=False)
    if duplicate.any():
        example = (
            df.loc[duplicate, ["year", *OUTCOME_KEY]]
            .head(1)
            .to_dict(orient="records")[0]
        )
        raise ValueError(
            "Observed cutoff data has duplicate year/outcome keys; "
            f"example={example}"
        )


def assert_past_only(history: pd.DataFrame, target_year: int) -> None:
    """Fail if a target/future observation could enter a forecast frame."""

    if history.empty:
        raise ValueError("Cannot construct a target frame from empty history")
    offending = history.loc[history["year"] >= target_year, "year"]
    if not offending.empty:
        raise ValueError(
            "Target frame history must be strictly past-only: "
            f"found year {int(offending.min())} for target_year={target_year}"
        )


def _add_prior_year_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add expanding statistics shifted by one year within each outcome key."""

    result = df.sort_values([*OUTCOME_KEY, "year"], kind="mergesort").copy()
    grouped = result.groupby(OUTCOME_KEY, sort=False, dropna=False)

    result["hist_year_count"] = grouped.cumcount().astype("int64")
    result["last_closing_rank"] = grouped[TARGET_COLUMN].shift(1)
    result["last_observed_year"] = grouped["year"].shift(1)
    result["years_since_observed"] = result["year"] - result["last_observed_year"]

    target = result[TARGET_COLUMN].astype(float)
    past_sum = grouped[TARGET_COLUMN].cumsum().astype(float) - target
    target_squared = target.pow(2)
    past_squared_sum = (
        target_squared.groupby(
            [result[column] for column in OUTCOME_KEY],
            sort=False,
            dropna=False,
        ).cumsum()
        - target_squared
    )
    count = result["hist_year_count"].astype(float)
    result["hist_mean_closing_rank"] = past_sum.div(count.where(count > 0))

    numerator = past_squared_sum - past_sum.pow(2).div(count.where(count > 0))
    variance = numerator.div((count - 1).where(count > 1)).clip(lower=0)
    result["hist_std_closing_rank"] = np.sqrt(variance)
    result["years_since_ews"] = (result["year"] - 2019).clip(lower=0)
    return result


def _add_past_only_frequency(
    df: pd.DataFrame,
    *,
    identity_column: str,
    feature_name: str,
) -> pd.DataFrame:
    """Count identity rows in complete years strictly before each row's year.

    A row-wise ``cumcount`` is not sufficient here: it would let later rows in
    the same target year see earlier rows from that year.  Aggregating to one
    identity/year count first makes every row in a year receive the same
    shifted cumulative value.
    """

    yearly = (
        df.groupby([identity_column, "year"], dropna=False, sort=False)
        .size()
        .rename("_rows_in_year")
        .reset_index()
        .sort_values([identity_column, "year"], kind="mergesort")
    )
    grouped = yearly.groupby(identity_column, dropna=False, sort=False)["_rows_in_year"]
    yearly[feature_name] = (grouped.cumsum() - yearly["_rows_in_year"]).astype("int64")
    return df.merge(
        yearly[[identity_column, "year", feature_name]],
        on=[identity_column, "year"],
        how="left",
        sort=False,
        validate="many_to_one",
    )


def _add_past_only_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    result = _add_past_only_frequency(
        df,
        identity_column="institute",
        feature_name="institute_freq",
    )
    return _add_past_only_frequency(
        result,
        identity_column="program",
        feature_name="program_freq",
    )


def _add_previous_round_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add only the latest strictly earlier completed round in the same year."""

    result = df.sort_values(["year", *CHOICE_KEY, "round"], kind="mergesort").copy()
    grouped = result.groupby(["year", *CHOICE_KEY], sort=False, dropna=False)
    result["prev_round_closing_rank"] = grouped[TARGET_COLUMN].shift(1)
    result["prev_round_number"] = grouped["round"].shift(1)
    invalid = result["prev_round_number"].notna() & (
        result["prev_round_number"] >= result["round"]
    )
    if invalid.any():
        raise AssertionError("Previous-round feature used a same or later round")
    return result


def build_longitudinal_features(
    observed: pd.DataFrame,
    mode: FeatureAvailabilityMode | str = FeatureAvailabilityMode.PRE_COUNSELLING,
) -> pd.DataFrame:
    """Featurize observed rows using only information available before each row.

    ``closing_rank`` remains in the returned frame solely as the supervised
    target.  Callers must select model inputs with :func:`features_for_mode`.
    """

    resolved = _as_mode(mode)
    validate_observed_cutoffs(observed)
    result = _add_prior_year_statistics(observed)
    result = _add_past_only_frequencies(result)
    if resolved is FeatureAvailabilityMode.IN_ROUND_UPDATE:
        result = _add_previous_round_statistics(result)
    assert_features_available(features_for_mode(resolved), resolved)
    return result


def _history_aggregates(history: pd.DataFrame) -> pd.DataFrame:
    ordered = history.sort_values([*OUTCOME_KEY, "year"], kind="mergesort")
    aggregate = (
        ordered.groupby(OUTCOME_KEY, dropna=False)[TARGET_COLUMN]
        .agg(
            hist_mean_closing_rank="mean",
            hist_std_closing_rank="std",
            hist_year_count="count",
            last_closing_rank="last",
        )
        .reset_index()
    )
    last_year = (
        ordered.groupby(OUTCOME_KEY, dropna=False)["year"]
        .last()
        .rename("last_observed_year")
        .reset_index()
    )
    return aggregate.merge(last_year, on=OUTCOME_KEY, how="left", validate="one_to_one")


def _add_history_frequencies(target: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach identity counts from an already validated strictly-past history."""

    result = target
    for identity_column, feature_name in (
        ("institute", "institute_freq"),
        ("program", "program_freq"),
    ):
        counts = (
            history.groupby(identity_column, dropna=False)
            .size()
            .rename(feature_name)
            .reset_index()
        )
        result = result.merge(
            counts,
            on=identity_column,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        result[feature_name] = result[feature_name].fillna(0).astype("int64")
    return result


def _prepare_completed_rounds(
    completed_rounds: pd.DataFrame,
    target_year: int,
    target_rounds: Iterable[int],
) -> pd.DataFrame:
    required = {"year", "round", *CHOICE_KEY, TARGET_COLUMN}
    missing = sorted(required - set(completed_rounds.columns))
    if missing:
        raise ValueError(f"Completed-round data is missing required columns: {missing}")
    if not (completed_rounds["year"] == target_year).all():
        raise ValueError("Completed-round data must belong only to the target year")
    maximum_target_round = max(int(value) for value in target_rounds)
    if (completed_rounds["round"] >= maximum_target_round).any():
        raise ValueError(
            "Completed-round data includes the target or a later round; "
            f"all completed rounds must be < {maximum_target_round}"
        )

    ordered = completed_rounds.sort_values([*CHOICE_KEY, "round"], kind="mergesort")
    latest = ordered.groupby(CHOICE_KEY, dropna=False).tail(1)
    return latest[[*CHOICE_KEY, "round", TARGET_COLUMN]].rename(
        columns={
            "round": "prev_round_number",
            TARGET_COLUMN: "prev_round_closing_rank",
        }
    )


def build_target_year_frame(
    history: pd.DataFrame,
    target_year: int,
    *,
    mode: FeatureAvailabilityMode | str = FeatureAvailabilityMode.PRE_COUNSELLING,
    snapshot_year: int | None = None,
    target_rounds: Iterable[int] | None = None,
    completed_rounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a forecast universe without loading observed target-year rows.

    The program/profile/round universe is copied from one explicitly selected
    prior snapshot.  Historical statistics are calculated only from ``history``
    where every year is strictly less than ``target_year``.
    """

    resolved = _as_mode(mode)
    validate_observed_cutoffs(history)
    assert_past_only(history, target_year)
    selected_snapshot = int(history["year"].max()) if snapshot_year is None else int(snapshot_year)
    if selected_snapshot >= target_year:
        raise ValueError("snapshot_year must be strictly earlier than target_year")
    if selected_snapshot not in set(history["year"].astype(int).unique()):
        raise ValueError(f"No rows found for requested snapshot_year={selected_snapshot}")

    snapshot = history.loc[history["year"] == selected_snapshot].copy()
    if target_rounds is not None:
        round_set = {int(value) for value in target_rounds}
        if not round_set:
            raise ValueError("target_rounds cannot be empty")
        snapshot = snapshot.loc[snapshot["round"].isin(round_set)].copy()
    if snapshot.empty:
        raise ValueError("The selected prior-year snapshot produced an empty target universe")

    keep = [*OUTCOME_KEY, *STATIC_COLUMNS]
    target = snapshot[keep].drop_duplicates(OUTCOME_KEY, keep="last")
    target["year"] = int(target_year)
    target["snapshot_year"] = selected_snapshot

    aggregate = _history_aggregates(history)
    target = target.merge(aggregate, on=OUTCOME_KEY, how="left", validate="one_to_one")
    target = _add_history_frequencies(target, history)
    target["years_since_observed"] = target["year"] - target["last_observed_year"]
    target["years_since_ews"] = (target["year"] - 2019).clip(lower=0)

    if resolved is FeatureAvailabilityMode.IN_ROUND_UPDATE:
        if target_rounds is None or len(set(int(v) for v in target_rounds)) != 1:
            raise ValueError("in_round_update target frames must select exactly one target round")
        target_round = next(iter(set(int(v) for v in target_rounds)))
        if target_round > 1 and completed_rounds is None:
            raise ValueError(
                "in_round_update requires completed_rounds for target rounds after round 1"
            )
        if completed_rounds is not None:
            previous = _prepare_completed_rounds(
                completed_rounds,
                target_year,
                [target_round],
            )
            target = target.merge(previous, on=CHOICE_KEY, how="left", validate="many_to_one")
        else:
            target["prev_round_number"] = np.nan
            target["prev_round_closing_rank"] = np.nan

    leaked = sorted(OBSERVED_ONLY_COLUMNS.intersection(target.columns))
    if leaked:
        raise AssertionError(f"Target frame contains observed-only columns: {leaked}")
    assert_features_available(features_for_mode(resolved), resolved)
    return target.sort_values(OUTCOME_KEY, kind="mergesort").reset_index(drop=True)


def match_target_outcomes(
    forecast_frame: pd.DataFrame,
    observed: pd.DataFrame,
    target_year: int,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Join a past-only forecast universe to outcomes for offline evaluation."""

    actual = observed.loc[observed["year"] == target_year, [*OUTCOME_KEY, TARGET_COLUMN]].copy()
    if actual.empty:
        raise ValueError(f"No observed outcomes found for target_year={target_year}")
    if actual.duplicated(OUTCOME_KEY).any():
        raise ValueError(f"Observed outcomes for {target_year} are not unique on OUTCOME_KEY")

    matched = forecast_frame.merge(
        actual.rename(columns={TARGET_COLUMN: "actual_closing_rank"}),
        on=OUTCOME_KEY,
        how="inner",
        validate="one_to_one",
    )
    counts: dict[str, int | float] = {
        "forecast_rows": int(len(forecast_frame)),
        "observed_rows": int(len(actual)),
        "matched_rows": int(len(matched)),
        "forecast_rows_without_observation": int(len(forecast_frame) - len(matched)),
        "observed_rows_outside_prior_universe": int(len(actual) - len(matched)),
        "observed_universe_coverage": float(len(matched) / len(actual)),
    }
    return matched, counts
