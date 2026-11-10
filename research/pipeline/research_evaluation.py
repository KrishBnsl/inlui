#!/usr/bin/env python3
"""Availability-correct rolling-origin evaluation for JoSAA closing ranks.

This script is the only professor-facing benchmark path in the repository.  It
does not read the legacy feature-engineered target-year rows or the historical
test-residual artifact.  Forecast universes are constructed from a prior-year
snapshot and all uncertainty radii come from earlier rolling-origin residuals.

The local dataset currently has unresolved source provenance and licensing.
Consequently generated metrics are labelled exploratory and hash-bound; they
must not be presented as independently reproducible public research evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from josaa_core.baselines import LastObservationRegressor
from josaa_core.feature_schema import (
    FeatureAvailabilityMode,
    HISTORICAL_STAT_FEATURES,
    PAST_ONLY_FREQUENCY_FEATURES,
    assert_features_available,
    feature_schema_payload,
    features_for_mode,
)
from josaa_core.research_features import (
    OUTCOME_KEY,
    REQUIRED_OBSERVED_COLUMNS,
    build_longitudinal_features,
    build_target_year_frame,
    match_target_outcomes,
    validate_observed_cutoffs,
)
try:  # Support both `python -m ...` and the documented script path.
    from .build_registry import build_registry
except ImportError:  # pragma: no cover - exercised by the script-path smoke check
    from build_registry import build_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


LOGGER = logging.getLogger("josaa_research_evaluation")
SEED = 42
TARGET_COVERAGE = 0.90
MIN_GROUP_CALIBRATION_ROWS = 100
RESULT_SCHEMA_VERSION = 2
DATASET_ID = "josaa-cleaned-cutoff-outcomes"
CANONICAL_IN_ROUND_YEAR = 2025
CANONICAL_IN_ROUND_ROUND = 5
CANONICAL_CASE_STUDY_REQUEST = {
    "main_rank": 3000,
    "advanced_rank": 6700,
    "category": "OPEN",
    "gender": "Gender-Neutral",
    "round": 6,
    "top_n": 100,
}
CANONICAL_BUCKET_ORDER = [
    "safe_backup",
    "best_realistic",
    "ambitious_reach",
    "unlikely_reach",
]

NUMERIC_FEATURES = [
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
    "prev_round_closing_rank",
    "prev_round_number",
    # Registered only so the explicitly ineligible leakage diagnostic can be
    # fit without mutating global preprocessing state. Availability guards
    # still reject both columns from every strict feature policy.
    "opening_rank",
    "log_opening_rank",
]

CATEGORICAL_FEATURES = [
    "institute_type",
    "institute",
    "program",
    "category",
    "quota",
    "gender",
    "degree_type",
]

RANK_BINS = [0, 5_000, 15_000, 50_000, 100_000, math.inf]
RANK_LABELS = ["1-5k", "5k-15k", "15k-50k", "50k-100k", "100k+"]

RIDGE_HYPERPARAMETERS = {
    "alpha": 10.0,
    "solver": "lsqr",
    "tol": 1e-4,
}
HIST_GRADIENT_BOOSTING_HYPERPARAMETERS = {
    "loss": "squared_error",
    "learning_rate": 0.06,
    "max_iter": 120,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
}


class ResearchAssetError(RuntimeError):
    """Raised when the local-only research data path is unavailable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def source_snapshot_identity() -> dict[str, Any]:
    """Capture the source state once, before evidence generation mutates files."""

    commit = git_commit()
    dirty = git_worktree_dirty()
    return {
        "source_commit": commit,
        "generated_from_commit": commit,
        "source_worktree_dirty": dirty,
        "worktree_dirty": dirty,
        # Backward-compatible aliases retained for existing consumers.
        "git_commit": commit,
        "git_worktree_dirty": dirty,
    }


def generated_at_utc() -> str:
    """Use SOURCE_DATE_EPOCH when byte-stable generated metadata is required."""

    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    instant = (
        datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        if epoch is not None
        else datetime.now(timezone.utc)
    )
    return instant.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_semantics() -> dict[str, Any]:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is None:
        return {
            "timestamp_source": "wall_clock",
            "timestamp_semantics": "Execution-time timestamp.",
        }
    return {
        "timestamp_source": "SOURCE_DATE_EPOCH",
        "source_date_epoch": int(epoch),
        "timestamp_semantics": (
            "Reproducible build timestamp supplied by SOURCE_DATE_EPOCH; "
            "it is not the wall-clock execution time."
        ),
    }


def environment_payload() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def dataset_quality_payload(
    full_frame: pd.DataFrame,
    *,
    data_cutoff_year: int,
) -> dict[str, Any]:
    """Return deterministic, source-level quality facts before year filtering."""

    required = [column for column in REQUIRED_OBSERVED_COLUMNS if column in full_frame]
    missing_required_cells = int(full_frame[required].isna().sum().sum())
    rows_missing_required = int(full_frame[required].isna().any(axis=1).sum())
    duplicate_mask = full_frame.duplicated(["year", *OUTCOME_KEY], keep=False)
    duplicate_groups = int(
        full_frame.loc[duplicate_mask, ["year", *OUTCOME_KEY]].drop_duplicates().shape[0]
    )

    if "opening_rank" in full_frame:
        opening = pd.to_numeric(full_frame["opening_rank"], errors="coerce")
        closing = pd.to_numeric(full_frame["closing_rank"], errors="coerce")
        opening_after_closing: int | None = int((opening > closing).fillna(False).sum())
    else:
        opening_after_closing = None

    full_years = sorted(int(value) for value in full_frame["year"].dropna().unique())
    selected = full_frame.loc[full_frame["year"] <= data_cutoff_year]
    selected_years = sorted(int(value) for value in selected["year"].dropna().unique())
    excluded = full_frame.loc[full_frame["year"] > data_cutoff_year]
    excluded_years = sorted(int(value) for value in excluded["year"].dropna().unique())

    coverage: list[dict[str, Any]] = []
    for year, group in full_frame.groupby("year", sort=True, dropna=False):
        round_counts = (
            group["round"].value_counts(dropna=False).sort_index().astype(int)
        )
        numeric_rounds = sorted(
            int(value) for value in group["round"].dropna().unique()
        )
        coverage.append(
            {
                "year": int(year),
                "rows": int(len(group)),
                "round_count": int(len(numeric_rounds)),
                "rounds": numeric_rounds,
                "rows_by_round": {
                    str(int(round_number)): int(count)
                    for round_number, count in round_counts.items()
                    if pd.notna(round_number)
                },
                "experiment_status": (
                    "selected" if int(year) <= data_cutoff_year else "post_cutoff_excluded"
                ),
            }
        )

    return {
        "full_local_row_count": int(len(full_frame)),
        "full_local_years": full_years,
        "full_local_period": {
            "start_year": min(full_years),
            "end_year": max(full_years),
        },
        "selected_row_count": int(len(selected)),
        "selected_years": selected_years,
        "selected_period": {
            "start_year": min(selected_years),
            "end_year": max(selected_years),
        },
        "post_cutoff_row_count": int(len(excluded)),
        "post_cutoff_years": excluded_years,
        "post_cutoff_note": (
            "Locally present post-cutoff years are reported for transparency but excluded "
            "from every experiment and artifact."
        ),
        "required_fields": list(REQUIRED_OBSERVED_COLUMNS),
        "missing_required_field_cell_count": missing_required_cells,
        "rows_with_missing_required_fields": rows_missing_required,
        "duplicate_outcome_key_row_count": int(duplicate_mask.sum()),
        "duplicate_outcome_key_group_count": duplicate_groups,
        "opening_rank_greater_than_closing_rank_count": opening_after_closing,
        "per_year_round_coverage": coverage,
    }


def load_dataset(path: Path, data_cutoff_year: int) -> pd.DataFrame:
    if not path.exists():
        raise ResearchAssetError(
            f"DATA ASSET BLOCKED: {path} does not exist. No verified public source URL or "
            "redistribution license is recorded, so this repository cannot download it "
            "automatically. Place provenance-verified JoSAA round files under research/data/ "
            "and run `.venv/bin/python -m research.pipeline.preprocessing "
            "--data-dir research/data`, then rerun this command. See "
            "docs/TECHNICAL_REPORT.md#reproducibility."
        )

    LOGGER.info("Loading %s", path)
    frame = pd.read_csv(path, low_memory=False)
    validate_observed_cutoffs(frame)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    frame["closing_rank"] = pd.to_numeric(frame["closing_rank"], errors="raise").astype(float)
    quality = dataset_quality_payload(frame, data_cutoff_year=data_cutoff_year)
    selected = frame.loc[frame["year"] <= data_cutoff_year].copy()
    if selected.empty:
        raise ResearchAssetError(
            f"DATA ASSET BLOCKED: no rows at or before data_cutoff_year={data_cutoff_year} in {path}"
        )
    if data_cutoff_year not in set(selected["year"].unique()):
        raise ResearchAssetError(
            f"DATA ASSET BLOCKED: requested data_cutoff_year={data_cutoff_year} is absent from {path}"
        )
    selected.attrs["dataset_quality"] = quality
    return selected


def _split_features(features: Iterable[str]) -> tuple[list[str], list[str]]:
    ordered = list(features)
    numeric = [feature for feature in ordered if feature in NUMERIC_FEATURES]
    categorical = [feature for feature in ordered if feature in CATEGORICAL_FEATURES]
    missing = sorted(set(ordered) - set(numeric) - set(categorical))
    if missing:
        raise ValueError(f"No preprocessing strategy declared for features: {missing}")
    return numeric, categorical


def ridge_pipeline(features: list[str]) -> Pipeline:
    numeric, categorical = _split_features(features)
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2, dtype=np.float64),
            ),
        ]
    )
    prep = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        [
            ("preprocess", prep),
            ("regressor", Ridge(**RIDGE_HYPERPARAMETERS)),
        ]
    )


def nonlinear_pipeline(features: list[str], seed: int) -> Pipeline:
    numeric, categorical = _split_features(features)
    numeric_pipe = SimpleImputer(strategy="median", add_indicator=True)
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    prep = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingRegressor(
        **HIST_GRADIENT_BOOSTING_HYPERPARAMETERS,
        random_state=seed,
    )
    return Pipeline([("preprocess", prep), ("regressor", model)])


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    actual_array = actual_array[valid]
    predicted_array = predicted_array[valid]
    if len(actual_array) == 0:
        return {"n": 0, "mae": None, "rmse": None, "median_absolute_error": None, "r2": None}
    return {
        "n": int(len(actual_array)),
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "median_absolute_error": float(median_absolute_error(actual_array, predicted_array)),
        "r2": float(r2_score(actual_array, predicted_array)) if len(actual_array) > 1 else None,
    }


def higher_quantile(values: pd.Series | np.ndarray, coverage: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("Cannot estimate an uncertainty quantile from zero residuals")
    adjusted = min(1.0, math.ceil((len(finite) + 1) * coverage) / len(finite))
    return float(np.quantile(finite, adjusted, method="higher"))


def uncertainty_payload(
    calibration: pd.DataFrame,
    *,
    prediction_year: int,
    model_training_years: list[int],
    dataset_hash: str,
    seed: int,
) -> dict[str, Any]:
    if calibration.empty:
        raise ValueError("No earlier rolling-origin residuals are available for uncertainty")
    if (calibration["year"] >= prediction_year).any():
        raise ValueError("Uncertainty calibration contains target-year or future residuals")

    global_quantile = higher_quantile(calibration["abs_residual"], TARGET_COVERAGE)
    groups: list[dict[str, Any]] = []
    for (institute_type, round_number), group in calibration.groupby(
        ["institute_type", "round"], dropna=False, sort=True
    ):
        groups.append(
            {
                "institute_type_str": str(institute_type),
                "round": int(round_number),
                "count": int(len(group)),
                "abs_residual_quantile": higher_quantile(group["abs_residual"], TARGET_COVERAGE),
            }
        )
    return {
        "schema_version": 1,
        "method": "rolling_origin_oof_absolute_residual",
        "source_split": "rolling_origin_oof",
        "interval_name": "empirical_prediction_interval",
        "formal_coverage_guarantee": False,
        "target_coverage": TARGET_COVERAGE,
        "calibration_years": sorted(int(year) for year in calibration["year"].unique()),
        "model_training_years": model_training_years,
        "prediction_year": int(prediction_year),
        "dataset_sha256": dataset_hash,
        "seed": int(seed),
        "global": {
            "count": int(len(calibration)),
            "abs_residual_quantile": global_quantile,
        },
        "minimum_group_count_for_serving": MIN_GROUP_CALIBRATION_ROWS,
        "groups": groups,
    }


def apply_prediction_interval(
    predicted: np.ndarray,
    rows: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    global_quantile = higher_quantile(calibration["abs_residual"], TARGET_COVERAGE)
    grouped = (
        calibration.groupby(["institute_type", "round"], dropna=False)["abs_residual"]
        .agg(["count", lambda values: higher_quantile(values, TARGET_COVERAGE)])
        .reset_index()
    )
    grouped.columns = ["institute_type", "round", "count", "group_quantile"]
    joined = rows[["institute_type", "round"]].merge(
        grouped,
        on=["institute_type", "round"],
        how="left",
        sort=False,
    )
    radii = joined["group_quantile"].where(
        joined["count"].fillna(0) >= MIN_GROUP_CALIBRATION_ROWS,
        global_quantile,
    ).fillna(global_quantile).to_numpy(dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return predicted_array - radii, predicted_array + radii, radii, global_quantile


def fit_predict(
    pipeline: Pipeline,
    training: pd.DataFrame,
    forecast: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    pipeline.fit(training[features], training["closing_rank"].to_numpy(dtype=float))
    predictions = pipeline.predict(forecast[features])
    return np.asarray(predictions, dtype=float), time.perf_counter() - started


def _metric_row(
    *,
    year: int,
    model: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    counts: dict[str, int | float],
    elapsed_seconds: float,
    interval_coverage: float | None = None,
    interval_mean_width: float | None = None,
    calibration_count: int = 0,
    interval_quantile_global: float | None = None,
) -> dict[str, Any]:
    return {
        "year": int(year),
        "feature_mode": FeatureAvailabilityMode.PRE_COUNSELLING.value,
        "model": model,
        **regression_metrics(actual, predicted),
        **counts,
        "prediction_interval_coverage": interval_coverage,
        "prediction_interval_mean_width": interval_mean_width,
        "calibration_count": int(calibration_count),
        "interval_quantile_global": interval_quantile_global,
    }


def _error_records(
    rows: pd.DataFrame,
    predicted: np.ndarray,
    year: int,
) -> pd.DataFrame:
    result = rows[
        ["institute_type", "category", "round", "actual_closing_rank"]
    ].copy()
    result["year"] = int(year)
    result["predicted_closing_rank"] = np.asarray(predicted, dtype=float)
    result["residual"] = result["actual_closing_rank"] - result["predicted_closing_rank"]
    result["abs_residual"] = result["residual"].abs()
    result["rank_band"] = pd.cut(
        result["actual_closing_rank"],
        bins=RANK_BINS,
        labels=RANK_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    return result


def subgroup_metrics(errors: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    dimensions = {
        "year": "year",
        "institute_type": "institute_type",
        "category": "category",
        "rank_band": "rank_band",
    }
    for dimension, column in dimensions.items():
        for value, group in errors.groupby(column, dropna=False, sort=True):
            metrics = regression_metrics(
                group["actual_closing_rank"].to_numpy(),
                group["predicted_closing_rank"].to_numpy(),
            )
            records.append({"dimension": dimension, "value": str(value), **metrics})
    return pd.DataFrame(records)


def evaluate_rolling_origins(
    observed: pd.DataFrame,
    longitudinal: pd.DataFrame,
    evaluation_years: list[int],
    *,
    seed: int,
    origin_start_year: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    primary_features = features_for_mode(FeatureAvailabilityMode.PRE_COUNSELLING)
    assert_features_available(primary_features, FeatureAvailabilityMode.PRE_COUNSELLING)

    available_years = sorted(int(value) for value in observed["year"].unique())
    first_origin = (
        int(origin_start_year)
        if origin_start_year is not None
        else max(available_years[0] + 2, min(evaluation_years) - 2)
    )
    origins = [
        year
        for year in available_years
        if first_origin <= year <= max(evaluation_years)
    ]
    residual_history: dict[str, list[pd.DataFrame]] = {
        "ridge": [],
        "naive_last_year": [],
    }
    evaluation_errors: dict[str, list[pd.DataFrame]] = {
        "ridge": [],
        "naive_last_year": [],
    }
    metric_rows: list[dict[str, Any]] = []

    for target_year in origins:
        LOGGER.info("Rolling origin %s", target_year)
        past = observed.loc[observed["year"] < target_year]
        training = longitudinal.loc[longitudinal["year"] < target_year]
        target_frame = build_target_year_frame(
            past,
            target_year,
            mode=FeatureAvailabilityMode.PRE_COUNSELLING,
            snapshot_year=target_year - 1,
        )
        matched, counts = match_target_outcomes(target_frame, observed, target_year)
        if matched.empty:
            LOGGER.warning("No matched target outcomes for %s; skipping", target_year)
            continue

        actual = matched["actual_closing_rank"].to_numpy(dtype=float)
        ridge = ridge_pipeline(primary_features)
        ridge_predictions, elapsed = fit_predict(
            ridge,
            training,
            matched,
            primary_features,
        )
        naive_predictions = matched["last_closing_rank"].to_numpy(dtype=float)

        ridge_calibration = (
            pd.concat(residual_history["ridge"], ignore_index=True)
            if residual_history["ridge"]
            else pd.DataFrame()
        )
        ridge_coverage = None
        ridge_mean_width = None
        ridge_quantile = None
        if not ridge_calibration.empty:
            lower, upper, _radii, ridge_quantile = apply_prediction_interval(
                ridge_predictions,
                matched,
                ridge_calibration,
            )
            ridge_coverage = float(np.mean((actual >= lower) & (actual <= upper)))
            ridge_mean_width = float(np.mean(upper - lower))

        naive_calibration = (
            pd.concat(residual_history["naive_last_year"], ignore_index=True)
            if residual_history["naive_last_year"]
            else pd.DataFrame()
        )
        naive_coverage = None
        naive_mean_width = None
        naive_quantile = None
        if not naive_calibration.empty:
            lower, upper, _radii, naive_quantile = apply_prediction_interval(
                naive_predictions,
                matched,
                naive_calibration,
            )
            naive_coverage = float(np.mean((actual >= lower) & (actual <= upper)))
            naive_mean_width = float(np.mean(upper - lower))

        if target_year in evaluation_years:
            metric_rows.append(
                _metric_row(
                    year=target_year,
                    model="ridge",
                    actual=actual,
                    predicted=ridge_predictions,
                    counts=counts,
                    elapsed_seconds=elapsed,
                    interval_coverage=ridge_coverage,
                    interval_mean_width=ridge_mean_width,
                    calibration_count=len(ridge_calibration),
                    interval_quantile_global=ridge_quantile,
                )
            )

            metric_rows.append(
                _metric_row(
                    year=target_year,
                    model="naive_last_year",
                    actual=actual,
                    predicted=naive_predictions,
                    counts=counts,
                    elapsed_seconds=0.0,
                    interval_coverage=naive_coverage,
                    interval_mean_width=naive_mean_width,
                    calibration_count=len(naive_calibration),
                    interval_quantile_global=naive_quantile,
                )
            )
            metric_rows.append(
                _metric_row(
                    year=target_year,
                    model="historical_mean",
                    actual=actual,
                    predicted=matched["hist_mean_closing_rank"].to_numpy(dtype=float),
                    counts=counts,
                    elapsed_seconds=0.0,
                )
            )

            nonlinear = nonlinear_pipeline(primary_features, seed)
            nonlinear_predictions, nonlinear_elapsed = fit_predict(
                nonlinear,
                training,
                matched,
                primary_features,
            )
            metric_rows.append(
                _metric_row(
                    year=target_year,
                    model="hist_gradient_boosting",
                    actual=actual,
                    predicted=nonlinear_predictions,
                    counts=counts,
                    elapsed_seconds=nonlinear_elapsed,
                )
            )
            evaluation_errors["ridge"].append(
                _error_records(matched, ridge_predictions, target_year)
            )
            evaluation_errors["naive_last_year"].append(
                _error_records(matched, naive_predictions, target_year)
            )

        # This target's residuals become calibration evidence only for later years.
        residual_history["ridge"].append(
            _error_records(matched, ridge_predictions, target_year)
        )
        residual_history["naive_last_year"].append(
            _error_records(matched, naive_predictions, target_year)
        )

    if not residual_history["ridge"]:
        raise RuntimeError("Rolling-origin evaluation produced no residuals")
    return pd.DataFrame(metric_rows), {
        model: pd.concat(records, ignore_index=True)
        for model, records in evaluation_errors.items()
    }, {
        model: pd.concat(records, ignore_index=True)
        for model, records in residual_history.items()
    }


def select_model_before_final_test(
    fold_metrics: pd.DataFrame,
    final_test_year: int,
) -> tuple[str, pd.DataFrame]:
    """Select by mean rolling-origin MAE without reading final-test metrics."""

    validation = fold_metrics.loc[fold_metrics["year"] < final_test_year]
    if validation.empty:
        raise ValueError("At least one pre-final rolling origin is required for model selection")
    summary = (
        validation.groupby("model", sort=True)
        .agg(
            validation_origin_count=("year", "nunique"),
            validation_mean_mae=("mae", "mean"),
            validation_mean_rmse=("rmse", "mean"),
            validation_mean_median_absolute_error=("median_absolute_error", "mean"),
        )
        .reset_index()
        .sort_values(["validation_mean_mae", "model"], kind="mergesort")
        .reset_index(drop=True)
    )
    return str(summary.iloc[0]["model"]), summary


def evaluate_with_isolated_final_holdout(
    observed: pd.DataFrame,
    longitudinal: pd.DataFrame,
    evaluation_years: list[int],
    final_test_year: int,
    *,
    seed: int,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    str,
    pd.DataFrame,
]:
    """Select on earlier origins, then and only then evaluate the final holdout."""

    available_years = sorted(int(value) for value in observed["year"].unique())
    pre_final_years = [year for year in evaluation_years if year < final_test_year]
    if not pre_final_years:
        raise ValueError("At least one pre-final rolling origin is required for model selection")
    origin_start_year = max(available_years[0] + 2, min(evaluation_years) - 2)

    pre_final_metrics, pre_final_errors, _ = evaluate_rolling_origins(
        observed,
        longitudinal,
        pre_final_years,
        seed=seed,
        origin_start_year=origin_start_year,
    )
    selected_model, selection_summary = select_model_before_final_test(
        pre_final_metrics,
        final_test_year,
    )

    final_metrics, final_errors, residuals_by_model = evaluate_rolling_origins(
        observed,
        longitudinal,
        [final_test_year],
        seed=seed,
        origin_start_year=origin_start_year,
    )
    if set(pre_final_errors) != set(final_errors):
        raise RuntimeError("Pre-final and final evaluation error sets do not match")
    fold_metrics = pd.concat([pre_final_metrics, final_metrics], ignore_index=True)
    errors_by_model = {
        model: pd.concat([pre_final_errors[model], final_errors[model]], ignore_index=True)
        for model in pre_final_errors
    }
    return (
        fold_metrics,
        errors_by_model,
        residuals_by_model,
        selected_model,
        selection_summary,
    )


def evaluate_ablations(
    observed: pd.DataFrame,
    longitudinal: pd.DataFrame,
    final_test_year: int,
    primary_fold_metrics: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    primary_features = features_for_mode(FeatureAvailabilityMode.PRE_COUNSELLING)
    assert_features_available(primary_features, FeatureAvailabilityMode.PRE_COUNSELLING)
    past = observed.loc[observed["year"] < final_test_year]
    training = longitudinal.loc[longitudinal["year"] < final_test_year]
    target = build_target_year_frame(past, final_test_year, snapshot_year=final_test_year - 1)
    matched, _counts = match_target_outcomes(target, observed, final_test_year)
    actual = matched["actual_closing_rank"].to_numpy(dtype=float)

    primary_row = primary_fold_metrics.loc[
        (primary_fold_metrics["year"] == final_test_year)
        & (primary_fold_metrics["model"] == "ridge")
    ]
    if primary_row.empty:
        raise RuntimeError("Primary final-test Ridge metrics are missing")
    primary_metrics = {
        key: primary_row.iloc[0][key]
        for key in ("n", "mae", "rmse", "median_absolute_error", "r2")
    }

    def strict_record(
        *,
        variant: str,
        status: str,
        note: str,
        features: list[str],
        removed_features: list[str],
        metrics: dict[str, Any],
        policy_equivalent_to: str | None = None,
    ) -> dict[str, Any]:
        return {
            "variant": variant,
            "status": status,
            "feature_mode": FeatureAvailabilityMode.PRE_COUNSELLING.value,
            "policy_compliant": True,
            "eligible_for_primary_claim": True,
            "eligible_for_strict_comparison": True,
            "comparable_to_full_pre_counselling": True,
            "selected_as_primary_model": False,
            "policy_equivalent_to": policy_equivalent_to,
            "features": list(features),
            "feature_count": int(len(features)),
            "removed_features": list(removed_features),
            "note": note,
            **metrics,
        }

    records: list[dict[str, Any]] = [
        strict_record(
            variant="ridge_full_pre_counselling",
            status="evaluated_strict_reference",
            note=(
                "Full availability-correct Ridge reference on the final held-out year; "
                "model selection remains based only on earlier origins."
            ),
            features=primary_features,
            removed_features=[],
            metrics=primary_metrics,
        ),
        strict_record(
            variant="ridge_without_opening_rank_features",
            status="policy_equivalent_to_full_pre_counselling",
            note=(
                "Same fitted policy as the full strict reference because same-round opening "
                "rank is already excluded before model fitting."
            ),
            features=primary_features,
            removed_features=["opening_rank", "log_opening_rank", "opening_percentile"],
            metrics=primary_metrics,
            policy_equivalent_to="ridge_full_pre_counselling",
        ),
        strict_record(
            variant="ridge_without_previous_round_features",
            status="policy_equivalent_to_full_pre_counselling",
            note=(
                "Same fitted policy as the full strict reference because target-season "
                "previous-round outcomes are unavailable before counselling."
            ),
            features=primary_features,
            removed_features=["prev_round_closing_rank", "prev_round_number"],
            metrics=primary_metrics,
            policy_equivalent_to="ridge_full_pre_counselling",
        ),
    ]

    without_history = [
        feature for feature in primary_features if feature not in HISTORICAL_STAT_FEATURES
    ]
    no_history_model = ridge_pipeline(without_history)
    no_history_predictions, _ = fit_predict(
        no_history_model,
        training,
        matched,
        without_history,
    )
    records.append(
        strict_record(
            variant="ridge_without_historical_statistics",
            status="evaluated_strict_refit",
            note=(
                "Actual refit after removing all lagged closing-rank statistics; frequency "
                "counts remain so this isolates target-history signal from prevalence."
            ),
            features=without_history,
            removed_features=sorted(HISTORICAL_STAT_FEATURES),
            metrics=regression_metrics(actual, no_history_predictions),
        )
    )

    without_frequency = [
        feature for feature in primary_features if feature not in PAST_ONLY_FREQUENCY_FEATURES
    ]
    no_frequency_model = ridge_pipeline(without_frequency)
    no_frequency_predictions, _ = fit_predict(
        no_frequency_model,
        training,
        matched,
        without_frequency,
    )
    records.append(
        strict_record(
            variant="ridge_without_frequency_encodings",
            status="evaluated_strict_refit",
            note=(
                "Actual refit without the institute/program counts computed from strictly "
                "earlier years."
            ),
            features=without_frequency,
            removed_features=sorted(PAST_ONLY_FREQUENCY_FEATURES),
            metrics=regression_metrics(actual, no_frequency_predictions),
        )
    )

    # Leakage stress test: deliberately reproduce the prohibited experimental
    # choice to quantify why same-round opening ranks cannot support a forecast.
    target_actual = longitudinal.loc[longitudinal["year"] == final_test_year].copy()
    if "opening_rank" not in target_actual.columns or "opening_rank" not in training.columns:
        raise ValueError("Leakage diagnostic requires the observed opening_rank column")
    target_actual["log_opening_rank"] = np.log1p(
        target_actual["opening_rank"].to_numpy(dtype=float)
    )
    leaky_training = training.copy()
    leaky_training["log_opening_rank"] = np.log1p(
        leaky_training["opening_rank"].to_numpy(dtype=float)
    )
    leaky_features = [*primary_features, "opening_rank", "log_opening_rank"]
    leaky_model = ridge_pipeline(leaky_features)
    leaky_predictions, _ = fit_predict(
        leaky_model,
        leaky_training,
        target_actual,
        leaky_features,
    )
    records.append(
        {
            "variant": "legacy_same_round_opening_diagnostic",
            "status": "evaluated_leakage_stress_test",
            "feature_mode": "observed_same_round_leakage_diagnostic",
            "policy_compliant": False,
            "eligible_for_primary_claim": False,
            "eligible_for_strict_comparison": False,
            "comparable_to_full_pre_counselling": False,
            "selected_as_primary_model": False,
            "policy_equivalent_to": None,
            "features": leaky_features,
            "feature_count": int(len(leaky_features)),
            "removed_features": [],
            "ineligibility_reason": (
                "Reads observed same-round opening ranks and the realized target-year row "
                "universe, neither of which exists at the primary decision time."
            ),
            "note": (
                "Leakage diagnostic only; it is deliberately excluded from model selection, "
                "strict figures, ablation deltas, and every primary claim."
            ),
            **regression_metrics(
                target_actual["closing_rank"].to_numpy(dtype=float),
                leaky_predictions,
            ),
        }
    )
    return pd.DataFrame(records)


def evaluate_in_round_diagnostic(
    observed: pd.DataFrame,
    *,
    target_year: int = CANONICAL_IN_ROUND_YEAR,
    target_round: int = CANONICAL_IN_ROUND_ROUND,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate a fixed-round update and its no-previous-round comparator.

    This is deliberately not pooled with the primary pre-counselling folds. It
    predicts one round after earlier rounds in the same season are complete,
    so its information set and target subset answer a different question.
    """

    del seed  # Ridge is deterministic; retained in the public experiment API.
    if target_year not in set(observed["year"].astype(int).unique()):
        raise ValueError(f"In-round diagnostic target year {target_year} is absent")
    actual_rows = observed.loc[
        (observed["year"] == target_year) & (observed["round"] == target_round)
    ]
    if actual_rows.empty:
        raise ValueError(
            f"In-round diagnostic requires observed round {target_round} in {target_year}"
        )
    history = observed.loc[observed["year"] < target_year]
    if target_year - 1 not in set(history["year"].astype(int).unique()):
        raise ValueError(
            f"In-round diagnostic requires snapshot year {target_year - 1}"
        )
    completed = observed.loc[
        (observed["year"] == target_year) & (observed["round"] < target_round)
    ]
    if completed.empty:
        raise ValueError(
            f"In-round diagnostic requires completed rounds before {target_round} in {target_year}"
        )

    target = build_target_year_frame(
        history,
        target_year,
        mode=FeatureAvailabilityMode.IN_ROUND_UPDATE,
        snapshot_year=target_year - 1,
        target_rounds=[target_round],
        completed_rounds=completed,
    )
    # Coverage for this diagnostic is defined on the declared target-round
    # cohort, not on every observed row from the counselling season.
    matched, counts = match_target_outcomes(target, actual_rows, target_year)
    if matched.empty:
        raise RuntimeError("In-round diagnostic produced zero matched target outcomes")

    longitudinal = build_longitudinal_features(
        observed,
        FeatureAvailabilityMode.IN_ROUND_UPDATE,
    )
    training = longitudinal.loc[longitudinal["year"] < target_year]
    actual = matched["actual_closing_rank"].to_numpy(dtype=float)
    full_features = features_for_mode(FeatureAvailabilityMode.IN_ROUND_UPDATE)
    no_previous_features = [
        feature
        for feature in full_features
        if feature not in {"prev_round_closing_rank", "prev_round_number"}
    ]
    assert_features_available(full_features, FeatureAvailabilityMode.IN_ROUND_UPDATE)
    assert_features_available(
        no_previous_features,
        FeatureAvailabilityMode.IN_ROUND_UPDATE,
    )

    variants = (
        (
            "ridge_in_round_with_previous_round",
            full_features,
            [],
            (
                "Uses only the latest completed earlier round from the target season in "
                "addition to strictly prior-year features."
            ),
        ),
        (
            "ridge_in_round_without_previous_round",
            no_previous_features,
            ["prev_round_closing_rank", "prev_round_number"],
            (
                f"Comparator refit on the identical round-{target_round} target subset after removing "
                "both completed-previous-round fields."
            ),
        ),
    )
    records: list[dict[str, Any]] = []
    for variant, features, removed, note in variants:
        predictions, _ = fit_predict(
            ridge_pipeline(features),
            training,
            matched,
            features,
        )
        records.append(
            {
                "variant": variant,
                "status": "evaluated_non_primary_in_round_diagnostic",
                "evidence_role": "non_primary_in_round_diagnostic",
                "feature_mode": FeatureAvailabilityMode.IN_ROUND_UPDATE.value,
                "target_year": int(target_year),
                "target_round": int(target_round),
                "completed_rounds": sorted(
                    int(value) for value in completed["round"].unique()
                ),
                "training_years": sorted(
                    int(value) for value in training["year"].unique()
                ),
                "features": list(features),
                "feature_count": int(len(features)),
                "removed_features": removed,
                "eligible_for_primary_claim": False,
                "comparable_to_primary_pre_counselling": False,
                "comparable_within_in_round_diagnostic": True,
                "incomparability_reason": (
                    f"This fixed round-{target_round} subset is forecast after target-season "
                    f"rounds 1-{target_round - 1} "
                    "are observed; the primary benchmark forecasts all rounds before the "
                    "season and therefore has a different target and information set."
                ),
                "note": note,
                **counts,
                **regression_metrics(actual, predictions),
            }
        )
    return pd.DataFrame(records)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def generate_strict_figures(
    *,
    errors: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    output_dir: Path,
    selected_model: str,
) -> list[Path]:
    """Render the six canonical figures from this run's strict evidence."""

    if errors.empty:
        raise ValueError("Cannot render strict figures from an empty error frame")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predicted_vs_actual": output_dir / "strict_pred_vs_actual.png",
        "residual_distribution": output_dir / "strict_residual_distribution.png",
        "error_by_year": output_dir / "strict_error_by_year.png",
        "error_by_institute_type": output_dir / "strict_error_by_institute_type.png",
        "interval_coverage": output_dir / "strict_interval_coverage.png",
        "ablation": output_dir / "strict_ablation.png",
    }
    style = {
        "figure.figsize": (7.2, 4.6),
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
    save_options = {
        "dpi": 160,
        "bbox_inches": "tight",
        "metadata": {
            "Software": "SeatCraft strict research evaluator",
            "Creation Time": generated_at_utc(),
        },
    }

    with plt.rc_context(style):
        figure, axis = plt.subplots()
        axis.scatter(
            errors["actual_closing_rank"],
            errors["predicted_closing_rank"],
            s=8,
            alpha=0.25,
            edgecolors="none",
            color="#2563eb",
        )
        lower = float(
            min(errors["actual_closing_rank"].min(), errors["predicted_closing_rank"].min())
        )
        upper = float(
            max(errors["actual_closing_rank"].max(), errors["predicted_closing_rank"].max())
        )
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#111827", linewidth=1)
        axis.set(title=f"Strict predicted vs actual ({selected_model})", xlabel="Actual closing rank", ylabel="Predicted closing rank")
        figure.savefig(paths["predicted_vs_actual"], **save_options)
        plt.close(figure)

        figure, axis = plt.subplots()
        residuals = errors["residual"].dropna().to_numpy(dtype=float)
        lower_residual, upper_residual = np.quantile(residuals, [0.01, 0.99])
        central_residuals = residuals[
            (residuals >= lower_residual) & (residuals <= upper_residual)
        ]
        axis.hist(central_residuals, bins=60, color="#2563eb", alpha=0.85)
        axis.axvline(0.0, linestyle="--", color="#111827", linewidth=1)
        axis.set(
            title=f"Strict residual distribution: central 98% ({selected_model})",
            xlabel="Actual minus predicted closing rank",
            ylabel="Rows",
        )
        axis.text(
            0.99,
            0.97,
            (
                f"{len(central_residuals):,} of {len(residuals):,} rows shown; "
                "tails retained in metrics"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#374151",
        )
        figure.savefig(paths["residual_distribution"], **save_options)
        plt.close(figure)

        by_year = (
            errors.groupby("year", sort=True)["abs_residual"].mean().reset_index()
        )
        figure, axis = plt.subplots()
        axis.plot(by_year["year"], by_year["abs_residual"], marker="o", color="#2563eb")
        axis.set(
            title=f"Strict MAE by evaluation year ({selected_model})",
            xlabel="Evaluation year",
            ylabel="Mean absolute error (rank)",
        )
        axis.set_xticks(by_year["year"].astype(int).tolist())
        figure.savefig(paths["error_by_year"], **save_options)
        plt.close(figure)

        by_type = (
            errors.groupby("institute_type", sort=True)["abs_residual"]
            .mean()
            .sort_values(kind="mergesort")
        )
        figure, axis = plt.subplots()
        axis.bar(by_type.index.astype(str), by_type.to_numpy(), color="#2563eb")
        axis.set(
            title=f"Strict MAE by institute type ({selected_model})",
            xlabel="Institute type",
            ylabel="Mean absolute error (rank)",
        )
        figure.savefig(paths["error_by_institute_type"], **save_options)
        plt.close(figure)

        coverage = fold_metrics.loc[
            (fold_metrics["model"] == selected_model)
            & fold_metrics["prediction_interval_coverage"].notna(),
            ["year", "prediction_interval_coverage"],
        ].sort_values("year", kind="mergesort")
        figure, axis = plt.subplots()
        if coverage.empty:
            axis.text(
                0.5,
                0.5,
                "No origin had earlier calibration residuals",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            axis.plot(
                coverage["year"],
                coverage["prediction_interval_coverage"],
                marker="o",
                color="#2563eb",
                label="Empirical coverage",
            )
            axis.set_xticks(coverage["year"].astype(int).tolist())
        axis.axhline(TARGET_COVERAGE, linestyle="--", color="#b91c1c", label="Nominal 90%")
        axis.set(
            title=f"Strict empirical interval coverage ({selected_model})",
            xlabel="Evaluation year",
            ylabel="Covered fraction",
            ylim=(0.0, 1.05),
        )
        axis.legend(loc="best")
        figure.savefig(paths["interval_coverage"], **save_options)
        plt.close(figure)

        strict_ablations = ablations.loc[
            ablations["eligible_for_strict_comparison"].fillna(False)
            & ablations["mae"].notna()
        ].copy()
        if strict_ablations.empty:
            raise ValueError("No eligible strict ablations are available for plotting")
        label_map = {
            "ridge_full_pre_counselling": "Full strict Ridge",
            "ridge_without_opening_rank_features": "No opening rank (policy-equivalent)",
            "ridge_without_previous_round_features": "No previous round (policy-equivalent)",
            "ridge_without_historical_statistics": "No historical rank statistics",
            "ridge_without_frequency_encodings": "No past-only frequencies",
        }
        labels = [label_map.get(value, str(value)) for value in strict_ablations["variant"]]
        figure, axis = plt.subplots(figsize=(8.4, 5.0))
        positions = np.arange(len(strict_ablations))
        axis.barh(positions, strict_ablations["mae"].astype(float), color="#2563eb")
        axis.set_yticks(positions, labels=labels)
        axis.invert_yaxis()
        axis.set(
            title="Strict pre-counselling Ridge ablations",
            xlabel="Final-test mean absolute error (rank)",
        )
        figure.savefig(paths["ablation"], **save_options)
        plt.close(figure)

    return list(paths.values())


def file_metadata(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "role": role,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def code_hash_payload() -> list[dict[str, Any]]:
    paths = [
        REPO_ROOT / "packages/josaa-core/src/josaa_core/feature_schema.py",
        REPO_ROOT / "packages/josaa-core/src/josaa_core/research_features.py",
        REPO_ROOT / "packages/josaa-core/src/josaa_core/baselines.py",
        REPO_ROOT / "research/pipeline/research_evaluation.py",
        REPO_ROOT / "research/pipeline/build_registry.py",
    ]
    return [file_metadata(path, role="experiment_code") for path in paths]


def model_specifications(seed: int) -> dict[str, Any]:
    return {
        "naive_last_year": {
            "class": "LastObservationRegressor",
            "hyperparameters": {},
            "features": ["last_closing_rank"],
        },
        "historical_mean": {
            "class": "deterministic_historical_mean_baseline",
            "hyperparameters": {},
            "features": ["hist_mean_closing_rank"],
        },
        "ridge": {
            "class": "sklearn.linear_model.Ridge",
            "hyperparameters": dict(RIDGE_HYPERPARAMETERS),
            "preprocessing": {
                "numeric": "median_imputation_with_missing_indicators_then_standard_scaling",
                "categorical": "most_frequent_imputation_then_one_hot_min_frequency_2",
            },
            "features": features_for_mode(FeatureAvailabilityMode.PRE_COUNSELLING),
        },
        "hist_gradient_boosting": {
            "class": "sklearn.ensemble.HistGradientBoostingRegressor",
            "hyperparameters": {
                **HIST_GRADIENT_BOOSTING_HYPERPARAMETERS,
                "random_state": int(seed),
            },
            "preprocessing": {
                "numeric": "median_imputation_with_missing_indicators",
                "categorical": "most_frequent_imputation_then_unknown_safe_ordinal_encoding",
            },
            "features": features_for_mode(FeatureAvailabilityMode.PRE_COUNSELLING),
        },
    }


def write_serving_artifacts(
    *,
    observed: pd.DataFrame,
    longitudinal: pd.DataFrame,
    residuals: pd.DataFrame,
    artifact_dir: Path,
    data_cutoff_year: int,
    prediction_year: int,
    dataset_hash: str,
    seed: int,
    timestamp: str,
    selected_model: str,
    source_identity: dict[str, Any],
) -> list[Path]:
    if prediction_year <= data_cutoff_year:
        raise ValueError("prediction_year must be strictly greater than data_cutoff_year")
    if selected_model != "naive_last_year":
        raise ValueError(
            "Serving artifact writer currently supports the scientifically selected "
            "naive_last_year estimator only"
        )
    model_features = ["last_closing_rank"]
    assert_features_available(model_features, FeatureAvailabilityMode.PRE_COUNSELLING)
    training = longitudinal.loc[longitudinal["year"] <= data_cutoff_year]
    history = observed.loc[observed["year"] <= data_cutoff_year]
    universe = build_target_year_frame(
        history,
        prediction_year,
        snapshot_year=data_cutoff_year,
        mode=FeatureAvailabilityMode.PRE_COUNSELLING,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "pre_counselling_model.joblib"
    universe_path = artifact_dir / "inference_universe.csv"
    schema_path = artifact_dir / "pre_counselling_feature_schema.json"
    calibration_path = artifact_dir / "uncertainty_calibration.json"
    metadata_path = artifact_dir / "model_metadata.json"

    final_model = LastObservationRegressor()
    LOGGER.info("Materializing selected last-observation estimator through %s", data_cutoff_year)
    final_model.fit(universe[model_features])
    joblib.dump(final_model, model_path, compress=3)
    universe.to_csv(universe_path, index=False, lineterminator="\n")
    write_json(
        schema_path,
        feature_schema_payload(model_features, FeatureAvailabilityMode.PRE_COUNSELLING),
    )

    calibration = residuals.loc[residuals["year"] <= data_cutoff_year].copy()
    calibration_payload = uncertainty_payload(
        calibration,
        prediction_year=prediction_year,
        model_training_years=sorted(int(year) for year in training["year"].unique()),
        dataset_hash=dataset_hash,
        seed=seed,
    )
    calibration_payload["generated_at_utc"] = timestamp
    calibration_payload.update(timestamp_semantics())
    calibration_payload.update(source_identity)
    write_json(calibration_path, calibration_payload)

    model_hash = sha256_file(model_path)
    model_version = (
        f"pre-counselling-last-observation-cutoff{data_cutoff_year}-"
        f"seed{seed}-{model_hash[:12]}"
    )
    metadata = {
        "schema_version": 1,
        "model_version": model_version,
        "model_family": "LastObservationRegressor",
        "selection_metric": "mean_mae_across_pre_final_rolling_origins",
        "selection_used_final_test": False,
        "artifact_type": "full_sklearn_estimator",
        "feature_mode": FeatureAvailabilityMode.PRE_COUNSELLING.value,
        "data_cutoff_year": int(data_cutoff_year),
        "prediction_year": int(prediction_year),
        "snapshot_year": int(data_cutoff_year),
        "training_years": sorted(int(year) for year in training["year"].unique()),
        "features": model_features,
        "seed": int(seed),
        "dataset_sha256": dataset_hash,
        "generated_at_utc": timestamp,
        **timestamp_semantics(),
        **source_identity,
        "provenance_status": "unverified_local_dataset",
        "decision_support_disclaimer": "Decision support only; not an admission guarantee.",
        "artifacts": {
            model_path.name: {"sha256": model_hash, "bytes": model_path.stat().st_size},
            universe_path.name: {
                "sha256": sha256_file(universe_path),
                "bytes": universe_path.stat().st_size,
                "rows": int(len(universe)),
            },
            schema_path.name: {"sha256": sha256_file(schema_path), "bytes": schema_path.stat().st_size},
            calibration_path.name: {
                "sha256": sha256_file(calibration_path),
                "bytes": calibration_path.stat().st_size,
            },
        },
        "environment": environment_payload(),
    }
    write_json(metadata_path, metadata)
    return [model_path, universe_path, schema_path, calibration_path, metadata_path]


def generate_canonical_case_study(artifact_dir: Path) -> dict[str, Any]:
    """Call the real FastAPI route and retain one deterministic row per bucket."""

    inference_source = REPO_ROOT / "apps" / "inference-api" / "src"
    if str(inference_source) not in sys.path:
        sys.path.insert(0, str(inference_source))

    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from inference_api.config import settings as inference_settings
        from inference_api.routes import router
        from inference_api.services.artifact_loader import load_all_artifacts
    except ImportError as exc:  # pragma: no cover - exercised by environment smoke tests
        raise RuntimeError(
            "Canonical case-study generation requires the local inference API and its "
            "pinned FastAPI/httpx dependencies"
        ) from exc

    previous_artifact_dir = inference_settings.artifacts_dir
    inference_settings.artifacts_dir = artifact_dir.resolve()
    try:
        artifacts = load_all_artifacts()
    finally:
        inference_settings.artifacts_dir = previous_artifact_dir

    app = FastAPI()
    app.state.artifacts = artifacts
    app.state.artifact_error = None
    app.include_router(router, prefix="/api/v1")
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json=CANONICAL_CASE_STUDY_REQUEST,
        )
    if response.status_code != 200:
        raise RuntimeError(
            "Canonical case-study API call failed: "
            f"status={response.status_code}, body={response.text[:500]}"
        )
    body = response.json()
    result_rows = body.get("results")
    if not isinstance(result_rows, list) or not result_rows:
        raise RuntimeError("Canonical case-study API returned no recommendations")

    selected: list[dict[str, Any]] = []
    for bucket in CANONICAL_BUCKET_ORDER:
        row = next(
            (
                candidate
                for candidate in result_rows
                if candidate.get("recommendation_bucket") == bucket
            ),
            None,
        )
        if row is not None:
            selected.append(row)
    returned_buckets = {
        str(row.get("recommendation_bucket")) for row in result_rows
    }
    if {str(row.get("recommendation_bucket")) for row in selected} != returned_buckets:
        raise RuntimeError("Case-study selection did not retain one row from every returned bucket")

    validation_records: list[dict[str, Any]] = []
    for row in selected:
        rank_used = int(row["rank_used"])
        predicted = int(row["projected_closing_rank"])
        expected_margin = predicted - rank_used
        reported_margin = row.get("safety_margin")
        if reported_margin is None or int(reported_margin) != expected_margin:
            raise RuntimeError(
                "Case-study row has a margin inconsistent with its routed per-row rank: "
                f"id={row.get('id')} rank_used={rank_used} predicted={predicted} "
                f"reported_margin={reported_margin}"
            )
        institute_type = str(row.get("institute_type"))
        expected_rank = (
            CANONICAL_CASE_STUDY_REQUEST["advanced_rank"]
            if institute_type == "IIT"
            else CANONICAL_CASE_STUDY_REQUEST["main_rank"]
        )
        if rank_used != expected_rank:
            raise RuntimeError(
                "Case-study row used the wrong rank source: "
                f"id={row.get('id')} institute_type={institute_type} rank_used={rank_used}"
            )
        explanation = str(row.get("explanation", ""))
        if f"{predicted:,}" not in explanation or f"{abs(expected_margin):,}" not in explanation:
            raise RuntimeError(
                "Case-study explanation is not grounded in the row's predicted cutoff and "
                f"routed-rank margin: id={row.get('id')}"
            )
        validation_records.append(
            {
                "id": row.get("id"),
                "rank_used": rank_used,
                "rank_type_used": row.get("rank_type_used"),
                "expected_margin": expected_margin,
                "reported_margin": int(reported_margin),
                "explanation_mentions_predicted_cutoff_and_margin": True,
            }
        )

    return {
        "schema_version": 1,
        "status": "generated_from_real_inference_api",
        "endpoint": "/api/v1/predict",
        "request": dict(CANONICAL_CASE_STUDY_REQUEST),
        "selection_rule": (
            "First API-ordered recommendation from each bucket present in the returned "
            "top-100 result, using a fixed canonical bucket order."
        ),
        "bucket_order": list(CANONICAL_BUCKET_ORDER),
        "available_returned_buckets": [
            bucket for bucket in CANONICAL_BUCKET_ORDER if bucket in returned_buckets
        ],
        "selected_recommendations": selected,
        "selected_row_validation": validation_records,
        "response_metadata": {
            key: body.get(key)
            for key in (
                "total_options",
                "bucket_counts",
                "returned_bucket_counts",
                "institute_type_counts",
                "candidate_counts",
                "data_cutoff",
                "prediction_year",
                "model_version",
                "interval_label",
                "probability_method",
                "decision_support_disclaimer",
            )
        },
        "evidence_scope": (
            "Illustrative deterministic API regression case; it is not evidence of "
            "individual admission accuracy or an admission guarantee."
        ),
    }


def output_manifest(
    *,
    dataset_path: Path,
    dataset_hash: str,
    timestamp: str,
    output_paths: list[Path],
    figure_paths: list[Path],
    artifact_paths: list[Path],
    data_cutoff_year: int,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    def entry(path: Path, *, committed: bool, role: str) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(REPO_ROOT)),
            "role": role,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "committed_expected": committed,
        }

    return {
        "schema_version": 2,
        "generated_at_utc": timestamp,
        **timestamp_semantics(),
        **source_identity,
        "data_cutoff_year": int(data_cutoff_year),
        "source_assets": [
            {
                "path": str(dataset_path.relative_to(REPO_ROOT)),
                "role": "cleaned_observed_cutoffs",
                "sha256": dataset_hash,
                "bytes": dataset_path.stat().st_size,
                "committed": False,
                "availability": "local_only",
                "provenance": "unverified",
                "license": "unresolved",
                "redistribution": "not_authorized",
            }
        ],
        "versioned_outputs": [entry(path, committed=True, role="research_evidence") for path in output_paths],
        "figures": [
            entry(path, committed=True, role="strict_research_figure")
            for path in figure_paths
        ],
        "generated_serving_artifacts": [
            entry(
                path,
                committed=path.name
                in {
                    "pre_counselling_feature_schema.json",
                    "uncertainty_calibration.json",
                    "model_metadata.json",
                },
                role="generated_asset",
            )
            for path in artifact_paths
        ],
        "fresh_clone_status": "blocked_pending_provenance_verified_source_data",
        "recovery": (
            "Place provenance-verified JoSAA round files under research/data/ and run "
            "`.venv/bin/python -m research.pipeline.preprocessing --data-dir research/data`; "
            "then rerun "
            "the documented research_evaluation.py command."
        ),
    }


def run(args: argparse.Namespace) -> int:
    np.random.seed(args.seed)
    source_identity = source_snapshot_identity()
    dataset_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at_utc()
    dataset_hash = sha256_file(dataset_path) if dataset_path.exists() else "unavailable"

    observed = load_dataset(dataset_path, args.data_cutoff_year)
    years = sorted(int(value) for value in observed["year"].unique())
    evaluation_years = sorted(set(args.evaluation_years))
    missing_evaluation_years = sorted(set(evaluation_years) - set(years))
    if missing_evaluation_years:
        raise ResearchAssetError(
            f"Evaluation years absent from selected dataset: {missing_evaluation_years}"
        )
    if args.final_test_year not in evaluation_years:
        raise ValueError("final_test_year must be included in evaluation_years")
    if args.final_test_year != max(evaluation_years):
        raise ValueError("final_test_year must be the latest evaluation year")
    if args.prediction_year <= args.data_cutoff_year and not args.skip_serving_artifacts:
        raise ValueError("prediction_year must be strictly greater than data_cutoff_year")

    LOGGER.info("Building strictly lagged longitudinal features")
    longitudinal = build_longitudinal_features(
        observed,
        FeatureAvailabilityMode.PRE_COUNSELLING,
    )
    (
        fold_metrics,
        errors_by_model,
        residuals_by_model,
        selected_model,
        selection_summary,
    ) = evaluate_with_isolated_final_holdout(
        observed,
        longitudinal,
        evaluation_years,
        args.final_test_year,
        seed=args.seed,
    )
    LOGGER.info(
        "Selected %s by mean MAE on origins before %s",
        selected_model,
        args.final_test_year,
    )

    if selected_model not in errors_by_model or selected_model not in residuals_by_model:
        raise RuntimeError(
            f"Selected model {selected_model!r} has no stored subgroup/uncertainty residuals"
        )
    subgroup = subgroup_metrics(errors_by_model[selected_model])
    ablations = evaluate_ablations(
        observed,
        longitudinal,
        args.final_test_year,
        fold_metrics,
        seed=args.seed,
    )
    canonical_round_rows = observed.loc[
        (observed["year"] == CANONICAL_IN_ROUND_YEAR)
        & (observed["round"] == CANONICAL_IN_ROUND_ROUND)
    ]
    if canonical_round_rows.empty:
        in_round = pd.DataFrame(
            [
                {
                    "variant": "canonical_2025_round_6_in_round_diagnostic",
                    "status": "not_evaluated_target_absent",
                    "evidence_role": "non_primary_in_round_diagnostic",
                    "feature_mode": FeatureAvailabilityMode.IN_ROUND_UPDATE.value,
                    "target_year": CANONICAL_IN_ROUND_YEAR,
                    "target_round": CANONICAL_IN_ROUND_ROUND,
                    "eligible_for_primary_claim": False,
                    "comparable_to_primary_pre_counselling": False,
                    "note": "The selected dataset does not contain the canonical target subset.",
                }
            ]
        )
    else:
        LOGGER.info(
            "Evaluating non-primary %s round-%s in-round diagnostic",
            CANONICAL_IN_ROUND_YEAR,
            CANONICAL_IN_ROUND_ROUND,
        )
        in_round = evaluate_in_round_diagnostic(
            observed,
            target_year=CANONICAL_IN_ROUND_YEAR,
            target_round=CANONICAL_IN_ROUND_ROUND,
            seed=args.seed,
        )

    fold_path = output_dir / "research_fold_metrics.csv"
    subgroup_path = output_dir / "research_subgroup_metrics.csv"
    ablation_path = output_dir / "research_ablation_metrics.csv"
    in_round_path = output_dir / "research_in_round_diagnostic.csv"
    selection_path = output_dir / "research_model_selection.csv"
    result_path = output_dir / "metrics.json"
    manifest_path = output_dir / "artifact_manifest.json"
    fold_metrics.to_csv(fold_path, index=False, lineterminator="\n", float_format="%.10g")
    subgroup.to_csv(subgroup_path, index=False, lineterminator="\n", float_format="%.10g")
    ablations.to_csv(ablation_path, index=False, lineterminator="\n", float_format="%.10g")
    in_round.to_csv(in_round_path, index=False, lineterminator="\n", float_format="%.10g")
    selection_summary.to_csv(
        selection_path,
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )
    figure_paths = generate_strict_figures(
        errors=errors_by_model[selected_model],
        fold_metrics=fold_metrics,
        ablations=ablations,
        output_dir=output_dir / "figures",
        selected_model=selected_model,
    )
    figure_records = [
        file_metadata(path, role="strict_research_figure") for path in figure_paths
    ]

    final_row = fold_metrics.loc[
        (fold_metrics["year"] == args.final_test_year)
        & (fold_metrics["model"] == selected_model)
    ].iloc[0]
    final_comparison = fold_metrics.loc[
        fold_metrics["year"] == args.final_test_year,
        ["model", "n", "mae", "rmse", "median_absolute_error", "r2"],
    ].sort_values(["mae", "model"], kind="mergesort")
    macro = (
        fold_metrics.groupby("model", sort=True)[
            ["mae", "rmse", "median_absolute_error", "r2"]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    training_years = [year for year in years if year < args.final_test_year]
    validation_years = [year for year in evaluation_years if year < args.final_test_year]
    experiment_fingerprint = hashlib.sha256(
        (
            f"{dataset_hash}|{FeatureAvailabilityMode.PRE_COUNSELLING.value}|"
            f"{evaluation_years}|{args.final_test_year}|{args.seed}|"
            f"{features_for_mode(FeatureAvailabilityMode.PRE_COUNSELLING)}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    experiment_id = (
        f"seatcraft-strict-pre-counselling-{min(evaluation_years)}-"
        f"{args.final_test_year}-seed{args.seed}-{experiment_fingerprint}"
    )
    dataset_version = (
        f"local-unverified-cutoff-{args.data_cutoff_year}-{dataset_hash[:12]}"
    )
    dataset_quality = observed.attrs.get("dataset_quality", {})
    results = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": "exploratory_unverified_data_provenance",
        "generated_at_utc": timestamp,
        **timestamp_semantics(),
        **source_identity,
        "research_question": (
            "How accurately can JoSAA closing ranks be forecast using only information "
            "available before a target counselling season?"
        ),
        "primary_feature_mode": FeatureAvailabilityMode.PRE_COUNSELLING.value,
        "primary_model": selected_model,
        "selection_policy": (
            f"Lowest mean MAE across rolling origins strictly before the {args.final_test_year} "
            "final test; "
            "the final test was opened only after selection."
        ),
        "model_selection": {
            "selection_years": validation_years,
            "criterion": "mean_mae",
            "selected_model": selected_model,
            "selection_table_path": str(selection_path.relative_to(REPO_ROOT)),
            "ridge_status": "availability-correct experimental comparator; not selected",
            "selection_table": _clean_for_json(
                selection_summary.to_dict(orient="records")
            ),
        },
        "random_seed": int(args.seed),
        "dataset": {
            "id": DATASET_ID,
            "dataset_id": DATASET_ID,
            "version": dataset_version,
            "dataset_version": dataset_version,
            "path": str(dataset_path.relative_to(REPO_ROOT)),
            "sha256": dataset_hash,
            "bytes": dataset_path.stat().st_size,
            "selected_rows": int(len(observed)),
            "years": years,
            "data_cutoff_year": int(args.data_cutoff_year),
            "provenance": "unverified",
            "license": "unresolved",
            "quality": _clean_for_json(dataset_quality),
        },
        "periods": {
            "training": {
                "role": "expanding-window fit data for the final test",
                "start_year": min(training_years),
                "end_year": max(training_years),
                "years": training_years,
            },
            "validation": {
                "role": "rolling origins used for model selection",
                "start_year": min(validation_years),
                "end_year": max(validation_years),
                "years": validation_years,
            },
            "test": {
                "role": "untouched final temporal holdout opened after selection",
                "start_year": int(args.final_test_year),
                "end_year": int(args.final_test_year),
                "years": [int(args.final_test_year)],
            },
            "overlap_note": (
                "Validation origins are expanding-window forecasts; their observed outcomes "
                "subsequently enter later training windows. The final test year never enters "
                "model selection."
            ),
        },
        "feature_sets": {
            "pre_counselling": features_for_mode(
                FeatureAvailabilityMode.PRE_COUNSELLING
            ),
            "in_round_update": features_for_mode(
                FeatureAvailabilityMode.IN_ROUND_UPDATE
            ),
            "past_only_frequency_features": sorted(PAST_ONLY_FREQUENCY_FEATURES),
            "historical_rank_stat_features": sorted(HISTORICAL_STAT_FEATURES),
        },
        "models": model_specifications(args.seed),
        "evaluation": {
            "design": "expanding_window_rolling_origin",
            "evaluation_years": evaluation_years,
            "final_test_year": int(args.final_test_year),
            "target_frame": "prior_year_program_universe_plus_strictly_past_statistics",
            "final_test_metrics": _clean_for_json(final_row.to_dict()),
            "final_test_model_comparison": _clean_for_json(
                final_comparison.to_dict(orient="records")
            ),
            "macro_average_across_origins": _clean_for_json(macro),
            "fold_metrics_path": str(fold_path.relative_to(REPO_ROOT)),
            "subgroup_metrics_path": str(subgroup_path.relative_to(REPO_ROOT)),
            "ablation_metrics_path": str(ablation_path.relative_to(REPO_ROOT)),
            "in_round_diagnostic_path": str(in_round_path.relative_to(REPO_ROOT)),
            "fold_metrics": _clean_for_json(fold_metrics.to_dict(orient="records")),
            "subgroup_results": _clean_for_json(subgroup.to_dict(orient="records")),
            "ablation_results": _clean_for_json(ablations.to_dict(orient="records")),
            "in_round_diagnostic": _clean_for_json(in_round.to_dict(orient="records")),
        },
        "subgroup_results": _clean_for_json(subgroup.to_dict(orient="records")),
        "ablation_results": _clean_for_json(ablations.to_dict(orient="records")),
        "in_round_diagnostic": _clean_for_json(in_round.to_dict(orient="records")),
        "uncertainty": {
            "method": "rolling_origin_oof_absolute_residual",
            "name": "empirical_prediction_interval",
            "nominal_coverage": TARGET_COVERAGE,
            "formal_coverage_guarantee": False,
            "final_test_empirical_coverage": _clean_for_json(final_row["prediction_interval_coverage"]),
            "final_test_calibration_count": int(final_row["calibration_count"]),
            "final_test_residuals_used_for_interval": False,
            "residual_model": selected_model,
        },
        "known_evidence_limits": [
            "Raw source URLs, retrieval records, licensing, and immutable raw files are absent.",
            "The generated frame evaluates only programs/profile rows present in the declared prior-year universe.",
            "Prediction intervals are empirical OOF intervals, not confidence intervals or formal conformal guarantees.",
            "The fixed-round in-round diagnostic uses a different target subset and later information set, so it is non-primary and incomparable to the all-round pre-counselling benchmark.",
            "Ridge and nonlinear models did not beat the last-year baseline on the pre-final MAE selection criterion.",
        ],
        "figures": figure_records,
        "code_hashes": code_hash_payload(),
        "environment": environment_payload(),
    }

    artifact_paths: list[Path] = []
    if not args.skip_serving_artifacts:
        artifact_paths = write_serving_artifacts(
            observed=observed,
            longitudinal=longitudinal,
            residuals=residuals_by_model[selected_model],
            artifact_dir=artifact_dir,
            data_cutoff_year=args.data_cutoff_year,
            prediction_year=args.prediction_year,
            dataset_hash=dataset_hash,
            seed=args.seed,
            timestamp=timestamp,
            selected_model=selected_model,
            source_identity=source_identity,
        )
        results["serving_artifacts"] = [
            file_metadata(path, role="generated_serving_artifact")
            for path in artifact_paths
        ]
        LOGGER.info("Calling real inference API for canonical case-study payload")
        results["case_study"] = generate_canonical_case_study(artifact_dir)
    else:
        results["serving_artifacts"] = []
        results["case_study"] = {
            "status": "skipped_by_cli",
            "reason": "--skip-serving-artifacts explicitly disables artifact and API generation",
            "request_would_have_been": dict(CANONICAL_CASE_STUDY_REQUEST),
        }

    write_json(result_path, _clean_for_json(results))

    registry_paths: list[Path] = []
    if not args.skip_serving_artifacts:
        registry_path = output_dir / "model_registry.csv"
        build_registry(
            result_path,
            artifact_dir / "model_metadata.json",
            registry_path,
        )
        registry_paths.append(registry_path)

    versioned_outputs = [
        result_path,
        fold_path,
        subgroup_path,
        ablation_path,
        in_round_path,
        selection_path,
        *figure_paths,
        *registry_paths,
    ]
    manifest = output_manifest(
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        timestamp=timestamp,
        output_paths=versioned_outputs,
        figure_paths=figure_paths,
        artifact_paths=artifact_paths,
        data_cutoff_year=args.data_cutoff_year,
        source_identity=source_identity,
    )
    write_json(manifest_path, manifest)

    LOGGER.info("Research results: %s", result_path)
    LOGGER.info("Final test selected-model metrics: %s", regression_metrics(
        np.array([0.0]), np.array([0.0])
    ) if False else _clean_for_json(final_row.to_dict()))
    LOGGER.info("Manifest: %s", manifest_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "research/data/processed/josaa_cleaned.csv",
        help="Cleaned observed cutoff CSV (local-only until provenance is resolved).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research/reports",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "research/artifacts",
    )
    parser.add_argument(
        "--evaluation-years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024, 2025],
    )
    parser.add_argument("--final-test-year", type=int, default=2025)
    parser.add_argument(
        "--data-cutoff-year",
        type=int,
        default=2025,
        help="Explicit last complete year; never inferred from the maximum observed year.",
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-serving-artifacts",
        action="store_true",
        help="Evaluate only; do not generate local model/universe/calibration assets.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(parse_args(argv))
    except (ResearchAssetError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
