"""
JoSAA Day 3 — Model Training & Benchmarking Pipeline
======================================================

Reads the Day 2 feature-engineered train/valid/test splits, trains multiple
regression models to predict ``closing_rank``, compares them fairly, tunes
the best, and produces a comprehensive evaluation report.

Secondary task: classification sanity-check on ``cutoff_difficulty``.

Outputs:
    reports/regression_benchmark.csv|md     – model comparison table
    reports/classification_benchmark.csv|md – classification comparison
    reports/day3_model_report.md            – full narrative report
    model_artifacts/final_regression_model.joblib
    model_artifacts/preprocessing_pipeline.joblib
    model_artifacts/all_models.joblib
    model_artifacts/predictions_test.csv
    model_artifacts/residuals_analysis.csv
    model_artifacts/feature_importance.csv
    figures/feature_importance.png
    figures/pred_vs_actual.png
    figures/residuals_plot.png
    figures/error_by_year.png
    figures/error_by_category.png
    figures/error_by_institute_type.png
    figures/confusion_matrix.png

Usage:
    python train_models.py                     # defaults
    python train_models.py --data-dir ./data   # custom data dir

The module is also fully importable:
    from train_models import run_day3_pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.josaa_core.feature_schema import (
    MODEL_FEATURES,
    assert_no_forbidden_features,
    feature_schema_payload,
)

# ---------------------------------------------------------------------------
# Optional imports (may not be installed)
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
sns.set_theme(style="whitegrid", palette="deep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGRESSION_TARGET = "closing_rank"
CLASSIFICATION_TARGET = "cutoff_difficulty"

# Columns used as model features (all others are identifiers or targets)
# year and round ARE features for the model (temporal signal)
FEATURE_COLS = MODEL_FEATURES

# Columns that benefit from scaling (for Ridge / MLP)
SCALE_SENSITIVE_FEATURES = [
    "opening_rank", "log_opening_rank",
    "institute_freq", "program_freq", "category_freq", "institute_type_freq",
    "hist_mean_closing_rank", "hist_std_closing_rank",
    "hist_min_closing_rank", "hist_max_closing_rank", "hist_year_count",
    "prev_round_closing_rank",
    "opening_percentile", "applicants",
    "year", "round", "duration_years", "years_since_ews",
]

# Label encoder mappings (for reverse-lookup in error analysis)
CATEGORY_DECODE = {0: "EWS", 1: "OBC-NCL", 2: "OPEN", 3: "SC", 4: "ST"}
INST_TYPE_DECODE = {0: "GFTI", 1: "IIIT", 2: "IIT", 3: "NIT"}


# =========================================================================
# 1. Data Loading
# =========================================================================

def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-split regression datasets from Day 2.

    Returns (train_df, valid_df, test_df) as raw DataFrames.
    """
    feature_dir = data_dir / "features"

    train = pd.read_csv(feature_dir / "regression_train.csv")
    valid = pd.read_csv(feature_dir / "regression_valid.csv")
    test = pd.read_csv(feature_dir / "regression_test.csv")

    log.info("Loaded regression splits:")
    log.info("  Train: %d rows × %d cols  (years %s)",
             len(train), len(train.columns), sorted(train["year"].unique()))
    log.info("  Valid: %d rows × %d cols  (years %s)",
             len(valid), len(valid.columns), sorted(valid["year"].unique()))
    log.info("  Test:  %d rows × %d cols  (years %s)",
             len(test), len(test.columns), sorted(test["year"].unique()))

    # Verify temporal integrity
    train_years = set(train["year"].unique())
    valid_years = set(valid["year"].unique())
    test_years = set(test["year"].unique())
    assert train_years.isdisjoint(valid_years), "Train/valid year leakage!"
    assert train_years.isdisjoint(test_years), "Train/test year leakage!"
    assert valid_years.isdisjoint(test_years), "Valid/test year leakage!"
    assert max(train_years) < min(valid_years), "Train must precede valid!"
    assert max(valid_years) < min(test_years), "Valid must precede test!"
    log.info("  ✓ Temporal integrity verified — no leakage")

    return train, valid, test


def load_classification_data(
    data_dir: Path,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load pre-split classification datasets from Day 2.

    Returns (train, valid, test) or (None, None, None) if files don't exist.
    """
    feature_dir = data_dir / "features"
    paths = [
        feature_dir / "classification_train.csv",
        feature_dir / "classification_valid.csv",
        feature_dir / "classification_test.csv",
    ]

    if not all(p.exists() for p in paths):
        log.warning("Classification datasets not found — skipping classification")
        return None, None, None

    train = pd.read_csv(paths[0])
    valid = pd.read_csv(paths[1])
    test = pd.read_csv(paths[2])

    if CLASSIFICATION_TARGET not in train.columns:
        log.warning("Classification target '%s' not found — skipping",
                    CLASSIFICATION_TARGET)
        return None, None, None

    log.info("Loaded classification splits:")
    log.info("  Train: %d rows (label=1: %.1f%%)",
             len(train), train[CLASSIFICATION_TARGET].mean() * 100)
    log.info("  Valid: %d rows (label=1: %.1f%%)",
             len(valid), valid[CLASSIFICATION_TARGET].mean() * 100)
    log.info("  Test:  %d rows (label=1: %.1f%%)",
             len(test), test[CLASSIFICATION_TARGET].mean() * 100)

    return train, valid, test


# =========================================================================
# 2. Feature / Target Splitting
# =========================================================================

def split_features_target(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix X and target vector y.

    Only keeps columns that actually exist in the DataFrame.
    Converts boolean columns to int for sklearn compatibility.
    """
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.debug("Feature cols not found (OK): %s", missing)

    X = df[available].copy()

    # Convert booleans to int
    bool_cols = X.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    y = df[target].copy()

    return X, y


# =========================================================================
# 3. Preprocessing Pipeline
# =========================================================================

def build_preprocessing_pipeline(
    X_train: pd.DataFrame,
) -> Pipeline:
    """
    Build and fit a preprocessing pipeline.

    - Imputes missing values with median
    - Does NOT scale (tree models don't need it)
    - For Ridge/MLP, scaling is applied inside model-specific wrappers

    Returns a fitted Pipeline.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    pipeline.fit(X_train)
    log.info("Preprocessing pipeline fitted on %d features", len(numeric_cols))

    return pipeline


def build_scaled_pipeline(
    X_train: pd.DataFrame,
) -> Pipeline:
    """
    Build a pipeline that imputes AND scales — used for Ridge and MLP.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    pipeline.fit(X_train)
    return pipeline


# =========================================================================
# 4. Metrics Computation
# =========================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute regression metrics: MAE, RMSE, R², MedAE, MAPE (safe).

    MAPE is guarded against zeros — computed only where y_true > 0.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)

    # MAPE — safe computation (skip zeros)
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}MAE": round(mae, 2),
        f"{p}RMSE": round(rmse, 2),
        f"{p}R2": round(r2, 6),
        f"{p}MedAE": round(med_ae, 2),
        f"{p}MAPE": round(mape, 2) if not np.isnan(mape) else np.nan,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC."""
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["ROC_AUC"] = np.nan
    return metrics


# =========================================================================
# 5. Regression Model Training
# =========================================================================

def _get_regression_models() -> dict[str, Any]:
    """
    Return a dict of {name: model_instance} for regression benchmarking.

    Models use sensible defaults — tuning comes later for top performers.
    """
    models: dict[str, Any] = {}

    # 1. Linear baseline
    models["Ridge"] = Ridge(alpha=1.0)

    # 2. Random Forest
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    # 3. Extra Trees
    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    # 4. HistGradientBoosting (fast, handles NaN natively)
    models["HistGradientBoosting"] = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
    )

    # 6. XGBoost (if available)
    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )

    # 7. LightGBM (if available)
    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )

    return models


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    X_train_scaled: np.ndarray,
    X_valid_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Train all regression models, evaluate on all splits.

    Returns:
        benchmark_df: DataFrame with metrics for each model × split
        trained_models: dict of {name: fitted_model}
    """
    models = _get_regression_models()
    results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    log.info("=" * 60)
    log.info("TRAINING %d REGRESSION MODELS", len(models))
    log.info("=" * 60)

    for name, model in models.items():
        log.info("─" * 40)
        log.info("Training: %s", name)
        t0 = time.time()

        try:
            # Ridge and MLP need scaled features
            needs_scaling = name in ("Ridge", "MLP")
            Xtr = X_train_scaled if needs_scaling else X_train
            Xva = X_valid_scaled if needs_scaling else X_valid
            Xte = X_test_scaled if needs_scaling else X_test

            model.fit(Xtr, y_train)
            elapsed = time.time() - t0

            # Predict on all splits
            pred_train = model.predict(Xtr)
            pred_valid = model.predict(Xva)
            pred_test = model.predict(Xte)

            # Compute metrics
            m_train = compute_regression_metrics(y_train, pred_train, "train")
            m_valid = compute_regression_metrics(y_valid, pred_valid, "valid")
            m_test = compute_regression_metrics(y_test, pred_test, "test")

            row = {"Model": name, "train_time_s": round(elapsed, 1)}
            row.update(m_train)
            row.update(m_valid)
            row.update(m_test)
            results.append(row)
            trained_models[name] = model

            log.info("  Train MAE=%8.1f  R²=%.4f", m_train["train_MAE"], m_train["train_R2"])
            log.info("  Valid MAE=%8.1f  R²=%.4f", m_valid["valid_MAE"], m_valid["valid_R2"])
            log.info("  Test  MAE=%8.1f  R²=%.4f", m_test["test_MAE"], m_test["test_R2"])
            log.info("  Time: %.1fs", elapsed)

        except Exception as e:
            log.error("  FAILED: %s — %s", name, e)
            results.append({"Model": name, "train_time_s": np.nan,
                            "valid_MAE": np.nan, "valid_R2": np.nan})

    benchmark_df = pd.DataFrame(results)

    # Sort by validation MAE (primary metric)
    if "valid_MAE" in benchmark_df.columns:
        benchmark_df = benchmark_df.sort_values("valid_MAE", na_position="last")

    log.info("=" * 60)
    log.info("BENCHMARK SUMMARY (sorted by valid MAE)")
    log.info("=" * 60)
    summary_cols = ["Model", "train_time_s",
                    "train_MAE", "train_R2",
                    "valid_MAE", "valid_R2",
                    "test_MAE", "test_R2"]
    summary_cols = [c for c in summary_cols if c in benchmark_df.columns]
    print(benchmark_df[summary_cols].to_string(index=False))
    print()

    return benchmark_df, trained_models


# =========================================================================
# 6. Hyperparameter Tuning
# =========================================================================

def _get_tuning_params(model_name: str) -> dict[str, list]:
    """Return a sensible hyperparameter search space for the given model."""
    params: dict[str, dict] = {
        "HistGradientBoosting": {
            "max_iter": [300, 500, 800],
            "max_depth": [6, 8, 12],
            "learning_rate": [0.05, 0.1, 0.15],
            "min_samples_leaf": [10, 20, 50],
            "max_leaf_nodes": [31, 63, 127],
        },
        "XGBoost": {
            "n_estimators": [300, 500, 800],
            "max_depth": [6, 8, 10],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [5, 10, 20],
        },
        "LightGBM": {
            "n_estimators": [300, 500, 800],
            "max_depth": [6, 8, 12],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [10, 20, 50],
        },
        "RandomForest": {
            "n_estimators": [200, 400, 600],
            "max_depth": [15, 20, 30, None],
            "min_samples_leaf": [3, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "ExtraTrees": {
            "n_estimators": [200, 400, 600],
            "max_depth": [15, 20, 30, None],
            "min_samples_leaf": [3, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "GradientBoosting": {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9],
            "min_samples_leaf": [5, 10, 20],
        },
    }
    return params.get(model_name, {})


def tune_top_models(
    benchmark_df: pd.DataFrame,
    trained_models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_train_scaled: np.ndarray,
    X_valid_scaled: np.ndarray,
    feature_names: list[str],
    n_top: int = 2,
    n_iter: int = 15,
) -> tuple[str, Any, dict[str, Any]]:
    """
    Tune the top-N models by validation MAE using RandomizedSearchCV.

    Uses the validation set directly (not cross-validation) to stay
    consistent with our temporal split philosophy.

    Returns:
        best_name: name of the best model after tuning
        best_model: the fitted best model
        tuned_models: dict of all tuned models
    """
    valid_benchmark = benchmark_df.dropna(subset=["valid_MAE"])
    top_models = valid_benchmark.head(n_top)["Model"].tolist()

    log.info("=" * 60)
    log.info("TUNING TOP-%d MODELS: %s", n_top, top_models)
    log.info("=" * 60)

    tuned_results: list[dict[str, Any]] = []
    tuned_models: dict[str, Any] = {}

    for name in top_models:
        params = _get_tuning_params(name)
        if not params:
            log.info("  No tuning params for %s — keeping default", name)
            tuned_models[name] = trained_models[name]
            continue

        log.info("  Tuning %s with %d-iter randomized search…", name, n_iter)
        t0 = time.time()

        needs_scaling = name in ("Ridge", "MLP")
        Xtr = X_train_scaled if needs_scaling else X_train
        Xva = X_valid_scaled if needs_scaling else X_valid

        # Get a fresh model instance
        base_models = _get_regression_models()
        base_model = base_models[name]

        # Use RandomizedSearchCV with a single pre-defined split
        # We pass combined train+valid as X, with cv=[(train_idx, valid_idx)]
        X_combined = np.vstack([Xtr, Xva])
        y_combined = np.concatenate([y_train, y_valid])
        train_idx = np.arange(len(Xtr))
        valid_idx = np.arange(len(Xtr), len(X_combined))
        cv_split = [(train_idx, valid_idx)]

        search = RandomizedSearchCV(
            base_model,
            param_distributions=params,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv_split,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            refit=True,
        )

        try:
            search.fit(X_combined, y_combined)
            best = search.best_estimator_
            elapsed = time.time() - t0

            pred_valid = best.predict(Xva)
            m_valid = compute_regression_metrics(y_valid, pred_valid, "valid")

            tuned_models[name] = best
            log.info("    Best params: %s", search.best_params_)
            log.info("    Valid MAE: %.1f  R²: %.4f  (%.1fs)",
                     m_valid["valid_MAE"], m_valid["valid_R2"], elapsed)

            tuned_results.append({
                "Model": f"{name} (tuned)",
                "valid_MAE": m_valid["valid_MAE"],
                "valid_R2": m_valid["valid_R2"],
                "best_params": search.best_params_,
            })

        except Exception as e:
            log.error("    Tuning FAILED for %s: %s", name, e)
            tuned_models[name] = trained_models[name]

    # Pick overall best by valid MAE
    all_candidates: list[tuple[str, float]] = []
    for name, model in tuned_models.items():
        needs_scaling = name in ("Ridge", "MLP")
        Xva = X_valid_scaled if needs_scaling else X_valid
        pred = model.predict(Xva)
        mae = mean_absolute_error(y_valid, pred)
        all_candidates.append((name, mae))

    # Also include un-tuned models that weren't in top-N
    for name, model in trained_models.items():
        if name not in tuned_models:
            needs_scaling = name in ("Ridge", "MLP")
            Xva = X_valid_scaled if needs_scaling else X_valid
            pred = model.predict(Xva)
            mae = mean_absolute_error(y_valid, pred)
            all_candidates.append((name, mae))

    all_candidates.sort(key=lambda x: x[1])
    best_name = all_candidates[0][0]
    best_model = tuned_models.get(best_name, trained_models[best_name])

    log.info("=" * 60)
    log.info("BEST MODEL: %s  (valid MAE = %.1f)", best_name, all_candidates[0][1])
    log.info("=" * 60)

    return best_name, best_model, tuned_models


# =========================================================================
# 7. Refit & Final Evaluation
# =========================================================================

def refit_and_evaluate(
    best_name: str,
    best_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_scaled: np.ndarray,
    X_valid_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_names: list[str],
) -> tuple[Any, np.ndarray, dict[str, float]]:
    """
    Refit the best model on train+valid, then evaluate once on test.

    Returns:
        final_model: refitted model
        test_preds: predictions on test set
        final_metrics: test metrics dict
    """
    log.info("Refitting %s on train + valid (%d + %d = %d rows)…",
             best_name, len(y_train), len(y_valid),
             len(y_train) + len(y_valid))

    needs_scaling = best_name in ("Ridge", "MLP")
    Xtr = X_train_scaled if needs_scaling else X_train
    Xva = X_valid_scaled if needs_scaling else X_valid
    Xte = X_test_scaled if needs_scaling else X_test

    # Clone the best model's params and refit on train+valid
    from sklearn.base import clone
    final_model = clone(best_model)

    X_combined = np.vstack([Xtr, Xva])
    y_combined = np.concatenate([y_train, y_valid])

    t0 = time.time()
    final_model.fit(X_combined, y_combined)
    elapsed = time.time() - t0
    log.info("  Refitting took %.1fs", elapsed)

    test_preds = final_model.predict(Xte)
    final_metrics = compute_regression_metrics(y_test, test_preds, "final_test")

    log.info("=" * 60)
    log.info("FINAL TEST EVALUATION (%s, refitted on train+valid)", best_name)
    log.info("=" * 60)
    for k, v in final_metrics.items():
        log.info("  %s = %s", k, v)

    return final_model, test_preds, final_metrics


# =========================================================================
# 8. Classification Sanity Check
# =========================================================================

def _get_classification_models() -> dict[str, Any]:
    """Return classification models for the sanity check."""
    models: dict[str, Any] = {}

    models["LogisticRegression"] = LogisticRegression(
        max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1, random_state=42,
    )
    models["RandomForest_Clf"] = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    )

    if HAS_XGBOOST:
        models["XGBoost_Clf"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric="logloss",
        )

    if HAS_LIGHTGBM:
        models["LightGBM_Clf"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbose=-1,
        )

    return models


def run_classification_sanity_check(
    data_dir: Path,
    preprocess_pipeline: Pipeline,
    scaled_pipeline: Pipeline,
    figures_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Train classification models on the cutoff_difficulty target.

    This is a secondary sanity-check — keeps regression as the main focus.
    """
    cls_train, cls_valid, cls_test = load_classification_data(data_dir)
    if cls_train is None:
        return None

    log.info("=" * 60)
    log.info("CLASSIFICATION SANITY CHECK")
    log.info("=" * 60)

    # Use same feature cols (minus regression-specific ones)
    cls_feature_cols = [c for c in FEATURE_COLS
                        if c in cls_train.columns
                        and c != "closing_rank_vs_hist_mean"
                        and c != "closing_rank_ratio_hist"
                        and c != "competitiveness_ratio"]

    X_train, y_train = split_features_target(cls_train, CLASSIFICATION_TARGET, cls_feature_cols)
    X_valid, y_valid = split_features_target(cls_valid, CLASSIFICATION_TARGET, cls_feature_cols)
    X_test, y_test = split_features_target(cls_test, CLASSIFICATION_TARGET, cls_feature_cols)

    # Preprocess (build new pipelines since features differ from regression)
    cls_preprocess = build_preprocessing_pipeline(X_train)
    cls_scaled = build_scaled_pipeline(X_train)

    X_train_p = cls_preprocess.transform(X_train)
    X_valid_p = cls_preprocess.transform(X_valid)
    X_test_p = cls_preprocess.transform(X_test)
    X_train_s = cls_scaled.transform(X_train)
    X_valid_s = cls_scaled.transform(X_valid)
    X_test_s = cls_scaled.transform(X_test)

    models = _get_classification_models()
    results: list[dict[str, Any]] = []
    best_clf_name, best_clf_model = None, None
    best_val_f1 = -1.0

    for name, model in models.items():
        log.info("  Training: %s", name)
        t0 = time.time()

        try:
            needs_scaling = name == "LogisticRegression"
            Xtr = X_train_s if needs_scaling else X_train_p
            Xva = X_valid_s if needs_scaling else X_valid_p
            Xte = X_test_s if needs_scaling else X_test_p

            model.fit(Xtr, y_train)
            elapsed = time.time() - t0

            pred_valid = model.predict(Xva)
            pred_test = model.predict(Xte)

            # Probabilities for ROC-AUC
            prob_valid = None
            prob_test = None
            if hasattr(model, "predict_proba"):
                prob_valid = model.predict_proba(Xva)[:, 1]
                prob_test = model.predict_proba(Xte)[:, 1]

            m_valid = compute_classification_metrics(y_valid, pred_valid, prob_valid)
            m_test = compute_classification_metrics(y_test, pred_test, prob_test)

            row = {"Model": name, "train_time_s": round(elapsed, 1)}
            row.update({f"valid_{k}": v for k, v in m_valid.items()})
            row.update({f"test_{k}": v for k, v in m_test.items()})
            results.append(row)

            if m_valid.get("F1", 0) > best_val_f1:
                best_val_f1 = m_valid["F1"]
                best_clf_name = name
                best_clf_model = model

            log.info("    Valid F1=%.4f  ROC-AUC=%.4f  (%.1fs)",
                     m_valid["F1"], m_valid.get("ROC_AUC", 0), elapsed)

        except Exception as e:
            log.error("    FAILED: %s — %s", name, e)

    if not results:
        return None

    # Confusion matrix for best classifier
    if best_clf_model is not None:
        needs_scaling = best_clf_name == "LogisticRegression"
        Xte = X_test_s if needs_scaling else X_test_p
        pred_test = best_clf_model.predict(Xte)
        _plot_confusion_matrix(y_test, pred_test, best_clf_name, figures_dir)

    cls_benchmark = pd.DataFrame(results)
    cls_benchmark = cls_benchmark.sort_values("valid_F1", ascending=False,
                                               na_position="last")

    log.info("Classification benchmark:")
    print(cls_benchmark.to_string(index=False))
    print()

    return cls_benchmark


# =========================================================================
# 9. Plots & Diagnostics
# =========================================================================

def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Predicted vs Actual scatter plot with y=x reference and density coloring."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Sample for performance
    n = len(y_true)
    if n > 30_000:
        idx = np.random.RandomState(42).choice(n, 30_000, replace=False)
        y_t, y_p = y_true[idx], y_pred[idx]
    else:
        y_t, y_p = y_true, y_pred

    ax.scatter(y_t, y_p, alpha=0.08, s=3, c="#3F51B5", rasterized=True)

    lim = max(y_t.max(), y_p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, alpha=0.7, label="y = x (perfect)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual Closing Rank")
    ax.set_ylabel("Predicted Closing Rank")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info("  Saved %s", save_path.name)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Residual analysis: scatter + histogram."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    ax = axes[0]
    n = len(y_pred)
    if n > 30_000:
        idx = np.random.RandomState(42).choice(n, 30_000, replace=False)
        ax.scatter(y_pred[idx], residuals[idx], alpha=0.08, s=3, c="#E91E63",
                   rasterized=True)
    else:
        ax.scatter(y_pred, residuals, alpha=0.08, s=3, c="#E91E63", rasterized=True)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Predicted Closing Rank")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"Residuals vs Predicted — {title}")
    ax.grid(True, alpha=0.3)

    # Residual distribution
    ax = axes[1]
    clipped = np.clip(residuals, np.percentile(residuals, 1),
                      np.percentile(residuals, 99))
    ax.hist(clipped, bins=100, alpha=0.7, color="#4CAF50", edgecolor="none")
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    ax.axvline(np.median(residuals), color="blue", linewidth=1, linestyle="--",
               label=f"Median: {np.median(residuals):+,.0f}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info("  Saved %s", save_path.name)


def plot_error_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_values: np.ndarray,
    group_name: str,
    decode_map: Optional[dict] = None,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Bar chart of MAE by group + return summary DataFrame."""
    df = pd.DataFrame({
        "actual": y_true, "pred": y_pred, "group": group_values,
    })
    df["abs_error"] = np.abs(df["actual"] - df["pred"])

    summary = (
        df.groupby("group")
        .agg(
            MAE=("abs_error", "mean"),
            MedAE=("abs_error", "median"),
            count=("abs_error", "size"),
            mean_rank=("actual", "mean"),
        )
        .reset_index()
        .sort_values("group")
    )

    if decode_map:
        summary["group_label"] = summary["group"].map(decode_map).fillna(
            summary["group"].astype(str)
        )
    else:
        summary["group_label"] = summary["group"].astype(str)

    if save_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(summary["group_label"], summary["MAE"],
                      color=sns.color_palette("Set2", len(summary)),
                      edgecolor="gray", linewidth=0.5)

        # Add count labels
        for bar, count in zip(bars, summary["count"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"n={count:,}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel(group_name)
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title(f"Prediction Error by {group_name}")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45 if len(summary) > 5 else 0, ha="right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        log.info("  Saved %s", save_path.name)

    return summary


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    save_path: Path,
    top_n: int = 20,
) -> pd.DataFrame:
    """Bar chart of feature importance (from tree-based model)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        log.warning("  Model has no feature_importances_ — skipping")
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top = fi_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), top["importance"].values,
            color=sns.color_palette("viridis", len(top)),
            edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top-{top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info("  Saved %s", save_path.name)

    return fi_df


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    figures_dir: Path,
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Relaxed (0)", "Competitive (1)"],
                yticklabels=["Relaxed (0)", "Competitive (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png")
    plt.close()
    log.info("  Saved confusion_matrix.png")


def compute_permutation_importance(
    model: Any,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Compute permutation importance on the validation set."""
    from sklearn.inspection import permutation_importance

    log.info("Computing permutation importance (%d repeats)…", n_repeats)
    t0 = time.time()
    result = permutation_importance(
        model, X_valid, y_valid,
        n_repeats=n_repeats,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    )
    elapsed = time.time() - t0
    log.info("  Permutation importance computed in %.1fs", elapsed)

    pi_df = pd.DataFrame({
        "feature": feature_names[:X_valid.shape[1]],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return pi_df


def compute_shap_values(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    figures_dir: Path,
) -> None:
    """Compute SHAP values on a small sample and save summary plot."""
    if not HAS_SHAP:
        log.warning("SHAP not installed — skipping SHAP analysis")
        return

    log.info("Computing SHAP values on %d samples…", len(X_sample))
    t0 = time.time()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        elapsed = time.time() - t0
        log.info("  SHAP computed in %.1fs", elapsed)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names[:X_sample.shape[1]],
            show=False, max_display=20,
        )
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_summary.png", bbox_inches="tight")
        plt.close()
        log.info("  Saved shap_summary.png")

    except Exception as e:
        log.warning("  SHAP failed: %s — skipping", e)


# =========================================================================
# 10. Report Generation
# =========================================================================

def save_benchmark_tables(
    reg_benchmark: pd.DataFrame,
    cls_benchmark: Optional[pd.DataFrame],
    reports_dir: Path,
) -> None:
    """Save benchmark tables as CSV and Markdown."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Regression
    reg_benchmark.to_csv(reports_dir / "regression_benchmark.csv", index=False)

    md_lines = ["# Regression Model Benchmark", "",
                "Sorted by validation MAE (lower is better).", ""]
    md_lines.append(reg_benchmark.to_markdown(index=False))
    (reports_dir / "regression_benchmark.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    log.info("  Saved regression_benchmark.csv|md")

    # Classification
    if cls_benchmark is not None:
        cls_benchmark.to_csv(reports_dir / "classification_benchmark.csv", index=False)

        md_lines = ["# Classification Sanity-Check Benchmark", "",
                    "Binary target: `cutoff_difficulty` (1 = competitive, "
                    "0 = relaxed). Sorted by validation F1.", ""]
        md_lines.append(cls_benchmark.to_markdown(index=False))
        (reports_dir / "classification_benchmark.md").write_text(
            "\n".join(md_lines), encoding="utf-8"
        )
        log.info("  Saved classification_benchmark.csv|md")


def generate_model_report(
    best_name: str,
    final_metrics: dict[str, float],
    reg_benchmark: pd.DataFrame,
    cls_benchmark: Optional[pd.DataFrame],
    fi_df: pd.DataFrame,
    pi_df: Optional[pd.DataFrame],
    error_by_year: pd.DataFrame,
    error_by_category: pd.DataFrame,
    error_by_inst_type: pd.DataFrame,
    reports_dir: Path,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    test_preds: np.ndarray,
) -> None:
    """Generate the comprehensive Day 3 model evaluation report."""
    lines: list[str] = []

    lines.extend([
        "# JoSAA Day 3 — Model Training & Evaluation Report",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ])

    # Find baseline (Ridge) metrics for comparison
    ridge_row = reg_benchmark[reg_benchmark["Model"] == "Ridge"]
    ridge_mae = ridge_row["valid_MAE"].values[0] if len(ridge_row) > 0 else None

    best_row = reg_benchmark[reg_benchmark["Model"] == best_name]
    best_valid_mae = best_row["valid_MAE"].values[0] if len(best_row) > 0 else None

    lines.append(f"**Best model**: `{best_name}`")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in final_metrics.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    if ridge_mae and best_valid_mae:
        improvement = (1 - best_valid_mae / ridge_mae) * 100
        lines.append(
            f"The best model achieves **{improvement:.1f}% lower validation MAE** "
            f"than the Ridge baseline ({best_valid_mae:,.0f} vs {ridge_mae:,.0f})."
        )
        lines.append("")

    # --- Data split ---
    lines.extend([
        "## Data Split",
        "",
        "| Split | Years | Purpose |",
        "|---|---|---|",
        "| Train | 2016–2023 | Model fitting |",
        "| Valid | 2024 | Model selection & tuning |",
        "| Test | 2025–2026 | Final one-shot evaluation |",
        "",
        "> **Temporal split** — no random shuffling across years. "
        "This prevents data leakage from future cutoff patterns "
        "influencing historical feature statistics.",
        "",
    ])

    # --- Regression benchmark ---
    lines.extend([
        "## Regression Benchmark",
        "",
        "All models predicting `closing_rank` (JEE rank of the last admitted student).",
        "",
    ])
    summary_cols = ["Model", "train_time_s",
                    "train_MAE", "train_R2",
                    "valid_MAE", "valid_R2",
                    "test_MAE", "test_R2"]
    summary_cols = [c for c in summary_cols if c in reg_benchmark.columns]
    lines.append(reg_benchmark[summary_cols].to_markdown(index=False))
    lines.append("")

    # --- Classification summary ---
    if cls_benchmark is not None:
        lines.extend([
            "## Classification Sanity Check",
            "",
            "Binary target: `cutoff_difficulty` — whether a branch's closing rank "
            "is tighter than its historical mean (1 = competitive, 0 = relaxed).",
            "",
            "> **Note**: This is a derived label, not based on individual student "
            "admission outcomes. It serves as a sanity check for feature quality.",
            "",
        ])
        cls_summary = ["Model", "valid_F1", "valid_ROC_AUC", "test_F1", "test_ROC_AUC"]
        cls_summary = [c for c in cls_summary if c in cls_benchmark.columns]
        lines.append(cls_benchmark[cls_summary].to_markdown(index=False))
        lines.append("")

    # --- Feature importance ---
    lines.extend([
        "## Feature Importance",
        "",
        "### Built-in Feature Importance (top 15)",
        "",
    ])
    if len(fi_df) > 0:
        top15 = fi_df.head(15).copy()
        top15["importance"] = top15["importance"].round(4)
        lines.append(top15.to_markdown(index=False))
    else:
        lines.append("*(Not available for this model type)*")
    lines.append("")

    if pi_df is not None and len(pi_df) > 0:
        lines.extend([
            "### Permutation Importance (top 15)",
            "",
        ])
        top15_pi = pi_df.head(15).copy()
        top15_pi["importance_mean"] = top15_pi["importance_mean"].round(4)
        top15_pi["importance_std"] = top15_pi["importance_std"].round(4)
        lines.append(top15_pi.to_markdown(index=False))
        lines.append("")

    # --- Feature interpretation ---
    lines.extend([
        "### Key Feature Insights",
        "",
    ])

    if len(fi_df) > 0:
        top3 = fi_df.head(3)["feature"].tolist()
        lines.append(
            f"The top-3 features driving predictions are: "
            f"**{top3[0]}**, **{top3[1]}**, and **{top3[2]}**."
        )
        lines.append("")

        if "hist_mean_closing_rank" in top3:
            lines.append(
                "The historical mean closing rank dominates, which is expected — "
                "branches tend to have sticky cutoffs year-over-year. The model "
                "effectively learns to predict next year's cutoff as a function "
                "of prior years' cutoffs, adjusted for temporal trends."
            )
        if "opening_rank" in top3 or "opening_rank" in fi_df.head(5)["feature"].tolist():
            lines.append(
                "Opening rank is highly predictive because within-round opening "
                "and closing ranks are strongly correlated (the opening rank "
                "partially reveals the closing rank's range)."
            )
        lines.append("")

    # --- Error analysis by group ---
    lines.extend([
        "## Error Analysis",
        "",
        "### Error by Year",
        "",
    ])
    error_by_year_display = error_by_year.copy()
    error_by_year_display.columns = ["Year", "MAE", "MedAE", "Count", "Mean Rank", "Label"]
    lines.append(error_by_year_display[["Year", "MAE", "MedAE", "Count", "Mean Rank"]].to_markdown(index=False))
    lines.append("")

    lines.extend([
        "### Error by Category",
        "",
    ])
    error_by_cat_display = error_by_category.copy()
    error_by_cat_display.columns = ["Code", "MAE", "MedAE", "Count", "Mean Rank", "Category"]
    lines.append(error_by_cat_display[["Category", "MAE", "MedAE", "Count", "Mean Rank"]].to_markdown(index=False))
    lines.append("")

    lines.extend([
        "### Error by Institute Type",
        "",
    ])
    error_by_it_display = error_by_inst_type.copy()
    error_by_it_display.columns = ["Code", "MAE", "MedAE", "Count", "Mean Rank", "Inst Type"]
    lines.append(error_by_it_display[["Inst Type", "MAE", "MedAE", "Count", "Mean Rank"]].to_markdown(index=False))
    lines.append("")

    # --- Bias / Error discussion ---
    lines.extend([
        "## Bias & Error Discussion",
        "",
    ])

    # Analyze which groups are hardest
    if len(error_by_category) > 0:
        hardest_cat = error_by_category.loc[error_by_category["MAE"].idxmax()]
        easiest_cat = error_by_category.loc[error_by_category["MAE"].idxmin()]
        lines.append(
            f"- **Hardest category to predict**: {hardest_cat.get('group_label', hardest_cat['group'])} "
            f"(MAE = {hardest_cat['MAE']:,.0f})"
        )
        lines.append(
            f"- **Easiest category to predict**: {easiest_cat.get('group_label', easiest_cat['group'])} "
            f"(MAE = {easiest_cat['MAE']:,.0f})"
        )

    if len(error_by_inst_type) > 0:
        hardest_it = error_by_inst_type.loc[error_by_inst_type["MAE"].idxmax()]
        easiest_it = error_by_inst_type.loc[error_by_inst_type["MAE"].idxmin()]
        lines.append(
            f"- **Hardest institute type**: {hardest_it.get('group_label', hardest_it['group'])} "
            f"(MAE = {hardest_it['MAE']:,.0f})"
        )
        lines.append(
            f"- **Easiest institute type**: {easiest_it.get('group_label', easiest_it['group'])} "
            f"(MAE = {easiest_it['MAE']:,.0f})"
        )

    lines.append("")
    lines.extend([
        "Categories with higher closing ranks (SC, ST, EWS) tend to have higher "
        "absolute errors because the rank scale is wider. In relative terms "
        "(MAPE), performance may be more uniform.",
        "",
        "GFTIs and NITs cover a wider range of institutes with more variable "
        "cutoffs, making them harder to predict than IITs which have more "
        "stable, well-established patterns.",
        "",
    ])

    # --- Model reliability ---
    lines.extend([
        "## Model Reliability for Downstream Use",
        "",
    ])

    if "final_test_R2" in final_metrics:
        r2_val = final_metrics["final_test_R2"]
        if r2_val > 0.95:
            lines.append(
                f"With R² = {r2_val:.4f} on the held-out test set, the model "
                f"explains >{r2_val*100:.1f}% of the variance in closing ranks. "
                f"This is strong enough to support Monte Carlo simulation and "
                f"recommendation ranking in later stages."
            )
        elif r2_val > 0.85:
            lines.append(
                f"With R² = {r2_val:.4f}, the model provides a solid basis for "
                f"downstream Monte Carlo simulation, though prediction intervals "
                f"should be used to convey uncertainty."
            )
        else:
            lines.append(
                f"With R² = {r2_val:.4f}, the model captures the general trend "
                f"but uncertainty is significant. Monte Carlo simulation should "
                f"incorporate wide prediction intervals."
            )
    lines.append("")

    lines.extend([
        "### Limitations",
        "",
        "1. **No individual student data** — the model predicts branch-level cutoffs, "
        "not individual admission probability directly.",
        "2. **2026 data is partial** — only early rounds available, test performance "
        "on 2026 may differ from 2025.",
        "3. **EWS category only exists from 2019** — limited historical data for "
        "this group.",
        "4. **PwD rows have different rank scales** — model treats them with a "
        "binary flag, but a separate model might perform better.",
        "",
        "---",
        "",
        "*Report generated by the Day 3 model training pipeline.*",
    ])

    path = reports_dir / "day3_model_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  Saved day3_model_report.md")


# =========================================================================
# 11. Artifact Saving
# =========================================================================

def save_artifacts(
    final_model: Any,
    best_name: str,
    preprocess_pipeline: Pipeline,
    scaled_pipeline: Pipeline,
    trained_models: dict[str, Any],
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    test_preds: np.ndarray,
    fi_df: pd.DataFrame,
    artifact_dir: Path,
) -> None:
    """Save all model artifacts and prediction outputs."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Final model
    model_path = artifact_dir / "final_regression_model.joblib"
    joblib.dump({"model": final_model, "model_name": best_name}, model_path)
    log.info("  Saved final_regression_model.joblib")

    # 2. Preprocessing pipelines
    joblib.dump(preprocess_pipeline, artifact_dir / "preprocessing_pipeline.joblib")
    joblib.dump(scaled_pipeline, artifact_dir / "scaled_pipeline.joblib")
    log.info("  Saved preprocessing_pipeline.joblib & scaled_pipeline.joblib")

    # 3. All trained models
    joblib.dump(trained_models, artifact_dir / "all_models.joblib")
    log.info("  Saved all_models.joblib")

    # 4. Test predictions
    pred_df = test_df[["year", "round"]].copy()
    if "category_encoded" in test_df.columns:
        pred_df["category_encoded"] = test_df["category_encoded"].values
    if "institute_type_encoded" in test_df.columns:
        pred_df["institute_type_encoded"] = test_df["institute_type_encoded"].values
    pred_df["actual_closing_rank"] = y_test
    pred_df["predicted_closing_rank"] = np.round(test_preds).astype(int)
    pred_df["residual"] = y_test - test_preds
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df.to_csv(artifact_dir / "predictions_test.csv", index=False)
    log.info("  Saved predictions_test.csv")

    # 5. Residuals analysis
    residuals_df = pred_df.copy()
    residuals_df["pct_error"] = np.where(
        y_test > 0,
        np.abs(y_test - test_preds) / y_test * 100,
        np.nan,
    )
    residuals_df.to_csv(artifact_dir / "residuals_analysis.csv", index=False)
    log.info("  Saved residuals_analysis.csv")

    # 6. Feature importance
    if len(fi_df) > 0:
        fi_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
        log.info("  Saved feature_importance.csv")


# =========================================================================
# 12. Master Pipeline
# =========================================================================

def run_day3_pipeline(data_dir: Path) -> None:
    """
    Run the complete Day 3 model training pipeline:
    load → preprocess → train → benchmark → tune → refit → evaluate →
    classify → interpret → report → save.
    """
    figures_dir = Path("figures")
    reports_dir = Path("reports")
    artifact_dir = Path("model_artifacts")

    for d in [figures_dir, reports_dir, artifact_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Log available optional packages
    log.info("Optional packages: XGBoost=%s  LightGBM=%s  SHAP=%s",
             HAS_XGBOOST, HAS_LIGHTGBM, HAS_SHAP)

    # ─── Step 1: Load data ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Loading data")
    log.info("=" * 60)
    train_df, valid_df, test_df = load_data(data_dir)

    # ─── Step 2: Prepare features ────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Preparing features")
    log.info("=" * 60)

    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    assert_no_forbidden_features(available_features)
    log.info("Using %d/%d features: %s", len(available_features), len(FEATURE_COLS),
             available_features)
    with open(artifact_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema_payload(available_features), f, indent=2)
    log.info("Saved feature_schema.json")

    X_train, y_train = split_features_target(train_df, REGRESSION_TARGET, available_features)
    X_valid, y_valid = split_features_target(valid_df, REGRESSION_TARGET, available_features)
    X_test, y_test = split_features_target(test_df, REGRESSION_TARGET, available_features)

    feature_names = list(X_train.columns)
    log.info("Feature matrix: train=%s  valid=%s  test=%s",
             X_train.shape, X_valid.shape, X_test.shape)

    # ─── Step 3: Build preprocessing pipelines ───────────────────────
    log.info("=" * 60)
    log.info("STEP 3: Building preprocessing pipelines")
    log.info("=" * 60)

    preprocess_pipeline = build_preprocessing_pipeline(X_train)
    scaled_pipeline = build_scaled_pipeline(X_train)

    X_train_p = preprocess_pipeline.transform(X_train)
    X_valid_p = preprocess_pipeline.transform(X_valid)
    X_test_p = preprocess_pipeline.transform(X_test)

    X_train_s = scaled_pipeline.transform(X_train)
    X_valid_s = scaled_pipeline.transform(X_valid)
    X_test_s = scaled_pipeline.transform(X_test)

    y_train_np = y_train.values.astype(float)
    y_valid_np = y_valid.values.astype(float)
    y_test_np = y_test.values.astype(float)

    log.info("Preprocessed shapes: train=%s  valid=%s  test=%s",
             X_train_p.shape, X_valid_p.shape, X_test_p.shape)

    # ─── Step 4: Train regression models ─────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4: Training regression models")
    log.info("=" * 60)

    reg_benchmark, trained_models = train_models(
        X_train_p, y_train_np,
        X_valid_p, y_valid_np,
        X_test_p, y_test_np,
        feature_names,
        X_train_s, X_valid_s, X_test_s,
    )

    # ─── Step 5: Tune top models ─────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 5: Tuning top models")
    log.info("=" * 60)

    best_name, best_model, tuned_models = tune_top_models(
        reg_benchmark, trained_models,
        X_train_p, y_train_np,
        X_valid_p, y_valid_np,
        X_train_s, X_valid_s,
        feature_names,
        n_top=2, n_iter=3,
    )

    # ─── Step 6: Refit & final evaluation ────────────────────────────
    log.info("=" * 60)
    log.info("STEP 6: Refit & final test evaluation")
    log.info("=" * 60)

    final_model, test_preds, final_metrics = refit_and_evaluate(
        best_name, best_model,
        X_train_p, y_train_np,
        X_valid_p, y_valid_np,
        X_test_p, y_test_np,
        X_train_s, X_valid_s, X_test_s,
        feature_names,
    )

    # ─── Step 7: Classification sanity check ─────────────────────────
    log.info("=" * 60)
    log.info("STEP 7: Classification sanity check")
    log.info("=" * 60)

    cls_benchmark = run_classification_sanity_check(
        data_dir, preprocess_pipeline, scaled_pipeline, figures_dir,
    )

    # ─── Step 8: Plots & diagnostics ─────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 8: Generating plots & diagnostics")
    log.info("=" * 60)

    # Pred vs actual
    plot_pred_vs_actual(
        y_test_np, test_preds,
        f"Predicted vs Actual — {best_name} (Test Set)",
        figures_dir / "pred_vs_actual.png",
    )

    # Residual plots
    plot_residuals(
        y_test_np, test_preds,
        best_name,
        figures_dir / "residuals_plot.png",
    )

    # Error by year
    error_by_year = plot_error_by_group(
        y_test_np, test_preds,
        test_df["year"].values,
        "Year", None,
        figures_dir / "error_by_year.png",
    )

    # Error by category
    error_by_category = plot_error_by_group(
        y_test_np, test_preds,
        test_df["category_encoded"].values if "category_encoded" in test_df.columns else np.zeros(len(y_test_np)),
        "Category", CATEGORY_DECODE,
        figures_dir / "error_by_category.png",
    )

    # Error by institute type
    error_by_inst_type = plot_error_by_group(
        y_test_np, test_preds,
        test_df["institute_type_encoded"].values if "institute_type_encoded" in test_df.columns else np.zeros(len(y_test_np)),
        "Institute Type", INST_TYPE_DECODE,
        figures_dir / "error_by_institute_type.png",
    )

    # ─── Step 9: Feature importance & interpretation ─────────────────
    log.info("=" * 60)
    log.info("STEP 9: Feature importance & interpretation")
    log.info("=" * 60)

    # Built-in importance
    fi_df = plot_feature_importance(
        final_model, feature_names,
        figures_dir / "feature_importance.png",
    )

    # Permutation importance
    pi_df = None
    try:
        needs_scaling = best_name in ("Ridge", "MLP")
        Xva_for_pi = X_valid_s if needs_scaling else X_valid_p
        pi_df = compute_permutation_importance(
            final_model, Xva_for_pi, y_valid_np, feature_names, n_repeats=5,
        )
        log.info("Top-5 permutation importances:")
        print(pi_df.head().to_string(index=False))
        print()
    except Exception as e:
        log.warning("Permutation importance failed: %s", e)

    # SHAP (on small sample)
    if HAS_SHAP and hasattr(final_model, "feature_importances_"):
        sample_size = min(5000, len(X_valid_p))
        idx = np.random.RandomState(42).choice(len(X_valid_p), sample_size, replace=False)
        needs_scaling = best_name in ("Ridge", "MLP")
        X_shap = X_valid_s[idx] if needs_scaling else X_valid_p[idx]
        compute_shap_values(final_model, X_shap, feature_names, figures_dir)

    # ─── Step 10: Save benchmark tables ──────────────────────────────
    log.info("=" * 60)
    log.info("STEP 10: Saving benchmark tables")
    log.info("=" * 60)

    save_benchmark_tables(reg_benchmark, cls_benchmark, reports_dir)

    # ─── Step 11: Generate model report ──────────────────────────────
    log.info("=" * 60)
    log.info("STEP 11: Generating model report")
    log.info("=" * 60)

    generate_model_report(
        best_name, final_metrics,
        reg_benchmark, cls_benchmark,
        fi_df, pi_df,
        error_by_year, error_by_category, error_by_inst_type,
        reports_dir,
        test_df, y_test_np, test_preds,
    )

    # ─── Step 12: Save all artifacts ─────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 12: Saving artifacts")
    log.info("=" * 60)

    save_artifacts(
        final_model, best_name,
        preprocess_pipeline, scaled_pipeline,
        trained_models, test_df, y_test_np, test_preds,
        fi_df, artifact_dir,
    )

    # ─── Final summary ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("DAY 3 PIPELINE COMPLETE")
    log.info("=" * 60)

    log.info("  Best model: %s", best_name)
    for k, v in final_metrics.items():
        log.info("    %s = %s", k, v)

    log.info("")
    log.info("  Output directories:")
    for d in [figures_dir, reports_dir, artifact_dir]:
        file_count = sum(1 for f in d.rglob("*") if f.is_file())
        total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        log.info("    %-20s  %2d files  %7.2f MB", d, file_count, total_size / 1e6)

    log.info("")
    log.info("  Key answers:")
    log.info("    → Best cutoff predictor: %s", best_name)
    if "final_test_R2" in final_metrics:
        log.info("    → Test R²: %.4f", final_metrics["final_test_R2"])
    if "final_test_MAE" in final_metrics:
        log.info("    → Test MAE: %.1f ranks", final_metrics["final_test_MAE"])
    if len(fi_df) > 0:
        top3 = fi_df.head(3)["feature"].tolist()
        log.info("    → Top features: %s", ", ".join(top3))
    log.info("    → Ready for Monte Carlo & recommendation: check report for details")


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    """Command-line entry point for the Day 3 pipeline."""
    parser = argparse.ArgumentParser(
        description="JoSAA Day 3 — Model Training & Benchmarking Pipeline",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing the feature-engineered data (default: ./data)",
    )
    args = parser.parse_args()

    run_day3_pipeline(args.data_dir)


if __name__ == "__main__":
    main()
