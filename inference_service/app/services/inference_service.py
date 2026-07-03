"""
Core inference service.

Responsibilities
----------------
1. Filter the universe of choices by the student's profile.
2. Encode categorical features using the pre-trained label encoders.
3. Run the preprocessing pipeline + Ridge regression model.
4. Return a DataFrame of choices with a 'predicted_closing_rank' column.

This module is intentionally stateless — it receives artifacts as arguments
so it is easy to unit-test without touching the filesystem.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.josaa_core.feature_schema import MODEL_FEATURES, assert_no_forbidden_features

logger = logging.getLogger("inference_service.inference_service")


# ── Feature encoding ────────────────────────────────────────────────────────────

def encode_features(
    df: pd.DataFrame,
    label_encoders: dict,
    feature_schema: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply label encoders to produce the integer-encoded feature columns
    expected by the preprocessing pipeline.

    Columns not present in the encoders dictionary are left unchanged.
    Unknown categorical values are mapped to -1 (treated as unseen).

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows from the universe (josaa_cleaned.csv).
    label_encoders : dict
        Mapping of {column_name: {str_value: int_code}}.

    Returns
    -------
    pd.DataFrame
        10-column DataFrame in the order expected by preprocessing_pipeline.
    """
    encoded = df.copy()

    # Convert any boolean columns to int
    bool_cols = encoded.select_dtypes(include=["bool"]).columns
    for c in bool_cols:
        encoded[c] = encoded[c].astype(int)

    required_features = feature_schema or MODEL_FEATURES
    assert_no_forbidden_features(required_features)

    # Ensure all required columns exist (fill with 0 if missing)
    for col in required_features:
        if col not in encoded.columns:
            logger.warning(f"Feature column '{col}' missing — defaulting to 0")
            encoded[col] = 0

    return encoded[required_features]


# ── Universe filtering ──────────────────────────────────────────────────────────

def filter_universe(
    universe: pd.DataFrame,
    category: str,
    gender: str,
    round_: int,
    pref_inst_types: list[str] | None,
    pref_branch_keywords: list[str] | None,
    advanced_rank: int | None,
    home_state: str | None = None,
    is_pwd: bool = False,
) -> pd.DataFrame:
    """
    Return the subset of the universe matching the student's profile.

    Parameters
    ----------
    universe : pd.DataFrame
        Full 2024 Round-6 universe (pre-loaded at startup).
    category : str
        e.g. "OBC-NCL"
    gender : str
        "Gender-Neutral" or "Female-only"
    round_ : int
        Which round's data to use (default 6).
    pref_inst_types : list[str] | None
        Optional institute type filter e.g. ["NIT", "IIIT"].
    pref_branch_keywords : list[str] | None
        Optional branch keyword filter e.g. ["Computer Science"].
    advanced_rank : int | None
        If None, IIT rows are excluded (student has no JEE Advanced rank).
    home_state : str | None
        Student home state. HS rows are only included when the universe has
        institute_state metadata that proves the row applies to the student.
    is_pwd : bool
        Whether to match PwD-specific rows.

    Returns
    -------
    pd.DataFrame
        Filtered subset (may be empty if no programs match).
    """
    available_rounds = sorted(universe["round"].dropna().unique())
    effective_round = round_
    if available_rounds and effective_round not in available_rounds:
        earlier_or_equal = [r for r in available_rounds if r <= round_]
        effective_round = earlier_or_equal[-1] if earlier_or_equal else available_rounds[-1]
        logger.info(
            "Requested round %s is unavailable; using round %s from loaded universe",
            round_,
            effective_round,
        )

    mask = (
        (universe["category"] == category)
        & (universe["gender"] == gender)
        & (universe["round"] == effective_round)
    )

    if "is_pwd" in universe.columns:
        mask = mask & (universe["is_pwd"] == is_pwd)

    if "quota" in universe.columns:
        quota = universe["quota"].astype(str).str.upper()
        if "institute_state" in universe.columns and home_state:
            institute_state = universe["institute_state"].astype(str).str.casefold()
            home_state_norm = home_state.casefold()
            mask = mask & (
                (quota != "HS")
                | ((quota == "HS") & (institute_state == home_state_norm))
            )
        else:
            # Without institute-state metadata, HS rows may belong to any state.
            # Keep AI/OS/global rows and avoid presenting unverifiable HS matches.
            mask = mask & (quota != "HS")

    # Exclude IITs if no Advanced rank is provided
    if advanced_rank is None:
        mask = mask & (universe["institute_type"] != "IIT")

    # Optional institute type filter
    if pref_inst_types:
        mask = mask & (universe["institute_type"].isin(pref_inst_types))

    # Optional branch keyword filter (case-insensitive OR match)
    if pref_branch_keywords:
        kw_mask = pd.Series([False] * len(universe), index=universe.index)
        for kw in pref_branch_keywords:
            kw_mask = kw_mask | universe["program"].str.contains(kw, case=False, na=False)
        mask = mask & kw_mask

    return universe[mask].copy()


def assign_rank_sources(
    choices: pd.DataFrame,
    main_rank: int | None,
    advanced_rank: int | None = None,
    iit_only: bool = False,
) -> pd.DataFrame:
    """
    Attach JoSAA rank source metadata per row.

    IIT cutoffs are keyed to JEE Advanced rank. NIT, IIIT, and GFTI cutoffs are
    keyed to JEE Main rank even when an Advanced rank is also present.
    """
    if choices.empty:
        return choices.copy()

    choices = choices.copy()
    inst_type = choices["institute_type"].astype(str).str.upper()
    is_iit = inst_type == "IIT"

    if is_iit.any() and advanced_rank is None:
        choices = choices.loc[~is_iit].copy()
        inst_type = choices["institute_type"].astype(str).str.upper()
        is_iit = inst_type == "IIT"

    if (~is_iit).any() and main_rank is None and not iit_only:
        raise ValueError("main_rank is required for NIT/IIIT/GFTI recommendations")

    if is_iit.any() and advanced_rank is None:
        raise ValueError("advanced_rank is required for IIT recommendations")

    choices["rank_used"] = np.where(is_iit, advanced_rank, main_rank).astype(int)
    choices["rank_type_used"] = np.where(is_iit, "JEE_ADVANCED", "JEE_MAIN")
    return choices


# ── Main inference call ─────────────────────────────────────────────────────────

def run_inference(
    choices: pd.DataFrame,
    model: Any,
    prep_pipeline: Any,
    label_encoders: dict,
    feature_schema: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the trained Ridge regression model on a filtered set of choices.

    Parameters
    ----------
    choices : pd.DataFrame
        Filtered universe rows for this student.
    model : sklearn estimator
        The final_regression_model.
    prep_pipeline : sklearn Pipeline
        The preprocessing pipeline.
    label_encoders : dict
        Label encoder mappings.

    Returns
    -------
    pd.DataFrame
        `choices` with an added 'predicted_closing_rank' column.
    """
    if choices.empty:
        return choices

    features_df = encode_features(choices, label_encoders, feature_schema)

    try:
        # Pass a DataFrame since ColumnTransformer expects it (with correct column names/order)
        X = prep_pipeline.transform(features_df)
        preds = model.predict(X)
    except Exception as exc:
        logger.error(f"Model inference failed: {exc}", exc_info=True)
        raise

    choices = choices.copy()
    choices["predicted_closing_rank"] = np.maximum(1, np.round(preds)).astype(int)
    return choices
