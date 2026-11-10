"""
Core inference service.

Responsibilities
----------------
1. Filter the universe of choices by the student's profile.
2. Encode categorical features using the pre-trained label encoders.
3. Run the selected pre-counselling sklearn-compatible estimator.
4. Return a DataFrame of choices with a 'predicted_closing_rank' column.

This module is intentionally stateless — it receives artifacts as arguments
so it is easy to unit-test without touching the filesystem.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from josaa_core.feature_schema import MODEL_FEATURES, assert_no_forbidden_features

logger = logging.getLogger("inference_api.inference")


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
    pref_quotas: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return the subset of the universe matching the student's profile.

    Parameters
    ----------
    universe : pd.DataFrame
        Artifact-backed prior-season candidate universe (pre-loaded at startup).
    category : str
        e.g. "OBC-NCL"
    gender : str
        "Gender-Neutral" or "Female-only"
    round_ : int
        Validated counselling-round filter supplied by the request (API default 5).
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
    mask = (
        (universe["category"] == category)
        & (universe["gender"] == gender)
        & (universe["round"] == round_)
    )

    if "is_pwd" in universe.columns:
        mask = mask & (universe["is_pwd"] == is_pwd)
    elif is_pwd:
        # Never present general-pool rows as PwD-specific eligibility when the
        # serving universe cannot prove the distinction.
        mask = mask & False

    if "quota" in universe.columns:
        quota = universe["quota"].astype(str).str.upper()
        state_column = next(
            (name for name in ("institute_state", "state") if name in universe.columns),
            None,
        )
        if state_column and home_state:
            institute_state = universe[state_column].astype(str).str.strip().str.casefold()
            home_state_norm = home_state.casefold()
            mask = mask & (
                (quota != "HS")
                | ((quota == "HS") & (institute_state == home_state_norm))
            )
        else:
            # Without institute-state metadata, HS rows may belong to any state.
            # Keep AI/OS/global rows and avoid presenting unverifiable HS matches.
            mask = mask & (quota != "HS")

        if pref_quotas:
            mask = mask & quota.isin(pref_quotas)

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
            kw_mask = kw_mask | universe["program"].str.contains(
                kw, case=False, na=False, regex=False
            )
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
    Run the selected pre-counselling estimator on a filtered set of choices.

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

    if feature_schema is None:
        if hasattr(model, "feature_names_in_"):
            feature_schema = [str(name) for name in model.feature_names_in_]
        elif isinstance(getattr(model, "feature_name", None), str):
            feature_schema = [model.feature_name]
    features_df = encode_features(choices, label_encoders, feature_schema)

    try:
        # Research-safe serving artifacts are full sklearn-compatible estimators
        # that accept the schema-ordered DataFrame directly. The optional
        # transformer branch remains for small synthetic unit-test fixtures.
        X = prep_pipeline.transform(features_df) if prep_pipeline is not None else features_df
        preds = model.predict(X)
    except Exception as exc:
        logger.error("Model inference failed error_type=%s", type(exc).__name__)
        raise

    choices = choices.copy()
    choices["predicted_closing_rank"] = np.maximum(1, np.round(preds)).astype(int)
    return choices
