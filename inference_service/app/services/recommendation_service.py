"""
Recommendation service — Monte Carlo simulation, probability calculation,
margin scoring, and Safe / Moderate / Ambitious classification.

All random state is seeded per-request when mc_enabled=True so results are
reproducible for the same input.  Deterministic mode (mc_enabled=False) uses
the predicted rank directly without sampling.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.services.explanation_service import attach_explanations

logger = logging.getLogger("inference_service.recommendation_service")


# ── Monte Carlo ─────────────────────────────────────────────────────────────────

def _get_std_dev(std_map: pd.DataFrame, inst_type: str, round_: int) -> float:
    """
    Look up the empirical residual std_dev for an (institute_type, round) pair.
    Falls back to the configured default if no match is found.
    """
    if std_map.empty:
        return settings.mc_std_dev_default

    match = std_map[
        (std_map["institute_type_str"] == inst_type) & (std_map["round"] == round_)
    ]
    if match.empty:
        # Try type-level fallback (any round)
        match = std_map[std_map["institute_type_str"] == inst_type]

    if match.empty or pd.isna(match["std_dev"].iloc[0]):
        return settings.mc_std_dev_default

    return float(match["std_dev"].iloc[0])


def run_monte_carlo_vectorised(
    predicted_cutoffs: np.ndarray,
    std_devs: np.ndarray,
    student_rank: int,
    n_sims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Monte Carlo simulation for all choices simultaneously.

    Samples `n_sims` cutoffs for each choice from N(pred, std_dev), then
    computes admission probability and 90 % confidence interval bounds.

    Parameters
    ----------
    predicted_cutoffs : np.ndarray  shape (n_choices,)
    std_devs          : np.ndarray  shape (n_choices,)
    student_rank      : int
    n_sims            : int

    Returns
    -------
    probs      : np.ndarray  shape (n_choices,)   — probability ∈ [0, 1]
    ci_lower   : np.ndarray  shape (n_choices,)   — 5th percentile
    ci_median  : np.ndarray  shape (n_choices,)   — 50th percentile
    ci_upper   : np.ndarray  shape (n_choices,)   — 95th percentile
    """
    rng = np.random.default_rng(seed=42)  # deterministic seed
    # Shape: (n_sims, n_choices)
    simulated = rng.normal(
        loc=predicted_cutoffs[np.newaxis, :],
        scale=std_devs[np.newaxis, :],
        size=(n_sims, len(predicted_cutoffs)),
    )
    # Ranks can't be negative
    simulated = np.maximum(1, simulated)

    probs = np.mean(student_rank <= simulated, axis=0)
    ci_lower = np.percentile(simulated, 5, axis=0)
    ci_median = np.percentile(simulated, 50, axis=0)
    ci_upper = np.percentile(simulated, 95, axis=0)

    return probs, ci_lower, ci_median, ci_upper


# ── Scoring & labelling ─────────────────────────────────────────────────────────

def _classify(prob: float) -> str:
    """Classify a probability into Safe / Moderate / Ambitious."""
    if prob >= settings.safe_threshold:
        return "Safe"
    elif prob >= settings.moderate_threshold:
        return "Moderate"
    return "Ambitious"


def _composite_score(prob: float, margin_score: float) -> float:
    """Composite ranking score — probability dominates, margin breaks ties."""
    return (prob * settings.score_prob_weight) + (margin_score * settings.score_margin_weight)


# ── Public entry point ──────────────────────────────────────────────────────────

def build_recommendations(
    choices: pd.DataFrame,
    student_rank: int,
    std_map: pd.DataFrame,
    round_: int,
    mc_enabled: bool = True,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Run the full recommendation pipeline on a filtered + inferred choice set.

    Steps:
    1. Look up std_dev per row (by institute_type + round).
    2. Run vectorised Monte Carlo (or deterministic fallback).
    3. Compute margin score and composite ranking score.
    4. Classify each choice as Safe / Moderate / Ambitious.
    5. Attach natural-language explanation.
    6. Sort by score descending and return top_n rows.

    Parameters
    ----------
    choices : pd.DataFrame
        Output of run_inference() — contains 'predicted_closing_rank'.
    student_rank : int
    std_map : pd.DataFrame
    round_ : int
    mc_enabled : bool
    top_n : int

    Returns
    -------
    pd.DataFrame
        Ranked recommendations with all metadata columns attached.
    """
    if choices.empty:
        return choices

    choices = choices.copy()

    # ── 1. Map std_dev per row ───────────────────────────────────────────────
    choices["_std_dev"] = choices.apply(
        lambda row: _get_std_dev(
            std_map,
            row.get("institute_type", "__default__"),
            round_,
        ),
        axis=1,
    )

    predicted = choices["predicted_closing_rank"].values.astype(float)
    std_devs = choices["_std_dev"].values.astype(float)

    # ── 2. Monte Carlo / deterministic probability ───────────────────────────
    if mc_enabled:
        probs, ci_lower, ci_median, ci_upper = run_monte_carlo_vectorised(
            predicted, std_devs, student_rank, settings.mc_n_sims
        )
    else:
        # Deterministic: probability = 1 if rank ≤ predicted cutoff else 0
        probs = (student_rank <= predicted).astype(float)
        ci_lower = predicted
        ci_median = predicted
        ci_upper = predicted

    choices["admission_probability"] = probs
    choices["ci_lower_5"] = np.maximum(1, ci_lower).astype(int)
    choices["expected_cutoff_50"] = ci_median.astype(int)
    choices["ci_upper_95"] = ci_upper.astype(int)

    # ── 3. Margin & composite score ─────────────────────────────────────────
    choices["margin"] = choices["predicted_closing_rank"] - student_rank
    choices["margin_score"] = choices["margin"] / choices["predicted_closing_rank"].clip(lower=1)
    choices["recommendation_score"] = choices.apply(
        lambda r: _composite_score(r["admission_probability"], r["margin_score"]), axis=1
    )

    # ── 4. Classify ─────────────────────────────────────────────────────────
    choices["confidence_label"] = choices["admission_probability"].map(_classify)

    # ── 5. Explanations ──────────────────────────────────────────────────────
    choices = attach_explanations(choices, student_rank)

    # ── 6. Sort and trim ─────────────────────────────────────────────────────
    choices = choices.sort_values("recommendation_score", ascending=False)
    return choices.head(top_n).reset_index(drop=True)
