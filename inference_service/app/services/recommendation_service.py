"""
Recommendation service — Monte Carlo simulation, probability calculation,
rank-sensitive fit scoring, and Safe / Moderate / Ambitious classification.

All random state is seeded per-request when mc_enabled=True so results are
reproducible for the same input.  Deterministic mode (mc_enabled=False) uses
the predicted rank directly without sampling.
"""

import logging
import numpy as np
import pandas as pd

from app.config import settings
from app.services.explanation_service import attach_explanations

logger = logging.getLogger("inference_service.recommendation_service")

INSTITUTE_SCORES = {
    "IIT": 1.00,
    "NIT": 0.78,
    "IIIT": 0.72,
    "GFTI": 0.48,
}

BRANCH_SCORES = {
    "computer science": 1.00,
    "artificial intelligence": 0.98,
    "machine learning": 0.96,
    "data science": 0.95,
    "mathematics and computing": 0.94,
    "electronics and communication": 0.88,
    "electronics": 0.84,
    "electrical": 0.78,
    "information technology": 0.86,
    "mechanical": 0.64,
    "chemical": 0.58,
    "civil": 0.52,
    "metallurgical": 0.48,
    "mining": 0.44,
}


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

    if match.empty:
        match = std_map[std_map["institute_type_str"] == "__default__"]

    if match.empty or pd.isna(match["std_dev"].iloc[0]):
        return settings.mc_std_dev_default

    return float(match["std_dev"].iloc[0])


def run_monte_carlo_vectorised(
    predicted_cutoffs: np.ndarray,
    std_devs: np.ndarray,
    student_rank: int | np.ndarray,
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

    ranks = np.asarray(student_rank, dtype=float)
    if ranks.ndim == 0:
        ranks = np.full(len(predicted_cutoffs), float(ranks))
    probs = np.mean(ranks[np.newaxis, :] <= simulated, axis=0)
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


def _bucket(prob: float, margin: float, rank: float, predicted: float) -> str:
    """Recommendation grouping used by the UI/RAG layer.

    ``ambitious_reach`` intentionally means realistic reach: the student is a
    little behind the projected cutoff or has non-trivial odds. Far-off,
    near-zero options are marked separately and excluded from normal results.
    """
    rank = max(float(rank), 1.0)
    predicted = max(float(predicted), 1.0)
    worse_than_cutoff_ratio = max((rank - predicted) / predicted, 0.0)
    better_than_cutoff_ratio = max(margin / predicted, 0.0)
    close_to_cutoff_ratio = abs(rank - predicted) / predicted

    if prob < 0.10 or (prob < 0.15 and worse_than_cutoff_ratio > 0.40):
        return "unlikely_reach"
    if prob >= 0.80 or better_than_cutoff_ratio >= 0.25:
        return "safe_backup"
    if worse_than_cutoff_ratio > 0 and (prob <= 0.50 or worse_than_cutoff_ratio <= 0.40):
        return "ambitious_reach"
    if prob >= 0.45 or close_to_cutoff_ratio <= 0.18:
        return "best_realistic"
    if prob >= 0.15 and worse_than_cutoff_ratio <= 0.40:
        return "ambitious_reach"
    return "unlikely_reach"


def _ensure_bucket_coverage(choices: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep the default list balanced when realistic bucket candidates exist."""
    if top_n <= 0 or choices.empty:
        return choices.head(top_n)

    selected = choices.head(top_n).copy()
    if len(selected) < top_n:
        return selected

    selected_ids = set(selected["_recommendation_row_id"])
    required = ["safe_backup", "best_realistic", "ambitious_reach"]

    for bucket in required:
        if bucket in set(selected["recommendation_bucket"]):
            continue

        candidate = choices[choices["recommendation_bucket"] == bucket].head(1)
        if candidate.empty:
            continue

        replaceable = selected[
            ~selected["recommendation_bucket"].isin(required)
            | selected["recommendation_bucket"].duplicated(keep="first")
        ]
        if replaceable.empty:
            continue

        drop_id = replaceable.iloc[-1]["_recommendation_row_id"]
        selected = pd.concat(
            [
                selected[selected["_recommendation_row_id"] != drop_id],
                candidate[~candidate["_recommendation_row_id"].isin(selected_ids)],
            ],
            ignore_index=False,
        )
        selected_ids = set(selected["_recommendation_row_id"])

    return selected.sort_values("_sort_position").head(top_n)


def _branch_score(program: str) -> float:
    """Map program names to a desirability score with a conservative default."""
    program_norm = str(program).casefold()
    for keyword, score in BRANCH_SCORES.items():
        if keyword in program_norm:
            return score
    return 0.55


def _admissibility_score(prob: np.ndarray) -> np.ndarray:
    """
    Reward reasonable admission odds without letting 100% safe choices dominate.

    Probability rises to full credit around 75%, then very safe options are gently
    flattened because they are usually backups rather than the best first picks.
    """
    score = np.minimum(prob / 0.75, 1.0)
    very_safe = prob > 0.85
    score[very_safe] = 1.0 - (0.18 * ((prob[very_safe] - 0.85) / 0.15).clip(0, 1))
    return score.clip(0, 1)


def _fit_score(predicted: np.ndarray, student_rank: int | np.ndarray, prob: np.ndarray) -> np.ndarray:
    """
    Prefer options close to the student's rank, with a mild positive margin.

    Choices far easier than the profile lose fit credit; impossible reaches also
    lose credit through the probability gate.
    """
    rank = np.maximum(np.asarray(student_rank, dtype=float), 1.0)
    ratio = predicted.clip(min=1) / rank
    proximity = np.exp(-((np.log(ratio) - np.log(1.18)) ** 2) / (2 * (0.75 ** 2)))
    probability_gate = np.minimum(prob / 0.35, 1.0)
    return (proximity * probability_gate).clip(0, 1)


def _competitiveness_score(predicted: np.ndarray) -> np.ndarray:
    """Score lower/tighter predicted closing ranks as more competitive."""
    log_pred = np.log1p(predicted.clip(min=1))
    span = log_pred.max() - log_pred.min()
    if span <= 0:
        return np.ones_like(log_pred)
    return (1.0 - ((log_pred - log_pred.min()) / span)).clip(0, 1)


# ── Public entry point ──────────────────────────────────────────────────────────

def build_recommendations(
    choices: pd.DataFrame,
    student_rank: int,
    std_map: pd.DataFrame,
    round_: int,
    mc_enabled: bool = True,
    top_n: int = 30,
    sort_mode: str = "best_fit",
) -> pd.DataFrame:
    """
    Run the full recommendation pipeline on a filtered + inferred choice set.

    Steps:
    1. Look up std_dev per row (by institute_type + round).
    2. Run vectorised Monte Carlo (or deterministic fallback).
    3. Compute rank-sensitive desirability, admissibility, and fit scores.
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
    choices["_recommendation_row_id"] = np.arange(len(choices))

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
    rank_used = (
        choices["rank_used"].to_numpy(dtype=float)
        if "rank_used" in choices.columns
        else np.full(len(choices), float(student_rank))
    )

    # ── 2. Monte Carlo / deterministic probability ───────────────────────────
    if mc_enabled:
        probs, ci_lower, ci_median, ci_upper = run_monte_carlo_vectorised(
            predicted, std_devs, rank_used, settings.mc_n_sims
        )
    else:
        # Deterministic: probability = 1 if rank ≤ predicted cutoff else 0
        probs = (rank_used <= predicted).astype(float)
        ci_lower = predicted
        ci_median = predicted
        ci_upper = predicted

    choices["admission_probability"] = probs
    choices["ci_lower_5"] = np.maximum(1, ci_lower).astype(int)
    choices["expected_cutoff_50"] = ci_median.astype(int)
    choices["ci_upper_95"] = ci_upper.astype(int)

    # ── 3. Rank-sensitive scoring ──────────────────────────────────────────
    choices["rank_used"] = rank_used.astype(int)
    if "rank_type_used" not in choices.columns:
        choices["rank_type_used"] = "main"
    choices["margin"] = choices["predicted_closing_rank"] - choices["rank_used"]
    choices["margin_score"] = choices["margin"] / choices["predicted_closing_rank"].clip(lower=1)

    competitiveness = _competitiveness_score(predicted)
    institute = (
        choices["institute_type"]
        .astype(str)
        .str.upper()
        .map(INSTITUTE_SCORES)
        .fillna(0.50)
        .to_numpy(dtype=float)
    )
    branch = choices["program"].map(_branch_score).to_numpy(dtype=float)
    desirability = (0.45 * competitiveness) + (0.25 * institute) + (0.30 * branch)

    admissibility = _admissibility_score(probs)
    fit = _fit_score(predicted, rank_used, probs)
    safety = probs * np.clip(choices["margin"].to_numpy(dtype=float) / np.maximum(predicted, 1), 0, 1)

    choices["competitiveness_score"] = competitiveness
    choices["institute_score"] = institute
    choices["branch_score"] = branch
    choices["desirability_score"] = desirability
    choices["admissibility_score"] = admissibility
    choices["fit_score"] = fit
    choices["safety_score"] = safety
    viable_desirability = desirability * np.maximum(admissibility, fit)
    choices["recommendation_score"] = (
        100.0 * ((0.45 * viable_desirability) + (0.30 * fit) + (0.25 * admissibility))
    ).round(4)
    choices["recommendation_bucket"] = [
        _bucket(prob, margin, rank, pred)
        for prob, margin, rank, pred in zip(
            choices["admission_probability"],
            choices["margin"],
            choices["rank_used"],
            choices["predicted_closing_rank"],
        )
    ]

    # ── 4. Classify ─────────────────────────────────────────────────────────
    choices["confidence_label"] = choices["admission_probability"].map(_classify)

    # ── 5. Explanations ──────────────────────────────────────────────────────
    choices = attach_explanations(choices, int(np.median(rank_used)))

    # ── 6. Sort and trim ─────────────────────────────────────────────────────
    choices = choices[choices["recommendation_bucket"] != "unlikely_reach"].copy()
    if choices.empty:
        return choices.drop(columns=["_recommendation_row_id"], errors="ignore").reset_index(drop=True)

    if sort_mode == "highest_probability":
        choices = choices.sort_values(["admission_probability", "recommendation_score"], ascending=[False, False])
    elif sort_mode == "most_competitive":
        choices = choices.sort_values(["predicted_closing_rank", "recommendation_score"], ascending=[True, False])
    elif sort_mode == "safest_backup":
        choices = choices.sort_values(["safety_score", "admission_probability"], ascending=[False, False])
    else:
        choices = choices.sort_values("recommendation_score", ascending=False)
    choices["_sort_position"] = np.arange(len(choices))
    if sort_mode == "best_fit":
        choices = _ensure_bucket_coverage(choices, top_n)
    else:
        choices = choices.head(top_n)
    return choices.drop(columns=["_recommendation_row_id", "_sort_position"], errors="ignore").reset_index(drop=True)
