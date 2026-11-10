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

from inference_api.config import settings
from inference_api.services.explanation_service import attach_explanations

logger = logging.getLogger("inference_api.recommendation")

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

DEFAULT_TOP_N = 100
BUCKET_QUOTAS = {
    "safe_backup": 30,
    "best_realistic": 50,
    "ambitious_reach": 20,
}
SAFE_MAX_RATIO = 0.80
TARGET_MAX_RATIO = 1.05
REACH_MAX_RATIO = 1.40
NORMAL_90_INTERVAL_Z = 1.6448536269514722


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


def _get_uncertainty_radius(std_map: pd.DataFrame, inst_type: str, round_: int) -> float:
    """Return a validated 90% prediction-interval radius for one row.

    New research artifacts provide a rolling-origin out-of-fold
    absolute-residual quantile. Legacy/synthetic test fixtures that only
    contain ``std_dev`` use the corresponding normal 90% radius.
    """
    if std_map.empty:
        return settings.mc_std_dev_default * NORMAL_90_INTERVAL_Z

    match = std_map[
        (std_map["institute_type_str"] == inst_type) & (std_map["round"] == round_)
    ]
    if match.empty:
        match = std_map[std_map["institute_type_str"] == inst_type]
    if match.empty:
        match = std_map[std_map["institute_type_str"] == "__default__"]
    if match.empty:
        return settings.mc_std_dev_default * NORMAL_90_INTERVAL_Z

    row = match.iloc[0]
    if "uncertainty_radius" in match.columns:
        radius = row.get("uncertainty_radius")
        if pd.notna(radius) and float(radius) > 0:
            return float(radius)
    std_dev = row.get("std_dev", settings.mc_std_dev_default)
    if pd.isna(std_dev) or float(std_dev) <= 0:
        std_dev = settings.mc_std_dev_default
    return float(std_dev) * NORMAL_90_INTERVAL_Z


def run_monte_carlo_vectorised(
    predicted_cutoffs: np.ndarray,
    std_devs: np.ndarray,
    student_rank: int | np.ndarray,
    n_sims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Monte Carlo simulation for all choices simultaneously.

    Samples `n_sims` cutoffs for each choice from N(pred, std_dev), then
    computes an uncalibrated admission-likelihood estimate and empirical
    prediction-interval samples.

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
    predicted_cutoffs = np.asarray(predicted_cutoffs, dtype=float)
    std_devs = np.asarray(std_devs, dtype=float)
    if predicted_cutoffs.ndim != 1 or std_devs.shape != predicted_cutoffs.shape:
        raise ValueError("predicted_cutoffs and std_devs must be equal-length 1D arrays")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if not np.isfinite(predicted_cutoffs).all() or (predicted_cutoffs < 1).any():
        raise ValueError("predicted_cutoffs must contain finite positive ranks")
    if not np.isfinite(std_devs).all() or (std_devs <= 0).any():
        raise ValueError("std_devs must contain finite positive values")

    rng = np.random.default_rng(seed=42)  # deterministic seed
    # Common random numbers preserve monotonic admission probabilities for rows
    # with equal uncertainty while retaining the correct marginal distribution.
    standard_draws = rng.normal(size=(n_sims, 1))
    simulated = (
        predicted_cutoffs[np.newaxis, :]
        + standard_draws * std_devs[np.newaxis, :]
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


def _rank_ratio(rank: float, predicted: float) -> float:
    return max(float(rank), 1.0) / max(float(predicted), 1.0)


def _bucket(rank: float, predicted: float) -> str:
    """Recommendation grouping used by the UI/RAG layer.

    Buckets are transparent rank-ratio bands. ``ambitious_reach`` means the
    student's rank is moderately above (worse than) the projected cutoff.
    Far-off options are marked separately and excluded from normal results.
    """
    ratio = _rank_ratio(rank, predicted)
    if ratio <= SAFE_MAX_RATIO:
        return "safe_backup"
    if ratio <= TARGET_MAX_RATIO:
        return "best_realistic"
    if ratio <= REACH_MAX_RATIO:
        return "ambitious_reach"
    return "unlikely_reach"


def _bucket_counts(choices: pd.DataFrame) -> dict[str, int]:
    counts = choices.get("recommendation_bucket", pd.Series(dtype=str)).value_counts()
    return {
        "safe_backup": int(counts.get("safe_backup", 0)),
        "best_realistic": int(counts.get("best_realistic", 0)),
        "ambitious_reach": int(counts.get("ambitious_reach", 0)),
        "unlikely_reach": int(counts.get("unlikely_reach", 0)),
    }


def _institute_type_counts(choices: pd.DataFrame) -> dict[str, int]:
    counts = choices.get("institute_type", pd.Series(dtype=str)).value_counts()
    return {
        "IIT": int(counts.get("IIT", 0)),
        "NIT": int(counts.get("NIT", 0)),
        "IIIT": int(counts.get("IIIT", 0)),
        "GFTI": int(counts.get("GFTI", 0)),
    }


def _scaled_bucket_quotas(top_n: int) -> dict[str, int]:
    if top_n <= 0:
        return {bucket: 0 for bucket in BUCKET_QUOTAS}
    if top_n == sum(BUCKET_QUOTAS.values()):
        return BUCKET_QUOTAS.copy()

    total = sum(BUCKET_QUOTAS.values())
    quotas = {
        bucket: int(top_n * quota / total)
        for bucket, quota in BUCKET_QUOTAS.items()
    }
    remaining = top_n - sum(quotas.values())
    for bucket in sorted(
        BUCKET_QUOTAS,
        key=lambda name: (top_n * BUCKET_QUOTAS[name] / total) - quotas[name],
        reverse=True,
    ):
        if remaining <= 0:
            break
        quotas[bucket] += 1
        remaining -= 1
    return quotas


def _select_balanced_recommendations(choices: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select a quota-balanced final list from the full scored candidate set."""
    if top_n <= 0 or choices.empty:
        return choices.head(top_n)

    quotas = _scaled_bucket_quotas(top_n)
    selected_parts = []
    selected_ids: set[int] = set()

    for bucket, quota in quotas.items():
        bucket_rows = choices[choices["recommendation_bucket"] == bucket]
        if bucket_rows.empty or quota <= 0:
            continue
        take = bucket_rows.head(quota)
        selected_parts.append(take)
        selected_ids.update(take["_recommendation_row_id"].astype(int).tolist())

    selected = (
        pd.concat(selected_parts, ignore_index=False)
        if selected_parts
        else choices.head(0).copy()
    )

    if len(selected) < top_n:
        remainder = choices[
            ~choices["_recommendation_row_id"].isin(selected_ids)
        ].head(top_n - len(selected))
        selected = pd.concat([selected, remainder], ignore_index=False)
        selected_ids.update(remainder["_recommendation_row_id"].astype(int).tolist())

    # If both JEE Main and JEE Advanced candidates exist, keep both represented.
    if "rank_type_used" in choices.columns and len(selected) >= top_n:
        available_rank_types = set(choices["rank_type_used"].dropna().astype(str))
        selected_rank_types = set(selected["rank_type_used"].dropna().astype(str))
        for rank_type in sorted(available_rank_types - selected_rank_types):
            candidate = choices[
                (choices["rank_type_used"].astype(str) == rank_type)
                & ~choices["_recommendation_row_id"].isin(selected_ids)
            ].head(1)
            if candidate.empty:
                continue

            replaceable = selected[
                selected["rank_type_used"].astype(str).map(
                    selected["rank_type_used"].astype(str).value_counts()
                )
                > 1
            ]
            if replaceable.empty:
                continue

            drop_id = replaceable.iloc[-1]["_recommendation_row_id"]
            selected = pd.concat(
                [
                    selected[selected["_recommendation_row_id"] != drop_id],
                    candidate,
                ],
                ignore_index=False,
            )
            selected_ids = set(selected["_recommendation_row_id"].astype(int).tolist())

    for bucket in BUCKET_QUOTAS:
        if bucket in set(selected["recommendation_bucket"]):
            continue
        candidate = choices[
            (choices["recommendation_bucket"] == bucket)
            & ~choices["_recommendation_row_id"].isin(selected_ids)
        ].head(1)
        if candidate.empty or len(selected) < top_n:
            if not candidate.empty:
                selected = pd.concat([selected, candidate], ignore_index=False)
            continue

        replaceable = selected[
            selected["recommendation_bucket"].map(
                selected["recommendation_bucket"].value_counts()
            )
            > 1
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
        selected_ids = set(selected["_recommendation_row_id"].astype(int).tolist())

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
    top_n: int = DEFAULT_TOP_N,
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
        choices.attrs["bucket_counts"] = _bucket_counts(choices)
        choices.attrs["institute_type_counts"] = _institute_type_counts(choices)
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
    choices["_uncertainty_radius"] = choices.apply(
        lambda row: _get_uncertainty_radius(
            std_map,
            row.get("institute_type", "__default__"),
            round_,
        ),
        axis=1,
    )

    predicted = choices["predicted_closing_rank"].values.astype(float)
    std_devs = choices["_std_dev"].values.astype(float)
    interval_radii = choices["_uncertainty_radius"].values.astype(float)
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

    rank_ratios = np.asarray(rank_used, dtype=float) / np.maximum(predicted, 1.0)
    choices["model_admission_probability"] = probs
    # This is the direct Monte Carlo estimate under the residual distribution
    # assumption. Distance-to-cutoff heuristics are kept in the separate
    # recommendation_score and never used as probability floors.
    choices["admission_probability"] = probs
    choices["model_probability_percent"] = (probs * 100.0).round(1)
    choices["calibrated_probability_percent"] = np.nan
    choices["rank_ratio"] = rank_ratios
    choices["probability_was_calibrated"] = False
    if mc_enabled:
        # The serving interval comes from rolling-origin OOF calibration when
        # available, rather than final-test residuals.
        ci_lower = predicted - interval_radii
        ci_median = predicted
        ci_upper = predicted + interval_radii
    choices["ci_lower_5"] = np.maximum(1, np.floor(ci_lower)).astype(int)
    choices["expected_cutoff_50"] = np.round(ci_median).astype(int)
    choices["ci_upper_95"] = np.maximum(1, np.ceil(ci_upper)).astype(int)

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
        _bucket(rank, pred)
        for rank, pred in zip(
            choices["rank_used"],
            choices["predicted_closing_rank"],
        )
    ]
    bucket_counts = _bucket_counts(choices)
    institute_type_counts = _institute_type_counts(choices)

    # ── 4. Classify ─────────────────────────────────────────────────────────
    choices["confidence_label"] = choices["admission_probability"].map(_classify)

    # ── 5. Explanations ──────────────────────────────────────────────────────
    choices = attach_explanations(choices, student_rank)
    # ── 6. Sort and trim ─────────────────────────────────────────────────────
    choices = choices[choices["recommendation_bucket"] != "unlikely_reach"].copy()
    if choices.empty:
        result = choices.drop(columns=["_recommendation_row_id"], errors="ignore").reset_index(drop=True)
        result.attrs["bucket_counts"] = bucket_counts
        result.attrs["institute_type_counts"] = institute_type_counts
        return result

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
        choices = _select_balanced_recommendations(choices, top_n)
    else:
        choices = choices.head(top_n)
    result = choices.drop(columns=["_recommendation_row_id", "_sort_position"], errors="ignore").reset_index(drop=True)
    result.attrs["bucket_counts"] = bucket_counts
    result.attrs["institute_type_counts"] = institute_type_counts
    return result
