"""
Explanation service — generates per-choice natural language explanations.

Explanations are kept short and factual (1–2 sentences) and are based purely
on model output values rather than LLM generation, so they are deterministic
and never hallucinate.
"""

import logging

import pandas as pd

logger = logging.getLogger("inference_api.explanation")


def generate_explanation(row: pd.Series, student_rank: int) -> str:
    """
    Generate a short explanation string for a single recommendation row.

    Parameters
    ----------
    row : pd.Series
        A row from the ranked recommendations DataFrame.  Must contain
        'institute', 'program', 'predicted_closing_rank', 'admission_probability',
        and 'confidence_label'.
    student_rank : int
        The student's JEE rank.

    Returns
    -------
    str
        A 1–2 sentence explanation grounded only in model outputs.
    """
    predicted_cutoff = int(row.get("predicted_closing_rank", 0))
    prob = float(row.get("admission_probability", 0.0))
    label = str(row.get("confidence_label", "Unknown"))
    margin = predicted_cutoff - student_rank

    institute = row.get("institute", "This institute")
    program = row.get("program", "this program")

    prob_pct = round(prob * 100, 1)

    if label == "Safe":
        return (
            f"{institute} — {program} is a Safe choice. "
            f"The predicted closing rank ({predicted_cutoff:,}) is {abs(margin):,} places "
            f"beyond your rank; the model's uncalibrated admission-likelihood estimate "
            f"is {prob_pct}% based on simulated closing cutoffs, not individual outcomes."
        )
    elif label == "Moderate":
        if margin > 0:
            return (
                f"{institute} — {program} is a Moderate choice. "
                f"Your rank is within {margin:,} ranks of the predicted cutoff ({predicted_cutoff:,}), "
                f"and the model's uncalibrated admission-likelihood estimate is {prob_pct}% "
                f"from simulated closing cutoffs; the actual cutoff remains uncertain."
            )
        else:
            return (
                f"{institute} — {program} is a Moderate choice. "
                f"Your rank is {abs(margin):,} places outside the predicted cutoff "
                f"({predicted_cutoff:,}); the model's uncalibrated admission-likelihood estimate "
                f"is {prob_pct}% from simulated closing cutoffs, not individual outcomes."
            )
    else:  # Ambitious
        return (
            f"{institute} — {program} is an Ambitious reach. "
            f"The predicted cutoff ({predicted_cutoff:,}) is {abs(margin):,} ranks tighter than "
            f"your rank; the model's uncalibrated admission-likelihood estimate is {prob_pct}% "
            f"from simulated closing cutoffs, not an individual admission probability."
        )


def attach_explanations(df: pd.DataFrame, student_rank: int) -> pd.DataFrame:
    """
    Add an 'explanation' column to the ranked recommendations DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Ranked choices with columns: institute, program,
        predicted_closing_rank, admission_probability, confidence_label.
    student_rank : int
        Fallback JEE rank for legacy rows without a scalar ``rank_used`` value.
        Routed per-row ranks take precedence so IIT explanations use JEE Advanced
        rank while NIT/IIIT/GFTI explanations use JEE Main rank.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 'explanation' column added.
    """
    df = df.copy()

    def rank_for_row(row: pd.Series) -> int:
        rank_used = row.get("rank_used")
        if pd.api.types.is_scalar(rank_used) and not pd.isna(rank_used):
            return int(rank_used)
        return int(student_rank)

    df["explanation"] = df.apply(
        lambda row: generate_explanation(row, rank_for_row(row)), axis=1
    )
    return df
