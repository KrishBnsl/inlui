"""
Chatbot context service.

Converts the recommendation engine output into a compact, structured text block
that the RAG chatbot can use as additional context in its prompt.

The block is grounded entirely in model output — no hallucinations.
The chatbot is instructed to answer questions only from this block and the
indexed documents, not from general web knowledge.
"""

import logging
from typing import Any

logger = logging.getLogger("inference_service.chatbot_context_service")


# ── Context block builder ───────────────────────────────────────────────────────

def build_context_block(recommendation_output: dict | None) -> tuple[str, dict]:
    """
    Convert a recommendation response dict into a chatbot context block.

    Parameters
    ----------
    recommendation_output : dict | None
        The JSON-serialisable dict returned by /predict or /recommend.
        May be None if the user hasn't run a simulation yet.

    Returns
    -------
    context_block : str
        Multi-line text to prepend to the RAG prompt.
    key_facts : dict
        Structured key-value pairs extracted from the output.
    """
    if not recommendation_output:
        return _no_results_block(), {}

    try:
        return _build_from_output(recommendation_output)
    except Exception as exc:
        logger.warning(f"Failed to build context block: {exc}", exc_info=True)
        return _no_results_block(), {}


def _no_results_block() -> str:
    return (
        "## Model Context\n"
        "No recommendation results are available yet for this session. "
        "The student has not submitted their profile, or the simulation has not completed.\n\n"
        + MODEL_NOTES
    )


def _build_from_output(output: dict) -> tuple[str, dict]:
    """Build context when recommendation output is present."""
    results: list[dict] = output.get("results", [])
    total = output.get("total_options", len(results))
    safest = output.get("safest_choice", "—")
    upgrade = output.get("top_upgrade", "—")
    most_ambitious = output.get("most_ambitious", "—")
    safe_count = output.get("safe_count", "—")
    moderate_count = output.get("moderate_count", "—")
    ambitious_count = output.get("ambitious_count", "—")

    lines: list[str] = [
        "## Student Recommendation Context",
        "",
        f"- **Total eligible options found:** {total}",
        f"- **Safe choices:** {safe_count}",
        f"- **Moderate choices:** {moderate_count}",
        f"- **Ambitious choices:** {ambitious_count}",
        f"- **Top Safe pick:** {safest}",
        f"- **Best Upgrade pick:** {upgrade}",
        f"- **Most Ambitious pick:** {most_ambitious}",
        "",
        "### Top Recommendations (sorted by admission probability)",
        "",
    ]

    # Emit up to the top 10 picks in human-readable form
    for i, item in enumerate(results[:10], start=1):
        institute = item.get("institute_name", "Unknown Institute")
        program = item.get("program_name", "Unknown Program")
        prob = item.get("probability_percent", 0.0)
        cutoff = item.get("projected_closing_rank", 0)
        label = item.get("confidence_label", "Unknown")
        lower = item.get("uncertainty_lower")
        upper = item.get("uncertainty_upper")
        margin = item.get("safety_margin")
        explanation = item.get("explanation", "")
        quota = item.get("quota_applied", "—")

        ci_text = f" | 90% CI: [{lower:,}–{upper:,}]" if lower and upper else ""
        margin_text = f" | Safety margin: {margin:+,}" if margin is not None else ""

        lines.append(
            f"{i}. **{institute}** — {program} "
            f"[{label}] | {prob:.1f}% admission probability | "
            f"Predicted cutoff: {cutoff:,}{ci_text}{margin_text} | Quota: {quota}"
        )
        if explanation:
            lines.append(f"   → {explanation}")
        lines.append("")

    lines.append("")
    lines.append(MODEL_NOTES)

    context_block = "\n".join(lines)

    key_facts = {
        "model_type": "Ridge regression (sklearn)",
        "training_data": "JoSAA cutoff data 2018–2024",
        "uncertainty_method": "Monte Carlo simulation (N=1000), 90% confidence interval",
        "safe_threshold": "≥80% admission probability",
        "moderate_threshold": "40–80% admission probability",
        "ambitious_threshold": "<40% admission probability",
        "total_options": total,
        "safe_count": safe_count,
        "moderate_count": moderate_count,
        "ambitious_count": ambitious_count,
    }

    return context_block, key_facts


# ── Static model notes ──────────────────────────────────────────────────────────

MODEL_NOTES = """\
## About the Prediction Model

- **Model type:** Ridge regression trained on JoSAA allotment data from 2018–2024.
- **What it predicts:** The expected closing rank for a given institute–program–category–gender–quota combination in the upcoming JoSAA round.
- **Uncertainty:** Each prediction is run through a Monte Carlo simulation (N=1000) sampling from a Gaussian centred on the predicted rank. The 5th and 95th percentiles form the 90% confidence interval.
- **Admission probability:** The fraction of Monte Carlo simulations in which the student's rank is ≤ the simulated cutoff. A probability of 80% means the student would be admitted in ~800 of 1000 simulated scenarios.
- **Classification labels:**
  - **Safe** (≥80%): High confidence of admission; the student's rank is comfortably within the predicted cutoff range.
  - **Moderate** (40–80%): Competitive but uncertain; the student is near the historical boundary.
  - **Ambitious** (<40%): The student's rank exceeds the expected cutoff; admission depends on unusually high cutoff volatility.
- **Limitations:** The model cannot account for sudden policy changes, seat matrix revisions, or extreme fluctuations in the applicant pool. Always verify final cutoffs on the official JoSAA portal (josaa.nic.in).
- **Data source:** JoSAA official allotment data 2018–2024, cleaned and aggregated by the inlui ML pipeline.
"""
