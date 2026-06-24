"""
Route handlers for the JoSAA Inference Service.

Handlers are intentionally thin — they:
1. Validate input
2. Pull the ArtifactStore from app.state
3. Delegate to service functions
4. Map the result to response models

All heavy logic lives in the services/ layer.
"""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.schemas.request_models import ChatContextRequest, PredictRequest
from app.schemas.response_models import (
    ChatContextResponse,
    HealthResponse,
    HistoricalDataPoint,
    RecommendationItem,
    RecommendResponse,
)
from app.services.chatbot_context_service import build_context_block
from app.services.inference_service import filter_universe, run_inference
from app.services.recommendation_service import build_recommendations
from app.utils.validation import validate_predict_request

logger = logging.getLogger("inference_service.routes")

router = APIRouter()


# ── Health ──────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness probe — confirms artifacts are loaded."""
    artifacts = getattr(request.app.state, "artifacts", None)
    loaded = artifacts is not None
    return HealthResponse(
        status="ok",
        artifacts_loaded=loaded,
        universe_size=artifacts.universe_size if loaded else 0,
    )


# ── Shared prediction logic ─────────────────────────────────────────────────────

def _run_full_pipeline(request: Request, req: PredictRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full inference + recommendation pipeline for a student profile.

    Returns
    -------
    ranked : pd.DataFrame
        Ranked recommendations with all metadata.
    choices_raw : pd.DataFrame
        The filtered universe rows (for historical data lookup).
    """
    artifacts = request.app.state.artifacts

    # ── Filter universe ────────────────────────────────────────────────────
    choices = filter_universe(
        universe=artifacts.universe,
        category=req.category,
        gender=req.gender,
        round_=req.round,
        pref_inst_types=req.pref_inst_types,
        pref_branch_keywords=req.pref_branch_keywords,
        advanced_rank=req.advanced_rank,
    )

    if choices.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No matching programs found for this profile. "
                "Try broadening your institute type or branch filters."
            ),
        )

    # ── Run model inference ────────────────────────────────────────────────
    choices = run_inference(
        choices=choices,
        model=artifacts.model,
        prep_pipeline=artifacts.prep_pipeline,
        label_encoders=artifacts.label_encoders,
    )

    # ── Run Monte Carlo + ranking ──────────────────────────────────────────
    student_rank = req.advanced_rank if req.advanced_rank else req.main_rank

    ranked = build_recommendations(
        choices=choices,
        student_rank=student_rank,
        std_map=artifacts.std_map,
        round_=req.round,
        mc_enabled=req.mc_enabled,
        top_n=req.top_n,
    )

    return ranked, choices


def _build_historical_data(row: pd.Series) -> list[HistoricalDataPoint]:
    """
    Extract historical opening/closing rank data from a universe row.

    The cleaned CSV has columns like opening_rank and closing_rank for 2024.
    For a richer view we'd need to query the full time-series, but for now
    we surface what's available.
    """
    points = []
    # If the universe has year/opening_rank/closing_rank columns
    if "year" in row.index and "opening_rank" in row.index and "closing_rank" in row.index:
        points.append(
            HistoricalDataPoint(
                year=int(row.get("year", 2024)),
                opening_rank=int(row.get("opening_rank", 0)),
                closing_rank=int(row.get("closing_rank", 0)),
            )
        )
    return points


def _row_to_item(row: pd.Series, idx: int, student_rank: int) -> RecommendationItem:
    """Convert a single ranked-recommendation DataFrame row to a response model."""
    historical = _build_historical_data(row)

    margin = row.get("margin")
    safety_margin = int(margin) if margin is not None and not pd.isna(margin) else None

    prob = float(row.get("admission_probability", 0.0))
    prob_pct = round(prob * 100, 1)

    lower = row.get("ci_lower_5")
    upper = row.get("ci_upper_95")

    return RecommendationItem(
        id=str(idx),
        institute_name=str(row.get("institute", "Unknown")),
        institute_type=str(row.get("institute_type", "Unknown")),
        program_name=str(row.get("program", "Unknown")),
        quota_applied=str(row.get("quota", "AI")),
        category=str(row.get("category", "")),
        probability_percent=prob_pct,
        projected_closing_rank=int(row.get("predicted_closing_rank", 0)),
        uncertainty_lower=int(lower) if lower is not None and not pd.isna(lower) else None,
        uncertainty_upper=int(upper) if upper is not None and not pd.isna(upper) else None,
        safety_margin=safety_margin,
        confidence_label=str(row.get("confidence_label", "Unknown")),
        explanation=str(row.get("explanation", "")),
        recommendation_score=float(row.get("recommendation_score", 0.0)),
        historical_data=historical,
    )


def _build_response(ranked: pd.DataFrame, req: PredictRequest) -> RecommendResponse:
    """Convert the ranked DataFrame to the full response model."""
    items = [
        _row_to_item(row, idx, req.advanced_rank or req.main_rank)
        for idx, row in ranked.iterrows()
    ]

    safe_items = [r for r in items if r.confidence_label == "Safe"]
    mod_items = [r for r in items if r.confidence_label == "Moderate"]
    amb_items = [r for r in items if r.confidence_label == "Ambitious"]

    safest_choice = (
        f"{safe_items[0].institute_name} — {safe_items[0].program_name}"
        if safe_items
        else (items[0].institute_name + " — " + items[0].program_name if items else "—")
    )
    top_upgrade = (
        f"{mod_items[0].institute_name} — {mod_items[0].program_name}"
        if mod_items
        else safest_choice
    )
    most_ambitious = (
        f"{amb_items[0].institute_name} — {amb_items[0].program_name}"
        if amb_items
        else (items[-1].institute_name + " — " + items[-1].program_name if items else "—")
    )

    return RecommendResponse(
        total_options=len(items),
        safest_choice=safest_choice,
        top_upgrade=top_upgrade,
        most_ambitious=most_ambitious,
        safe_count=len(safe_items),
        moderate_count=len(mod_items),
        ambitious_count=len(amb_items),
        results=items,
    )


# ── POST /predict ───────────────────────────────────────────────────────────────

@router.post("/predict", response_model=RecommendResponse)
async def predict(request: Request, req: PredictRequest) -> RecommendResponse:
    """
    Run ML inference for a student profile and return ranked recommendations.

    Input mirrors the frontend's SimulationInput type.
    Output is a superset of SimulationResponse with additional ML metadata.
    """
    validate_predict_request(req)

    logger.info(
        f"Predict request: rank={req.main_rank}, category={req.category}, "
        f"gender={req.gender}, inst_types={req.pref_inst_types}"
    )

    try:
        ranked, _ = _run_full_pipeline(request, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return _build_response(ranked, req)


# ── POST /recommend ─────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: Request, req: PredictRequest) -> RecommendResponse:
    """
    Alias for /predict — returns the same ranked recommendation table.
    Exposed as a separate endpoint for semantic clarity in frontend code.
    """
    return await predict(request, req)


# ── POST /chat-context ──────────────────────────────────────────────────────────

@router.post("/chat-context", response_model=ChatContextResponse)
async def chat_context(request: Request, req: ChatContextRequest) -> ChatContextResponse:
    """
    Convert recommendation output into a compact context block for the RAG chatbot.

    The context block is grounded purely in model outputs and static project docs —
    never hallucinated content.
    """
    logger.info(f"Chat-context request: question={req.question[:60]!r}…")

    context_block, key_facts = build_context_block(req.recommendation_output)

    return ChatContextResponse(context_block=context_block, key_facts=key_facts)
