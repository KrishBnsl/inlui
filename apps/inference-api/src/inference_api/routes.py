"""
Route handlers for the JoSAA Inference Service.

Handlers are intentionally thin — they:
1. Validate input
2. Pull the ArtifactStore from application state
3. Delegate to service functions
4. Map the result to response models

All heavy logic lives in the services/ layer.
"""

import hashlib
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response

from inference_api.schemas.request_models import ChatContextRequest, PredictRequest
from inference_api.schemas.response_models import (
    ChatContextResponse,
    HealthResponse,
    HistoricalDataPoint,
    RecommendationItem,
    RecommendResponse,
)
from inference_api.services.chatbot_context_service import build_context_block
from inference_api.services.artifact_loader import ARTIFACT_RECOVERY_COMMAND
from inference_api.services.prediction_service import assign_rank_sources, filter_universe, run_inference
from inference_api.services.recommendation_service import build_recommendations
from inference_api.utils.validation import validate_predict_request

logger = logging.getLogger("inference_api.routes")

router = APIRouter()


# ── Health ──────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness probe — confirms artifacts are loaded."""
    artifacts = getattr(request.app.state, "artifacts", None)
    loaded = artifacts is not None
    missing = {
        "model": "missing",
        "preprocessing_pipeline": "missing",
        "label_encoders": "not_required",
        "feature_schema": "missing",
        "residual_uncertainty": "missing",
        "program_universe": "missing",
        "model_metadata": "missing",
    }
    return HealthResponse(
        status="ok" if loaded else "degraded",
        ready=loaded,
        artifacts_loaded=loaded,
        universe_size=artifacts.universe_size if loaded else 0,
        artifact_status=artifacts.artifact_status if loaded else missing,
        artifact_error=getattr(request.app.state, "artifact_error", None),
        recovery_command=None if loaded else ARTIFACT_RECOVERY_COMMAND,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness(request: Request, response: Response) -> HealthResponse:
    """Readiness probe; returns 503 until every validated artifact is loaded."""
    result = await health(request)
    if not result.ready:
        response.status_code = 503
    return result


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
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Inference artifacts are not loaded. Check ARTIFACTS_DIR and DATA_DIR.",
        )

    # ── Filter universe ────────────────────────────────────────────────────
    choices = filter_universe(
        universe=artifacts.universe,
        category=req.category,
        gender=req.gender,
        round_=req.round,
        pref_inst_types=req.pref_inst_types,
        pref_branch_keywords=req.pref_branch_keywords,
        advanced_rank=req.advanced_rank,
        home_state=req.home_state,
        is_pwd=req.is_pwd,
        pref_quotas=req.pref_quotas,
    )
    candidate_counts = {
        "universe": len(artifacts.universe),
        "after_profile_filters": len(choices),
    }
    logger.info(
        "Candidate counts: universe=%s after_profile_filters=%s",
        len(artifacts.universe),
        len(choices),
    )

    if choices.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No matching programs found for this profile. "
                "Try broadening your institute type or branch filters."
            ),
        )

    try:
        choices = assign_rank_sources(
            choices,
            main_rank=req.main_rank,
            advanced_rank=req.advanced_rank,
        )
        candidate_counts["after_rank_routing"] = len(choices)
        logger.info(
            "Candidate counts after rank-source assignment: %s",
            len(choices),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Run model inference ────────────────────────────────────────────────
    choices = run_inference(
        choices=choices,
        model=artifacts.model,
        prep_pipeline=artifacts.prep_pipeline,
        label_encoders=artifacts.label_encoders,
        feature_schema=artifacts.feature_schema,
    )

    # ── Run Monte Carlo + ranking ──────────────────────────────────────────
    student_rank = req.main_rank

    ranked = build_recommendations(
        choices=choices,
        student_rank=student_rank,
        std_map=artifacts.std_map,
        round_=req.round,
        mc_enabled=req.mc_enabled,
        top_n=req.top_n,
        sort_mode=req.sort_mode,
    )
    full_bucket_counts = ranked.attrs.get("bucket_counts", {})
    candidate_counts["scored"] = sum(int(value) for value in full_bucket_counts.values())
    candidate_counts["eligible_after_reach_filter"] = sum(
        int(full_bucket_counts.get(name, 0))
        for name in ("safe_backup", "best_realistic", "ambitious_reach")
    )
    candidate_counts["returned"] = len(ranked)
    ranked.attrs["candidate_counts"] = candidate_counts
    ranked.attrs["serving_metadata"] = artifacts.metadata

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


def _stable_choice_id(row: pd.Series) -> str:
    """Return an order-independent ID for one programme/seat-pool identity."""

    fields = (
        "institute",
        "institute_type",
        "program",
        "quota",
        "category",
        "gender",
        "is_pwd",
        "round",
    )
    identity = "\x1f".join(str(row.get(field, "")).strip().casefold() for field in fields)
    return f"choice-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _row_to_item(row: pd.Series, student_rank: int) -> RecommendationItem:
    """Convert a single ranked-recommendation DataFrame row to a response model."""
    historical = _build_historical_data(row)

    margin = row.get("margin")
    safety_margin = int(margin) if margin is not None and not pd.isna(margin) else None

    prob = float(row.get("admission_probability", 0.0))
    prob_pct = round(prob * 100, 1)

    lower = row.get("ci_lower_5")
    upper = row.get("ci_upper_95")

    return RecommendationItem(
        id=_stable_choice_id(row),
        institute_name=str(row.get("institute", "Unknown")),
        institute_type=str(row.get("institute_type", "Unknown")),
        program_name=str(row.get("program", "Unknown")),
        quota_applied=str(row.get("quota", "AI")),
        category=str(row.get("category", "")),
        probability_percent=prob_pct,
        admission_probability=prob,
        model_probability_percent=float(row.get("model_probability_percent", prob_pct)),
        calibrated_probability_percent=(
            float(row.get("calibrated_probability_percent"))
            if row.get("calibrated_probability_percent") is not None
            and not pd.isna(row.get("calibrated_probability_percent"))
            else None
        ),
        projected_closing_rank=int(row.get("predicted_closing_rank", 0)),
        uncertainty_lower=int(lower) if lower is not None and not pd.isna(lower) else None,
        uncertainty_upper=int(upper) if upper is not None and not pd.isna(upper) else None,
        safety_margin=safety_margin,
        rank_ratio=float(row.get("rank_ratio")) if row.get("rank_ratio") is not None and not pd.isna(row.get("rank_ratio")) else None,
        confidence_label=str(row.get("confidence_label", "Unknown")),
        explanation=str(row.get("explanation", "")),
        recommendation_score=float(row.get("recommendation_score", 0.0)),
        fit_score=float(row.get("fit_score", 0.0)),
        competitiveness_score=float(row.get("competitiveness_score", 0.0)),
        institute_score=float(row.get("institute_score", 0.0)),
        branch_score=float(row.get("branch_score", 0.0)),
        safety_score=float(row.get("safety_score", 0.0)),
        recommendation_bucket=str(row.get("recommendation_bucket", "")) or None,
        rank_used=int(row.get("rank_used", student_rank)),
        rank_type_used=str(row.get("rank_type_used", "JEE_MAIN")),
        historical_data=historical,
    )


def _build_response(ranked: pd.DataFrame, req: PredictRequest) -> RecommendResponse:
    """Convert the ranked DataFrame to the full response model."""
    items = [
        _row_to_item(row, req.advanced_rank or req.main_rank)
        for _, row in ranked.iterrows()
    ]

    safe_items = [r for r in items if r.recommendation_bucket == "safe_backup"]
    target_items = [r for r in items if r.recommendation_bucket == "best_realistic"]
    reach_items = [r for r in items if r.recommendation_bucket == "ambitious_reach"]
    bucket_counts = ranked.attrs.get(
        "bucket_counts",
        {
            "safe_backup": len(safe_items),
            "best_realistic": len(target_items),
            "ambitious_reach": len(reach_items),
            "unlikely_reach": 0,
        },
    )
    institute_type_counts = ranked.attrs.get(
        "institute_type_counts",
        {"IIT": 0, "NIT": 0, "IIIT": 0, "GFTI": 0},
    )
    returned_bucket_counts = {
        "safe_backup": len(safe_items),
        "best_realistic": len(target_items),
        "ambitious_reach": len(reach_items),
        "unlikely_reach": 0,
    }
    metadata = ranked.attrs.get("serving_metadata", {})
    rank_window_debug = {
        "main_rank": req.main_rank,
        "advanced_rank": req.advanced_rank,
        "main_reach_window": [round(req.main_rank * 0.60), round(req.main_rank * 1.05)],
        "advanced_reach_window": (
            [round(req.advanced_rank * 0.60), round(req.advanced_rank * 1.05)]
            if req.advanced_rank
            else None
        ),
    }

    safest_choice = (
        f"{safe_items[0].institute_name} — {safe_items[0].program_name}"
        if safe_items
        else (items[0].institute_name + " — " + items[0].program_name if items else "—")
    )
    top_upgrade = (
        f"{reach_items[0].institute_name} — {reach_items[0].program_name}"
        if reach_items
        else "No realistic reach found"
    )
    most_ambitious = (
        f"{reach_items[0].institute_name} — {reach_items[0].program_name}"
        if reach_items
        else None
    )

    return RecommendResponse(
        total_options=len(items),
        safest_choice=safest_choice,
        top_upgrade=top_upgrade,
        most_ambitious=most_ambitious,
        safe_count=len(safe_items),
        moderate_count=len(target_items),
        ambitious_count=len(reach_items),
        bucket_counts=bucket_counts,
        institute_type_counts=institute_type_counts,
        rank_window_debug=rank_window_debug,
        candidate_counts=ranked.attrs.get("candidate_counts", {"returned": len(items)}),
        returned_bucket_counts=returned_bucket_counts,
        data_cutoff=(
            str(metadata.get("data_cutoff_year"))
            if metadata.get("data_cutoff_year") is not None
            else None
        ),
        model_version=metadata.get("model_version"),
        prediction_year=metadata.get("prediction_year"),
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
        logger.info(
            "Recommendation bucket distribution: %s; institute types: %s; returned=%s; rank_windows=%s",
            ranked.attrs.get("bucket_counts", {}),
            ranked.attrs.get("institute_type_counts", {}),
            len(ranked),
            {
                "main": [round(req.main_rank * 0.60), round(req.main_rank * 1.05)],
                "advanced": (
                    [round(req.advanced_rank * 0.60), round(req.advanced_rank * 1.05)]
                    if req.advanced_rank
                    else None
                ),
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during inference")
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Check the inference-service logs for diagnostics.",
        )

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
    logger.info("Chat-context request accepted question_length=%s", len(req.question))

    context_block, key_facts = build_context_block(req.recommendation_output)

    return ChatContextResponse(context_block=context_block, key_facts=key_facts)
