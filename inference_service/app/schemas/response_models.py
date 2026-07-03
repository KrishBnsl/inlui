"""
Pydantic response models for the JoSAA Inference Service.

The shapes are designed as a strict superset of the frontend's existing
SimulationResponse / PredictionResult types in seatcraft/src/lib/types.ts.
All new fields are optional so the frontend degrades gracefully when running
against the older Rust sim_engine.
"""

from pydantic import BaseModel, Field


# ── Sub-models ─────────────────────────────────────────────────────────────────

class HistoricalDataPoint(BaseModel):
    """One year's opening / closing rank data for a program."""

    year: int
    opening_rank: int
    closing_rank: int


class RecommendationItem(BaseModel):
    """
    Single recommended college/program choice.

    Fields prefixed with a comment are new (not in the Rust response).
    All new fields are Optional so the frontend type-checks against both backends.
    """

    id: str = Field(..., description="Unique identifier for this recommendation.")

    # ── Core identity ─────────────────────────────────────────────────────
    institute_name: str
    institute_type: str = Field(..., description="IIT / NIT / IIIT / GFTI")
    program_name: str
    quota_applied: str = Field(..., description="AI or HS quota used.")
    category: str

    # ── Probability & rank prediction ─────────────────────────────────────
    probability_percent: float = Field(
        ..., description="Admission probability as a percentage [0, 100]."
    )
    admission_probability: float = Field(
        ..., description="Admission probability as a decimal [0, 1]."
    )
    model_probability_percent: float | None = Field(
        default=None,
        description="Raw model/Monte Carlo admission probability as a percentage [0, 100].",
    )
    calibrated_probability_percent: float | None = Field(
        default=None,
        description="Displayed probability after cutoff-distance calibration [0, 100].",
    )
    projected_closing_rank: int = Field(
        ..., description="Model-predicted closing rank for the upcoming round."
    )

    # ── New: uncertainty bounds (90 % CI from Monte Carlo) ────────────────
    uncertainty_lower: int | None = Field(
        default=None,
        description="5th-percentile simulated closing rank (optimistic scenario).",
    )
    uncertainty_upper: int | None = Field(
        default=None,
        description="95th-percentile simulated closing rank (pessimistic scenario).",
    )

    # ── New: recommendation metadata ──────────────────────────────────────
    safety_margin: int | None = Field(
        default=None,
        description=(
            "projected_closing_rank − student_rank. "
            "Positive → student is safer; negative → student is at reach."
        ),
    )
    rank_ratio: float | None = Field(
        default=None,
        description="rank_used / projected_closing_rank. Greater than 1 means the option is harder than the student's rank.",
    )
    confidence_label: str | None = Field(
        default=None,
        description="Safe | Moderate | Ambitious",
    )
    explanation: str | None = Field(
        default=None,
        description="Short natural-language reason for the recommendation.",
    )
    recommendation_score: float | None = Field(
        default=None,
        description="Internal composite ranking score (higher is better).",
    )
    fit_score: float | None = Field(default=None, description="Rank-fit score [0, 1].")
    competitiveness_score: float | None = Field(
        default=None, description="Competitiveness/desirability of projected cutoff [0, 1]."
    )
    institute_score: float | None = Field(default=None, description="Institute desirability score [0, 1].")
    branch_score: float | None = Field(default=None, description="Branch desirability score [0, 1].")
    safety_score: float | None = Field(default=None, description="Safety score [0, 1].")
    recommendation_bucket: str | None = Field(
        default=None,
        description="safe_backup | best_realistic | ambitious_reach | unlikely_reach",
    )
    rank_used: int = Field(..., description="The exam rank used for this recommendation row.")
    rank_type_used: str = Field(..., description="JEE_MAIN for NIT/IIIT/GFTI, JEE_ADVANCED for IIT.")

    # ── Historical cutoff data (same as Rust response) ────────────────────
    historical_data: list[HistoricalDataPoint] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    """
    Full recommendation response.  Compatible superset of SimulationResponse.
    """

    # ── Fields matching SimulationResponse in types.ts ────────────────────
    total_options: int
    safest_choice: str = Field(..., description="Institute – program of the top safe pick.")
    top_upgrade: str = Field(..., description="Institute – program of the best upgrade pick.")

    # ── New summary fields ────────────────────────────────────────────────
    most_ambitious: str | None = Field(
        default=None,
        description="Institute – program of the highest-reach pick.",
    )
    safe_count: int | None = Field(default=None, description="Number of Safe recommendations.")
    moderate_count: int | None = Field(default=None, description="Number of Moderate recommendations.")
    ambitious_count: int | None = Field(default=None, description="Number of Ambitious recommendations.")
    bucket_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts across the full candidate set before normal result trimming.",
    )
    institute_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts by institute type across the full candidate set before normal result trimming.",
    )
    rank_window_debug: dict = Field(
        default_factory=dict,
        description="Rank-specific reach windows used for this request.",
    )

    # ── Results list ─────────────────────────────────────────────────────
    results: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    artifacts_loaded: bool
    universe_size: int
    artifact_status: dict[str, str] = Field(default_factory=dict)


class ChatContextResponse(BaseModel):
    """Response for POST /chat-context."""

    context_block: str = Field(
        ...,
        description=(
            "Compact text block to prepend to the RAG chatbot prompt. "
            "Contains student profile, top picks, and model notes."
        ),
    )
    key_facts: dict = Field(
        default_factory=dict,
        description="Structured key-value facts extracted from the recommendation output.",
    )
