"""
Pydantic response models for the JoSAA Inference Service.

The shapes are the authoritative Python inference contract consumed by the
SeatCraft frontend. Optional metadata remains nullable so missing artifact
facts are shown as unknown rather than guessed.
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

    Optional fields are explicit research/serving metadata.
    """

    id: str = Field(..., description="Unique identifier for this recommendation.")

    # ── Core identity ─────────────────────────────────────────────────────
    institute_name: str
    institute_type: str = Field(..., description="IIT / NIT / IIIT / GFTI")
    program_name: str
    quota_applied: str = Field(..., description="AI, HS, OS, or STATE quota used.")
    category: str

    # ── Admission-likelihood estimate & rank prediction ──────────────────
    probability_percent: float = Field(
        ...,
        description=(
            "Uncalibrated admission-likelihood estimate as a percentage [0, 100]. "
            "This is simulated cutoff exceedance, not an individual admission probability."
        ),
    )
    admission_probability: float = Field(
        ...,
        description=(
            "Uncalibrated admission-likelihood estimate as a decimal [0, 1], "
            "derived from Monte Carlo closing-cutoff simulations. The field name is "
            "retained for API compatibility; it is not an individual admission probability."
        ),
    )
    model_probability_percent: float | None = Field(
        default=None,
        description=(
            "Raw uncalibrated admission-likelihood estimate from Monte Carlo "
            "closing-cutoff simulations as a percentage [0, 100]."
        ),
    )
    calibrated_probability_percent: float | None = Field(
        default=None,
        description=(
            "Reserved for a future empirically calibrated probability. "
            "Null for the current uncalibrated model."
        ),
    )
    projected_closing_rank: int = Field(
        ..., description="Model-predicted closing rank for the upcoming round."
    )

    # ── Uncertainty bounds (90% empirical prediction interval) ───────────
    uncertainty_lower: int | None = Field(
        default=None,
        description="Lower bound of the empirical 90% closing-rank prediction interval.",
    )
    uncertainty_upper: int | None = Field(
        default=None,
        description="Upper bound of the empirical 90% closing-rank prediction interval.",
    )

    # ── Recommendation metadata ───────────────────────────────────────────
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

    # ── Historical cutoff data ────────────────────────────────────────────
    historical_data: list[HistoricalDataPoint] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    """Full recommendation response for the Python inference API."""

    # ── Frontend summary fields ────────────────────────────────────────────
    total_options: int
    safest_choice: str = Field(..., description="Institute – program of the top safe pick.")
    top_upgrade: str = Field(..., description="Institute – program of the best upgrade pick.")

    # ── Extended summary fields ───────────────────────────────────────────
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
    candidate_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Candidate counts at each filtering, scoring, and return stage.",
    )
    returned_bucket_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Bucket counts in the returned result list.",
    )

    # Artifact-backed serving metadata. Unknown values are null, never guessed.
    data_cutoff: str | None = None
    model_version: str | None = None
    prediction_year: int | None = None
    probability_method: str = (
        "Uncalibrated admission-likelihood estimate from a Monte Carlo normal "
        "approximation scaled to a rolling-origin OOF 90% residual radius; "
        "not an individual admission probability"
    )
    interval_label: str = "90% empirical prediction interval"
    decision_support_disclaimer: str = (
        "Decision support only; the uncalibrated admission-likelihood estimate "
        "is not an individual admission probability and is not an admission guarantee."
    )

    # ── Results list ─────────────────────────────────────────────────────
    results: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    ready: bool = False
    artifacts_loaded: bool
    universe_size: int
    artifact_status: dict[str, str] = Field(default_factory=dict)
    artifact_error: str | None = None
    recovery_command: str | None = None


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
