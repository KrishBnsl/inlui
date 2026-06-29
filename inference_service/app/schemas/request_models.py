"""
Pydantic request models for the JoSAA Inference Service.

All fields map 1-to-1 with the frontend's SimulationInput type in
seatcraft/src/lib/types.ts, with additional optional fields for
ML-specific preferences.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Input for POST /predict and POST /recommend.

    Required fields match the frontend SimulationInput interface.
    Optional fields enable more fine-grained filtering and control.
    """

    # ── Required — mirrors SimulationInput in types.ts ─────────────────────
    main_rank: int = Field(
        ...,
        ge=1,
        le=1_200_000,
        description="JEE Main CRL rank — used for NIT / IIIT / GFTI eligibility.",
        examples=[15000],
    )
    category: str = Field(
        ...,
        description="Reservation category (OPEN, OBC-NCL, EWS, SC, ST).",
        examples=["OBC-NCL"],
    )
    gender: str = Field(
        ...,
        description="Seat pool (Gender-Neutral or Female-only).",
        examples=["Gender-Neutral"],
    )

    # ── Optional — present in SimulationInput ──────────────────────────────
    advanced_rank: int | None = Field(
        default=None,
        ge=1,
        le=50_000,
        description="JEE Advanced rank — required to see IIT options.",
    )
    home_state: str | None = Field(
        default=None,
        description=(
            "Student's home state. Used for HS routing when the loaded universe "
            "contains institute-state metadata."
        ),
        examples=["Tamil Nadu"],
    )
    is_pwd: bool = Field(
        default=False,
        description="PwD (Person with Disability) flag.",
    )

    # ── ML-specific preferences (fully optional) ───────────────────────────
    pref_inst_types: list[str] | None = Field(
        default=None,
        description=(
            "Filter by institute types, e.g. ['NIT', 'IIIT']. "
            "Leave null for all types."
        ),
        examples=[["NIT", "IIIT"]],
    )
    pref_branch_keywords: list[str] | None = Field(
        default=None,
        description=(
            "Case-insensitive keywords to filter program names, "
            "e.g. ['Computer Science', 'Electronics']. "
            "Leave null for all branches."
        ),
        examples=[["Computer Science"]],
    )
    round: int = Field(
        default=5,
        ge=1,
        le=6,
        description="JoSAA counselling round to use as the universe baseline (default 5).",
    )
    mc_enabled: bool = Field(
        default=True,
        description=(
            "If true, run Monte Carlo uncertainty simulation. "
            "Set false for deterministic (faster) predictions."
        ),
    )
    top_n: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Maximum number of recommendations to return (sorted by score).",
    )


class ChatContextRequest(BaseModel):
    """Input for POST /chat-context."""

    question: str = Field(
        ...,
        min_length=1,
        description="The user's question to the chatbot.",
    )
    recommendation_output: dict | None = Field(
        default=None,
        description=(
            "The full JSON response from POST /predict or /recommend. "
            "Used to generate context the chatbot can reference."
        ),
    )
