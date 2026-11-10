"""
Pydantic request models for the JoSAA Inference Service.

All fields map 1-to-1 with the frontend's SimulationInput type in
apps/web/src/lib/types.ts, with additional optional fields for
ML-specific preferences.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """
    Input for POST /predict and POST /recommend.

    Required fields match the frontend SimulationInput interface.
    Optional fields enable more fine-grained filtering and control.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Required — mirrors SimulationInput in types.ts ─────────────────────
    main_rank: int = Field(
        ...,
        strict=True,
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
        strict=True,
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
        strict=True,
        description="PwD (Person with Disability) flag.",
    )

    # ── ML-specific preferences (fully optional) ───────────────────────────
    pref_inst_types: list[str] | None = Field(
        default=None,
        max_length=4,
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
    pref_quotas: list[str] | None = Field(
        default=None,
        max_length=4,
        description="Optional quota filter: AI, HS, OS, or STATE.",
    )
    round: int = Field(
        default=5,
        strict=True,
        ge=1,
        le=6,
        description="JoSAA counselling round to use as the universe baseline (default 5).",
    )
    mc_enabled: Literal[True] = Field(
        default=True,
        description=(
            "Must remain true for the public API so probability and interval "
            "metadata always describe the returned calculation."
        ),
    )
    top_n: int = Field(
        default=100,
        strict=True,
        ge=1,
        le=200,
        description="Maximum number of recommendations to return (balanced by bucket).",
    )
    sort_mode: Literal[
        "best_fit",
        "highest_probability",
        "most_competitive",
        "safest_backup",
    ] = Field(
        default="best_fit",
        description="Backend ranking mode. Default is composite best realistic fit.",
    )

    @field_validator("home_state")
    @classmethod
    def validate_home_state_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or len(value) > 100:
            raise ValueError("home_state must contain 1 to 100 characters")
        return value

    @field_validator("pref_branch_keywords")
    @classmethod
    def validate_branch_keywords(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if len(value) > 20:
            raise ValueError("At most 20 branch keywords are allowed")
        cleaned = [keyword.strip() for keyword in value]
        if any(not keyword or len(keyword) > 100 for keyword in cleaned):
            raise ValueError("Branch keywords must contain 1 to 100 characters")
        return cleaned


class ChatContextRequest(BaseModel):
    """Input for POST /chat-context."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="The user's question to the chatbot.",
    )

    @field_validator("question")
    @classmethod
    def validate_question_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    recommendation_output: dict | None = Field(
        default=None,
        description=(
            "The full JSON response from POST /predict or /recommend. "
            "Used to generate context the chatbot can reference."
        ),
    )
