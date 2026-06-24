"""
Input validation utilities for the inference service.

All public functions raise `fastapi.HTTPException` with a 422 status and a
clear error message so the frontend can surface the problem to the user.
"""

from fastapi import HTTPException

# Valid categorical values (must match label_encoders.json exactly)
VALID_CATEGORIES = {"EWS", "OBC-NCL", "OPEN", "SC", "ST"}
VALID_GENDERS = {"Female-only", "Gender-Neutral"}
VALID_INST_TYPES = {"IIT", "NIT", "IIIT", "GFTI"}
VALID_QUOTAS = {"AI", "HS", "OS", "STATE"}

# JEE rank boundaries
MAIN_RANK_MIN = 1
MAIN_RANK_MAX = 1_200_000   # JEE Main CRL max
ADVANCED_RANK_MIN = 1
ADVANCED_RANK_MAX = 50_000  # JEE Advanced max (approx. seats × buffer)


def validate_main_rank(rank: int | None) -> None:
    """Raise 422 if main_rank is outside [1, 1_200_000]."""
    if rank is None:
        raise HTTPException(
            status_code=422,
            detail="main_rank is required and must be a positive integer.",
        )
    if not isinstance(rank, int) or rank < MAIN_RANK_MIN or rank > MAIN_RANK_MAX:
        raise HTTPException(
            status_code=422,
            detail=(
                f"main_rank must be an integer between {MAIN_RANK_MIN:,} "
                f"and {MAIN_RANK_MAX:,}, got {rank!r}."
            ),
        )


def validate_advanced_rank(rank: int | None) -> None:
    """Raise 422 if advanced_rank is present but outside [1, 50_000]."""
    if rank is None:
        return  # optional field
    if not isinstance(rank, int) or rank < ADVANCED_RANK_MIN or rank > ADVANCED_RANK_MAX:
        raise HTTPException(
            status_code=422,
            detail=(
                f"advanced_rank must be an integer between {ADVANCED_RANK_MIN:,} "
                f"and {ADVANCED_RANK_MAX:,}, got {rank!r}."
            ),
        )


def validate_category(category: str) -> None:
    """Raise 422 if category is not one of the known JoSAA reservation categories."""
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid category {category!r}. "
                f"Must be one of: {sorted(VALID_CATEGORIES)}."
            ),
        )


def validate_gender(gender: str) -> None:
    """Raise 422 if gender is not a recognised JoSAA seat-pool label."""
    if gender not in VALID_GENDERS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid gender {gender!r}. "
                f"Must be one of: {sorted(VALID_GENDERS)}."
            ),
        )


def validate_inst_types(inst_types: list[str] | None) -> None:
    """Raise 422 if any preferred institute type is unrecognised."""
    if not inst_types:
        return  # optional
    invalid = [t for t in inst_types if t not in VALID_INST_TYPES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown institute type(s): {invalid}. "
                f"Valid values: {sorted(VALID_INST_TYPES)}."
            ),
        )


def validate_predict_request(request: object) -> None:
    """
    Run all field-level validations for a PredictRequest.

    Parameters
    ----------
    request : PredictRequest
        The incoming request object (uses duck-typing for testability).
    """
    validate_main_rank(getattr(request, "main_rank", None))
    validate_advanced_rank(getattr(request, "advanced_rank", None))
    validate_category(getattr(request, "category", ""))
    validate_gender(getattr(request, "gender", ""))
    validate_inst_types(getattr(request, "pref_inst_types", None))
