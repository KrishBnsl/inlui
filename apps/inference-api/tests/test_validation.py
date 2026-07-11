"""
Tests for input validation utilities.

These tests are pure unit tests — no ML artifacts or filesystem access needed.

Run with:
    cd apps/inference-api && pytest tests/test_validation.py -v
"""

import pytest
from fastapi import HTTPException

from inference_api.utils.validation import (
    validate_advanced_rank,
    validate_category,
    validate_gender,
    validate_inst_types,
    validate_main_rank,
    validate_predict_request,
    validate_quotas,
)


# ── validate_main_rank ──────────────────────────────────────────────────────────

class TestValidateMainRank:
    def test_valid_rank(self):
        validate_main_rank(15000)  # Should not raise

    def test_valid_rank_boundary_low(self):
        validate_main_rank(1)

    def test_valid_rank_boundary_high(self):
        validate_main_rank(1_200_000)

    def test_none_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_main_rank(None)
        assert exc_info.value.status_code == 422

    def test_zero_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_main_rank(0)
        assert exc_info.value.status_code == 422

    def test_negative_raises(self):
        with pytest.raises(HTTPException):
            validate_main_rank(-500)

    def test_too_high_raises(self):
        with pytest.raises(HTTPException):
            validate_main_rank(2_000_000)

    def test_float_raises(self):
        with pytest.raises((HTTPException, TypeError)):
            validate_main_rank(1500.5)  # type: ignore

    def test_bool_raises(self):
        with pytest.raises(HTTPException):
            validate_main_rank(True)


# ── validate_advanced_rank ──────────────────────────────────────────────────────

class TestValidateAdvancedRank:
    def test_none_is_ok(self):
        validate_advanced_rank(None)  # Optional field, should not raise

    def test_valid_advanced_rank(self):
        validate_advanced_rank(5000)

    def test_zero_raises(self):
        with pytest.raises(HTTPException):
            validate_advanced_rank(0)

    def test_too_high_raises(self):
        with pytest.raises(HTTPException):
            validate_advanced_rank(100_000)


# ── validate_category ───────────────────────────────────────────────────────────

class TestValidateCategory:
    @pytest.mark.parametrize("cat", ["OPEN", "OBC-NCL", "EWS", "SC", "ST"])
    def test_valid_categories(self, cat):
        validate_category(cat)

    def test_invalid_category_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_category("GENERAL")
        assert exc_info.value.status_code == 422
        assert "GENERAL" in exc_info.value.detail

    def test_lowercase_raises(self):
        with pytest.raises(HTTPException):
            validate_category("open")

    def test_empty_string_raises(self):
        with pytest.raises(HTTPException):
            validate_category("")


# ── validate_gender ─────────────────────────────────────────────────────────────

class TestValidateGender:
    def test_gender_neutral(self):
        validate_gender("Gender-Neutral")

    def test_female_only(self):
        validate_gender("Female-only")

    def test_invalid_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_gender("Male")
        assert exc_info.value.status_code == 422


# ── validate_inst_types ─────────────────────────────────────────────────────────

class TestValidateInstTypes:
    def test_none_is_ok(self):
        validate_inst_types(None)

    def test_empty_list_is_ok(self):
        validate_inst_types([])

    def test_valid_types(self):
        validate_inst_types(["NIT", "IIT", "IIIT", "GFTI"])

    def test_invalid_type_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_inst_types(["NIT", "CFTI"])
        assert "CFTI" in exc_info.value.detail


class TestValidateQuotas:
    @pytest.mark.parametrize("quota", ["AI", "HS", "OS", "STATE"])
    def test_valid_quota(self, quota):
        validate_quotas([quota])

    def test_invalid_quota_raises(self):
        with pytest.raises(HTTPException, match="quota"):
            validate_quotas(["LOCAL"])


# ── validate_predict_request ────────────────────────────────────────────────────

class TestValidatePredictRequest:
    """Integration test — validates a duck-typed object."""

    class _MockRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _make(self, **overrides):
        defaults = dict(
            main_rank=15000,
            advanced_rank=None,
            category="OBC-NCL",
            gender="Gender-Neutral",
            pref_inst_types=None,
            pref_quotas=None,
        )
        defaults.update(overrides)
        return self._MockRequest(**defaults)

    def test_valid_request(self):
        validate_predict_request(self._make())  # Should not raise

    def test_bad_rank_fails(self):
        with pytest.raises(HTTPException):
            validate_predict_request(self._make(main_rank=0))

    def test_bad_category_fails(self):
        with pytest.raises(HTTPException):
            validate_predict_request(self._make(category="INVALID"))

    def test_bad_gender_fails(self):
        with pytest.raises(HTTPException):
            validate_predict_request(self._make(gender="Non-binary"))

    def test_bad_inst_type_fails(self):
        with pytest.raises(HTTPException):
            validate_predict_request(self._make(pref_inst_types=["PRIVATE"]))

    def test_bad_quota_fails(self):
        with pytest.raises(HTTPException):
            validate_predict_request(self._make(pref_quotas=["PRIVATE"]))
