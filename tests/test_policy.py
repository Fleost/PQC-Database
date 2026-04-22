"""Unit tests for app.policy — workload-aware PQ parameter selection."""

import pytest

from app.policy import (
    BASELINE_KEM_ID,
    BASELINE_SIG_ID,
    DEFAULT_THRESHOLD_BYTES,
    STRONG_KEM_ID,
    STRONG_SIG_ID,
    PolicyDecision,
    ThresholdPolicyConfig,
    format_threshold_label,
    is_strong_assignment,
    select_policy_for_record,
)

_50_KIB = 50 * 1024   # 51200  — current default threshold
_10_MIB = 10 * 1024 * 1024  # 10485760 — used only in custom-threshold / format tests


# ---------------------------------------------------------------------------
# Below threshold → baseline
# ---------------------------------------------------------------------------


class TestBelowThreshold:
    def test_kem_id(self):
        decision = select_policy_for_record(_50_KIB - 1)
        assert decision.kem_id == "ML-KEM-768"

    def test_sig_id(self):
        decision = select_policy_for_record(_50_KIB - 1)
        assert decision.sig_id == "ML-DSA-65"

    def test_tier_label(self):
        decision = select_policy_for_record(_50_KIB - 1)
        assert decision.tier_label == "baseline"

    def test_escalated_false(self):
        decision = select_policy_for_record(_50_KIB - 1)
        assert decision.escalated is False

    def test_zero_bytes(self):
        decision = select_policy_for_record(0)
        assert decision.tier_label == "baseline"
        assert decision.escalated is False


# ---------------------------------------------------------------------------
# Equal to threshold → baseline (strictly greater-than rule)
# ---------------------------------------------------------------------------


class TestAtThreshold:
    def test_exact_threshold_is_baseline(self):
        decision = select_policy_for_record(_50_KIB)
        assert decision.tier_label == "baseline"
        assert decision.kem_id == "ML-KEM-768"
        assert decision.sig_id == "ML-DSA-65"
        assert decision.escalated is False

    def test_rule_is_strictly_greater_than(self):
        """Confirm the boundary: threshold itself → baseline, threshold+1 → strong."""
        at = select_policy_for_record(_50_KIB)
        above = select_policy_for_record(_50_KIB + 1)
        assert at.escalated is False
        assert above.escalated is True


# ---------------------------------------------------------------------------
# Above threshold → strong
# ---------------------------------------------------------------------------


class TestAboveThreshold:
    def test_kem_id(self):
        decision = select_policy_for_record(_50_KIB + 1)
        assert decision.kem_id == "ML-KEM-1024"

    def test_sig_id(self):
        decision = select_policy_for_record(_50_KIB + 1)
        assert decision.sig_id == "ML-DSA-87"

    def test_tier_label(self):
        decision = select_policy_for_record(_50_KIB + 1)
        assert decision.tier_label == "strong"

    def test_escalated_true(self):
        decision = select_policy_for_record(_50_KIB + 1)
        assert decision.escalated is True

    def test_large_record(self):
        decision = select_policy_for_record(100 * 1024 * 1024)
        assert decision.tier_label == "strong"


# ---------------------------------------------------------------------------
# Default threshold
# ---------------------------------------------------------------------------


class TestDefaultThreshold:
    def test_default_threshold_is_50_kib(self):
        assert DEFAULT_THRESHOLD_BYTES == _50_KIB

    def test_no_config_uses_default(self):
        decision = select_policy_for_record(_50_KIB + 1)
        assert decision.threshold_bytes == _50_KIB

    def test_default_config_threshold_matches_constant(self):
        config = ThresholdPolicyConfig()
        assert config.threshold_bytes == DEFAULT_THRESHOLD_BYTES


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    def test_custom_lower_threshold_escalates_smaller_record(self):
        config = ThresholdPolicyConfig(threshold_bytes=1024)
        decision = select_policy_for_record(2048, config)
        assert decision.escalated is True
        assert decision.kem_id == "ML-KEM-1024"

    def test_custom_lower_threshold_baseline_for_equal(self):
        config = ThresholdPolicyConfig(threshold_bytes=1024)
        decision = select_policy_for_record(1024, config)
        assert decision.escalated is False

    def test_custom_higher_threshold_keeps_baseline(self):
        config = ThresholdPolicyConfig(threshold_bytes=50 * 1024 * 1024)
        decision = select_policy_for_record(_10_MIB + 1, config)
        assert decision.escalated is False
        assert decision.tier_label == "baseline"

    def test_zero_threshold_always_strong_for_nonzero(self):
        config = ThresholdPolicyConfig(threshold_bytes=0)
        decision = select_policy_for_record(1, config)
        assert decision.escalated is True

    def test_zero_threshold_zero_size_is_baseline(self):
        config = ThresholdPolicyConfig(threshold_bytes=0)
        decision = select_policy_for_record(0, config)
        assert decision.escalated is False


# ---------------------------------------------------------------------------
# Decision fields are correct
# ---------------------------------------------------------------------------


class TestDecisionFields:
    def test_record_size_reflected(self):
        size = 42_000
        decision = select_policy_for_record(size)
        assert decision.record_size_bytes == size

    def test_threshold_reflected(self):
        config = ThresholdPolicyConfig(threshold_bytes=999)
        decision = select_policy_for_record(1, config)
        assert decision.threshold_bytes == 999

    def test_returns_policy_decision_instance(self):
        decision = select_policy_for_record(0)
        assert isinstance(decision, PolicyDecision)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


class TestInvalidInputs:
    def test_negative_record_size_raises_value_error(self):
        with pytest.raises(ValueError, match="record_size_bytes must be >= 0"):
            select_policy_for_record(-1)

    def test_negative_threshold_raises_value_error(self):
        config = ThresholdPolicyConfig(threshold_bytes=-1)
        with pytest.raises(ValueError, match="threshold_bytes must be >= 0"):
            select_policy_for_record(0, config)

    def test_float_record_size_raises_type_error(self):
        with pytest.raises(TypeError, match="record_size_bytes must be an int"):
            select_policy_for_record(1.5)  # type: ignore[arg-type]

    def test_string_record_size_raises_type_error(self):
        with pytest.raises(TypeError, match="record_size_bytes must be an int"):
            select_policy_for_record("large")  # type: ignore[arg-type]

    def test_none_record_size_raises_type_error(self):
        with pytest.raises(TypeError, match="record_size_bytes must be an int"):
            select_policy_for_record(None)  # type: ignore[arg-type]

    def test_float_threshold_raises_type_error(self):
        config = ThresholdPolicyConfig(threshold_bytes=1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="threshold_bytes must be an int"):
            select_policy_for_record(0, config)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


class TestIsStrongAssignment:
    def test_below_threshold(self):
        assert is_strong_assignment(_50_KIB - 1) is False

    def test_at_threshold(self):
        assert is_strong_assignment(_50_KIB) is False

    def test_above_threshold(self):
        assert is_strong_assignment(_50_KIB + 1) is True

    def test_with_custom_config(self):
        config = ThresholdPolicyConfig(threshold_bytes=100)
        assert is_strong_assignment(101, config) is True
        assert is_strong_assignment(100, config) is False


class TestFormatThresholdLabel:
    def test_10_mib(self):
        assert format_threshold_label(_10_MIB) == "10.0 MiB"

    def test_1_mib(self):
        assert format_threshold_label(1024 * 1024) == "1.0 MiB"

    def test_kib(self):
        assert format_threshold_label(2048) == "2.0 KiB"

    def test_bytes(self):
        assert format_threshold_label(512) == "512 B"

    def test_zero(self):
        assert format_threshold_label(0) == "0 B"

    def test_non_int_raises_type_error(self):
        with pytest.raises(TypeError):
            format_threshold_label("10mb")  # type: ignore[arg-type]

    def test_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            format_threshold_label(-1)


# ---------------------------------------------------------------------------
# Constants are aligned with the crypto layer
# ---------------------------------------------------------------------------


class TestConstants:
    def test_baseline_kem(self):
        assert BASELINE_KEM_ID == "ML-KEM-768"

    def test_baseline_sig(self):
        assert BASELINE_SIG_ID == "ML-DSA-65"

    def test_strong_kem(self):
        assert STRONG_KEM_ID == "ML-KEM-1024"

    def test_strong_sig(self):
        assert STRONG_SIG_ID == "ML-DSA-87"
