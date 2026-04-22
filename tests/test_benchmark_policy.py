"""
Unit tests for the benchmark policy integration in app/benchmark.py.

These tests cover the policy-resolution helper and the Sample/CellResult
policy metadata fields without requiring a live database connection.
"""

import pytest

from app.benchmark import (
    POLICY_MODES,
    Sample,
    _resolve_policy_decision,
)
from app.policy import (
    BASELINE_KEM_ID,
    BASELINE_SIG_ID,
    DEFAULT_THRESHOLD_BYTES,
    STRONG_KEM_ID,
    STRONG_SIG_ID,
    PolicyDecision,
)

_10_MIB = 10 * 1024 * 1024  # 10485760


# ---------------------------------------------------------------------------
# _resolve_policy_decision — adaptive_threshold
# ---------------------------------------------------------------------------


class TestAdaptiveThreshold:
    def test_small_record_selects_baseline_kem(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB - 1, _10_MIB)
        assert decision.kem_id == "ML-KEM-768"

    def test_small_record_selects_baseline_sig(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB - 1, _10_MIB)
        assert decision.sig_id == "ML-DSA-65"

    def test_small_record_tier_label(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB - 1, _10_MIB)
        assert decision.tier_label == "baseline"

    def test_small_record_not_escalated(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB - 1, _10_MIB)
        assert decision.escalated is False

    def test_large_record_selects_strong_kem(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB + 1, _10_MIB)
        assert decision.kem_id == "ML-KEM-1024"

    def test_large_record_selects_strong_sig(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB + 1, _10_MIB)
        assert decision.sig_id == "ML-DSA-87"

    def test_large_record_tier_label(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB + 1, _10_MIB)
        assert decision.tier_label == "strong"

    def test_large_record_escalated(self):
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB + 1, _10_MIB)
        assert decision.escalated is True

    def test_at_threshold_is_baseline(self):
        # Strictly-greater-than rule: equal to threshold → baseline.
        decision = _resolve_policy_decision("adaptive_threshold", _10_MIB, _10_MIB)
        assert decision.tier_label == "baseline"
        assert decision.escalated is False

    def test_threshold_reflected_in_decision(self):
        custom_threshold = 512 * 1024
        decision = _resolve_policy_decision("adaptive_threshold", 0, custom_threshold)
        assert decision.threshold_bytes == custom_threshold

    def test_custom_threshold_escalates_above(self):
        custom_threshold = 1024
        decision = _resolve_policy_decision("adaptive_threshold", 1025, custom_threshold)
        assert decision.escalated is True

    def test_custom_threshold_baseline_at(self):
        custom_threshold = 1024
        decision = _resolve_policy_decision("adaptive_threshold", 1024, custom_threshold)
        assert decision.escalated is False


# ---------------------------------------------------------------------------
# _resolve_policy_decision — uniform_baseline
# ---------------------------------------------------------------------------


class TestUniformBaseline:
    def test_always_baseline_kem(self):
        for size in [0, 1024, _10_MIB, _10_MIB + 1, 100 * 1024 * 1024]:
            d = _resolve_policy_decision("uniform_baseline", size, _10_MIB)
            assert d.kem_id == "ML-KEM-768", f"failed for size={size}"

    def test_always_baseline_sig(self):
        for size in [0, 1024, _10_MIB, _10_MIB + 1]:
            d = _resolve_policy_decision("uniform_baseline", size, _10_MIB)
            assert d.sig_id == "ML-DSA-65", f"failed for size={size}"

    def test_tier_label(self):
        d = _resolve_policy_decision("uniform_baseline", _10_MIB + 1, _10_MIB)
        assert d.tier_label == "baseline"

    def test_not_escalated(self):
        d = _resolve_policy_decision("uniform_baseline", _10_MIB + 1, _10_MIB)
        assert d.escalated is False

    def test_returns_policy_decision_instance(self):
        d = _resolve_policy_decision("uniform_baseline", 1024, _10_MIB)
        assert isinstance(d, PolicyDecision)


# ---------------------------------------------------------------------------
# _resolve_policy_decision — uniform_strong
# ---------------------------------------------------------------------------


class TestUniformStrong:
    def test_always_strong_kem(self):
        for size in [0, 1024, _10_MIB, _10_MIB + 1, 100 * 1024 * 1024]:
            d = _resolve_policy_decision("uniform_strong", size, _10_MIB)
            assert d.kem_id == "ML-KEM-1024", f"failed for size={size}"

    def test_always_strong_sig(self):
        for size in [0, 1024, _10_MIB, _10_MIB + 1]:
            d = _resolve_policy_decision("uniform_strong", size, _10_MIB)
            assert d.sig_id == "ML-DSA-87", f"failed for size={size}"

    def test_tier_label(self):
        d = _resolve_policy_decision("uniform_strong", 0, _10_MIB)
        assert d.tier_label == "strong"

    def test_escalated_true(self):
        d = _resolve_policy_decision("uniform_strong", 0, _10_MIB)
        assert d.escalated is True

    def test_returns_policy_decision_instance(self):
        d = _resolve_policy_decision("uniform_strong", 1024, _10_MIB)
        assert isinstance(d, PolicyDecision)


# ---------------------------------------------------------------------------
# _resolve_policy_decision — invalid mode
# ---------------------------------------------------------------------------


class TestInvalidPolicyMode:
    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown policy_mode"):
            _resolve_policy_decision("nonexistent_mode", 1024, _10_MIB)


# ---------------------------------------------------------------------------
# POLICY_MODES constant
# ---------------------------------------------------------------------------


class TestPolicyModesConstant:
    def test_contains_expected_modes(self):
        assert "uniform_baseline" in POLICY_MODES
        assert "uniform_strong" in POLICY_MODES
        assert "adaptive_threshold" in POLICY_MODES

    def test_is_complete(self):
        assert len(POLICY_MODES) == 3


# ---------------------------------------------------------------------------
# Sample dataclass — policy metadata fields
# ---------------------------------------------------------------------------


class TestSamplePolicyMetadata:
    def _make_sample(self, **overrides) -> Sample:
        defaults = dict(
            size_bytes=1024,
            scheme="hybrid",
            encrypt_ms=1.0,
            sign_ms=0.5,
            db_insert_ms=2.0,
            db_fetch_ms=1.5,
            verify_ms=0.4,
            decrypt_ms=0.8,
            end_to_end_ms=6.2,
        )
        defaults.update(overrides)
        return Sample(**defaults)

    def test_default_policy_fields(self):
        s = self._make_sample()
        assert s.selected_kem_id == ""
        assert s.selected_sig_id == ""
        assert s.tier_label == "n/a"
        assert s.escalated is False
        assert s.threshold_bytes == DEFAULT_THRESHOLD_BYTES
        assert s.policy_mode == "n/a"
        assert s.payload_name == ""

    def test_baseline_metadata(self):
        s = self._make_sample(
            selected_kem_id="ML-KEM-768",
            selected_sig_id="ML-DSA-65",
            tier_label="baseline",
            escalated=False,
            threshold_bytes=_10_MIB,
            policy_mode="uniform_baseline",
            payload_name="test_record.json",
        )
        assert s.selected_kem_id == "ML-KEM-768"
        assert s.selected_sig_id == "ML-DSA-65"
        assert s.tier_label == "baseline"
        assert s.escalated is False
        assert s.threshold_bytes == _10_MIB
        assert s.policy_mode == "uniform_baseline"
        assert s.payload_name == "test_record.json"

    def test_strong_metadata(self):
        s = self._make_sample(
            selected_kem_id="ML-KEM-1024",
            selected_sig_id="ML-DSA-87",
            tier_label="strong",
            escalated=True,
            threshold_bytes=_10_MIB,
            policy_mode="adaptive_threshold",
        )
        assert s.selected_kem_id == "ML-KEM-1024"
        assert s.selected_sig_id == "ML-DSA-87"
        assert s.tier_label == "strong"
        assert s.escalated is True

    def test_policy_mode_adaptive(self):
        s = self._make_sample(policy_mode="adaptive_threshold")
        assert s.policy_mode == "adaptive_threshold"


# ---------------------------------------------------------------------------
# Verify key selection correctness (unit-level, no DB)
# ---------------------------------------------------------------------------


class TestVerifyKeyAlignment:
    """
    Verify that the sig ID carried on a PolicyDecision matches the expected
    verify-key source for that profile.  This is a lightweight check that
    the mapping is consistent without exercising real crypto.
    """

    def test_baseline_decision_sig_id_matches_baseline_constant(self):
        d = _resolve_policy_decision("uniform_baseline", 1024, _10_MIB)
        assert d.sig_id == BASELINE_SIG_ID

    def test_strong_decision_sig_id_matches_strong_constant(self):
        d = _resolve_policy_decision("uniform_strong", 1024, _10_MIB)
        assert d.sig_id == STRONG_SIG_ID

    def test_adaptive_small_uses_baseline_sig(self):
        d = _resolve_policy_decision("adaptive_threshold", 100, _10_MIB)
        assert d.sig_id == BASELINE_SIG_ID

    def test_adaptive_large_uses_strong_sig(self):
        d = _resolve_policy_decision("adaptive_threshold", _10_MIB + 1, _10_MIB)
        assert d.sig_id == STRONG_SIG_ID

    def test_baseline_decision_kem_id_matches_baseline_constant(self):
        d = _resolve_policy_decision("uniform_baseline", 1024, _10_MIB)
        assert d.kem_id == BASELINE_KEM_ID

    def test_strong_decision_kem_id_matches_strong_constant(self):
        d = _resolve_policy_decision("uniform_strong", 1024, _10_MIB)
        assert d.kem_id == STRONG_KEM_ID


# ---------------------------------------------------------------------------
# Default threshold is 10 MiB
# ---------------------------------------------------------------------------


class TestDefaultThresholdInBenchmark:
    def test_adaptive_uses_10_mib_default(self):
        # When called with DEFAULT_THRESHOLD_BYTES, behaviour matches 10-MiB expectation.
        d_at = _resolve_policy_decision("adaptive_threshold", _10_MIB, DEFAULT_THRESHOLD_BYTES)
        assert d_at.tier_label == "baseline"   # at threshold → baseline

        d_above = _resolve_policy_decision("adaptive_threshold", _10_MIB + 1, DEFAULT_THRESHOLD_BYTES)
        assert d_above.tier_label == "strong"  # above threshold → strong

    def test_default_threshold_bytes_value(self):
        assert DEFAULT_THRESHOLD_BYTES == 10 * 1024 * 1024
