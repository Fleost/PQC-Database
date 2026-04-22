"""
Unit tests for workload-level policy aggregation in app/benchmark.py.

All tests use lightweight synthetic fixtures — no DB, no crypto.
"""

from __future__ import annotations

import pytest

from app.benchmark import (
    BenchmarkReport,
    CellResult,
    EnvelopeMetrics,
    PolicyRunSummary,
    PolicyTierSummary,
    Stats,
    _weighted_mean,
    report_to_dict,
    summarize_policy_run,
    format_report,
)
from app.policy import DEFAULT_THRESHOLD_BYTES


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _stats(mean: float, p50: float | None = None, p95: float | None = None) -> Stats:
    p50 = p50 if p50 is not None else mean
    p95 = p95 if p95 is not None else mean * 1.2
    return Stats(
        mean=mean,
        minimum=mean * 0.8,
        maximum=mean * 1.5,
        p50=p50,
        p95=p95,
        p99=mean * 1.6,
    )


def _envelope(plaintext_bytes: int, extra_overhead: int = 100) -> EnvelopeMetrics:
    total_stored = plaintext_bytes + extra_overhead
    return EnvelopeMetrics(
        plaintext_bytes=plaintext_bytes,
        ciphertext_bytes=plaintext_bytes,
        nonce_bytes=12,
        tag_bytes=16,
        wrapped_dek_bytes=40,
        eph_pubkey_bytes=32,
        salt_bytes=32,
        hkdf_info_bytes=19,
        aad_bytes=30,
        pq_ct_bytes=1088,
        signature_bytes=extra_overhead - 100 if extra_overhead > 100 else 0,
        header_bytes=181,
        overhead_bytes=extra_overhead,
        total_stored_bytes=total_stored,
    )


def _cell(
    size_bytes: int,
    tier_label: str,
    escalated: bool,
    iterations: int = 10,
    scheme: str = "hybrid",
    policy_mode: str = "adaptive_threshold",
    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
    envelope_extra_overhead: int = 100,
    end_to_end_mean: float = 10.0,
    end_to_end_p50: float | None = None,
    end_to_end_p95: float | None = None,
    encrypt_mean: float = 2.0,
    sign_mean: float = 1.0,
    verify_mean: float = 0.8,
    db_insert_mean: float = 3.0,
    db_fetch_mean: float = 2.0,
    decrypt_mean: float = 1.5,
) -> CellResult:
    sign_stats = _stats(sign_mean) if sign_mean > 0 else None
    verify_stats = _stats(verify_mean) if verify_mean > 0 else None
    return CellResult(
        scheme=scheme,
        size_bytes=size_bytes,
        size_label=f"{size_bytes} B",
        payload_name=f"payload_{size_bytes}.bin",
        iterations=iterations,
        pq_sig_id="ML-DSA-65" if tier_label == "baseline" else "ML-DSA-87",
        encrypt=_stats(encrypt_mean),
        sign=sign_stats,
        db_insert=_stats(db_insert_mean),
        db_fetch=_stats(db_fetch_mean),
        verify=verify_stats,
        decrypt=_stats(decrypt_mean),
        end_to_end=_stats(end_to_end_mean, end_to_end_p50, end_to_end_p95),
        envelope=_envelope(size_bytes, envelope_extra_overhead),
        policy_mode=policy_mode,
        threshold_bytes=threshold_bytes,
        selected_kem_id="ML-KEM-768" if tier_label == "baseline" else "ML-KEM-1024",
        selected_sig_id="ML-DSA-65" if tier_label == "baseline" else "ML-DSA-87",
        tier_label=tier_label,
        escalated=escalated,
    )


def _report(cells: list[CellResult], policy_mode: str = "adaptive_threshold") -> BenchmarkReport:
    return BenchmarkReport(
        run_at="2026-01-01T00:00:00+00:00",
        host="test-host",
        iterations=10,
        warmup_iterations=2,
        payload_source="synthetic",
        payload_selection="synthetic_defaults",
        sizes_bytes=sorted(set(c.size_bytes for c in cells)),
        schemes=["hybrid"],
        pq_kem_id="adaptive" if policy_mode == "adaptive_threshold" else "ML-KEM-768",
        pq_sig_id=None,
        policy_mode=policy_mode,
        policy_threshold_bytes=150,   # threshold between 100B and 200B in fixtures
        results=cells,
    )


# Two-cell adaptive fixture: 100B baseline, 200B strong
# Custom threshold of 150B — set on the report so summarize_policy_run reads it.
@pytest.fixture
def two_cell_adaptive() -> BenchmarkReport:
    cells = [
        _cell(size_bytes=100, tier_label="baseline", escalated=False, iterations=10,
              policy_mode="adaptive_threshold", threshold_bytes=150,
              envelope_extra_overhead=10),   # stored = 110B each
        _cell(size_bytes=200, tier_label="strong",   escalated=True,  iterations=10,
              policy_mode="adaptive_threshold", threshold_bytes=150,
              envelope_extra_overhead=20),   # stored = 220B each
    ]
    return _report(cells, policy_mode="adaptive_threshold")


@pytest.fixture
def uniform_baseline_report() -> BenchmarkReport:
    cells = [
        _cell(size_bytes=100, tier_label="baseline", escalated=False, iterations=10,
              policy_mode="uniform_baseline", threshold_bytes=DEFAULT_THRESHOLD_BYTES),
        _cell(size_bytes=200, tier_label="baseline", escalated=False, iterations=10,
              policy_mode="uniform_baseline", threshold_bytes=DEFAULT_THRESHOLD_BYTES),
    ]
    return _report(cells, policy_mode="uniform_baseline")


@pytest.fixture
def uniform_strong_report() -> BenchmarkReport:
    cells = [
        _cell(size_bytes=100, tier_label="strong", escalated=True, iterations=10,
              policy_mode="uniform_strong", threshold_bytes=DEFAULT_THRESHOLD_BYTES),
        _cell(size_bytes=200, tier_label="strong", escalated=True, iterations=10,
              policy_mode="uniform_strong", threshold_bytes=DEFAULT_THRESHOLD_BYTES),
    ]
    return _report(cells, policy_mode="uniform_strong")


# ---------------------------------------------------------------------------
# _weighted_mean
# ---------------------------------------------------------------------------


class TestWeightedMean:
    def test_equal_weights(self):
        assert _weighted_mean([1.0, 3.0], [1, 1]) == pytest.approx(2.0)

    def test_unequal_weights(self):
        # 10 * 1.0 + 10 * 3.0 = 40 / 20 = 2.0
        assert _weighted_mean([1.0, 3.0], [10, 10]) == pytest.approx(2.0)

    def test_single_value(self):
        assert _weighted_mean([5.5], [7]) == pytest.approx(5.5)

    def test_zero_weight_returns_zero(self):
        assert _weighted_mean([1.0, 2.0], [0, 0]) == pytest.approx(0.0)

    def test_skewed_weights(self):
        # weight 1 on value 0, weight 9 on value 10 → mean = 9.0
        assert _weighted_mean([0.0, 10.0], [1, 9]) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# summarize_policy_run — totals
# ---------------------------------------------------------------------------


class TestSummarizeTotals:
    def test_total_records(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # 10 iterations × 2 cells = 20 total records
        assert s.total_records == 20

    def test_total_plaintext_bytes(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # 10 × 100B + 10 × 200B = 3000B
        assert s.total_plaintext_bytes == 3000

    def test_total_stored_bytes(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # 10 × 110B + 10 × 220B = 3300B
        assert s.total_stored_bytes == 3300

    def test_overall_storage_amplification(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # 3300 / 3000 = 1.1
        assert s.overall_storage_amplification == pytest.approx(1.1)

    def test_returns_policy_run_summary(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        assert isinstance(s, PolicyRunSummary)


# ---------------------------------------------------------------------------
# summarize_policy_run — escalation coverage
# ---------------------------------------------------------------------------


class TestEscalationCoverage:
    def test_escalated_record_count(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # only strong cell (200B, 10 iterations) is escalated
        assert s.escalated_records == 10

    def test_escalated_record_fraction(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # 10 / 20 = 0.5
        assert s.escalated_record_fraction == pytest.approx(0.5)

    def test_escalated_plaintext_bytes(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # 10 × 200B = 2000B
        assert s.escalated_plaintext_bytes == 2000

    def test_escalated_plaintext_byte_fraction(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        # 2000 / 3000 ≈ 0.6667
        assert s.escalated_plaintext_byte_fraction == pytest.approx(2000 / 3000)


# ---------------------------------------------------------------------------
# summarize_policy_run — uniform_baseline (zero escalation)
# ---------------------------------------------------------------------------


class TestUniformBaselineSummary:
    def test_zero_escalated_records(self, uniform_baseline_report):
        s = summarize_policy_run(uniform_baseline_report)
        assert s.escalated_records == 0

    def test_zero_escalated_fraction(self, uniform_baseline_report):
        s = summarize_policy_run(uniform_baseline_report)
        assert s.escalated_record_fraction == pytest.approx(0.0)

    def test_zero_escalated_plaintext_bytes(self, uniform_baseline_report):
        s = summarize_policy_run(uniform_baseline_report)
        assert s.escalated_plaintext_bytes == 0

    def test_single_tier_baseline(self, uniform_baseline_report):
        s = summarize_policy_run(uniform_baseline_report)
        assert len(s.tier_summaries) == 1
        assert s.tier_summaries[0].tier_label == "baseline"

    def test_policy_mode(self, uniform_baseline_report):
        s = summarize_policy_run(uniform_baseline_report)
        assert s.policy_mode == "uniform_baseline"


# ---------------------------------------------------------------------------
# summarize_policy_run — uniform_strong (all escalated)
# ---------------------------------------------------------------------------


class TestUniformStrongSummary:
    def test_all_records_escalated(self, uniform_strong_report):
        s = summarize_policy_run(uniform_strong_report)
        assert s.escalated_records == s.total_records

    def test_escalated_fraction_is_one(self, uniform_strong_report):
        s = summarize_policy_run(uniform_strong_report)
        assert s.escalated_record_fraction == pytest.approx(1.0)

    def test_single_tier_strong(self, uniform_strong_report):
        s = summarize_policy_run(uniform_strong_report)
        assert len(s.tier_summaries) == 1
        assert s.tier_summaries[0].tier_label == "strong"

    def test_policy_mode(self, uniform_strong_report):
        s = summarize_policy_run(uniform_strong_report)
        assert s.policy_mode == "uniform_strong"


# ---------------------------------------------------------------------------
# Per-tier summaries — adaptive fixture
# ---------------------------------------------------------------------------


class TestTierSummaries:
    def test_two_tiers_present(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        labels = [t.tier_label for t in s.tier_summaries]
        assert "baseline" in labels
        assert "strong" in labels

    def test_baseline_tier_order(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        assert s.tier_summaries[0].tier_label == "baseline"
        assert s.tier_summaries[1].tier_label == "strong"

    def test_baseline_record_count(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        baseline = next(t for t in s.tier_summaries if t.tier_label == "baseline")
        assert baseline.record_count == 10

    def test_strong_record_count(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        strong = next(t for t in s.tier_summaries if t.tier_label == "strong")
        assert strong.record_count == 10

    def test_baseline_record_fraction(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        baseline = next(t for t in s.tier_summaries if t.tier_label == "baseline")
        assert baseline.record_fraction == pytest.approx(0.5)

    def test_baseline_plaintext_bytes(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        baseline = next(t for t in s.tier_summaries if t.tier_label == "baseline")
        assert baseline.plaintext_bytes == 10 * 100  # 1000B

    def test_strong_plaintext_bytes(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        strong = next(t for t in s.tier_summaries if t.tier_label == "strong")
        assert strong.plaintext_bytes == 10 * 200  # 2000B

    def test_baseline_total_stored_bytes(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        baseline = next(t for t in s.tier_summaries if t.tier_label == "baseline")
        assert baseline.total_stored_bytes == 10 * 110  # 1100B

    def test_strong_total_stored_bytes(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        strong = next(t for t in s.tier_summaries if t.tier_label == "strong")
        assert strong.total_stored_bytes == 10 * 220  # 2200B

    def test_tier_storage_amplification(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        for ts in s.tier_summaries:
            assert ts.storage_amplification == pytest.approx(1.1)

    def test_tier_plaintext_byte_fractions_sum_to_one(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        total = sum(t.plaintext_byte_fraction for t in s.tier_summaries)
        assert total == pytest.approx(1.0)

    def test_tier_record_fractions_sum_to_one(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        total = sum(t.record_fraction for t in s.tier_summaries)
        assert total == pytest.approx(1.0)

    def test_tier_summaries_are_policy_tier_summary_instances(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        for ts in s.tier_summaries:
            assert isinstance(ts, PolicyTierSummary)


# ---------------------------------------------------------------------------
# Per-tier latency aggregation
# ---------------------------------------------------------------------------


class TestTierLatency:
    def test_baseline_mean_end_to_end(self, two_cell_adaptive):
        # The baseline cell has end_to_end_mean=10.0 (default from _cell fixture)
        s = summarize_policy_run(two_cell_adaptive)
        baseline = next(t for t in s.tier_summaries if t.tier_label == "baseline")
        assert baseline.mean_end_to_end_ms == pytest.approx(10.0)

    def test_strong_mean_end_to_end(self, two_cell_adaptive):
        s = summarize_policy_run(two_cell_adaptive)
        strong = next(t for t in s.tier_summaries if t.tier_label == "strong")
        assert strong.mean_end_to_end_ms == pytest.approx(10.0)

    def test_overall_mean_end_to_end_weighted(self):
        # Use different latencies to verify weighting.
        cells = [
            _cell(100, "baseline", False, iterations=10, end_to_end_mean=5.0),
            _cell(200, "strong",   True,  iterations=10, end_to_end_mean=15.0),
        ]
        report = _report(cells)
        s = summarize_policy_run(report)
        # 10 * 5 + 10 * 15 = 200 / 20 = 10.0
        assert s.mean_end_to_end_ms == pytest.approx(10.0)

    def test_unequal_iteration_weighting(self):
        # baseline: 5 iterations × mean=4ms; strong: 15 iterations × mean=12ms
        # weighted mean = (5*4 + 15*12) / 20 = (20 + 180) / 20 = 10.0
        cells = [
            _cell(100, "baseline", False, iterations=5,  end_to_end_mean=4.0),
            _cell(200, "strong",   True,  iterations=15, end_to_end_mean=12.0),
        ]
        report = _report(cells)
        s = summarize_policy_run(report)
        assert s.mean_end_to_end_ms == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Empty report
# ---------------------------------------------------------------------------


class TestEmptyReport:
    def test_empty_report_returns_zeros(self):
        report = _report([], policy_mode="adaptive_threshold")
        s = summarize_policy_run(report)
        assert s.total_records == 0
        assert s.escalated_records == 0
        assert s.escalated_record_fraction == pytest.approx(0.0)
        assert s.overall_storage_amplification == pytest.approx(0.0)
        assert s.tier_summaries == []


# ---------------------------------------------------------------------------
# JSON output includes policy_summary
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_policy_summary_key_present(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        assert "policy_summary" in d

    def test_policy_summary_has_tier_summaries(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        ps = d["policy_summary"]
        assert "tier_summaries" in ps
        assert len(ps["tier_summaries"]) == 2

    def test_policy_summary_escalated_record_fraction(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        ps = d["policy_summary"]
        assert ps["escalated_record_fraction"] == pytest.approx(0.5, rel=1e-4)

    def test_policy_summary_escalated_plaintext_byte_fraction(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        ps = d["policy_summary"]
        assert ps["escalated_plaintext_byte_fraction"] == pytest.approx(2000 / 3000, rel=1e-4)

    def test_policy_summary_total_records(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        assert d["policy_summary"]["total_records"] == 20

    def test_policy_summary_overall_amplification(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        assert d["policy_summary"]["overall_storage_amplification"] == pytest.approx(1.1, rel=1e-4)

    def test_tier_summary_fields_present(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        tier = d["policy_summary"]["tier_summaries"][0]
        required_keys = {
            "tier_label", "record_count", "record_fraction",
            "plaintext_bytes", "plaintext_byte_fraction",
            "total_stored_bytes", "storage_amplification",
            "mean_end_to_end_ms", "p50_end_to_end_ms", "p95_end_to_end_ms",
            "mean_encrypt_ms", "mean_sign_ms", "mean_verify_ms",
            "mean_db_insert_ms", "mean_db_fetch_ms", "mean_decrypt_ms",
        }
        assert required_keys.issubset(tier.keys())

    def test_per_cell_results_preserved(self, two_cell_adaptive):
        d = report_to_dict(two_cell_adaptive)
        assert len(d["results"]) == 2
        assert all("tier_label" in r for r in d["results"])

    def test_zero_escalation_json(self, uniform_baseline_report):
        d = report_to_dict(uniform_baseline_report)
        assert d["policy_summary"]["escalated_records"] == 0
        assert d["policy_summary"]["escalated_record_fraction"] == 0.0

    def test_full_escalation_json(self, uniform_strong_report):
        d = report_to_dict(uniform_strong_report)
        assert d["policy_summary"]["escalated_record_fraction"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Text report includes policy summary section
# ---------------------------------------------------------------------------


class TestTextReport:
    def test_policy_tier_summary_section_present(self, two_cell_adaptive):
        text = format_report(two_cell_adaptive)
        assert "POLICY TIER SUMMARY" in text

    def test_policy_mode_shown(self, two_cell_adaptive):
        text = format_report(two_cell_adaptive)
        assert "adaptive_threshold" in text

    def test_escalated_count_shown(self, two_cell_adaptive):
        text = format_report(two_cell_adaptive)
        # 10/20 escalated
        assert "10 / 20" in text

    def test_tier_labels_in_report(self, two_cell_adaptive):
        text = format_report(two_cell_adaptive)
        assert "baseline" in text
        assert "strong" in text

    def test_no_policy_section_for_legacy(self):
        cells = [
            _cell(100, "n/a", False, policy_mode="n/a"),
        ]
        report = BenchmarkReport(
            run_at="2026-01-01T00:00:00+00:00",
            host="test-host",
            iterations=10,
            warmup_iterations=2,
            payload_source="synthetic",
            payload_selection="synthetic_defaults",
            sizes_bytes=[100],
            schemes=["hybrid"],
            pq_kem_id="ML-KEM-768",
            pq_sig_id=None,
            policy_mode="n/a",
            policy_threshold_bytes=DEFAULT_THRESHOLD_BYTES,
            results=cells,
        )
        text = format_report(report)
        assert "POLICY TIER SUMMARY" not in text


# ---------------------------------------------------------------------------
# extract_policy_chart_data (lightweight)
# ---------------------------------------------------------------------------


class TestExtractPolicyChartData:
    def test_returns_expected_keys(self, two_cell_adaptive):
        from visualize import extract_policy_chart_data
        d = report_to_dict(two_cell_adaptive)
        cd = extract_policy_chart_data(d)
        assert "tier_labels" in cd
        assert "record_fractions" in cd
        assert "plaintext_byte_fractions" in cd
        assert "storage_amplifications" in cd
        assert "mean_e2e_ms" in cd
        assert "component_ms" in cd
        assert "escalated_record_fraction" in cd
        assert "escalated_plaintext_byte_fraction" in cd

    def test_tier_labels(self, two_cell_adaptive):
        from visualize import extract_policy_chart_data
        d = report_to_dict(two_cell_adaptive)
        cd = extract_policy_chart_data(d)
        assert cd["tier_labels"] == ["baseline", "strong"]

    def test_escalated_fraction(self, two_cell_adaptive):
        from visualize import extract_policy_chart_data
        d = report_to_dict(two_cell_adaptive)
        cd = extract_policy_chart_data(d)
        assert cd["escalated_record_fraction"] == pytest.approx(0.5, rel=1e-4)

    def test_component_ms_has_all_components(self, two_cell_adaptive):
        from visualize import extract_policy_chart_data
        d = report_to_dict(two_cell_adaptive)
        cd = extract_policy_chart_data(d)
        for comp in ["encrypt", "sign", "db_insert", "db_fetch", "verify", "decrypt"]:
            assert comp in cd["component_ms"]
