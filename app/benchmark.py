"""
app/benchmark.py
================
Performance benchmark for the PQC encrypted database system.

Measures encryption, DB insert, DB fetch, decryption, and end-to-end
latency for classical (X25519 + AES-256-GCM) and hybrid PQC
(X25519 + ML-KEM-{512,768,1024} + AES-256-GCM) encryption schemes
across record sizes ranging from 1 KB to 1 MB.

Run inside Docker:
    python -m app.benchmark

Run with options:
    python -m app.benchmark --iterations 20 --warmup 3 --output results.json

Policy modes (per-record adaptive PQ parameter assignment):
    python -m app.benchmark --policy-mode adaptive_threshold
    python -m app.benchmark --policy-mode uniform_baseline
    python -m app.benchmark --policy-mode uniform_strong --policy-threshold-bytes 5242880
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import db, stage2
from .service import ServiceContext, put_record, get_record
from .policy import (
    ThresholdPolicyConfig,
    PolicyDecision,
    select_policy_for_record,
    BASELINE_KEM_ID,
    BASELINE_SIG_ID,
    STRONG_KEM_ID,
    STRONG_SIG_ID,
    DEFAULT_THRESHOLD_BYTES,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Record sizes to benchmark (bytes)
DEFAULT_SIZES_BYTES: List[int] = [
    1 * 1024,         #   1 KB
    4 * 1024,         #   4 KB
    16 * 1024,        #  16 KB
    64 * 1024,        #  64 KB
    256 * 1024,       # 256 KB
    1 * 1024 * 1024,  #   1 MB
]

SCHEMES = ["classical", "hybrid"]
PAYLOAD_SOURCES = ["synthetic", "faers", "all_ggvp"]
POLICY_MODES = ["uniform_baseline", "uniform_strong", "adaptive_threshold"]


def _validate_schemes(schemes_str: str) -> List[str]:
    """Parse and validate a comma-separated schemes string.

    Returns a deduplicated list preserving canonical order (classical before hybrid).
    Raises ValueError for any unrecognised scheme token.
    """
    tokens = [t.strip() for t in schemes_str.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--schemes must not be empty")
    unknown = [t for t in tokens if t not in SCHEMES]
    if unknown:
        raise ValueError(
            f"Unknown scheme(s): {', '.join(unknown)!r}. "
            f"Must be one or more of: {', '.join(SCHEMES)}"
        )
    # Deduplicate while preserving canonical order.
    seen = set()
    ordered = []
    for s in SCHEMES:
        if s in tokens and s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered


TENANT_ID = "benchmark-tenant"
KEY_ID    = "benchmark-key-v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"


def _clear_benchmark_data(conn) -> None:
    """Delete all rows written by the benchmark tenant so each cell starts
    from a stable, near-empty table.  Runs before warmup, so the DELETE
    itself is never included in measured timings."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM encrypted_records WHERE tenant_id = %s", (TENANT_ID,))
        cur.execute("DELETE FROM audit_events    WHERE tenant_id = %s", (TENANT_ID,))
    conn.commit()


def _human_size(n: int) -> str:
    """Return a compact, human-readable byte size string."""
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)} MB"
    if n >= 1024:
        return f"{n // 1024} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EnvelopeMetrics:
    """Byte sizes of each stored field in one encrypted envelope.

    Values are deterministic for a given (scheme, plaintext_size) pair, so
    only one sample is kept per cell rather than aggregating across iterations.
    """
    plaintext_bytes:    int
    ciphertext_bytes:   int   # AES-GCM: equal to plaintext (no expansion)
    nonce_bytes:        int   # always 12
    tag_bytes:          int   # always 16
    wrapped_dek_bytes:  int   # always 40 (32-byte DEK + 8-byte AES-KW overhead)
    eph_pubkey_bytes:   int   # always 32  (X25519)
    salt_bytes:         int   # always 32
    hkdf_info_bytes:    int   # small constant (~19–21 B)
    aad_bytes:          int   # per-record AAD
    pq_ct_bytes:        int   # ML-KEM ciphertext (0 for classical)
    signature_bytes:    int   # ML-DSA signature (0 for unsigned)
    header_bytes:       int   # nonce+tag+dek+pk+salt+info+aad (fixed crypto metadata)
    overhead_bytes:     int   # header + pq_ct + signature (everything except the payload)
    total_stored_bytes: int   # ciphertext + overhead

    @staticmethod
    def from_dict(plaintext_size: int, d: Dict) -> "EnvelopeMetrics":
        return EnvelopeMetrics(
            plaintext_bytes=plaintext_size,
            ciphertext_bytes=d["ciphertext_bytes"],
            nonce_bytes=d["nonce_bytes"],
            tag_bytes=d["tag_bytes"],
            wrapped_dek_bytes=d["wrapped_dek_bytes"],
            eph_pubkey_bytes=d["eph_pubkey_bytes"],
            salt_bytes=d["salt_bytes"],
            hkdf_info_bytes=d["hkdf_info_bytes"],
            aad_bytes=d["aad_bytes"],
            pq_ct_bytes=d["pq_ct_bytes"],
            signature_bytes=d.get("signature_bytes", 0),
            header_bytes=d["header_bytes"],
            overhead_bytes=d["overhead_bytes"],
            total_stored_bytes=d["total_stored_bytes"],
        )


@dataclass
class Sample:
    """Raw timings (ms) for a single put+get round-trip."""
    size_bytes: int
    scheme: str
    encrypt_ms: float
    sign_ms: float
    db_insert_ms: float
    db_fetch_ms: float
    verify_ms: float
    decrypt_ms: float
    end_to_end_ms: float
    envelope: Dict = field(default=None)       # type: ignore[assignment]
    encrypt_ops: Dict = field(default_factory=dict)
    decrypt_ops: Dict = field(default_factory=dict)
    # Policy metadata — populated from the PolicyDecision used for this record.
    selected_kem_id: str = ""
    selected_sig_id: str = ""
    tier_label: str = "n/a"
    escalated: bool = False
    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES
    policy_mode: str = "n/a"
    payload_name: str = ""


@dataclass
class Stats:
    """Aggregate statistics for a list of float samples."""
    mean: float
    minimum: float
    maximum: float
    p50: float
    p95: float
    p99: float

    @staticmethod
    def from_values(values: List[float]) -> "Stats":
        s = sorted(values)
        n = len(s)

        def percentile(p: float) -> float:
            idx = (p / 100) * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            frac = idx - lo
            return s[lo] + frac * (s[hi] - s[lo])

        return Stats(
            mean=statistics.mean(values),
            minimum=min(values),
            maximum=max(values),
            p50=percentile(50),
            p95=percentile(95),
            p99=percentile(99),
        )


@dataclass
class CellResult:
    """All statistics for one (scheme, size) cell."""
    scheme: str
    size_bytes: int
    size_label: str
    payload_name: str
    iterations: int
    pq_sig_id: Optional[str]
    encrypt: Stats
    sign: Optional[Stats]
    db_insert: Stats
    db_fetch: Stats
    verify: Optional[Stats]
    decrypt: Stats
    end_to_end: Stats
    throughput_bytes_per_sec: Dict[str, Stats] = field(default_factory=dict)
    envelope: EnvelopeMetrics = field(default=None)       # type: ignore[assignment]
    encrypt_breakdown: Dict[str, Stats] = field(default_factory=dict)
    decrypt_breakdown: Dict[str, Stats] = field(default_factory=dict)
    # Policy metadata — taken from the first sample (all samples in a cell share the same decision).
    policy_mode: str = "n/a"
    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES
    selected_kem_id: str = ""
    selected_sig_id: str = ""
    tier_label: str = "n/a"
    escalated: bool = False


@dataclass
class BenchmarkReport:
    """Top-level container for the full benchmark run."""
    run_at: str
    host: str
    iterations: int
    warmup_iterations: int
    payload_source: str
    payload_selection: str
    sizes_bytes: List[int]
    schemes: List[str]
    pq_kem_id: str
    pq_sig_id: Optional[str] = None   # None = unsigned; "ML-DSA-*" = signed
    policy_mode: str = "n/a"
    policy_threshold_bytes: int = DEFAULT_THRESHOLD_BYTES
    results: List[CellResult] = field(default_factory=list)


@dataclass
class PolicyTierSummary:
    """Aggregated metrics for one policy tier (baseline or strong) across a run."""
    tier_label: str
    record_count: int
    record_fraction: float
    plaintext_bytes: int
    plaintext_byte_fraction: float
    total_stored_bytes: int
    storage_amplification: float
    mean_end_to_end_ms: float
    p50_end_to_end_ms: float
    p95_end_to_end_ms: float
    mean_encrypt_ms: float
    mean_sign_ms: float
    mean_verify_ms: float
    mean_db_insert_ms: float
    mean_db_fetch_ms: float
    mean_decrypt_ms: float


@dataclass
class PolicyRunSummary:
    """Workload-level policy summary across the entire benchmark run.

    Aggregates all result cells by tier so callers can make statements such as:
    "X% of records were escalated to the strong profile."
    "Strong-tier records accounted for Y% of total plaintext bytes."
    """
    policy_mode: Optional[str]
    threshold_bytes: Optional[int]
    total_records: int
    total_plaintext_bytes: int
    total_stored_bytes: int
    overall_storage_amplification: float
    escalated_records: int
    escalated_record_fraction: float
    escalated_plaintext_bytes: int
    escalated_plaintext_byte_fraction: float
    mean_end_to_end_ms: float
    p50_end_to_end_ms: float
    p95_end_to_end_ms: float
    tier_summaries: List[PolicyTierSummary] = field(default_factory=list)


@dataclass(frozen=True)
class PayloadSpec:
    name: str
    size_bytes: int
    path: Optional[Path] = None
    plaintext: Optional[bytes] = None

    def read_bytes(self) -> bytes:
        if self.plaintext is not None:
            return self.plaintext
        if self.path is None:
            raise ValueError(f"Payload '{self.name}' has neither plaintext nor path")
        return self.path.read_bytes()


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _resolve_policy_decision(
    policy_mode: str,
    plaintext_size: int,
    policy_threshold_bytes: int,
) -> PolicyDecision:
    """Return the PolicyDecision for one record given the benchmark policy mode.

    For ``adaptive_threshold``, calls ``select_policy_for_record`` with a
    ``ThresholdPolicyConfig`` built from ``policy_threshold_bytes``.  For
    uniform modes, constructs the decision directly without evaluating a
    threshold.
    """
    if policy_mode == "adaptive_threshold":
        config = ThresholdPolicyConfig(threshold_bytes=policy_threshold_bytes)
        return select_policy_for_record(plaintext_size, config)
    if policy_mode == "uniform_baseline":
        return PolicyDecision(
            record_size_bytes=plaintext_size,
            threshold_bytes=policy_threshold_bytes,
            kem_id=BASELINE_KEM_ID,
            sig_id=BASELINE_SIG_ID,
            tier_label="baseline",
            escalated=False,
        )
    if policy_mode == "uniform_strong":
        return PolicyDecision(
            record_size_bytes=plaintext_size,
            threshold_bytes=policy_threshold_bytes,
            kem_id=STRONG_KEM_ID,
            sig_id=STRONG_SIG_ID,
            tier_label="strong",
            escalated=True,
        )
    raise ValueError(
        f"Unknown policy_mode: {policy_mode!r}. "
        f"Must be one of: {', '.join(POLICY_MODES)}"
    )


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def _aggregate_ops(ops_list: List[Dict]) -> Dict[str, Stats]:
    """Aggregate a list of per-operation timing dicts into Stats objects."""
    if not ops_list:
        return {}
    all_keys: set = set().union(*(d.keys() for d in ops_list))
    result: Dict[str, Stats] = {}
    for key in sorted(all_keys):
        vals = [d[key] for d in ops_list if key in d]
        if vals:
            result[key] = Stats.from_values(vals)
    return result


def _throughput_bytes_per_sec(size_bytes: int, duration_ms: float) -> float:
    if duration_ms <= 0:
        return 0.0
    return (size_bytes * 1000.0) / duration_ms


def _aggregate_throughput(samples: List[Sample]) -> Dict[str, Stats]:
    metric_map: Dict[str, List[float]] = {
        "encrypt":   [_throughput_bytes_per_sec(s.size_bytes, s.encrypt_ms)   for s in samples],
        "db_insert": [_throughput_bytes_per_sec(s.size_bytes, s.db_insert_ms) for s in samples],
        "db_fetch":  [_throughput_bytes_per_sec(s.size_bytes, s.db_fetch_ms)  for s in samples],
        "decrypt":   [_throughput_bytes_per_sec(s.size_bytes, s.decrypt_ms)   for s in samples],
        "end_to_end":[_throughput_bytes_per_sec(s.size_bytes, s.end_to_end_ms)for s in samples],
    }
    # Only include sign/verify throughput when signature mode is active.
    if any(s.sign_ms > 0 for s in samples):
        metric_map["sign"]   = [_throughput_bytes_per_sec(s.size_bytes, s.sign_ms)   for s in samples]
        metric_map["verify"] = [_throughput_bytes_per_sec(s.size_bytes, s.verify_ms) for s in samples]
    return {metric: Stats.from_values(values) for metric, values in metric_map.items()}


def _load_payload_candidates(payload_source: str) -> List[PayloadSpec]:
    if payload_source == "synthetic":
        return []

    if payload_source == "faers":
        paths = sorted((DATA_DIR / "faers_records").glob("*.json"))
    elif payload_source == "all_ggvp":
        paths = sorted(DATA_DIR.glob("ALL_GGVP*.vcf.gz"))
    else:
        raise ValueError(f"Unsupported payload source: {payload_source}")

    candidates: List[PayloadSpec] = []
    for path in paths:
        candidates.append(PayloadSpec(name=path.name, size_bytes=path.stat().st_size, path=path))
    if not candidates:
        raise ValueError(f"No payload files found for source '{payload_source}'")
    return candidates


def _select_real_payloads(candidates: List[PayloadSpec], requested_sizes: List[int]) -> List[PayloadSpec]:
    if len(requested_sizes) > len(candidates):
        raise ValueError(
            f"Requested {len(requested_sizes)} size buckets but only {len(candidates)} payload files are available"
        )

    remaining = list(candidates)
    selected: List[PayloadSpec] = []
    for requested_size in sorted(requested_sizes):
        idx = min(
            range(len(remaining)),
            key=lambda i: (abs(remaining[i].size_bytes - requested_size), remaining[i].size_bytes, remaining[i].name),
        )
        selected.append(remaining.pop(idx))
    random.shuffle(selected)
    return selected


def _select_distribution_payloads(candidates: List[PayloadSpec], count: int) -> List[PayloadSpec]:
    if count <= 0:
        raise ValueError("real_payload_count must be positive")
    if count >= len(candidates):
        selected = list(candidates)
        random.shuffle(selected)
        return selected

    ordered = sorted(candidates, key=lambda p: (p.size_bytes, p.name))
    selected: List[PayloadSpec] = []
    used_indices: set[int] = set()

    for rank in range(count):
        # Evenly sample order statistics across the empirical size distribution.
        idx = round(rank * (len(ordered) - 1) / (count - 1)) if count > 1 else len(ordered) // 2
        while idx in used_indices and idx + 1 < len(ordered):
            idx += 1
        while idx in used_indices and idx - 1 >= 0:
            idx -= 1
        used_indices.add(idx)
        selected.append(ordered[idx])

    random.shuffle(selected)
    return selected


def _resolve_payloads(
    payload_source: str,
    requested_sizes: Optional[List[int]],
    *,
    real_payload_count: Optional[int],
) -> List[PayloadSpec]:
    if payload_source == "synthetic":
        if requested_sizes is None:
            requested_sizes = DEFAULT_SIZES_BYTES
        return [
            PayloadSpec(
                name=f"synthetic_{size}B.bin",
                size_bytes=size,
                plaintext=os.urandom(size),
            )
            for size in requested_sizes
        ]

    candidates = _load_payload_candidates(payload_source)
    print(f"  Payload files discovered : {len(candidates)}")
    if requested_sizes:
        selected = _select_real_payloads(candidates, requested_sizes)
        print(f"  Payload files selected   : {len(selected)}  (explicit size buckets)")
        return selected
    if real_payload_count is None:
        selected = list(candidates)
        random.shuffle(selected)
        print(f"  Payload files selected   : {len(selected)}  (all files — use --real-payload-count N to limit)")
        return selected
    selected = _select_distribution_payloads(candidates, real_payload_count)
    print(f"  Payload files selected   : {len(selected)}  (sampled across empirical distribution)")
    return selected


def _run_single(
    conn,
    *,
    scheme: str,
    plaintext: bytes,
    aad: bytes,
    classical_keys: Any,
    hybrid_keys_by_kem: Dict[str, Any],
    sig_keys_by_sig: Dict[str, stage2.SigningKeys],
    decision: PolicyDecision,
    effective_policy_mode: str,
    payload_name: str = "",
) -> Sample:
    """Execute one put+get round-trip and return a raw timing Sample.

    Key material is selected per-record from ``hybrid_keys_by_kem`` and
    ``sig_keys_by_sig`` using the algorithm IDs carried on ``decision``.
    """
    ctx = ServiceContext(tenant_id=TENANT_ID, key_id=KEY_ID, version=1)

    # Resolve per-record keys from the policy decision.
    sig_keys: Optional[stage2.SigningKeys] = sig_keys_by_sig.get(decision.sig_id)
    sig_verify_key = sig_keys.verify_key if sig_keys is not None else None

    t_wall_start = time.perf_counter()

    if scheme == "classical":
        record_id, put_timings = put_record(
            conn,
            ctx=ctx,
            plaintext=plaintext,
            aad=aad,
            scheme="classical",
            recipient_keys=classical_keys,
            sig_keys=sig_keys,
        )
        _, get_timings = get_record(
            conn,
            tenant_id=TENANT_ID,
            record_id=record_id,
            recipient_keys=classical_keys,
            sig_verify_key=sig_verify_key,
        )
    else:
        hybrid_keys = hybrid_keys_by_kem[decision.kem_id]
        record_id, put_timings = put_record(
            conn,
            ctx=ctx,
            plaintext=plaintext,
            aad=aad,
            scheme="hybrid",
            recipient_keys=hybrid_keys,
            pq_kem_id=decision.kem_id,
            sig_keys=sig_keys,
        )
        _, get_timings = get_record(
            conn,
            tenant_id=TENANT_ID,
            record_id=record_id,
            recipient_keys=hybrid_keys,
            pq_sk=hybrid_keys.pq_sk,
            sig_verify_key=sig_verify_key,
        )

    wall_ms = (time.perf_counter() - t_wall_start) * 1000.0

    return Sample(
        size_bytes=len(plaintext),
        scheme=scheme,
        encrypt_ms=put_timings["encrypt"],
        sign_ms=put_timings.get("sign", 0.0),
        db_insert_ms=put_timings["db_insert"],
        db_fetch_ms=get_timings["db_fetch"],
        verify_ms=get_timings.get("verify", 0.0),
        decrypt_ms=get_timings["decrypt"],
        end_to_end_ms=wall_ms,
        envelope=put_timings.get("envelope"),
        encrypt_ops=put_timings.get("encrypt_ops") or {},
        decrypt_ops=get_timings.get("decrypt_ops") or {},
        selected_kem_id=decision.kem_id,
        selected_sig_id=decision.sig_id,
        tier_label=decision.tier_label,
        escalated=decision.escalated,
        threshold_bytes=decision.threshold_bytes,
        policy_mode=effective_policy_mode,
        payload_name=payload_name,
    )


def run_benchmark(
    conn,
    *,
    sizes_bytes: Optional[List[int]] = None,
    iterations: int = 10,
    warmup: int = 2,
    pq_kem_id: str = stage2.DEFAULT_PQ_KEM_ID,
    pq_sig_id: Optional[str] = None,
    payload_source: str = "synthetic",
    real_payload_count: Optional[int] = None,
    schemes: Optional[List[str]] = None,
    policy_mode: Optional[str] = None,
    policy_threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
) -> BenchmarkReport:
    """Run the benchmark for the selected schemes and return a BenchmarkReport.

    Args:
        schemes:                List of scheme names to measure.
        pq_kem_id:              ML-KEM parameter set for legacy (non-policy) hybrid runs.
        pq_sig_id:              ML-DSA parameter set for legacy (non-policy) signing, or None.
        policy_mode:            One of ``POLICY_MODES`` or None for legacy fixed-param behavior.
                                When set, overrides pq_kem_id/pq_sig_id with per-record
                                policy-driven selection.
        policy_threshold_bytes: Threshold for ``adaptive_threshold`` mode (default 10 MiB).
    """
    if schemes is None:
        schemes = list(SCHEMES)

    run_classical = "classical" in schemes
    run_hybrid    = "hybrid" in schemes

    if policy_mode is not None and policy_mode not in POLICY_MODES:
        raise ValueError(
            f"Unknown policy_mode: {policy_mode!r}. "
            f"Must be one of: {', '.join(POLICY_MODES)}"
        )

    # ── Key generation ────────────────────────────────────────────────────────
    print()
    print("  Generating key pairs...", end=" ", flush=True)

    classical_keys = stage2.generate_classical_recipient_keys() if run_classical else None

    # Build a dict of hybrid key bundles keyed by KEM ID.
    # Policy modes require both baseline and strong bundles; legacy uses only
    # the single run-level KEM.
    hybrid_keys_by_kem: Dict[str, Any] = {}
    if run_hybrid:
        if policy_mode in ("uniform_baseline", "adaptive_threshold"):
            hybrid_keys_by_kem[BASELINE_KEM_ID] = stage2.generate_hybrid_recipient_keys(BASELINE_KEM_ID)
        if policy_mode in ("uniform_strong", "adaptive_threshold"):
            hybrid_keys_by_kem[STRONG_KEM_ID] = stage2.generate_hybrid_recipient_keys(STRONG_KEM_ID)
        if policy_mode is None:
            hybrid_keys_by_kem[pq_kem_id] = stage2.generate_hybrid_recipient_keys(pq_kem_id)

    # Build a dict of signing key bundles keyed by sig ID.
    sig_keys_by_sig: Dict[str, stage2.SigningKeys] = {}
    if policy_mode in ("uniform_baseline", "adaptive_threshold"):
        sig_keys_by_sig[BASELINE_SIG_ID] = stage2.generate_sig_keys(
            BASELINE_SIG_ID, key_id="bench-sig-key-v1"
        )
    if policy_mode in ("uniform_strong", "adaptive_threshold"):
        sig_keys_by_sig[STRONG_SIG_ID] = stage2.generate_sig_keys(
            STRONG_SIG_ID, key_id="bench-sig-key-v1"
        )
    if policy_mode is None and pq_sig_id is not None:
        sig_keys_by_sig[pq_sig_id] = stage2.generate_sig_keys(
            pq_sig_id, key_id="bench-sig-key-v1"
        )

    print("done.")

    # ── Determine report-level KEM/sig labels ─────────────────────────────────
    # For policy modes, record-level selection varies; use a descriptive label.
    if policy_mode == "uniform_baseline":
        report_pq_kem_id = BASELINE_KEM_ID
        report_pq_sig_id: Optional[str] = BASELINE_SIG_ID
    elif policy_mode == "uniform_strong":
        report_pq_kem_id = STRONG_KEM_ID
        report_pq_sig_id = STRONG_SIG_ID
    elif policy_mode == "adaptive_threshold":
        report_pq_kem_id = "adaptive"
        report_pq_sig_id = "adaptive"
    else:
        report_pq_kem_id = pq_kem_id
        report_pq_sig_id = pq_sig_id

    effective_policy_mode = policy_mode or "n/a"

    report = BenchmarkReport(
        run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        host=os.uname().nodename,
        iterations=iterations,
        warmup_iterations=warmup,
        payload_source=payload_source,
        payload_selection=(
            "explicit_sizes" if sizes_bytes else ("synthetic_defaults" if payload_source == "synthetic" else "real_distribution")
        ),
        sizes_bytes=[],
        schemes=schemes,
        pq_kem_id=report_pq_kem_id,
        pq_sig_id=report_pq_sig_id,
        policy_mode=effective_policy_mode,
        policy_threshold_bytes=policy_threshold_bytes,
    )

    # Randomise the order sizes are measured to avoid confounding thermal/frequency
    # state with the size dimension.  Crucially, classical and hybrid are measured
    # in an interleaved fashion within each size so both schemes always experience
    # the same CPU/system state on every iteration, eliminating the between-cell
    # variance that caused spurious classical "underperformance" at large sizes.
    payload_specs = _resolve_payloads(
        payload_source,
        sizes_bytes,
        real_payload_count=real_payload_count,
    )
    random.shuffle(payload_specs)

    total_sizes = len(payload_specs)
    verbose_progress = total_sizes <= 100

    for size_num, payload in enumerate(payload_specs, 1):
        size = payload.size_bytes
        label = _human_size(size)
        if verbose_progress:
            print(
                f"  [{size_num:>2}/{total_sizes}] size={label:<7}"
                f"  source={payload.name[:36]:<36}"
                f"  warmup={warmup}  iterations={iterations}  ...",
                end=" ",
                flush=True,
            )
        elif size_num == 1 or size_num == total_sizes or size_num % 250 == 0:
            print(
                f"  [{size_num:>5}/{total_sizes}] size={label:<7}"
                f"  source={payload.name[:36]:<36}"
                f"  warmup={warmup}  iterations={iterations}",
                flush=True,
            )

        plaintext = payload.read_bytes()
        aad_by_scheme = {
            "classical": f"benchmark|scheme=classical|size={size}".encode(),
            "hybrid":    f"benchmark|scheme=hybrid|size={size}".encode(),
        }
        report.sizes_bytes.append(size)

        # Compute the policy decision once per payload — all iterations within
        # this cell use the same plaintext size, so the decision is constant.
        if policy_mode is not None:
            decision = _resolve_policy_decision(policy_mode, len(plaintext), policy_threshold_bytes)
        else:
            # Legacy mode: synthesize a decision from run-level KEM/sig params.
            decision = PolicyDecision(
                record_size_bytes=len(plaintext),
                threshold_bytes=policy_threshold_bytes,
                kem_id=pq_kem_id,
                sig_id=pq_sig_id or "",
                tier_label="n/a",
                escalated=False,
            )

        # Clear once per size block so all schemes share the same starting
        # table state.
        _clear_benchmark_data(conn)

        # Warmup (interleaved when multiple schemes are active) — results discarded.
        for _ in range(warmup):
            for scheme in schemes:
                _run_single(
                    conn,
                    scheme=scheme,
                    plaintext=plaintext,
                    aad=aad_by_scheme[scheme],
                    classical_keys=classical_keys,
                    hybrid_keys_by_kem=hybrid_keys_by_kem,
                    sig_keys_by_sig=sig_keys_by_sig,
                    decision=decision,
                    effective_policy_mode=effective_policy_mode,
                    payload_name=payload.name,
                )

        # Measured rounds: when both schemes are active, alternate which goes first
        # each iteration so any residual ordering bias averages out.
        samples_by_scheme: Dict[str, List[Sample]] = {s: [] for s in schemes}
        for i in range(iterations):
            iteration_schemes = list(schemes) if i % 2 == 0 else list(reversed(schemes))
            for scheme in iteration_schemes:
                samples_by_scheme[scheme].append(
                    _run_single(
                        conn,
                        scheme=scheme,
                        plaintext=plaintext,
                        aad=aad_by_scheme[scheme],
                        classical_keys=classical_keys,
                        hybrid_keys_by_kem=hybrid_keys_by_kem,
                        sig_keys_by_sig=sig_keys_by_sig,
                        decision=decision,
                        effective_policy_mode=effective_policy_mode,
                        payload_name=payload.name,
                    )
                )

        # Build one CellResult per active scheme.
        size_cells: Dict[str, CellResult] = {}
        for scheme in schemes:
            samples = samples_by_scheme[scheme]
            # Envelope sizes are deterministic per (scheme, size); use first sample.
            env = None
            if samples[0].envelope:
                env = EnvelopeMetrics.from_dict(size, samples[0].envelope)

            sign_stats   = Stats.from_values([s.sign_ms   for s in samples]) if any(s.sign_ms   > 0 for s in samples) else None
            verify_stats = Stats.from_values([s.verify_ms for s in samples]) if any(s.verify_ms > 0 for s in samples) else None

            # Policy metadata is constant within a cell (same plaintext size → same decision).
            first = samples[0]

            cell = CellResult(
                scheme=scheme,
                size_bytes=size,
                size_label=label,
                payload_name=payload.name,
                iterations=iterations,
                pq_sig_id=report_pq_sig_id,
                encrypt=Stats.from_values([s.encrypt_ms    for s in samples]),
                sign=sign_stats,
                db_insert=Stats.from_values([s.db_insert_ms for s in samples]),
                db_fetch=Stats.from_values([s.db_fetch_ms   for s in samples]),
                verify=verify_stats,
                decrypt=Stats.from_values([s.decrypt_ms    for s in samples]),
                end_to_end=Stats.from_values([s.end_to_end_ms for s in samples]),
                throughput_bytes_per_sec=_aggregate_throughput(samples),
                envelope=env,
                encrypt_breakdown=_aggregate_ops([s.encrypt_ops for s in samples if s.encrypt_ops]),
                decrypt_breakdown=_aggregate_ops([s.decrypt_ops for s in samples if s.decrypt_ops]),
                policy_mode=first.policy_mode,
                threshold_bytes=first.threshold_bytes,
                selected_kem_id=first.selected_kem_id,
                selected_sig_id=first.selected_sig_id,
                tier_label=first.tier_label,
                escalated=first.escalated,
            )
            report.results.append(cell)
            size_cells[scheme] = cell

        # Print one summary line per size.
        if verbose_progress:
            if run_classical and run_hybrid:
                cl_cell = size_cells["classical"]
                hy_cell = size_cells["hybrid"]
                print(
                    f"classical p50={cl_cell.end_to_end.p50:>7.2f} ms"
                    f"  hybrid p50={hy_cell.end_to_end.p50:>7.2f} ms"
                    f"  (mean: {cl_cell.end_to_end.mean:>7.2f} / {hy_cell.end_to_end.mean:>7.2f})"
                )
            else:
                active_scheme = schemes[0]
                sc = size_cells[active_scheme]
                print(f"{active_scheme} p50={sc.end_to_end.p50:>7.2f} ms  (mean: {sc.end_to_end.mean:>7.2f})")

    # Sort results into a canonical (scheme order, then ascending size) order so
    # report tables and charts receive cells in a predictable sequence regardless
    # of the randomised measurement order.
    scheme_order = {s: i for i, s in enumerate(SCHEMES)}
    report.results.sort(key=lambda c: (scheme_order.get(c.scheme, 99), c.size_bytes, c.payload_name))
    report.sizes_bytes.sort()

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_SIZE_W   = 8    # width of size column
_DIVIDER  = "─"
_HEAVY    = "═"
_MAX_RENDER_CELLS = 40


def _fmt_cell(stat: Stats) -> str:
    """'mean / p95' in a fixed-width column."""
    return f"{stat.mean:>7.3f} / {stat.p95:<7.3f}"


def _cell_identity(cell: CellResult) -> Tuple[int, str]:
    return (cell.size_bytes, cell.payload_name)


def _summarize_sizes(sizes: List[int], max_items: int = 12) -> str:
    if not sizes:
        return "(none)"
    labels = [_human_size(s) for s in sizes]
    if len(labels) <= max_items:
        return ", ".join(labels)
    head = ", ".join(labels[:6])
    tail = ", ".join(labels[-3:])
    return f"{head}, ... , {tail}  ({len(labels)} total)"


def _truncate_cells(cells: List[CellResult], max_cells: int = _MAX_RENDER_CELLS) -> List[CellResult]:
    if len(cells) <= max_cells:
        return cells
    head = cells[: max_cells // 2]
    tail = cells[-(max_cells - len(head)) :]
    return head + tail


def _table_for_scheme(cells: List[CellResult]) -> str:
    metric_names = ["Encrypt", "DB Insert", "DB Fetch", "Decrypt", "End-to-End"]
    col_w = 19  # wide enough for "mean / p95"

    # Header row
    size_col = f"{'Size':<{_SIZE_W}}"
    headers  = "  ".join(f"{m:<{col_w}}" for m in metric_names)
    subhdr   = "  ".join(f"{'mean(ms) / p95(ms)':<{col_w}}" for _ in metric_names)
    sep_line = _DIVIDER * (_SIZE_W + 2 + len(headers))

    lines = [
        sep_line,
        f"{size_col}  {headers}",
        f"{' ' * _SIZE_W}  {subhdr}",
        sep_line,
    ]

    for cell in cells:
        row_parts = [
            _fmt_cell(cell.encrypt),
            _fmt_cell(cell.db_insert),
            _fmt_cell(cell.db_fetch),
            _fmt_cell(cell.decrypt),
            _fmt_cell(cell.end_to_end),
        ]
        row = "  ".join(f"{p:<{col_w}}" for p in row_parts)
        lines.append(f"{cell.size_label:<{_SIZE_W}}  {row}")

    lines.append(sep_line)
    return "\n".join(lines)


def _storage_table(cells: List[CellResult]) -> str:
    """Render a per-size storage breakdown table for a single scheme."""
    col_w = 12
    headers = ["Size", "Plaintext", "Ciphertext", "Header", "PQ-CT", "Sig", "Overhead", "Total-Stored", "Amplification"]
    sep_line = _DIVIDER * (sum(col_w for _ in headers) + 2 * (len(headers) - 1))

    lines = [
        sep_line,
        "  ".join(f"{h:<{col_w}}" for h in headers),
        "  ".join(
            f"{'(bytes)':<{col_w}}" if h != "Amplification" else f"{'(total/plain)':<{col_w}}"
            for h in headers
        ),
        sep_line,
    ]

    for cell in cells:
        e = cell.envelope
        if e is None:
            lines.append(f"{cell.size_label:<{col_w}}  (no envelope data)")
            continue
        amp = e.total_stored_bytes / e.plaintext_bytes if e.plaintext_bytes else 0.0
        row = "  ".join(f"{v:<{col_w}}" for v in [
            cell.size_label,
            e.plaintext_bytes,
            e.ciphertext_bytes,
            e.header_bytes,
            e.pq_ct_bytes if e.pq_ct_bytes else "—",
            e.signature_bytes if e.signature_bytes else "—",
            e.overhead_bytes,
            e.total_stored_bytes,
            f"{amp:.4f}x",
        ])
        lines.append(row)

    lines.append(sep_line)
    return "\n".join(lines)


def _storage_overhead_comparison(
    classical_cells: List[CellResult],
    hybrid_cells: List[CellResult],
) -> str:
    """Show the extra bytes hybrid adds over classical, per size."""
    col_w = 14
    headers = ["Size", "Classical", "Hybrid", "PQ-CT", "Delta (B)", "Delta (%)"]
    sep_line = _DIVIDER * (sum(col_w for _ in headers) + 2 * (len(headers) - 1))

    lines = [
        sep_line,
        "  ".join(f"{h:<{col_w}}" for h in headers),
        "  ".join(f"{'(total bytes)':<{col_w}}" if h in ("Classical", "Hybrid") else
                  f"{'(ML-KEM ct)':<{col_w}}" if h == "PQ-CT" else
                  f"{'':<{col_w}}" for h in headers),
        sep_line,
    ]

    for cl, hy in zip(classical_cells, hybrid_cells):
        if cl.envelope is None or hy.envelope is None:
            lines.append(f"{cl.size_label:<{col_w}}  (no envelope data)")
            continue
        delta = hy.envelope.total_stored_bytes - cl.envelope.total_stored_bytes
        pct = (delta / cl.envelope.total_stored_bytes * 100) if cl.envelope.total_stored_bytes else 0.0
        row = "  ".join(f"{v:<{col_w}}" for v in [
            cl.size_label,
            cl.envelope.total_stored_bytes,
            hy.envelope.total_stored_bytes,
            hy.envelope.pq_ct_bytes,
            f"+{delta}",
            f"+{pct:.2f}%",
        ])
        lines.append(row)

    lines.append(sep_line)
    return "\n".join(lines)


def _overhead_table(
    classical_cells: List[CellResult],
    hybrid_cells: List[CellResult],
) -> str:
    """Show absolute (ms) and relative (%) overhead of hybrid vs classical."""
    metric_names  = ["Encrypt", "DB Insert", "DB Fetch", "Decrypt", "End-to-End"]
    col_w = 22  # wide enough for "+12.345 ms / +123.4%"

    def get_means(cells: List[CellResult]) -> List[Tuple[float, float, float, float, float]]:
        return [
            (c.encrypt.mean, c.db_insert.mean, c.db_fetch.mean, c.decrypt.mean, c.end_to_end.mean)
            for c in cells
        ]

    cl_means = get_means(classical_cells)
    hy_means = get_means(hybrid_cells)

    size_col = f"{'Size':<{_SIZE_W}}"
    headers  = "  ".join(f"{m:<{col_w}}" for m in metric_names)
    subhdr   = "  ".join(f"{'delta(ms) / overhead':<{col_w}}" for _ in metric_names)
    sep_line = _DIVIDER * (_SIZE_W + 2 + len(headers))

    lines = [
        sep_line,
        f"{size_col}  {headers}",
        f"{' ' * _SIZE_W}  {subhdr}",
        sep_line,
    ]

    for cl_row, hy_row, cl_cell in zip(cl_means, hy_means, classical_cells):
        parts = []
        for cl_v, hy_v in zip(cl_row, hy_row):
            delta = hy_v - cl_v
            pct   = (delta / cl_v * 100) if cl_v > 0 else 0.0
            sign  = "+" if delta >= 0 else ""
            parts.append(f"{sign}{delta:>6.3f} ms / {sign}{pct:>5.1f}%")
        row = "  ".join(f"{p:<{col_w}}" for p in parts)
        lines.append(f"{cl_cell.size_label:<{_SIZE_W}}  {row}")

    lines.append(sep_line)
    return "\n".join(lines)


def format_report(report: BenchmarkReport) -> str:
    """Render the full benchmark report as a human-readable string."""
    out: List[str] = []
    W = 120  # total report width

    sig_label = report.pq_sig_id if report.pq_sig_id else "none (unsigned)"

    # ── Title block ──────────────────────────────────────────────────────────
    out.append(_HEAVY * W)
    out.append(f"  PQC ENCRYPTED DATABASE — PERFORMANCE BENCHMARK REPORT")
    out.append(f"  Run at  : {report.run_at}   Host: {report.host}")
    out.append(f"  Schemes : classical (X25519-ECDH + AES-256-GCM)")
    out.append(f"          : hybrid   (X25519-ECDH + {report.pq_kem_id} + AES-256-GCM)")
    out.append(f"  Signing : {sig_label}")
    out.append(f"  Policy  : {report.policy_mode}  (threshold: {report.policy_threshold_bytes} B)")
    out.append(f"  Payload : {report.payload_source}")
    out.append(f"  Select  : {report.payload_selection}")
    out.append(f"  Sizes   : {_summarize_sizes(report.sizes_bytes)}")
    out.append(f"  Runs    : {report.iterations} measured + {report.warmup_iterations} warmup per cell")
    out.append(f"  Metric  : All latencies in milliseconds (ms).  'mean / p95' = average / 95th-percentile")
    out.append(_HEAVY * W)

    seen_payloads = set()
    payload_lines: List[str] = []
    for cell in report.results:
        key = (cell.size_bytes, cell.payload_name)
        if key in seen_payloads:
            continue
        seen_payloads.add(key)
        payload_lines.append(f"  {_human_size(cell.size_bytes):<8} {cell.payload_name}")
    if payload_lines:
        out.append("")
        out.append("  PAYLOAD FILES")
        if len(payload_lines) <= 20:
            out.extend(payload_lines)
        else:
            out.extend(payload_lines[:10])
            out.append(f"  ... ({len(payload_lines) - 15} more payload files omitted) ...")
            out.extend(payload_lines[-5:])

    by_scheme: Dict[str, List[CellResult]] = {}
    for cell in report.results:
        by_scheme.setdefault(cell.scheme, []).append(cell)
    for cells in by_scheme.values():
        cells.sort(key=lambda c: (c.size_bytes, c.payload_name))

    # ── Per-scheme tables ─────────────────────────────────────────────────────
    sig_suffix = f" + {report.pq_sig_id}" if report.pq_sig_id else ""
    scheme_labels = {
        "classical": f"CLASSICAL ENCRYPTION  (X25519-ECDH + AES-256-GCM{sig_suffix})",
        "hybrid":    f"HYBRID PQC ENCRYPTION  (X25519-ECDH + {report.pq_kem_id} + AES-256-GCM{sig_suffix})",
    }

    for scheme in report.schemes:
        cells = by_scheme.get(scheme, [])
        if not cells:
            continue
        out.append("")
        out.append(f"  SCHEME: {scheme_labels.get(scheme, scheme.upper())}")
        if len(cells) > _MAX_RENDER_CELLS:
            out.append(f"  Showing first/last {_MAX_RENDER_CELLS // 2} rows only ({len(cells)} total cells).")
        out.append(_table_for_scheme(_truncate_cells(cells)))

    # ── Storage / envelope size tables ───────────────────────────────────────
    cl_cells = by_scheme.get("classical", [])
    hy_cells = by_scheme.get("hybrid",    [])

    for scheme in report.schemes:
        cells = by_scheme.get(scheme, [])
        if cells and cells[0].envelope is not None:
            out.append("")
            out.append(f"  STORAGE BREAKDOWN — {scheme.upper()}")
            out.append(f"  Header = nonce + tag + wrapped-DEK + eph-pubkey + salt + hkdf-info + AAD  |  Amplification = total-stored / plaintext")
            if len(cells) > _MAX_RENDER_CELLS:
                out.append(f"  Showing first/last {_MAX_RENDER_CELLS // 2} rows only ({len(cells)} total cells).")
            out.append(_storage_table(_truncate_cells(cells)))

    if cl_cells and hy_cells and len(cl_cells) == len(hy_cells):
        if cl_cells[0].envelope is not None and hy_cells[0].envelope is not None:
            out.append("")
            out.append(f"  STORAGE OVERHEAD COMPARISON  (Hybrid PQC vs Classical)")
            out.append(f"  Delta = bytes added by the PQ KEM ciphertext layer.")
            if len(cl_cells) > _MAX_RENDER_CELLS:
                out.append(f"  Showing first/last {_MAX_RENDER_CELLS // 2} rows only ({len(cl_cells)} total payloads).")
            out.append(_storage_overhead_comparison(_truncate_cells(cl_cells), _truncate_cells(hy_cells)))

    # ── Overhead comparison ───────────────────────────────────────────────────

    if cl_cells and hy_cells and len(cl_cells) == len(hy_cells):
        out.append("")
        out.append(f"  OVERHEAD COMPARISON  (Hybrid PQC vs Classical)")
        out.append(f"  Positive values indicate additional latency introduced by the PQC KEM layer.")
        if len(cl_cells) > _MAX_RENDER_CELLS:
            out.append(f"  Showing first/last {_MAX_RENDER_CELLS // 2} rows only ({len(cl_cells)} total payloads).")
        out.append(_overhead_table(_truncate_cells(cl_cells), _truncate_cells(hy_cells)))

    # ── Crypto operation breakdown ────────────────────────────────────────────
    out.append("")
    out.append(f"  CRYPTO OPERATION BREAKDOWN  (mean ms per primitive, averaged over {report.iterations} iterations)")
    out.append(f"  Encrypt-side ops: t_dek_gen, t_ecdh, t_pq_encaps (hybrid only), t_hkdf, t_wrap, t_aes_gcm, t_sig_sign (signed only)")
    out.append(f"  Decrypt-side ops: t_ecdh, t_pq_decaps (hybrid only), t_hkdf, t_unwrap, t_aes_gcm, t_sig_verify (signed only)")
    out.append(f"  Note: sub-op means will slightly exceed total encrypt/decrypt due to perf_counter() call overhead (~0.1–0.3 µs each).")

    _OP_LABELS = {
        "t_dek_gen_ms":    "DEK-gen",
        "t_aes_gcm_ms":    "AES-GCM",
        "t_eph_keygen_ms": "Eph-keygen",
        "t_ecdh_ms":       "X25519-ECDH",
        "t_pq_encaps_ms":  "ML-KEM-encaps",
        "t_pq_decaps_ms":  "ML-KEM-decaps",
        "t_hkdf_ms":       "HKDF-SHA256",
        "t_wrap_ms":       "AES-KW-wrap",
        "t_unwrap_ms":     "AES-KW-unwrap",
        "t_sig_sign_ms":   "ML-DSA-sign",
        "t_sig_verify_ms": "ML-DSA-verify",
        "t_total_crypto_ms": "TOTAL-CRYPTO",
    }
    _ENC_KEYS = ["t_dek_gen_ms", "t_aes_gcm_ms", "t_eph_keygen_ms", "t_ecdh_ms",
                 "t_pq_encaps_ms", "t_hkdf_ms", "t_wrap_ms", "t_sig_sign_ms", "t_total_crypto_ms"]
    _DEC_KEYS = ["t_ecdh_ms", "t_pq_decaps_ms", "t_hkdf_ms", "t_unwrap_ms",
                 "t_aes_gcm_ms", "t_sig_verify_ms", "t_total_crypto_ms"]

    for phase, keys in [("ENCRYPT", _ENC_KEYS), ("DECRYPT", _DEC_KEYS)]:
        col_w = 15
        op_labels = [_OP_LABELS.get(k, k) for k in keys]
        size_col_w = 8
        sep = _DIVIDER * (size_col_w + 2 + len(op_labels) * (col_w + 2))
        display_results = _truncate_cells(report.results)
        out.append("")
        out.append(f"  {phase} breakdown  (mean ms)")
        if len(report.results) > _MAX_RENDER_CELLS:
            out.append(f"  Showing first/last {_MAX_RENDER_CELLS // 2} rows only ({len(report.results)} total cells).")
        out.append("  " + sep)
        header = f"  {'Scheme':<10} {'Size':<{size_col_w}}  " + "  ".join(f"{lbl:<{col_w}}" for lbl in op_labels)
        out.append(header)
        out.append("  " + sep)
        for cell in display_results:
            bk = cell.encrypt_breakdown if phase == "ENCRYPT" else cell.decrypt_breakdown
            vals = []
            for k in keys:
                s = bk.get(k)
                vals.append(f"{s.mean:>{col_w}.4f}" if s else f"{'—':>{col_w}}")
            out.append(
                f"  {cell.scheme:<10} {cell.size_label:<{size_col_w}}  " + "  ".join(vals)
            )
        out.append("  " + sep)

    # ── Per-metric summary ────────────────────────────────────────────────────
    out.append("")
    out.append(f"  DETAILED STATISTICS  (mean / min / max / p50 / p95 / p99 — all in ms)")

    stat_header = (
        f"  {'Scheme':<10} {'Size':<8} {'Metric':<12}"
        f" {'mean':>9} {'min':>9} {'max':>9} {'p50':>9} {'p95':>9} {'p99':>9}"
    )
    out.append("  " + _DIVIDER * (len(stat_header) - 2))
    out.append(stat_header)
    out.append("  " + _DIVIDER * (len(stat_header) - 2))

    metric_map = [
        ("encrypt",    "Encrypt"),
        ("sign",       "Sign"),
        ("db_insert",  "DB Insert"),
        ("db_fetch",   "DB Fetch"),
        ("verify",     "Verify"),
        ("decrypt",    "Decrypt"),
        ("end_to_end", "End-to-End"),
    ]

    display_results = _truncate_cells(report.results)
    if len(report.results) > _MAX_RENDER_CELLS:
        out.append(f"  Showing first/last {_MAX_RENDER_CELLS // 2} rows only ({len(report.results)} total cells).")
        out.append("  " + _DIVIDER * (len(stat_header) - 2))

    for cell in display_results:
        first = True
        for attr, label in metric_map:
            s: Optional[Stats] = getattr(cell, attr)
            if s is None:
                continue  # skip sign/verify rows when not running in signed mode
            prefix = (
                f"  {cell.scheme:<10} {cell.size_label:<8}"
                if first
                else f"  {'':<10} {'':<8}"
            )
            out.append(
                f"{prefix} {label:<12}"
                f" {s.mean:>9.4f} {s.minimum:>9.4f} {s.maximum:>9.4f}"
                f" {s.p50:>9.4f} {s.p95:>9.4f} {s.p99:>9.4f}"
            )
            first = False
        out.append("  " + _DIVIDER * (len(stat_header) - 2))

    # ── Policy Tier Summary ───────────────────────────────────────────────────
    if report.policy_mode != "n/a":
        summary = summarize_policy_run(report)
        out.append("")
        out.append("  POLICY TIER SUMMARY")
        sep = _DIVIDER * W
        out.append("  " + sep)
        out.append(f"  Policy mode   : {summary.policy_mode or 'n/a'}")
        if summary.threshold_bytes is not None:
            from .policy import format_threshold_label
            out.append(
                f"  Threshold     : {format_threshold_label(summary.threshold_bytes)}"
                f"  ({summary.threshold_bytes} B)"
            )
        out.append(
            f"  Total records : {summary.total_records}"
            f"   Escalated: {summary.escalated_records} / {summary.total_records}"
            f"  ({summary.escalated_record_fraction * 100:.1f}%)"
        )
        _tb = summary.total_plaintext_bytes
        _eb = summary.escalated_plaintext_bytes
        out.append(
            f"  Plaintext     : {_tb:,} B total"
            f"   Escalated: {_eb:,} B"
            f"  ({summary.escalated_plaintext_byte_fraction * 100:.1f}%)"
        )
        out.append(
            f"  Storage amp.  : {summary.overall_storage_amplification:.4f}x overall"
            f"   Mean E2E: {summary.mean_end_to_end_ms:.3f} ms"
        )
        if summary.tier_summaries:
            out.append("  " + sep)
            col_w = 10
            hdr = (
                f"  {'Tier':<{col_w}}"
                f" {'Records':>9} {'Rec%':>7}"
                f" {'PlainMiB':>10} {'Byte%':>7}"
                f" {'StoredMiB':>10} {'Amp':>7}"
                f" {'MeanE2E':>9} {'P50 E2E':>9} {'P95 E2E':>9}"
            )
            out.append(hdr)
            out.append("  " + sep)
            for ts in summary.tier_summaries:
                plain_mib = ts.plaintext_bytes / (1024 * 1024)
                stored_mib = ts.total_stored_bytes / (1024 * 1024)
                out.append(
                    f"  {ts.tier_label:<{col_w}}"
                    f" {ts.record_count:>9} {ts.record_fraction * 100:>6.1f}%"
                    f" {plain_mib:>10.2f} {ts.plaintext_byte_fraction * 100:>6.1f}%"
                    f" {stored_mib:>10.2f} {ts.storage_amplification:>6.4f}x"
                    f" {ts.mean_end_to_end_ms:>9.3f} {ts.p50_end_to_end_ms:>9.3f}"
                    f" {ts.p95_end_to_end_ms:>9.3f}"
                )
        out.append("  " + sep)

    out.append(_HEAVY * W)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Policy aggregation
# ---------------------------------------------------------------------------

def _weighted_mean(values: List[float], weights: List[int]) -> float:
    """Iteration-count-weighted mean of per-cell stats values."""
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _build_tier_summary(
    tier: str,
    tier_cells: List[CellResult],
    total_records: int,
    total_plaintext_bytes: int,
) -> PolicyTierSummary:
    """Build a PolicyTierSummary by aggregating all cells in one tier."""
    weights = [c.iterations for c in tier_cells]
    record_count = sum(weights)
    plaintext_bytes = sum(c.iterations * c.size_bytes for c in tier_cells)
    total_stored = sum(
        c.iterations * c.envelope.total_stored_bytes
        for c in tier_cells if c.envelope is not None
    )
    amp = total_stored / plaintext_bytes if plaintext_bytes else 0.0

    return PolicyTierSummary(
        tier_label=tier,
        record_count=record_count,
        record_fraction=record_count / total_records if total_records else 0.0,
        plaintext_bytes=plaintext_bytes,
        plaintext_byte_fraction=plaintext_bytes / total_plaintext_bytes if total_plaintext_bytes else 0.0,
        total_stored_bytes=total_stored,
        storage_amplification=amp,
        mean_end_to_end_ms=_weighted_mean([c.end_to_end.mean for c in tier_cells], weights),
        p50_end_to_end_ms=_weighted_mean([c.end_to_end.p50  for c in tier_cells], weights),
        p95_end_to_end_ms=_weighted_mean([c.end_to_end.p95  for c in tier_cells], weights),
        mean_encrypt_ms=_weighted_mean([c.encrypt.mean for c in tier_cells], weights),
        mean_sign_ms=_weighted_mean(
            [c.sign.mean if c.sign else 0.0 for c in tier_cells], weights
        ),
        mean_verify_ms=_weighted_mean(
            [c.verify.mean if c.verify else 0.0 for c in tier_cells], weights
        ),
        mean_db_insert_ms=_weighted_mean([c.db_insert.mean for c in tier_cells], weights),
        mean_db_fetch_ms=_weighted_mean([c.db_fetch.mean  for c in tier_cells], weights),
        mean_decrypt_ms=_weighted_mean([c.decrypt.mean   for c in tier_cells], weights),
    )


def summarize_policy_run(report: BenchmarkReport) -> PolicyRunSummary:
    """Aggregate all result cells into a workload-level PolicyRunSummary.

    Pure function — does not rerun any benchmarks.  Works for all policy modes
    and for legacy mode (where ``tier_label`` is "n/a" for all cells).

    Latency values (mean, p50, p95) are iteration-count-weighted averages of
    per-cell Stats values across the full result set.
    """
    cells = report.results

    effective_policy_mode: Optional[str] = (
        report.policy_mode if report.policy_mode != "n/a" else None
    )

    if not cells:
        return PolicyRunSummary(
            policy_mode=effective_policy_mode,
            threshold_bytes=report.policy_threshold_bytes,
            total_records=0,
            total_plaintext_bytes=0,
            total_stored_bytes=0,
            overall_storage_amplification=0.0,
            escalated_records=0,
            escalated_record_fraction=0.0,
            escalated_plaintext_bytes=0,
            escalated_plaintext_byte_fraction=0.0,
            mean_end_to_end_ms=0.0,
            p50_end_to_end_ms=0.0,
            p95_end_to_end_ms=0.0,
            tier_summaries=[],
        )

    weights = [c.iterations for c in cells]
    total_records = sum(weights)
    total_plaintext_bytes = sum(c.iterations * c.size_bytes for c in cells)
    total_stored_bytes = sum(
        c.iterations * c.envelope.total_stored_bytes
        for c in cells if c.envelope is not None
    )
    escalated_records = sum(c.iterations for c in cells if c.escalated)
    escalated_plaintext_bytes = sum(
        c.iterations * c.size_bytes for c in cells if c.escalated
    )
    overall_amp = total_stored_bytes / total_plaintext_bytes if total_plaintext_bytes else 0.0

    # Per-tier summaries — preserve insertion order by sorting tier labels so
    # "baseline" always appears before "strong", with "n/a" last.
    _TIER_ORDER = {"baseline": 0, "strong": 1}
    tier_labels = sorted(
        set(c.tier_label for c in cells),
        key=lambda t: (_TIER_ORDER.get(t, 99), t),
    )
    tier_summaries = [
        _build_tier_summary(tier, [c for c in cells if c.tier_label == tier],
                            total_records, total_plaintext_bytes)
        for tier in tier_labels
    ]

    return PolicyRunSummary(
        policy_mode=effective_policy_mode,
        threshold_bytes=report.policy_threshold_bytes,
        total_records=total_records,
        total_plaintext_bytes=total_plaintext_bytes,
        total_stored_bytes=total_stored_bytes,
        overall_storage_amplification=overall_amp,
        escalated_records=escalated_records,
        escalated_record_fraction=escalated_records / total_records if total_records else 0.0,
        escalated_plaintext_bytes=escalated_plaintext_bytes,
        escalated_plaintext_byte_fraction=(
            escalated_plaintext_bytes / total_plaintext_bytes if total_plaintext_bytes else 0.0
        ),
        mean_end_to_end_ms=_weighted_mean([c.end_to_end.mean for c in cells], weights),
        p50_end_to_end_ms=_weighted_mean([c.end_to_end.p50  for c in cells], weights),
        p95_end_to_end_ms=_weighted_mean([c.end_to_end.p95  for c in cells], weights),
        tier_summaries=tier_summaries,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _stats_to_dict(s: Stats) -> Dict:
    return {
        "mean_ms": round(s.mean, 6),
        "min_ms":  round(s.minimum, 6),
        "max_ms":  round(s.maximum, 6),
        "p50_ms":  round(s.p50, 6),
        "p95_ms":  round(s.p95, 6),
        "p99_ms":  round(s.p99, 6),
    }


def _throughput_stats_to_dict(s: Stats) -> Dict:
    return {
        "mean_bps": round(s.mean, 6),
        "min_bps":  round(s.minimum, 6),
        "max_bps":  round(s.maximum, 6),
        "p50_bps":  round(s.p50, 6),
        "p95_bps":  round(s.p95, 6),
        "p99_bps":  round(s.p99, 6),
    }


def _envelope_to_dict(e: "EnvelopeMetrics") -> Dict:
    return {
        "plaintext_bytes":    e.plaintext_bytes,
        "ciphertext_bytes":   e.ciphertext_bytes,
        "nonce_bytes":        e.nonce_bytes,
        "tag_bytes":          e.tag_bytes,
        "wrapped_dek_bytes":  e.wrapped_dek_bytes,
        "eph_pubkey_bytes":   e.eph_pubkey_bytes,
        "salt_bytes":         e.salt_bytes,
        "hkdf_info_bytes":    e.hkdf_info_bytes,
        "aad_bytes":          e.aad_bytes,
        "pq_ct_bytes":        e.pq_ct_bytes,
        "signature_bytes":    e.signature_bytes,
        "header_bytes":       e.header_bytes,
        "overhead_bytes":     e.overhead_bytes,
        "total_stored_bytes": e.total_stored_bytes,
        "amplification":      round(e.total_stored_bytes / e.plaintext_bytes, 6) if e.plaintext_bytes else None,
    }


def _breakdown_to_dict(breakdown: Dict[str, Stats]) -> Dict:
    return {k: _stats_to_dict(v) for k, v in breakdown.items()}


def _throughput_breakdown_to_dict(breakdown: Dict[str, Stats]) -> Dict:
    return {k: _throughput_stats_to_dict(v) for k, v in breakdown.items()}


def _policy_tier_summary_to_dict(t: PolicyTierSummary) -> Dict:
    return {
        "tier_label":              t.tier_label,
        "record_count":            t.record_count,
        "record_fraction":         round(t.record_fraction, 6),
        "plaintext_bytes":         t.plaintext_bytes,
        "plaintext_byte_fraction": round(t.plaintext_byte_fraction, 6),
        "total_stored_bytes":      t.total_stored_bytes,
        "storage_amplification":   round(t.storage_amplification, 6),
        "mean_end_to_end_ms":      round(t.mean_end_to_end_ms, 6),
        "p50_end_to_end_ms":       round(t.p50_end_to_end_ms, 6),
        "p95_end_to_end_ms":       round(t.p95_end_to_end_ms, 6),
        "mean_encrypt_ms":         round(t.mean_encrypt_ms, 6),
        "mean_sign_ms":            round(t.mean_sign_ms, 6),
        "mean_verify_ms":          round(t.mean_verify_ms, 6),
        "mean_db_insert_ms":       round(t.mean_db_insert_ms, 6),
        "mean_db_fetch_ms":        round(t.mean_db_fetch_ms, 6),
        "mean_decrypt_ms":         round(t.mean_decrypt_ms, 6),
    }


def _policy_run_summary_to_dict(s: PolicyRunSummary) -> Dict:
    return {
        "policy_mode":                       s.policy_mode,
        "threshold_bytes":                   s.threshold_bytes,
        "total_records":                     s.total_records,
        "total_plaintext_bytes":             s.total_plaintext_bytes,
        "total_stored_bytes":                s.total_stored_bytes,
        "overall_storage_amplification":     round(s.overall_storage_amplification, 6),
        "escalated_records":                 s.escalated_records,
        "escalated_record_fraction":         round(s.escalated_record_fraction, 6),
        "escalated_plaintext_bytes":         s.escalated_plaintext_bytes,
        "escalated_plaintext_byte_fraction": round(s.escalated_plaintext_byte_fraction, 6),
        "mean_end_to_end_ms":                round(s.mean_end_to_end_ms, 6),
        "p50_end_to_end_ms":                 round(s.p50_end_to_end_ms, 6),
        "p95_end_to_end_ms":                 round(s.p95_end_to_end_ms, 6),
        "tier_summaries":                    [_policy_tier_summary_to_dict(t) for t in s.tier_summaries],
    }


def report_to_dict(report: BenchmarkReport) -> Dict:
    results = []
    for cell in report.results:
        results.append({
            "scheme":              cell.scheme,
            "pq_sig_id":           cell.pq_sig_id,
            "size_bytes":          cell.size_bytes,
            "size_label":          cell.size_label,
            "payload_name":        cell.payload_name,
            "iterations":          cell.iterations,
            "policy_mode":         cell.policy_mode,
            "threshold_bytes":     cell.threshold_bytes,
            "selected_kem_id":     cell.selected_kem_id,
            "selected_sig_id":     cell.selected_sig_id,
            "tier_label":          cell.tier_label,
            "escalated":           cell.escalated,
            "encrypt":             _stats_to_dict(cell.encrypt),
            "sign":                _stats_to_dict(cell.sign) if cell.sign else None,
            "db_insert":           _stats_to_dict(cell.db_insert),
            "db_fetch":            _stats_to_dict(cell.db_fetch),
            "verify":              _stats_to_dict(cell.verify) if cell.verify else None,
            "decrypt":             _stats_to_dict(cell.decrypt),
            "end_to_end":          _stats_to_dict(cell.end_to_end),
            "throughput_bytes_per_sec": _throughput_breakdown_to_dict(cell.throughput_bytes_per_sec),
            "envelope_bytes":      _envelope_to_dict(cell.envelope) if cell.envelope else None,
            "encrypt_breakdown":   _breakdown_to_dict(cell.encrypt_breakdown),
            "decrypt_breakdown":   _breakdown_to_dict(cell.decrypt_breakdown),
        })
    return {
        "run_at":                report.run_at,
        "host":                  report.host,
        "iterations":            report.iterations,
        "warmup_iterations":     report.warmup_iterations,
        "payload_source":        report.payload_source,
        "payload_selection":     report.payload_selection,
        "sizes_bytes":           report.sizes_bytes,
        "schemes":               report.schemes,
        "pq_kem_id":             report.pq_kem_id,
        "pq_sig_id":             report.pq_sig_id,
        "policy_mode":           report.policy_mode,
        "policy_threshold_bytes": report.policy_threshold_bytes,
        "policy_summary":        _policy_run_summary_to_dict(summarize_policy_run(report)),
        "results":               results,
    }


def save_json(report: BenchmarkReport, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    print(f"\n  Results saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the PQC encrypted database system."
    )
    p.add_argument(
        "--iterations", type=int, default=10,
        help="Number of measured iterations per (scheme, size) cell (default: 10).",
    )
    p.add_argument(
        "--warmup", type=int, default=2,
        help="Number of warmup iterations to discard before measuring (default: 2).",
    )
    p.add_argument(
        "--sizes", type=str, default="",
        help=(
            "Comma-separated record sizes in bytes, e.g. '1024,65536,1048576'. "
            "For synthetic payloads, defaults to: 1 KB, 4 KB, 16 KB, 64 KB, 256 KB, 1 MB. "
            "For real payloads, passing --sizes overrides the natural distribution sampling."
        ),
    )
    p.add_argument(
        "--schemes", type=str, default="classical,hybrid",
        help=(
            "Comma-separated list of schemes to benchmark. "
            "Valid values: classical, hybrid (default: classical,hybrid). "
            "Examples: --schemes classical   --schemes hybrid   --schemes classical,hybrid"
        ),
    )
    p.add_argument(
        "--pq-kem", type=str, default=stage2.DEFAULT_PQ_KEM_ID,
        help=(
            f"PQ KEM algorithm ID, used when 'hybrid' is in --schemes and no --policy-mode is set. "
            f"Supported: {', '.join(stage2.SUPPORTED_PQ_KEM_IDS)} "
            f"(default: {stage2.DEFAULT_PQ_KEM_ID})."
        ),
    )
    p.add_argument(
        "--pq-sig", type=str, default="none",
        help=(
            "PQ signature algorithm for ML-DSA envelope signing (legacy non-policy mode). "
            "'none' disables signing (default). "
            f"Supported: none, {', '.join(stage2.SUPPORTED_PQ_SIG_IDS)}. "
            "Ignored when --policy-mode is set (policy drives sig selection)."
        ),
    )
    p.add_argument(
        "--payload-source", type=str, default="synthetic", choices=PAYLOAD_SOURCES,
        help="Payload source: synthetic, faers, or all_ggvp (default: synthetic).",
    )
    p.add_argument(
        "--real-payload-count", type=int, default=None,
        help=(
            "When using a real payload source without --sizes, benchmark only this many real "
            "files sampled across the empirical size distribution. "
            "Omit to use all discovered files (default: use all)."
        ),
    )
    p.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Path to write JSON results (default: benchmark_results.json).",
    )
    p.add_argument(
        "--no-json", action="store_true",
        help="Skip writing JSON output file.",
    )
    p.add_argument(
        "--policy-mode", type=str, default=None, choices=POLICY_MODES,
        help=(
            "Per-record PQ parameter assignment policy. "
            f"Choices: {', '.join(POLICY_MODES)}. "
            "When set, overrides --pq-kem and --pq-sig with policy-driven per-record selection. "
            "adaptive_threshold uses --policy-threshold-bytes to split baseline vs strong profiles."
        ),
    )
    p.add_argument(
        "--policy-threshold-bytes", type=int, default=DEFAULT_THRESHOLD_BYTES,
        help=(
            f"Byte threshold for adaptive_threshold policy mode (default: {DEFAULT_THRESHOLD_BYTES} = 50 KiB). "
            "Records strictly larger than this threshold receive ML-KEM-1024 + ML-DSA-87; "
            "others receive ML-KEM-768 + ML-DSA-65."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate --schemes
    try:
        schemes = _validate_schemes(args.schemes)
    except ValueError as exc:
        import sys
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    import sys as _sys

    # Validate --policy-threshold-bytes
    if args.policy_threshold_bytes < 0:
        print("error: --policy-threshold-bytes must be >= 0", file=_sys.stderr)
        _sys.exit(1)

    # Validate --pq-kem only when hybrid is selected and no policy mode overrides it.
    pq_sig_id: Optional[str] = None
    if args.policy_mode is None:
        if "hybrid" in schemes:
            try:
                stage2.validate_pq_kem_id(args.pq_kem)
            except ValueError as exc:
                print(f"error: {exc}", file=_sys.stderr)
                _sys.exit(1)

        # Validate --pq-sig
        if args.pq_sig.lower() != "none":
            try:
                stage2.validate_pq_sig_id(args.pq_sig)
                pq_sig_id = args.pq_sig
            except ValueError as exc:
                print(f"error: {exc}", file=_sys.stderr)
                _sys.exit(1)

    sizes = (
        [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
        if args.sizes
        else None
    )

    print()
    print("  " + "═" * 60)
    print("  PQC ENCRYPTED DATABASE — PERFORMANCE BENCHMARK")
    print("  " + "═" * 60)
    print(f"  Iterations per cell : {args.iterations}  (+ {args.warmup} warmup)")
    print(f"  Payload source      : {args.payload_source}")
    if sizes:
        print(f"  Payload selection   : explicit size buckets")
        print(f"  Record sizes        : {', '.join(_human_size(s) for s in sizes)}")
    elif args.payload_source == "synthetic":
        print(f"  Payload selection   : synthetic default size buckets")
        print(f"  Record sizes        : {', '.join(_human_size(s) for s in DEFAULT_SIZES_BYTES)}")
    else:
        if args.real_payload_count is None:
            print(f"  Payload selection   : all real files")
        else:
            print(f"  Payload selection   : real file distribution sample ({args.real_payload_count} files)")
        print(f"  Record sizes        : derived from real files")
    if args.policy_mode:
        print(f"  Policy mode         : {args.policy_mode}  (threshold: {args.policy_threshold_bytes} B)")
        print(f"  Schemes             : {', '.join(schemes)}")
    else:
        if "hybrid" in schemes:
            print(f"  Schemes             : {', '.join(schemes)}  (PQ KEM: {args.pq_kem})")
        else:
            print(f"  Schemes             : {', '.join(schemes)}")
        print(f"  Signing             : {pq_sig_id if pq_sig_id else 'none (unsigned)'}")
    print(f"  Output              : {'(suppressed)' if args.no_json else args.output}")
    print()

    cfg = db.config_from_env()
    print(f"  Connecting to PostgreSQL at {cfg.host}:{cfg.port}/{cfg.dbname} ...", end=" ", flush=True)
    with db.connect(cfg) as conn:
        db.init_schema(conn)
        print("connected.")
        print()

        report = run_benchmark(
            conn,
            sizes_bytes=sizes,
            iterations=args.iterations,
            warmup=args.warmup,
            pq_kem_id=args.pq_kem,
            pq_sig_id=pq_sig_id,
            payload_source=args.payload_source,
            real_payload_count=args.real_payload_count,
            schemes=schemes,
            policy_mode=args.policy_mode,
            policy_threshold_bytes=args.policy_threshold_bytes,
        )

    print()
    print(format_report(report))

    if not args.no_json:
        save_json(report, args.output)


if __name__ == "__main__":
    main()
