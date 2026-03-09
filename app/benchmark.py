"""
app/benchmark.py
================
Performance benchmark for the PQC encrypted database system.

Measures encryption, DB insert, DB fetch, decryption, and end-to-end
latency for both classical (X25519 + AES-256-GCM) and hybrid PQC
(X25519 + ML-KEM-768 + AES-256-GCM) encryption schemes across record
sizes ranging from 1 KB to 1 MB.

Run inside Docker:
    python -m app.benchmark

Run with options:
    python -m app.benchmark --iterations 20 --warmup 3 --output results.json
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
from typing import Dict, List, Tuple

from . import db, stage2
from .service import ServiceContext, put_record, get_record


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

TENANT_ID = "benchmark-tenant"
KEY_ID    = "benchmark-key-v1"


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
    plaintext_bytes:   int
    ciphertext_bytes:  int   # AES-GCM: equal to plaintext (no expansion)
    nonce_bytes:       int   # always 12
    tag_bytes:         int   # always 16
    wrapped_dek_bytes: int   # always 40 (32-byte DEK + 8-byte AES-KW overhead)
    eph_pubkey_bytes:  int   # always 32  (X25519)
    salt_bytes:        int   # always 32
    hkdf_info_bytes:   int   # small constant (~19–21 B)
    aad_bytes:         int   # per-record AAD
    pq_ct_bytes:       int   # ML-KEM ciphertext (0 for classical)
    header_bytes:      int   # nonce+tag+dek+pk+salt+info+aad (fixed crypto metadata)
    overhead_bytes:    int   # header + pq_ct  (everything except the payload)
    total_stored_bytes: int  # ciphertext + overhead

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
    db_insert_ms: float
    db_fetch_ms: float
    decrypt_ms: float
    end_to_end_ms: float
    envelope: Dict = field(default=None)       # type: ignore[assignment]
    encrypt_ops: Dict = field(default_factory=dict)
    decrypt_ops: Dict = field(default_factory=dict)


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
    iterations: int
    encrypt: Stats
    db_insert: Stats
    db_fetch: Stats
    decrypt: Stats
    end_to_end: Stats
    envelope: EnvelopeMetrics = field(default=None)       # type: ignore[assignment]
    encrypt_breakdown: Dict[str, Stats] = field(default_factory=dict)
    decrypt_breakdown: Dict[str, Stats] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Top-level container for the full benchmark run."""
    run_at: str
    host: str
    iterations: int
    warmup_iterations: int
    sizes_bytes: List[int]
    schemes: List[str]
    pq_kem_id: str
    results: List[CellResult] = field(default_factory=list)


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


def _run_single(
    conn,
    *,
    scheme: str,
    plaintext: bytes,
    aad: bytes,
    classical_keys,
    hybrid_keys,
    pq_kem_id: str,
) -> Sample:
    """Execute one put+get round-trip and return raw timing Sample."""
    ctx = ServiceContext(tenant_id=TENANT_ID, key_id=KEY_ID, version=1)

    t_wall_start = time.perf_counter()

    if scheme == "classical":
        record_id, put_timings = put_record(
            conn,
            ctx=ctx,
            plaintext=plaintext,
            aad=aad,
            scheme="classical",
            recipient_keys=classical_keys,
        )
        _, get_timings = get_record(
            conn,
            tenant_id=TENANT_ID,
            record_id=record_id,
            recipient_keys=classical_keys,
        )
    else:
        record_id, put_timings = put_record(
            conn,
            ctx=ctx,
            plaintext=plaintext,
            aad=aad,
            scheme="hybrid",
            recipient_keys=hybrid_keys,
            pq_kem_id=pq_kem_id,
        )
        _, get_timings = get_record(
            conn,
            tenant_id=TENANT_ID,
            record_id=record_id,
            recipient_keys=hybrid_keys,
            pq_sk=hybrid_keys.pq_sk,
        )

    wall_ms = (time.perf_counter() - t_wall_start) * 1000.0

    return Sample(
        size_bytes=len(plaintext),
        scheme=scheme,
        encrypt_ms=put_timings["encrypt"],
        db_insert_ms=put_timings["db_insert"],
        db_fetch_ms=get_timings["db_fetch"],
        decrypt_ms=get_timings["decrypt"],
        end_to_end_ms=wall_ms,
        envelope=put_timings.get("envelope"),
        encrypt_ops=put_timings.get("encrypt_ops") or {},
        decrypt_ops=get_timings.get("decrypt_ops") or {},
    )


def run_benchmark(
    conn,
    *,
    sizes_bytes: List[int] = DEFAULT_SIZES_BYTES,
    iterations: int = 10,
    warmup: int = 2,
    pq_kem_id: str = stage2.DEFAULT_PQ_KEM_ID,
) -> BenchmarkReport:
    """Run the full benchmark and return a BenchmarkReport."""
    print()
    print("  Generating key pairs...", end=" ", flush=True)
    classical_keys = stage2.generate_classical_recipient_keys()
    hybrid_keys    = stage2.generate_hybrid_recipient_keys(pq_kem_id)
    print("done.")

    report = BenchmarkReport(
        run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        host=os.uname().nodename,
        iterations=iterations,
        warmup_iterations=warmup,
        sizes_bytes=sizes_bytes,
        schemes=SCHEMES,
        pq_kem_id=pq_kem_id,
    )

    # Randomise the order sizes are measured to avoid confounding thermal/frequency
    # state with the size dimension.  Crucially, classical and hybrid are measured
    # in an interleaved fashion within each size so both schemes always experience
    # the same CPU/system state on every iteration, eliminating the between-cell
    # variance that caused spurious classical "underperformance" at large sizes.
    size_order = list(sizes_bytes)
    random.shuffle(size_order)

    total_sizes = len(size_order)

    for size_num, size in enumerate(size_order, 1):
        label = _human_size(size)
        print(
            f"  [{size_num:>2}/{total_sizes}] size={label:<7}"
            f"  warmup={warmup}  iterations={iterations}  ...",
            end=" ",
            flush=True,
        )

        plaintext = os.urandom(size)
        aad_cl = f"benchmark|scheme=classical|size={size}".encode()
        aad_hy = f"benchmark|scheme=hybrid|size={size}".encode()

        # Clear once per size block so both schemes share the same starting
        # table state.
        _clear_benchmark_data(conn)

        # Warmup both schemes (interleaved) — results discarded.
        for _ in range(warmup):
            _run_single(conn, scheme="classical", plaintext=plaintext, aad=aad_cl,
                        classical_keys=classical_keys, hybrid_keys=hybrid_keys, pq_kem_id=pq_kem_id)
            _run_single(conn, scheme="hybrid",    plaintext=plaintext, aad=aad_hy,
                        classical_keys=classical_keys, hybrid_keys=hybrid_keys, pq_kem_id=pq_kem_id)

        # Measured rounds: alternate which scheme goes first each iteration so
        # any residual ordering bias averages out.
        samples_cl: List[Sample] = []
        samples_hy: List[Sample] = []
        for i in range(iterations):
            if i % 2 == 0:
                samples_cl.append(_run_single(conn, scheme="classical", plaintext=plaintext, aad=aad_cl,
                                              classical_keys=classical_keys, hybrid_keys=hybrid_keys, pq_kem_id=pq_kem_id))
                samples_hy.append(_run_single(conn, scheme="hybrid",    plaintext=plaintext, aad=aad_hy,
                                              classical_keys=classical_keys, hybrid_keys=hybrid_keys, pq_kem_id=pq_kem_id))
            else:
                samples_hy.append(_run_single(conn, scheme="hybrid",    plaintext=plaintext, aad=aad_hy,
                                              classical_keys=classical_keys, hybrid_keys=hybrid_keys, pq_kem_id=pq_kem_id))
                samples_cl.append(_run_single(conn, scheme="classical", plaintext=plaintext, aad=aad_cl,
                                              classical_keys=classical_keys, hybrid_keys=hybrid_keys, pq_kem_id=pq_kem_id))

        for scheme, samples in [("classical", samples_cl), ("hybrid", samples_hy)]:
            # Envelope sizes are deterministic per (scheme, size); use first sample.
            env = None
            if samples[0].envelope:
                env = EnvelopeMetrics.from_dict(size, samples[0].envelope)

            cell = CellResult(
                scheme=scheme,
                size_bytes=size,
                size_label=label,
                iterations=iterations,
                encrypt=Stats.from_values([s.encrypt_ms    for s in samples]),
                db_insert=Stats.from_values([s.db_insert_ms for s in samples]),
                db_fetch=Stats.from_values([s.db_fetch_ms   for s in samples]),
                decrypt=Stats.from_values([s.decrypt_ms    for s in samples]),
                end_to_end=Stats.from_values([s.end_to_end_ms for s in samples]),
                envelope=env,
                encrypt_breakdown=_aggregate_ops([s.encrypt_ops for s in samples if s.encrypt_ops]),
                decrypt_breakdown=_aggregate_ops([s.decrypt_ops for s in samples if s.decrypt_ops]),
            )
            report.results.append(cell)

        # Print one summary line per size showing both schemes side-by-side.
        # Retrieve the two cells just added (they may not be at a fixed index
        # because results are appended in scheme loop order).
        cl_cell = next(c for c in reversed(report.results) if c.scheme == "classical" and c.size_bytes == size)
        hy_cell = next(c for c in reversed(report.results) if c.scheme == "hybrid"    and c.size_bytes == size)
        print(
            f"classical p50={cl_cell.end_to_end.p50:>7.2f} ms"
            f"  hybrid p50={hy_cell.end_to_end.p50:>7.2f} ms"
            f"  (mean: {cl_cell.end_to_end.mean:>7.2f} / {hy_cell.end_to_end.mean:>7.2f})"
        )

    # Sort results into a canonical (scheme order, then ascending size) order so
    # report tables and charts receive cells in a predictable sequence regardless
    # of the randomised measurement order.
    scheme_order = {s: i for i, s in enumerate(SCHEMES)}
    report.results.sort(key=lambda c: (scheme_order.get(c.scheme, 99), c.size_bytes))

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_SIZE_W   = 8    # width of size column
_DIVIDER  = "─"
_HEAVY    = "═"


def _fmt_cell(stat: Stats) -> str:
    """'mean / p95' in a fixed-width column."""
    return f"{stat.mean:>7.3f} / {stat.p95:<7.3f}"


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
    headers = ["Size", "Plaintext", "Ciphertext", "Header", "PQ-CT", "Overhead", "Total-Stored", "Amplification"]
    sep_line = _DIVIDER * (sum(col_w for _ in headers) + 2 * (len(headers) - 1))

    lines = [
        sep_line,
        "  ".join(f"{h:<{col_w}}" for h in headers),
        "  ".join(f"{'(bytes)':<{col_w}}" if h != "Amplification" else f"{'(total/plain)':<{col_w}}" for h in headers),
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

    # ── Title block ──────────────────────────────────────────────────────────
    out.append(_HEAVY * W)
    out.append(f"  PQC ENCRYPTED DATABASE — PERFORMANCE BENCHMARK REPORT")
    out.append(f"  Run at  : {report.run_at}   Host: {report.host}")
    out.append(f"  Schemes : classical (X25519-ECDH + AES-256-GCM)")
    out.append(f"          : hybrid   (X25519-ECDH + {report.pq_kem_id} + AES-256-GCM)")
    out.append(f"  Sizes   : {', '.join(_human_size(s) for s in report.sizes_bytes)}")
    out.append(f"  Runs    : {report.iterations} measured + {report.warmup_iterations} warmup per cell")
    out.append(f"  Metric  : All latencies in milliseconds (ms).  'mean / p95' = average / 95th-percentile")
    out.append(_HEAVY * W)

    by_scheme: Dict[str, List[CellResult]] = {}
    for cell in report.results:
        by_scheme.setdefault(cell.scheme, []).append(cell)
    for cells in by_scheme.values():
        cells.sort(key=lambda c: c.size_bytes)

    # ── Per-scheme tables ─────────────────────────────────────────────────────
    scheme_labels = {
        "classical": "CLASSICAL ENCRYPTION  (X25519-ECDH + AES-256-GCM)",
        "hybrid":    f"HYBRID PQC ENCRYPTION  (X25519-ECDH + {report.pq_kem_id} + AES-256-GCM)",
    }

    for scheme in report.schemes:
        cells = by_scheme.get(scheme, [])
        if not cells:
            continue
        out.append("")
        out.append(f"  SCHEME: {scheme_labels.get(scheme, scheme.upper())}")
        out.append(_table_for_scheme(cells))

    # ── Storage / envelope size tables ───────────────────────────────────────
    cl_cells = by_scheme.get("classical", [])
    hy_cells = by_scheme.get("hybrid",    [])

    for scheme in report.schemes:
        cells = by_scheme.get(scheme, [])
        if cells and cells[0].envelope is not None:
            out.append("")
            out.append(f"  STORAGE BREAKDOWN — {scheme.upper()}")
            out.append(f"  Header = nonce + tag + wrapped-DEK + eph-pubkey + salt + hkdf-info + AAD  |  Amplification = total-stored / plaintext")
            out.append(_storage_table(cells))

    if cl_cells and hy_cells and len(cl_cells) == len(hy_cells):
        if cl_cells[0].envelope is not None and hy_cells[0].envelope is not None:
            out.append("")
            out.append(f"  STORAGE OVERHEAD COMPARISON  (Hybrid PQC vs Classical)")
            out.append(f"  Delta = bytes added by the PQ KEM ciphertext layer.")
            out.append(_storage_overhead_comparison(cl_cells, hy_cells))

    # ── Overhead comparison ───────────────────────────────────────────────────

    if cl_cells and hy_cells and len(cl_cells) == len(hy_cells):
        out.append("")
        out.append(f"  OVERHEAD COMPARISON  (Hybrid PQC vs Classical)")
        out.append(f"  Positive values indicate additional latency introduced by the PQC KEM layer.")
        out.append(_overhead_table(cl_cells, hy_cells))

    # ── Crypto operation breakdown ────────────────────────────────────────────
    out.append("")
    out.append(f"  CRYPTO OPERATION BREAKDOWN  (mean ms per primitive, averaged over {report.iterations} iterations)")
    out.append(f"  Encrypt-side ops: t_dek_gen, t_ecdh, t_pq_encaps (hybrid only), t_hkdf, t_wrap, t_aes_gcm")
    out.append(f"  Decrypt-side ops: t_ecdh, t_pq_decaps (hybrid only), t_hkdf, t_unwrap, t_aes_gcm")
    out.append(f"  Note: sub-op means will slightly exceed total encrypt/decrypt due to perf_counter() call overhead (~0.1–0.3 µs each).")

    _OP_LABELS = {
        "t_dek_gen_ms":   "DEK-gen",
        "t_aes_gcm_ms":   "AES-GCM",
        "t_eph_keygen_ms":"Eph-keygen",
        "t_ecdh_ms":      "X25519-ECDH",
        "t_pq_encaps_ms": "ML-KEM-encaps",
        "t_pq_decaps_ms": "ML-KEM-decaps",
        "t_hkdf_ms":      "HKDF-SHA256",
        "t_wrap_ms":      "AES-KW-wrap",
        "t_unwrap_ms":    "AES-KW-unwrap",
        "t_total_crypto_ms": "TOTAL-CRYPTO",
    }
    _ENC_KEYS = ["t_dek_gen_ms", "t_aes_gcm_ms", "t_eph_keygen_ms", "t_ecdh_ms",
                 "t_pq_encaps_ms", "t_hkdf_ms", "t_wrap_ms", "t_total_crypto_ms"]
    _DEC_KEYS = ["t_ecdh_ms", "t_pq_decaps_ms", "t_hkdf_ms", "t_unwrap_ms",
                 "t_aes_gcm_ms", "t_total_crypto_ms"]

    for phase, keys in [("ENCRYPT", _ENC_KEYS), ("DECRYPT", _DEC_KEYS)]:
        col_w = 15
        op_labels = [_OP_LABELS.get(k, k) for k in keys]
        size_col_w = 8
        sep = _DIVIDER * (size_col_w + 2 + len(op_labels) * (col_w + 2))
        out.append("")
        out.append(f"  {phase} breakdown  (mean ms)")
        out.append("  " + sep)
        header = f"  {'Scheme':<10} {'Size':<{size_col_w}}  " + "  ".join(f"{lbl:<{col_w}}" for lbl in op_labels)
        out.append(header)
        out.append("  " + sep)
        for cell in report.results:
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
        ("db_insert",  "DB Insert"),
        ("db_fetch",   "DB Fetch"),
        ("decrypt",    "Decrypt"),
        ("end_to_end", "End-to-End"),
    ]

    for cell in report.results:
        first = True
        for attr, label in metric_map:
            s: Stats = getattr(cell, attr)
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

    out.append(_HEAVY * W)
    return "\n".join(out)


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
        "header_bytes":       e.header_bytes,
        "overhead_bytes":     e.overhead_bytes,
        "total_stored_bytes": e.total_stored_bytes,
        "amplification":      round(e.total_stored_bytes / e.plaintext_bytes, 6) if e.plaintext_bytes else None,
    }


def _breakdown_to_dict(breakdown: Dict[str, Stats]) -> Dict:
    return {k: _stats_to_dict(v) for k, v in breakdown.items()}


def report_to_dict(report: BenchmarkReport) -> Dict:
    results = []
    for cell in report.results:
        results.append({
            "scheme":              cell.scheme,
            "size_bytes":          cell.size_bytes,
            "size_label":          cell.size_label,
            "iterations":          cell.iterations,
            "encrypt":             _stats_to_dict(cell.encrypt),
            "db_insert":           _stats_to_dict(cell.db_insert),
            "db_fetch":            _stats_to_dict(cell.db_fetch),
            "decrypt":             _stats_to_dict(cell.decrypt),
            "end_to_end":          _stats_to_dict(cell.end_to_end),
            "envelope_bytes":      _envelope_to_dict(cell.envelope) if cell.envelope else None,
            "encrypt_breakdown":   _breakdown_to_dict(cell.encrypt_breakdown),
            "decrypt_breakdown":   _breakdown_to_dict(cell.decrypt_breakdown),
        })
    return {
        "run_at":            report.run_at,
        "host":              report.host,
        "iterations":        report.iterations,
        "warmup_iterations": report.warmup_iterations,
        "sizes_bytes":       report.sizes_bytes,
        "schemes":           report.schemes,
        "pq_kem_id":         report.pq_kem_id,
        "results":           results,
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
            "Defaults to: 1 KB, 4 KB, 16 KB, 64 KB, 256 KB, 1 MB."
        ),
    )
    p.add_argument(
        "--pq-kem", type=str, default=stage2.DEFAULT_PQ_KEM_ID,
        help=f"PQ KEM algorithm ID (default: {stage2.DEFAULT_PQ_KEM_ID}).",
    )
    p.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Path to write JSON results (default: benchmark_results.json).",
    )
    p.add_argument(
        "--no-json", action="store_true",
        help="Skip writing JSON output file.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    sizes = (
        [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
        if args.sizes
        else DEFAULT_SIZES_BYTES
    )

    print()
    print("  " + "═" * 60)
    print("  PQC ENCRYPTED DATABASE — PERFORMANCE BENCHMARK")
    print("  " + "═" * 60)
    print(f"  Iterations per cell : {args.iterations}  (+ {args.warmup} warmup)")
    print(f"  Record sizes        : {', '.join(_human_size(s) for s in sizes)}")
    print(f"  Schemes             : classical, hybrid ({args.pq_kem})")
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
        )

    print()
    print(format_report(report))

    if not args.no_json:
        save_json(report, args.output)


if __name__ == "__main__":
    main()
