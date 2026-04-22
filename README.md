# PQC Database Evaluation

Benchmarks **classical** (X25519 + AES-256-GCM) versus **hybrid post-quantum** (X25519 + ML-KEM + AES-256-GCM) envelope encryption in a PostgreSQL-backed service, measuring end-to-end latency, per-primitive timing, and storage overhead across record sizes from 1 KB to 1 MB.

The experiment supports a **KEM parameter sweep** across all three standardized ML-KEM parameter sets (ML-KEM-512, ML-KEM-768, ML-KEM-1024), each runnable as a separate, independent configuration against the classical baseline.

The experiment also runs against real genomic (1000 Genomes VCF) and medical (FDA FAERS) datasets to characterize performance under realistic payload size distributions, not just synthetic round numbers.

---

## Background

Post-quantum cryptography (PQC) introduces new key encapsulation mechanisms (KEMs) intended to remain secure against cryptographically relevant quantum computers. ML-KEM (standardized in NIST FIPS 203, formerly known as Kyber) defines three parameter sets targeting NIST security levels 1, 3, and 5. A **hybrid** design combines ML-KEM with classical X25519 so security holds as long as either primitive is unbroken — the practical migration path for systems with existing key material.

This project measures the real-world overhead of hybrid encryption across a full database round-trip, and compares overhead across all three ML-KEM parameter sets.

---

## KEM Parameter Study

The benchmark supports three hybrid configurations in addition to the classical baseline:

| Configuration | Scheme | KEM | NIST Level | PQ CT size |
|---------------|--------|-----|------------|------------|
| Classical baseline | classical | X25519-ECDH | — | — |
| Hybrid ML-KEM-512 | hybrid | X25519-ECDH + ML-KEM-512 | 1 | ~768 B |
| Hybrid ML-KEM-768 | hybrid | X25519-ECDH + ML-KEM-768 | 3 | ~1,088 B |
| Hybrid ML-KEM-1024 | hybrid | X25519-ECDH + ML-KEM-1024 | 5 | ~1,568 B |

Each configuration is independently runnable via the `--schemes` and `--pq-kem` flags (see [Benchmark Options](#benchmark-options) and [Running Each Configuration Separately](#running-each-configuration-separately)).

---

## ML-DSA Signature Study

The benchmark includes a second experiment dimension: **envelope signing with ML-DSA** (standardized in NIST FIPS 204, historically known as Dilithium).

### What signing adds

Unsigned mode stores only the encrypted envelope (ciphertext + KEM header). Signed mode additionally:

1. Constructs a **canonical commitment** over all security-relevant envelope fields.
2. Signs that commitment with an ML-DSA key pair.
3. Stores the `sig_alg_id`, `signature`, and `sig_key_id` alongside the row.

On the read path, signature verification is performed **before decryption**. Decryption is only attempted if verification succeeds.

### Supported ML-DSA parameter sets

| Parameter set | NIST Security Level | Signature size | Signing key | Verify key |
|--------------|---------------------|---------------|-------------|------------|
| ML-DSA-44    | 2                   | 2,420 B       | 2,528 B     | 1,312 B    |
| ML-DSA-65    | 3                   | 3,309 B       | 4,032 B     | 1,952 B    |
| ML-DSA-87    | 5                   | 4,627 B       | 4,896 B     | 2,592 B    |

### What the signature covers

The commitment includes (in this order):

- Domain separator (`pqc-db-eval:envelope-commitment:v1`)
- `version`, `scheme`, `payload_cipher`, `kem_id`, `kdf_id`, `wrap_id`
- `ciphertext`, `nonce`, `tag`, `aad`
- `wrapped_dek`, `ephemeral_pubkey`, `salt`, `hkdf_info`
- `pq_kem_id`, `pq_ciphertext` (empty when unsigned or classical)

Encoding is length-prefixed (4-byte big-endian length + value bytes) for all fields, making the serialization canonical and unambiguous. See `stage2.build_envelope_commitment()` for the exact encoding.

Modifying **any** of the above fields invalidates the signature.

### Effect on measured stages

When signing is enabled, two additional stages appear in the benchmark output:

| Stage | What it measures |
|-------|-----------------|
| `sign` | ML-DSA sign over the canonical commitment after encrypt |
| `verify` | ML-DSA verify before decrypt; decryption is skipped on failure |

The `sign` stage is included in the write-path wall clock. The `verify` stage is included in the read-path wall clock.

### Storage impact

The signature is stored as a `BYTEA` column (`signature`) alongside `sig_alg_id` and `sig_key_id`. Its size is added to `overhead_bytes` and `total_stored_bytes` in the envelope metrics. The `Sig` column in the storage breakdown table shows this cost explicitly:

| Parameter set | Signature overhead |
|--------------|--------------------|
| ML-DSA-44    | +2,420 B per record |
| ML-DSA-65    | +3,309 B per record |
| ML-DSA-87    | +4,627 B per record |

For large payloads (≥ 64 KB), the signature overhead becomes a small fraction of total storage. For small payloads (1 KB), it dominates.

### Key rotation and signatures

Rotation changes `eph_pubkey`, `salt`, `pq_ct`, and `wrapped_dek` — all fields covered by the commitment. The existing signature becomes invalid after rotation. Two behaviors are supported:

- **Default (no new signing key):** signature columns are cleared to NULL; the rotated record becomes unsigned.
- **With `new_sig_keys`:** the envelope is re-signed after rotation and the new signature is stored.

### Running the signed experiment

**Unsigned (baseline):**
```bash
BENCHMARK_ARGS="--schemes hybrid --pq-sig none --output /app/results/bench_unsigned.json" \
  docker compose --profile benchmark run benchmark
```

**Hybrid + ML-DSA-44:**
```bash
BENCHMARK_ARGS="--schemes hybrid --pq-kem ML-KEM-768 --pq-sig ML-DSA-44 --output /app/results/bench_mldsa44.json" \
  docker compose --profile benchmark run benchmark
```

**Hybrid + ML-DSA-65:**
```bash
BENCHMARK_ARGS="--schemes hybrid --pq-kem ML-KEM-768 --pq-sig ML-DSA-65 --output /app/results/bench_mldsa65.json" \
  docker compose --profile benchmark run benchmark
```

**Hybrid + ML-DSA-87:**
```bash
BENCHMARK_ARGS="--schemes hybrid --pq-kem ML-KEM-768 --pq-sig ML-DSA-87 --output /app/results/bench_mldsa87.json" \
  docker compose --profile benchmark run benchmark
```

**Combined KEM + DSA (all four in one loop):**
```bash
for SIG in none ML-DSA-44 ML-DSA-65 ML-DSA-87; do
  BENCHMARK_ARGS="--schemes hybrid --pq-kem ML-KEM-768 --pq-sig ${SIG} \
    --output /app/results/bench_mldsa_${SIG//-/_}.json" \
    docker compose --profile benchmark run benchmark
done
```

---

## Policy-Driven Parameter Selection

`app/policy.py` implements a **workload-aware PQ parameter assignment policy** that selects between two security profiles on a per-record basis, without requiring application-level changes to decide which records need stronger cryptographic protection.

### Motivation

Not all records in a biomedical database carry the same sensitivity or size. A genomic VCF record can exceed 10 MiB, while a typical FAERS adverse-event record is under 100 KB. Applying ML-KEM-1024 + ML-DSA-87 to every record may be unnecessarily expensive for small records, while applying ML-KEM-768 + ML-DSA-65 to every record may leave large, high-value records under-protected relative to available NIST security levels.

The policy module lets the benchmark evaluate a size-threshold rule: records above a configurable byte threshold receive the **strong** profile (ML-KEM-1024 + ML-DSA-87); all others receive the **baseline** profile (ML-KEM-768 + ML-DSA-65).

### Policy profiles

| Profile | KEM | DSA | NIST Level |
|---------|-----|-----|------------|
| Baseline | ML-KEM-768 | ML-DSA-65 | 3 |
| Strong   | ML-KEM-1024 | ML-DSA-87 | 5 |

### Policy modes

Three modes are exposed via `--policy-mode`:

| Mode | Description |
|------|-------------|
| `uniform_baseline` | Every record uses the baseline profile regardless of size |
| `uniform_strong`   | Every record uses the strong profile regardless of size |
| `adaptive_threshold` | Records > `--policy-threshold-bytes` use the strong profile; all others use baseline |

When `--policy-mode` is set, the `--pq-kem` and `--pq-sig` flags are ignored — the policy drives per-record key selection instead.

### Key API (`app/policy.py`)

```python
from app.policy import ThresholdPolicyConfig, select_policy_for_record

config = ThresholdPolicyConfig(threshold_bytes=50 * 1024)  # 50 KiB
decision = select_policy_for_record(record_size_bytes=len(plaintext), config=config)
# decision.kem_id   → "ML-KEM-768" or "ML-KEM-1024"
# decision.sig_id   → "ML-DSA-65"  or "ML-DSA-87"
# decision.tier_label → "baseline" or "strong"
# decision.escalated  → False or True
```

The policy layer is pure — it performs no cryptographic operations and has no database dependency.

### Running policy experiments

**Uniform baseline (all records at NIST Level 3):**
```bash
BENCHMARK_ARGS="--schemes hybrid --policy-mode uniform_baseline \
  --output /app/results/bench_policy_baseline.json" \
  docker compose --profile benchmark run benchmark
```

**Uniform strong (all records at NIST Level 5):**
```bash
BENCHMARK_ARGS="--schemes hybrid --policy-mode uniform_strong \
  --output /app/results/bench_policy_strong.json" \
  docker compose --profile benchmark run benchmark
```

**Adaptive threshold (50 KiB default):**
```bash
BENCHMARK_ARGS="--schemes hybrid --policy-mode adaptive_threshold \
  --output /app/results/bench_policy_adaptive.json" \
  docker compose --profile benchmark run benchmark
```

**Adaptive threshold with custom cutoff (1 MiB):**
```bash
BENCHMARK_ARGS="--schemes hybrid --policy-mode adaptive_threshold \
  --policy-threshold-bytes 1048576 \
  --output /app/results/bench_policy_adaptive_1mib.json" \
  docker compose --profile benchmark run benchmark
```

### Full policy pilot experiment

`run_policy_pilot.sh` orchestrates the complete policy evaluation: 9 benchmark jobs (3 workloads × 3 policy modes) run sequentially, then `visualize.py` generates charts for each run:

```bash
./run_policy_pilot.sh
# Optional environment overrides:
# ITERATIONS=10 FAERS_COUNT=50 GGVP_COUNT=20 ./run_policy_pilot.sh
```

Results land in `results/policy_pilot/<timestamp>/`:

```
json/     — one JSON per job (e.g. faers_adaptive_threshold.json)
logs/     — stdout+stderr per job
plots/    — per-job chart directories
manifest.txt — run parameters and job list
```

### Policy figures

`plot_policy_figures.py` generates a 6-figure comparison suite from the pilot output:

```bash
python plot_policy_figures.py \
    --pilot-dir  results/policy_pilot/<timestamp>/json \
    --output-dir results/policy_pilot/<timestamp>/policy_figures
```

| Figure | Description |
|--------|-------------|
| `01_cost_vs_coverage.png` | Latency and storage cost vs byte coverage by policy mode |
| `02_escalation_coverage_bars.png` | Escalated records vs escalated bytes (adaptive mode) |
| `03_policy_cost_comparison.png` | Latency and storage amplification by policy mode |
| `04_record_size_vs_tier.png` | Per-record tier assignment under adaptive policy |
| `05_per_tier_latency_adaptive.png` | Baseline-tier vs strong-tier latency inside adaptive |
| `06_time_composition.png` | Stacked component latencies by policy mode |

### Policy summary in benchmark output

When a policy mode is active, the JSON output includes a `policy_summary` block and the text report appends a **POLICY TIER SUMMARY** table showing escalation rates, per-tier storage amplification, and weighted-mean latencies:

```json
"policy_summary": {
  "policy_mode": "adaptive_threshold",
  "threshold_bytes": 51200,
  "total_records": 60,
  "escalated_records": 10,
  "escalated_record_fraction": 0.1667,
  "overall_storage_amplification": 1.002341,
  ...
}
```

---

## Project Structure

```
pqc-db-eval/
├── app/
│   ├── stage2.py       # Crypto layer: classical + hybrid envelope encryption
│   ├── stage3.py       # Integration smoke tests (DB round-trips, all KEM variants)
│   ├── db.py           # PostgreSQL schema, insert/fetch/update helpers
│   ├── service.py      # High-level put_record / get_record / rotate API
│   ├── benchmark.py    # Benchmark runner, statistics, report formatter
│   └── policy.py       # Workload-aware PQ parameter assignment policy
├── tests/
│   ├── test_policy.py             # Unit tests for policy selection logic
│   ├── test_benchmark_policy.py   # Integration tests for policy in benchmark harness
│   └── test_policy_summary.py     # Tests for PolicyRunSummary aggregation
├── scripts/
│   ├── wait-for-postgres.sh  # Docker startup healthcheck
│   └── roles.sql             # Least-privilege DB role (run once as superuser)
├── Data/               # Real payload datasets (not committed)
│   ├── ALL_GGVP.chr*.vcf.gz  # 1000 Genomes Project VCF files (+ .tbi indexes)
│   └── faers_records/*.json  # FDA FAERS adverse event records
├── results/            # Benchmark JSON output + generated charts
├── visualize.py                      # 8-chart suite from raw benchmark JSON
├── aggregate_benchmark_results.py    # Bin raw results into quantile summaries
├── visualize_aggregated_benchmark.py # Multi-chart suite from aggregated JSON
├── plot_data_size_distributions.py   # Histogram of real payload sizes
├── plot_derived_throughput.py        # Throughput comparison (bytes/sec)
├── plot_storage_footprint_totals.py  # Total storage per scheme
├── plot_policy_figures.py            # Policy-comparison charts (6-figure suite)
├── render_storage_footprint_table.py # ASCII storage metrics table
├── summarize_real_payloads.py        # Summarize real dataset properties
├── run_policy_pilot.sh               # End-to-end policy pilot experiment runner
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

**Prerequisites:** Docker and Docker Compose.

### 1. Run integration smoke tests

Starts PostgreSQL, then runs classical and hybrid DB round-trips (all three ML-KEM parameter sets) with tenant isolation, tamper detection, soft delete, and key rotation checks:

```bash
docker compose up --build
```

### 2. Run the benchmark (both schemes, default ML-KEM-768)

```bash
docker compose --profile benchmark run benchmark
```

Results are written to `results/benchmark_results.json` and a summary is printed to stdout.

Override parameters via the `BENCHMARK_ARGS` environment variable:

```bash
BENCHMARK_ARGS="--iterations 50 --warmup 10 --payload-source all_ggvp" \
  docker compose --profile benchmark run benchmark
```

### 3. Generate charts

```bash
pip install matplotlib numpy
python visualize.py --input results/benchmark_results.json --output-dir results/
```

For real-payload runs (binned across the empirical size distribution):

```bash
python aggregate_benchmark_results.py \
  --input results/benchmark_results_all_ggvp_full.json \
  --output results/benchmark_results_all_ggvp_full_aggregated.json

python visualize_aggregated_benchmark.py \
  --input results/benchmark_results_all_ggvp_full_aggregated.json \
  --output-dir results/all_ggvp_charts/
```

---

## Running Each Configuration Separately

Each configuration can be launched independently. All examples below use `BENCHMARK_ARGS` with Docker Compose; direct `python -m app.benchmark` invocations inside the container work identically.

### Classical baseline only

```bash
BENCHMARK_ARGS="--schemes classical --output /app/results/benchmark_classical.json" \
  docker compose --profile benchmark run benchmark
```

### Hybrid with ML-KEM-512

```bash
BENCHMARK_ARGS="--schemes hybrid --pq-kem ML-KEM-512 --output /app/results/benchmark_hybrid_512.json" \
  docker compose --profile benchmark run benchmark
```

### Hybrid with ML-KEM-768 (default)

```bash
BENCHMARK_ARGS="--schemes hybrid --pq-kem ML-KEM-768 --output /app/results/benchmark_hybrid_768.json" \
  docker compose --profile benchmark run benchmark
```

### Hybrid with ML-KEM-1024

```bash
BENCHMARK_ARGS="--schemes hybrid --pq-kem ML-KEM-1024 --output /app/results/benchmark_hybrid_1024.json" \
  docker compose --profile benchmark run benchmark
```

### Both schemes together (default behavior)

```bash
# ML-KEM-768 (default)
docker compose --profile benchmark run benchmark

# ML-KEM-512
BENCHMARK_ARGS="--pq-kem ML-KEM-512 --output /app/results/benchmark_results_512.json" \
  docker compose --profile benchmark run benchmark

# ML-KEM-1024
BENCHMARK_ARGS="--pq-kem ML-KEM-1024 --output /app/results/benchmark_results_1024.json" \
  docker compose --profile benchmark run benchmark
```

---

## Benchmark Options

| Flag | Default | Description |
|------|---------|-------------|
| `--schemes LIST` | `classical,hybrid` | Comma-separated schemes to benchmark: `classical`, `hybrid`, or both |
| `--pq-kem ID` | `ML-KEM-768` | ML-KEM parameter set when `hybrid` is selected: `ML-KEM-512`, `ML-KEM-768`, `ML-KEM-1024` |
| `--pq-sig ID` | `none` | ML-DSA parameter set for envelope signing, or `none` to disable: `none`, `ML-DSA-44`, `ML-DSA-65`, `ML-DSA-87` |
| `--iterations N` | 10 | Measured iterations per (scheme × size) cell |
| `--warmup N` | 2 | Warmup rounds discarded before measurement |
| `--sizes BYTES` | 1 KB–1 MB (6 sizes) | Comma-separated record sizes, e.g. `1024,65536` |
| `--payload-source` | `synthetic` | `synthetic`, `faers`, or `all_ggvp` |
| `--real-payload-count N` | (all files) | Limit real-payload runs to N files sampled across the empirical distribution; omit to use all discovered files |
| `--output PATH` | `benchmark_results.json` | JSON output path |
| `--no-json` | — | Skip writing JSON output |
| `--policy-mode MODE` | (none) | Per-record PQ parameter policy: `uniform_baseline`, `uniform_strong`, or `adaptive_threshold`. When set, overrides `--pq-kem` and `--pq-sig` with policy-driven per-record selection |
| `--policy-threshold-bytes N` | `51200` (50 KiB) | Byte threshold for `adaptive_threshold` mode. Records strictly larger than this value receive ML-KEM-1024 + ML-DSA-87; all others receive ML-KEM-768 + ML-DSA-65 |

**`--schemes` examples:**
- `--schemes classical` — classical baseline only
- `--schemes hybrid` — hybrid only (uses `--pq-kem` to select the parameter set)
- `--schemes classical,hybrid` — both schemes interleaved (default)

**`--pq-kem` only applies when `hybrid` is in `--schemes`.** Passing an unsupported identifier exits with an error.

**`--pq-sig`** enables ML-DSA envelope signing. `none` (default) disables signing; a valid ML-DSA identifier enables sign+verify stages in every round-trip. Passing an unsupported identifier exits with an error.

**Payload sources:**
- `synthetic` — random bytes at each specified size
- `faers` — real FDA adverse event JSON records; all discovered files are used by default, or a size-matched subset when `--sizes` is given
- `all_ggvp` — real genomic VCF records from the 1000 Genomes Project; all discovered files are used by default, or a size-matched subset when `--sizes` is given

Use `--real-payload-count N` to limit a real-payload run to N files sampled across the empirical size distribution instead of using the full dataset.

---

## Measurement Methodology

Each (scheme × payload size) pair is one **cell**. Within a cell:

- When both schemes are active, classical and hybrid are measured in **alternating iterations** so both see equivalent CPU thermal and cache state.
- Odd iterations start with hybrid, even with classical, averaging out ordering bias.
- Warmup rounds are collected separately and discarded before statistics.
- The database table is cleared between size blocks to eliminate row-count carryover.

This controls for the main confounds in microbenchmarks: thermal throttling, cache warming, and iteration-order bias.

When only one scheme is selected (`--schemes classical` or `--schemes hybrid`), the warmup and measurement loops run that scheme exclusively — no interleaving overhead.

---

## Architecture

### Encryption Schemes

| Scheme | KEM | KDF | Payload Cipher | DEK Wrap |
|--------|-----|-----|----------------|----------|
| Classical | X25519-ECDH | HKDF-SHA256 | AES-256-GCM | AES-KW-RFC3394 |
| Hybrid PQC | X25519-ECDH + ML-KEM-{512,768,1024} | HKDF-SHA256 | AES-256-GCM | AES-KW-RFC3394 |

The hybrid scheme is parameterized by the selected ML-KEM variant; the key derivation and envelope structure are otherwise identical across all three parameter sets.

**Classical key derivation:**
```
ECDH(eph_sk, recipient_pk)  →  HKDF  →  KEK  →  AES-KW(DEK)
```

**Hybrid key derivation:**
```
ECDH(eph_sk, recipient_pk) ‖ ML-KEM.Encaps(recipient_pq_pk)  →  HKDF  →  KEK  →  AES-KW(DEK)
```

The two shared secrets are concatenated before HKDF so security is maintained as long as *either* primitive is unbroken.

### Record Envelope Fields

Each stored row contains:

| Field | Unsigned | Signed | Description |
|-------|----------|--------|-------------|
| `ciphertext` | = plaintext | = plaintext | AES-256-GCM encrypted payload |
| `nonce` | 12 B | 12 B | GCM nonce |
| `tag` | 16 B | 16 B | GCM authentication tag |
| `wrapped_dek` | 40 B | 40 B | AES-KW wrapped 32-byte DEK |
| `eph_pubkey` | 32 B | 32 B | Ephemeral X25519 public key |
| `salt` | 32 B | 32 B | Per-record HKDF salt |
| `hkdf_info` | ~21 B | ~21 B | Per-record HKDF context string |
| `aad` | ~36 B | ~36 B | Additional authenticated data |
| `pq_ct` | — / ~768–1,568 B | — / ~768–1,568 B | ML-KEM ciphertext (hybrid only; size by parameter set) |
| `signature` | — | 2,420–4,627 B | ML-DSA signature over canonical commitment |
| `sig_alg_id` | NULL | `"ML-DSA-*"` | Which ML-DSA parameter set was used |
| `sig_key_id` | NULL | string | Signing key identifier |

The ML-KEM ciphertext is the dominant source of storage amplification in the hybrid scheme. Its size varies by parameter set:

| Parameter set | PQ CT size |
|---------------|------------|
| ML-KEM-512 | ~768 B |
| ML-KEM-768 | ~1,088 B |
| ML-KEM-1024 | ~1,568 B |

The `pq_kem_id` column in the database records which parameter set was used, and decryption reads this field to select the correct backend — no out-of-band configuration needed.

---

## Module Overview

| File | Purpose |
|------|---------|
| `app/stage2.py` | Core crypto: `encrypt()`, `decrypt()`, key generation, `validate_pq_kem_id()`, `validate_pq_sig_id()`, `sign_envelope()`, `verify_envelope()`, `build_envelope_commitment()`, `envelope_sizes()` |
| `app/db.py` | Schema migration, `insert_record`, `fetch_record`, `update_wrap_fields`, `audit_event`; includes `sig_alg_id`/`signature`/`sig_key_id` columns |
| `app/service.py` | Service API: `put_record` (optional signing), `get_record` (optional pre-decrypt verify), `rotate_record_key` (re-sign or clear signature), `delete_record` |
| `app/stage3.py` | Smoke tests: round-trips for all ML-KEM variants, isolation, tamper detection, rotation; plus full signed round-trip suite for all three ML-DSA parameter sets |
| `app/benchmark.py` | Interleaved measurement harness, `Stats`, `CellResult`, `--schemes` / `--pq-kem` / `--pq-sig` / `--policy-mode` CLI |
| `app/policy.py` | Workload-aware PQ parameter policy: `ThresholdPolicyConfig`, `PolicyDecision`, `select_policy_for_record()`, `is_strong_assignment()`, `format_threshold_label()` |
| `visualize.py` | 8-chart visualisation from `benchmark_results.json` |
| `aggregate_benchmark_results.py` | Bin raw results into quantile summaries for trend plots |
| `visualize_aggregated_benchmark.py` | Multi-chart suite from aggregated JSON |
| `plot_policy_figures.py` | 6-figure policy comparison suite from `run_policy_pilot.sh` output |

---

## Integration Tests

`app/stage3.py` verifies correctness before benchmark runs:

1. **Classical round-trip** — encrypt, store, fetch, decrypt; assert plaintext matches
2. **Tenant isolation** — cross-tenant fetch raises `KeyError`
3. **Tamper detection** — bit-flip in ciphertext raises `InvalidTag`
4. **Soft delete** — deleted records are hidden from normal queries
5. **Hybrid round-trip (ML-KEM-512)** — encrypt, store, fetch, decrypt; assert plaintext matches
6. **Hybrid round-trip (ML-KEM-768)** — same with ML-KEM-768
7. **Hybrid round-trip (ML-KEM-1024)** — same with ML-KEM-1024
8. **Key rotation** — DEK rewrapped without re-encrypting payload; round-trip still succeeds (tested for each hybrid variant)
9. **Signed round-trip (ML-DSA-44)** — sign, store, verify, decrypt; assert plaintext matches; assert signature bytes = 2,420
10. **Signed round-trip (ML-DSA-65)** — same with ML-DSA-65 (sig = 3,309 B)
11. **Signed round-trip (ML-DSA-87)** — same with ML-DSA-87 (sig = 4,627 B)
12. **Ciphertext tamper → verify blocks decrypt** — bit-flip in ciphertext raises `SigVerificationError` before decryption is attempted
13. **Metadata tamper → verify blocks decrypt** — AAD mutation raises `SigVerificationError`
14. **Tenant isolation for signed records** — cross-tenant fetch raises `KeyError`
15. **Soft delete for signed records** — deleted signed records are hidden
16. **Key rotation clears signature** — rotation without new signing key NULLs signature columns; record becomes unsigned
17. **Key rotation with re-sign** — rotation with new signing key stores fresh signature; round-trip with new verify key succeeds

---

## Metrics

**Latency** (milliseconds, per iteration):

| Metric | Present when | What it covers |
|--------|--------------|----------------|
| `encrypt` | always | DEK generation, ECDH, ML-KEM encapsulation (hybrid only), HKDF, AES-KW wrap, AES-GCM encrypt |
| `sign` | `--pq-sig` ≠ `none` | ML-DSA sign over canonical envelope commitment |
| `db_insert` | always | Parameterized INSERT + audit event |
| `db_fetch` | always | Indexed SELECT by record ID and tenant |
| `verify` | `--pq-sig` ≠ `none` | ML-DSA verify before decryption; blocks decrypt on failure |
| `decrypt` | always | ECDH, ML-KEM decapsulation (hybrid only), HKDF, AES-KW unwrap, AES-GCM decrypt |
| `end_to_end` | always | Wall-clock time for the full round-trip including all stages |

Statistics: mean, min, max, p50, p95, p99.

**Per-primitive breakdown** within encrypt and decrypt is also recorded (ECDH, ML-KEM, HKDF, AES-GCM, AES-KW, ML-DSA-sign, ML-DSA-verify individually).

**Throughput** (bytes/sec) derived from latency for each stage.

**Storage** (bytes per record): plaintext, ciphertext, per-field header bytes, PQ ciphertext, signature bytes, total stored, and amplification ratio (total ÷ plaintext).

---

## Results

### Raw benchmark JSON (`benchmark_results.json`)

One entry per (scheme × size) cell. Top-level metadata fields:

```json
{
  "run_at": "2024-01-01T00:00:00+00:00",
  "host": "hostname",
  "iterations": 10,
  "warmup_iterations": 2,
  "payload_source": "synthetic",
  "schemes": ["classical", "hybrid"],
  "pq_kem_id": "ML-KEM-768",
  "pq_sig_id": null,
  "results": [...]
}
```

`pq_sig_id` is `null` for unsigned runs, or `"ML-DSA-44"` / `"ML-DSA-65"` / `"ML-DSA-87"` for signed runs.

Each result entry includes `"scheme"`, `"pq_sig_id"`, optional `"sign"` and `"verify"` stats objects (null when unsigned), and `envelope_bytes.signature_bytes` (0 when unsigned). The top-level `pq_kem_id` field records which ML-KEM variant was used for the run.

When running separate per-KEM configurations, save each to a distinct output file (e.g. `benchmark_hybrid_512.json`, `benchmark_hybrid_768.json`, `benchmark_hybrid_1024.json`) to keep results distinguishable.

### Aggregated JSON (`*_aggregated.json`)

Produced by `aggregate_benchmark_results.py`. Groups real-payload results into quantile bins and computes p25/p50/p75/p95 per bin per scheme, plus hybrid overhead percentages. Used for trend plots across the empirical size distribution.

### Charts (`results/`)

**From `visualize.py` (synthetic / explicit sizes):**

| File | Description |
|------|-------------|
| `01_e2e_comparison.png` | End-to-end latency — Classical vs Hybrid (bars = p50, ◆ = mean, tick = p95) |
| `02_per_metric_lines.png` | Per-metric latency across sizes (p50 line + p50→p95 band) |
| `03_overhead_pct.png` | Hybrid overhead vs Classical per metric (%) |
| `04_stacked_breakdown.png` | Operation breakdown: Encrypt + DB Insert + DB Fetch + Decrypt |
| `05_percentile_fan.png` | End-to-end percentile fan (p50 / p95 / p99) |
| `06_storage_total.png` | Total stored bytes: Classical vs Hybrid |
| `07_storage_composition.png` | Envelope layers: payload / header / PQ-KEM ciphertext |
| `08_storage_amplification.png` | Storage amplification ratio (total stored ÷ plaintext) |

**From `visualize_aggregated_benchmark.py` (real datasets, binned):**

| File | Description |
|------|-------------|
| `12_*_latency_trends.png` | Latency trends across the empirical size distribution |
| `13_*_overhead_trends.png` | Hybrid overhead (%) across sizes |
| `14_*_storage_amplification_trends.png` | Storage amplification across sizes |
| `15_*_time_composition.png` | Time composition breakdown |
| `17_derived_throughput.png` | Throughput comparison (bytes/sec) |
| `18_*_latency_comparison.png` | Classical vs hybrid latency side-by-side |

---

## Database Schema

Two tables created idempotently at startup via `app/db.py`:

- **`encrypted_records`** — one row per stored ciphertext envelope; columns mirror the `EncryptedRecord` dataclass. The `pq_kem_id` column records which ML-KEM parameter set was used for each hybrid record. Three additional columns hold signature data: `sig_alg_id` (`"ML-DSA-44"`, `"ML-DSA-65"`, or `"ML-DSA-87"`), `signature` (the signature bytes), and `sig_key_id` (key identifier). CHECK constraints ensure: (1) classical records carry no PQ fields; (2) hybrid records carry both PQ fields non-NULL; (3) signature columns are either all NULL (unsigned) or all non-NULL (signed). Indexes on `(tenant_id, created_at)`, `(tenant_id, key_id)`, `scheme`, and `created_at`.
- **`audit_events`** — append-only log of `put`, `delete`, and `rotate` operations per tenant.

Tenant isolation is enforced at the query level: every `SELECT`/`UPDATE` includes a `WHERE tenant_id = %s` filter.

To create a least-privilege application role, run `scripts/roles.sql` once as a PostgreSQL superuser.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `cryptography >= 42` | X25519, HKDF-SHA256, AES-GCM, AES-KW primitives |
| `liboqs-python == 0.14.1` | ML-KEM (FIPS 203) and ML-DSA (FIPS 204) via liboqs |
| `psycopg[binary] >= 3.1` | PostgreSQL driver (psycopg v3) |

`liboqs` is compiled from source inside the Docker image (pinned to tag `0.14.0`).
Both ML-KEM and ML-DSA use the standardized NIST names (`ML-KEM-*`, `ML-DSA-*`).
On builds that pre-date the standardized names, the library falls back to the historical identifiers (`Kyber*`, `Dilithium*`) automatically.
