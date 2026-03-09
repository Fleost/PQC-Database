# PQC Database Evaluation

Benchmarks **classical** (X25519 + AES-256-GCM) versus **hybrid post-quantum** (X25519 + ML-KEM-768 + AES-256-GCM) envelope encryption in a PostgreSQL-backed service, measuring end-to-end latency, per-primitive timing, and storage overhead across record sizes from 1 KB to 1 MB.

---

## Project Structure

```
pqc-db-eval/
├── app/
│   ├── __init__.py
│   ├── stage2.py       # Crypto layer: classical + hybrid envelope encryption
│   ├── stage3.py       # Integration smoke tests (DB round-trips)
│   ├── db.py           # PostgreSQL schema, insert/fetch/update helpers
│   ├── service.py      # High-level put_record / get_record / rotate API
│   └── benchmark.py    # Benchmark runner, statistics, report formatter
├── scripts/
│   ├── wait-for-postgres.sh  # Docker startup healthcheck
│   └── roles.sql             # Least-privilege DB role (run once as superuser)
├── results/            # Benchmark JSON output + generated charts
├── visualize.py        # 8-chart visualisation suite from benchmark JSON
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose

### 1. Run integration smoke tests

Starts PostgreSQL, runs classical and hybrid DB round-trips (tenant isolation,
tamper detection, soft delete, key rotation):

```bash
docker compose up --build
```

### 2. Run the benchmark

```bash
docker compose --profile benchmark run benchmark
```

Results are written to `results/benchmark_results.json` and a summary report
is printed to stdout.

Override parameters via the `BENCHMARK_ARGS` environment variable:

```bash
BENCHMARK_ARGS="--iterations 20 --warmup 5" \
  docker compose --profile benchmark run benchmark
```

### 3. Generate charts

```bash
pip install matplotlib numpy
python visualize.py --input results/benchmark_results.json --output-dir results/
```

---

## Benchmark Options

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations N` | 10 | Measured iterations per (scheme × size) cell |
| `--warmup N` | 2 | Warmup rounds discarded before measurement |
| `--sizes BYTES` | 1 KB–1 MB (6 sizes) | Comma-separated record sizes, e.g. `1024,65536` |
| `--pq-kem ID` | `ML-KEM-768` | Post-quantum KEM algorithm ID |
| `--output PATH` | `benchmark_results.json` | JSON output path |
| `--no-json` | — | Skip writing JSON output |

---

## Architecture

### Encryption Schemes

| Scheme | KEM | KDF | Payload Cipher | DEK Wrap |
|--------|-----|-----|----------------|----------|
| Classical | X25519-ECDH | HKDF-SHA256 | AES-256-GCM | AES-KW-RFC3394 |
| Hybrid PQC | X25519-ECDH + ML-KEM-768 | HKDF-SHA256 | AES-256-GCM | AES-KW-RFC3394 |

### Key Derivation

**Classical:**
```
ECDH(eph_sk, recipient_pk)  →  HKDF  →  KEK  →  AES-KW(DEK)
```

**Hybrid PQC:**
```
ECDH(eph_sk, recipient_pk) ‖ ML-KEM.Encaps(recipient_pq_pk)  →  HKDF  →  KEK  →  AES-KW(DEK)
```

The two shared secrets are concatenated before HKDF so security is maintained as long as *either* primitive is unbroken.

### Record Envelope Fields

Each stored row contains:

| Field | Size | Description |
|-------|------|-------------|
| `ciphertext` | = plaintext | AES-256-GCM encrypted payload |
| `nonce` | 12 B | GCM nonce |
| `tag` | 16 B | GCM authentication tag |
| `wrapped_dek` | 40 B | AES-KW wrapped 32-byte DEK |
| `eph_pubkey` | 32 B | Ephemeral X25519 public key |
| `salt` | 32 B | Per-record HKDF salt |
| `hkdf_info` | ~20 B | Per-record HKDF context string |
| `aad` | variable | Additional authenticated data |
| `pq_ct` | ~1,088 B | *(Hybrid only)* ML-KEM-768 ciphertext |

---

## Module Overview

| File | Purpose |
|------|---------|
| `app/stage2.py` | Core crypto: `encrypt()`, `decrypt()`, key generation, `envelope_sizes()` |
| `app/db.py` | Schema migration, `insert_record`, `fetch_record`, `audit_event` |
| `app/service.py` | Service API: `put_record`, `get_record`, `rotate_record_key`, `delete_record` |
| `app/stage3.py` | Smoke tests: round-trips, isolation, tamper detection, rotation |
| `app/benchmark.py` | Interleaved measurement harness, `Stats`, `CellResult`, report formatting |
| `visualize.py` | 8-chart visualisation from `benchmark_results.json` |

---

## Results

Charts saved to `results/`:

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

---

## Database Schema

Two tables are created idempotently at startup via `app/db.py:init_schema()`:

- **`encrypted_records`** — one row per stored ciphertext envelope; columns mirror the `EncryptedRecord` dataclass.
- **`audit_events`** — append-only log of `put`, `delete`, and `rotate` operations per tenant.

Tenant isolation is enforced at the query level: every `SELECT`/`UPDATE` includes a `WHERE tenant_id = %s` filter.

To create a least-privilege application role, run `scripts/roles.sql` once as a PostgreSQL superuser.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `cryptography >= 42` | X25519, HKDF-SHA256, AES-GCM, AES-KW primitives |
| `liboqs-python == 0.14.1` | ML-KEM-768 via liboqs |
| `psycopg[binary] >= 3.1` | PostgreSQL driver (psycopg v3) |

`liboqs` is compiled from source inside the Docker image (pinned to tag `0.14.0`).
