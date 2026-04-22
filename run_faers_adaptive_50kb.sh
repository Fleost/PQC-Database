#!/usr/bin/env bash
# run_faers_adaptive_50kb.sh
#
# Single-job experiment: FAERS workload, adaptive_threshold policy, 50 KiB threshold.
#
# This is a focused experiment to compare adaptive-threshold behaviour at
# 51,200 bytes (50 KiB) against the earlier 10,485,760-byte (10 MiB) pilot run.
# Only ONE benchmark job is executed.  Baseline, strong, GGVP, and synthetic
# runs are intentionally NOT repeated here; compare against the existing pilot:
#
#   results/policy_pilot/20260417_011356/json/faers_uniform_baseline.json
#   results/policy_pilot/20260417_011356/json/faers_uniform_strong.json
#   results/policy_pilot/20260417_011356/json/faers_adaptive_threshold.json  (10 MiB)
#
# Usage:
#   ./run_faers_adaptive_50kb.sh
#
# Environment overrides (all optional):
#   ITERATIONS             measured iterations per cell  (default: 10)
#   WARMUP                 warmup iterations             (default: 2)
#   FAERS_COUNT            FAERS records to load         (default: 50000 — full dataset)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
FAERS_COUNT="${FAERS_COUNT:-50000}"
POLICY_THRESHOLD_BYTES=51200          # 50 KiB — the only intentional change
THRESHOLD_LABEL="50KiB"
POLICY_MODE="adaptive_threshold"
PAYLOAD_SOURCE="faers"
LABEL="faers_adaptive_threshold_50kb"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/results/faers_adaptive_50kb/${TIMESTAMP}"
JSON_DIR="${RUN_DIR}/json"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${JSON_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Write manifest before running (exists even if the job fails mid-way)
# ---------------------------------------------------------------------------
cat > "${RUN_DIR}/manifest.txt" <<MANIFEST
FAERS Adaptive Threshold — 50 KiB Experiment
=============================================
timestamp             : ${TIMESTAMP}
run_dir               : ${RUN_DIR}
threshold_bytes       : ${POLICY_THRESHOLD_BYTES}
threshold_label       : ${THRESHOLD_LABEL}
workload              : ${PAYLOAD_SOURCE}
policy_mode           : ${POLICY_MODE}
iterations            : ${ITERATIONS}
warmup                : ${WARMUP}
faers_record_count    : ${FAERS_COUNT}  (full dataset)
output_json           : ${JSON_DIR}/${LABEL}.json
output_log            : ${LOG_DIR}/${LABEL}.log

Threshold rule
--------------
  strong tier  : record_size_bytes >  51200
  baseline tier: record_size_bytes <= 51200

Full-dataset escalation rate (pre-computed, no benchmark needed)
----------------------------------------------------------------
  FAERS total records : 50,000
  Above 51,200 B      : 6,919  (13.84%)
  At/below 51,200 B   : 43,081 (86.16%)
  (10 MiB pilot rate for comparison: 2 / 50,000 = 0.004%)

Runs intentionally NOT repeated
--------------------------------
  faers_uniform_baseline    — use results/policy_pilot/20260417_011356/json/faers_uniform_baseline.json
  faers_uniform_strong      — use results/policy_pilot/20260417_011356/json/faers_uniform_strong.json
  faers_adaptive_threshold  — use results/policy_pilot/20260417_011356/json/faers_adaptive_threshold.json (10 MiB)
  synthetic (all modes)     — not rerun
  ggvp (all modes)          — not rerun

Jobs (1 total)
--------------
  ${LABEL}

Benchmark command
-----------------
  python -m app.benchmark \\
      --output /app/results/${LABEL}.json \\
      --payload-source ${PAYLOAD_SOURCE} \\
      --real-payload-count ${FAERS_COUNT} \\
      --schemes hybrid \\
      --iterations ${ITERATIONS} \\
      --warmup ${WARMUP} \\
      --policy-mode ${POLICY_MODE} \\
      --policy-threshold-bytes ${POLICY_THRESHOLD_BYTES}

Future comparison
-----------------
  Load the following four JSON files together to compare adaptive at 50 KiB
  vs adaptive at 10 MiB vs the two uniform baselines:
    ${JSON_DIR}/${LABEL}.json
    results/policy_pilot/20260417_011356/json/faers_adaptive_threshold.json
    results/policy_pilot/20260417_011356/json/faers_uniform_baseline.json
    results/policy_pilot/20260417_011356/json/faers_uniform_strong.json
MANIFEST

log "=== FAERS Adaptive 50 KiB Experiment Runner ==="
log "Run directory : ${RUN_DIR}"
log "Threshold     : ${POLICY_THRESHOLD_BYTES} B (${THRESHOLD_LABEL})"
log "Iterations    : ${ITERATIONS}  Warmup: ${WARMUP}"
log "FAERS records : ${FAERS_COUNT} (full dataset)"
log "Output JSON   : ${JSON_DIR}/${LABEL}.json"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found in PATH" >&2; exit 1
fi
if ! docker compose version &>/dev/null; then
    echo "ERROR: docker compose not available" >&2; exit 1
fi

cd "${SCRIPT_DIR}"

# Guard: skip if output already exists (safe re-run behaviour)
if [[ -f "${JSON_DIR}/${LABEL}.json" ]]; then
    log "SKIP  ${LABEL}  (output already exists at ${JSON_DIR}/${LABEL}.json)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Ensure DB is running
# ---------------------------------------------------------------------------
log "Ensuring DB container is running..."
docker compose up -d db 2>&1 | grep -v "^time=" || true

log "Waiting for PostgreSQL to accept connections..."
for i in $(seq 1 30); do
    if docker compose exec -T db pg_isready -U postgres -q 2>/dev/null; then
        log "PostgreSQL is ready."
        break
    fi
    if [[ "${i}" -eq 30 ]]; then
        log "ERROR: PostgreSQL did not become ready within 150 s" >&2; exit 1
    fi
    sleep 5
done

echo ""

# ---------------------------------------------------------------------------
# Single benchmark job
# ---------------------------------------------------------------------------
log "START ${LABEL}"

local_start="$(date +%s)"
set +e
docker compose run --rm \
    -e PGHOST=db \
    -e PGPORT=5432 \
    -e PGDATABASE=pqc_eval \
    -e PGUSER=postgres \
    -e PGPASSWORD=postgres \
    --volume "${JSON_DIR}:/app/results" \
    benchmark \
    bash -lc "/app/scripts/wait-for-postgres.sh db 5432 && \
        python -m app.benchmark \
            --output /app/results/${LABEL}.json \
            --payload-source ${PAYLOAD_SOURCE} \
            --real-payload-count ${FAERS_COUNT} \
            --schemes hybrid \
            --iterations ${ITERATIONS} \
            --warmup ${WARMUP} \
            --policy-mode ${POLICY_MODE} \
            --policy-threshold-bytes ${POLICY_THRESHOLD_BYTES}" \
    2>&1 | tee "${LOG_DIR}/${LABEL}.log"
rc="${PIPESTATUS[0]}"
set -e

local_end="$(date +%s)"
elapsed=$(( local_end - local_start ))

if [[ "${rc}" -eq 0 ]]; then
    log "PASS  ${LABEL}  (${elapsed}s)"
    log ""
    log "=== Done ==="
    log "JSON : ${JSON_DIR}/${LABEL}.json"
    log "Log  : ${LOG_DIR}/${LABEL}.log"
    log "Manifest: ${RUN_DIR}/manifest.txt"
else
    log "FAIL  ${LABEL}  exit_code=${rc}  (${elapsed}s)"
    log "Check log: ${LOG_DIR}/${LABEL}.log"
    exit 1
fi
