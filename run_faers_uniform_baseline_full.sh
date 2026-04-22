#!/usr/bin/env bash
# run_faers_uniform_baseline_full.sh
#
# Single-job matched run: FAERS workload, uniform_baseline policy, full 50,000-record dataset.
#
# Purpose: produce a matched FAERS baseline result using the same record count,
# iteration count, warmup count, and scheme as run_faers_uniform_strong_full.sh
# and run_faers_adaptive_50kb.sh so all three can be compared apples-to-apples.
#
# Usage:
#   ./run_faers_uniform_baseline_full.sh
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
POLICY_MODE="uniform_baseline"
PAYLOAD_SOURCE="faers"
LABEL="faers_uniform_baseline"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/results/faers_full_matched_baseline/${TIMESTAMP}"
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
FAERS Uniform Baseline — Full Dataset Matched Run
==================================================
purpose               : matched full-dataset comparison (baseline leg)
timestamp             : ${TIMESTAMP}
run_dir               : ${RUN_DIR}
workload              : ${PAYLOAD_SOURCE}
policy_mode           : ${POLICY_MODE}
iterations            : ${ITERATIONS}
warmup                : ${WARMUP}
faers_record_count    : ${FAERS_COUNT}  (full dataset)
scheme                : hybrid
output_json           : ${JSON_DIR}/${LABEL}.json
output_log            : ${LOG_DIR}/${LABEL}.log

Matched comparison set
----------------------
  This run (baseline):
    ${JSON_DIR}/${LABEL}.json
  Matched strong run:
    results/faers_full_matched_strong/<timestamp>/json/faers_uniform_strong.json
  Matched adaptive 50 KiB run (already exists):
    results/faers_adaptive_50kb/20260418_153046/json/faers_adaptive_threshold_50kb.json

All three runs share:
  workload              : faers
  faers_record_count    : 50000
  iterations            : 10
  warmup                : 2
  scheme                : hybrid

Benchmark command
-----------------
  python -m app.benchmark \\
      --output /app/results/${LABEL}.json \\
      --payload-source ${PAYLOAD_SOURCE} \\
      --real-payload-count ${FAERS_COUNT} \\
      --schemes hybrid \\
      --iterations ${ITERATIONS} \\
      --warmup ${WARMUP} \\
      --policy-mode ${POLICY_MODE}
MANIFEST

log "=== FAERS Uniform Baseline Full-Dataset Runner ==="
log "Run directory : ${RUN_DIR}"
log "Policy mode   : ${POLICY_MODE}"
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
            --policy-mode ${POLICY_MODE}" \
    2>&1 | tee "${LOG_DIR}/${LABEL}.log"
rc="${PIPESTATUS[0]}"
set -e

local_end="$(date +%s)"
elapsed=$(( local_end - local_start ))

if [[ "${rc}" -eq 0 ]]; then
    log "PASS  ${LABEL}  (${elapsed}s)"
    log ""
    log "=== Done ==="
    log "JSON    : ${JSON_DIR}/${LABEL}.json"
    log "Log     : ${LOG_DIR}/${LABEL}.log"
    log "Manifest: ${RUN_DIR}/manifest.txt"
else
    log "FAIL  ${LABEL}  exit_code=${rc}  (${elapsed}s)"
    log "Check log: ${LOG_DIR}/${LABEL}.log"
    exit 1
fi
