#!/usr/bin/env bash
# run_faers_threshold_sweep.sh
#
# Threshold sweep: FAERS workload, adaptive_threshold policy, five thresholds:
#   10 KiB (10,240 B), 25 KiB (25,600 B), 50 KiB (51,200 B),
#   100 KiB (102,400 B), 250 KiB (256,000 B)
#
# Designed for unattended overnight execution.  Jobs run sequentially; a failed
# job is logged and skipped so the sweep continues.  A summary is appended to
# the manifest and printed at the end.
#
# Usage:
#   ./run_faers_threshold_sweep.sh
#   nohup ./run_faers_threshold_sweep.sh > sweep_$(date +%Y%m%d).log 2>&1 &
#
# Environment overrides (all optional):
#   ITERATIONS    measured iterations per cell  (default: 10)
#   WARMUP        warmup iterations             (default: 2)
#   FAERS_COUNT   FAERS records to load         (default: 50000 — full dataset)

set -uo pipefail

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
FAERS_COUNT="${FAERS_COUNT:-50000}"
POLICY_MODE="adaptive_threshold"
PAYLOAD_SOURCE="faers"

# Parallel arrays: bytes, short label (used in filenames), display string
THRESHOLD_BYTES=(  10240    25600    51200    102400    256000  )
THRESHOLD_LABELS=( "10kb"   "25kb"   "50kb"   "100kb"   "250kb" )
THRESHOLD_DISPLAY=(
    "10 KiB  (10,240 B)"
    "25 KiB  (25,600 B)"
    "50 KiB  (51,200 B)"
    "100 KiB (102,400 B)"
    "250 KiB (256,000 B)"
)
NUM_THRESHOLDS="${#THRESHOLD_BYTES[@]}"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/results/faers_threshold_sweep/${TIMESTAMP}"
JSON_DIR="${RUN_DIR}/json"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${JSON_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

elapsed_fmt() {
    local s=$1
    printf "%dh %02dm %02ds" $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

# ---------------------------------------------------------------------------
# Write manifest before any jobs start (exists even if the run dies mid-sweep)
# ---------------------------------------------------------------------------
{
    cat <<MANIFEST
FAERS Adaptive Threshold — Sweep Experiment
===========================================
timestamp             : ${TIMESTAMP}
run_dir               : ${RUN_DIR}
workload              : ${PAYLOAD_SOURCE}
policy_mode           : ${POLICY_MODE}
iterations            : ${ITERATIONS}
warmup                : ${WARMUP}
faers_record_count    : ${FAERS_COUNT}  (full dataset)

Threshold rule (per job)
------------------------
  strong tier  : record_size_bytes >  <threshold>
  baseline tier: record_size_bytes <= <threshold>

Thresholds
----------
MANIFEST
    for i in "${!THRESHOLD_BYTES[@]}"; do
        printf "  %-8s : %s\n" "${THRESHOLD_LABELS[$i]}" "${THRESHOLD_DISPLAY[$i]}"
    done

    cat <<MANIFEST

Runs intentionally NOT repeated
--------------------------------
  faers_uniform_baseline  — use results/policy_pilot/20260417_011356/json/faers_uniform_baseline.json
  faers_uniform_strong    — use results/policy_pilot/20260417_011356/json/faers_uniform_strong.json
  synthetic (all modes)   — not rerun
  ggvp (all modes)        — not rerun

Jobs (${NUM_THRESHOLDS} total)
--------------
MANIFEST
    for i in "${!THRESHOLD_BYTES[@]}"; do
        echo "  faers_adaptive_threshold_${THRESHOLD_LABELS[$i]}"
    done

    echo ""
    echo "Benchmark commands"
    echo "------------------"
    for i in "${!THRESHOLD_BYTES[@]}"; do
        tb="${THRESHOLD_BYTES[$i]}"
        tl="${THRESHOLD_LABELS[$i]}"
        label="faers_adaptive_threshold_${tl}"
        cat <<CMD
  [${tl}]
  python -m app.benchmark \\
      --output /app/results/${label}.json \\
      --payload-source ${PAYLOAD_SOURCE} \\
      --real-payload-count ${FAERS_COUNT} \\
      --schemes hybrid \\
      --iterations ${ITERATIONS} \\
      --warmup ${WARMUP} \\
      --policy-mode ${POLICY_MODE} \\
      --policy-threshold-bytes ${tb}

CMD
    done
} > "${RUN_DIR}/manifest.txt"

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

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
log "=== FAERS Adaptive Threshold Sweep — ${NUM_THRESHOLDS} jobs ==="
log "Run directory : ${RUN_DIR}"
log "Thresholds    : ${THRESHOLD_LABELS[*]}"
log "Iterations    : ${ITERATIONS}  Warmup: ${WARMUP}"
log "FAERS records : ${FAERS_COUNT} (full dataset)"
log ""

# ---------------------------------------------------------------------------
# Ensure DB is running once before the sweep begins
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
log ""

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
PASS_LIST=()
FAIL_LIST=()
SKIP_LIST=()
SWEEP_START="$(date +%s)"

for i in "${!THRESHOLD_BYTES[@]}"; do
    tb="${THRESHOLD_BYTES[$i]}"
    tl="${THRESHOLD_LABELS[$i]}"
    label="faers_adaptive_threshold_${tl}"
    job_num=$(( i + 1 ))

    log "--- Job ${job_num}/${NUM_THRESHOLDS}: ${label}  (threshold = ${tb} B) ---"

    # Skip if output already exists — safe to re-run a partial sweep
    if [[ -f "${JSON_DIR}/${label}.json" ]]; then
        log "SKIP  ${label}  (output already exists)"
        SKIP_LIST+=("${label}")
        log ""
        continue
    fi

    job_start="$(date +%s)"
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
                --output /app/results/${label}.json \
                --payload-source ${PAYLOAD_SOURCE} \
                --real-payload-count ${FAERS_COUNT} \
                --schemes hybrid \
                --iterations ${ITERATIONS} \
                --warmup ${WARMUP} \
                --policy-mode ${POLICY_MODE} \
                --policy-threshold-bytes ${tb}" \
        2>&1 | tee "${LOG_DIR}/${label}.log"
    rc="${PIPESTATUS[0]}"
    set -e

    job_end="$(date +%s)"
    elapsed=$(( job_end - job_start ))

    if [[ "${rc}" -eq 0 ]]; then
        log "PASS  ${label}  ($(elapsed_fmt ${elapsed}))"
        PASS_LIST+=("${label}")
    else
        log "FAIL  ${label}  exit_code=${rc}  ($(elapsed_fmt ${elapsed}))"
        log "      Log: ${LOG_DIR}/${label}.log"
        FAIL_LIST+=("${label}")
    fi
    log ""
done

SWEEP_END="$(date +%s)"
SWEEP_ELAPSED=$(( SWEEP_END - SWEEP_START ))

# ---------------------------------------------------------------------------
# Summary — printed to stdout and appended to manifest
# ---------------------------------------------------------------------------
summary() {
    echo ""
    echo "=== Sweep Summary ================================================================"
    printf "  Total wall time : %s\n" "$(elapsed_fmt ${SWEEP_ELAPSED})"
    printf "  Passed          : %d / %d\n" "${#PASS_LIST[@]}" "${NUM_THRESHOLDS}"
    printf "  Failed          : %d\n" "${#FAIL_LIST[@]}"
    printf "  Skipped         : %d\n" "${#SKIP_LIST[@]}"
    echo ""
    if [[ "${#PASS_LIST[@]}" -gt 0 ]]; then
        echo "  PASSED:"
        for j in "${PASS_LIST[@]}"; do
            echo "    [OK]   ${j}  ->  ${JSON_DIR}/${j}.json"
        done
    fi
    if [[ "${#SKIP_LIST[@]}" -gt 0 ]]; then
        echo "  SKIPPED (output existed):"
        for j in "${SKIP_LIST[@]}"; do
            echo "    [SKIP] ${j}"
        done
    fi
    if [[ "${#FAIL_LIST[@]}" -gt 0 ]]; then
        echo "  FAILED:"
        for j in "${FAIL_LIST[@]}"; do
            echo "    [FAIL] ${j}  ->  ${LOG_DIR}/${j}.log"
        done
    fi
    echo "=================================================================================="
}

summary | tee -a "${RUN_DIR}/manifest.txt"

log ""
log "Run directory : ${RUN_DIR}"
log "Manifest      : ${RUN_DIR}/manifest.txt"

if [[ "${#FAIL_LIST[@]}" -gt 0 ]]; then
    exit 1
fi
