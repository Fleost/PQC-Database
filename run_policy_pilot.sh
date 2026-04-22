#!/usr/bin/env bash
# run_policy_pilot.sh
#
# Pilot experiment runner for the workload-aware PQ policy benchmark.
# Runs 12 benchmark jobs (4 workloads × 3 policy modes) sequentially,
# writes one JSON + one log per job, and generates plots when done.
#
# Usage:
#   ./run_policy_pilot.sh            # uses defaults
#   ITERATIONS=5 ./run_policy_pilot.sh
#
# Environment overrides (all optional):
#   ITERATIONS              measured iterations per cell  (default: 5)
#   WARMUP                  warmup iterations             (default: 2)
#   FAERS_COUNT             FAERS files to sample         (default: 20)
#   GGVP_COUNT              GGVP  files to sample         (default: 10)
#   POLICY_THRESHOLD_BYTES  adaptive threshold in bytes   (default: 10485760)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ITERATIONS="${ITERATIONS:-5}"
WARMUP="${WARMUP:-2}"
FAERS_COUNT="${FAERS_COUNT:-20}"
GGVP_COUNT="${GGVP_COUNT:-10}"
POLICY_THRESHOLD_BYTES="${POLICY_THRESHOLD_BYTES:-51200}"

# Synthetic sizes straddle the 50 KiB threshold:
#   1 KB, 64 KB, 1 MB, 16 MB, 64 MB
# (64 KB, 1 MB, 16 MB, and 64 MB will escalate; 1 KB stays on baseline)
SYNTHETIC_SIZES="1024,65536,1048576,16777216,67108864"

POLICY_MODES=(uniform_baseline adaptive_threshold uniform_strong)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/results/policy_pilot/${TIMESTAMP}"
JSON_DIR="${RUN_DIR}/json"
LOG_DIR="${RUN_DIR}/logs"
PLOT_DIR="${RUN_DIR}/plots"

mkdir -p "${JSON_DIR}" "${LOG_DIR}" "${PLOT_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

PASS_COUNT=0
FAIL_COUNT=0
declare -a FAILED_JOBS=()

# Run one benchmark job.
#   run_job <label> <docker-benchmark-args...>
#
# The label becomes the base filename for the JSON and log.
# The benchmark container always writes to /app/results/<name>.json;
# we mount ${JSON_DIR} as /app/results inside the container.
run_job() {
    local label="$1"; shift
    local json_file="${JSON_DIR}/${label}.json"
    local log_file="${LOG_DIR}/${label}.log"

    # Guard: skip if output already exists (safe re-run behaviour)
    if [[ -f "${json_file}" ]]; then
        log "SKIP  ${label}  (output already exists)"
        return 0
    fi

    log "START ${label}"
    echo "  CMD: docker compose run --rm \\"
    echo "         -e PGHOST=db -e PGPORT=5432 -e PGDATABASE=pqc_eval \\"
    echo "         -e PGUSER=postgres -e PGPASSWORD=postgres \\"
    echo "         --volume \"${JSON_DIR}:/app/results\" \\"
    echo "         benchmark \\"
    echo "         bash -lc \"/app/scripts/wait-for-postgres.sh db 5432 && \\"
    echo "           python -m app.benchmark \\"
    echo "             --output /app/results/${label}.json \\"
    echo "             $*\""

    local start_ts
    start_ts="$(date +%s)"

    # Run and tee stdout+stderr to the log; capture exit code without aborting.
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
                $*" \
        2>&1 | tee "${log_file}"
    local rc="${PIPESTATUS[0]}"
    set -e

    local end_ts elapsed
    end_ts="$(date +%s)"
    elapsed=$(( end_ts - start_ts ))

    if [[ "${rc}" -eq 0 ]]; then
        log "PASS  ${label}  (${elapsed}s)"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    else
        log "FAIL  ${label}  exit_code=${rc}  (${elapsed}s)"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        FAILED_JOBS+=("${label}")
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
log "=== Policy Pilot Experiment Runner ==="
log "Run directory : ${RUN_DIR}"
log "Iterations    : ${ITERATIONS}  Warmup: ${WARMUP}"
log "FAERS sample  : ${FAERS_COUNT} files"
log "GGVP  sample  : ${GGVP_COUNT} files"
log "Threshold     : ${POLICY_THRESHOLD_BYTES} B"
log "Synthetic sizes: ${SYNTHETIC_SIZES}"
echo ""

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found in PATH" >&2; exit 1
fi
if ! docker compose version &>/dev/null; then
    echo "ERROR: docker compose not available" >&2; exit 1
fi

cd "${SCRIPT_DIR}"

# Ensure DB is running
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
# Write manifest (before runs so it exists even if runs fail mid-way)
# ---------------------------------------------------------------------------
cat > "${RUN_DIR}/manifest.txt" <<MANIFEST
Pilot experiment run
====================
timestamp             : ${TIMESTAMP}
run_dir               : ${RUN_DIR}
iterations            : ${ITERATIONS}
warmup                : ${WARMUP}
policy_threshold_bytes: ${POLICY_THRESHOLD_BYTES}
faers_sample_count    : ${FAERS_COUNT}
ggvp_sample_count     : ${GGVP_COUNT}
synthetic_sizes       : ${SYNTHETIC_SIZES}
policy_modes          : ${POLICY_MODES[*]}
workloads             : synthetic faers ggvp

Note on "mixed" workload
------------------------
The benchmark CLI does not have a dedicated --payload-source mixed flag.
"Mixed" is approximated here by running all three real-payload sources
(synthetic, faers, all_ggvp) under the same policy mode and analysing
their results together in post-processing.  This is consistent with the
existing codebase and requires no code changes.

Jobs (12 total)
---------------
MANIFEST

# ---------------------------------------------------------------------------
# Job 1-3: Synthetic
# ---------------------------------------------------------------------------
log "--- Workload group 1/4: synthetic ---"
for mode in "${POLICY_MODES[@]}"; do
    label="synthetic_${mode}"
    echo "${label}" >> "${RUN_DIR}/manifest.txt"
    run_job "${label}" \
        --payload-source synthetic \
        --sizes "${SYNTHETIC_SIZES}" \
        --schemes hybrid \
        --iterations "${ITERATIONS}" \
        --warmup "${WARMUP}" \
        --policy-mode "${mode}" \
        --policy-threshold-bytes "${POLICY_THRESHOLD_BYTES}"
done

# ---------------------------------------------------------------------------
# Job 4-6: FAERS
# ---------------------------------------------------------------------------
log "--- Workload group 2/4: faers ---"
for mode in "${POLICY_MODES[@]}"; do
    label="faers_${mode}"
    echo "${label}" >> "${RUN_DIR}/manifest.txt"
    run_job "${label}" \
        --payload-source faers \
        --real-payload-count "${FAERS_COUNT}" \
        --schemes hybrid \
        --iterations "${ITERATIONS}" \
        --warmup "${WARMUP}" \
        --policy-mode "${mode}" \
        --policy-threshold-bytes "${POLICY_THRESHOLD_BYTES}"
done

# ---------------------------------------------------------------------------
# Job 7-9: GGVP
# ---------------------------------------------------------------------------
log "--- Workload group 3/4: ggvp ---"
for mode in "${POLICY_MODES[@]}"; do
    label="ggvp_${mode}"
    echo "${label}" >> "${RUN_DIR}/manifest.txt"
    run_job "${label}" \
        --payload-source all_ggvp \
        --real-payload-count "${GGVP_COUNT}" \
        --schemes hybrid \
        --iterations "${ITERATIONS}" \
        --warmup "${WARMUP}" \
        --policy-mode "${mode}" \
        --policy-threshold-bytes "${POLICY_THRESHOLD_BYTES}"
done

# ---------------------------------------------------------------------------
# Jobs 10-12: "Mixed" = run all three sources under adaptive only, then
# produce a combined summary in post-processing.  Here we re-use the three
# separate JSON files already produced above (no extra benchmark run needed).
# We record which files constitute the mixed group in the manifest.
# ---------------------------------------------------------------------------
log "--- Workload group 4/4: mixed (post-processing group, no new runs) ---"
cat >> "${RUN_DIR}/manifest.txt" <<'MIXED'

Mixed workload group
--------------------
The "mixed" workload is the union of the synthetic, faers, and ggvp result
files.  No additional benchmark runs are needed.  To analyse them together,
load all three JSON files in the analysis notebook or aggregation script:
  synthetic_adaptive_threshold.json
  faers_adaptive_threshold.json
  ggvp_adaptive_threshold.json
MIXED

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
log "--- Generating plots ---"
for json in "${JSON_DIR}"/*.json; do
    base="$(basename "${json}" .json)"
    plot_subdir="${PLOT_DIR}/${base}"
    mkdir -p "${plot_subdir}"
    log "  Plotting ${base}..."
    python visualize.py \
        --input "${json}" \
        --output-dir "${plot_subdir}" \
        2>&1 | tee "${LOG_DIR}/${base}_plots.log" || \
        log "  WARNING: plotting failed for ${base} (non-fatal)"
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
log "=== Run complete ==="
log "Output directory : ${RUN_DIR}"
log "JSON files       : ${JSON_DIR}"
log "Logs             : ${LOG_DIR}"
log "Plots            : ${PLOT_DIR}"
log "Passed           : ${PASS_COUNT}"
log "Failed           : ${FAIL_COUNT}"

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    log "Failed jobs:"
    for j in "${FAILED_JOBS[@]}"; do
        log "  - ${j}  (log: ${LOG_DIR}/${j}.log)"
    done
    exit 1
fi

log "All jobs passed."
