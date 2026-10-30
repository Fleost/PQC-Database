#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c pilot.sif.sha256
args=()
if [[ $# -gt 0 ]]; then
    [[ $# == 1 && $1 == --quick ]] || { echo 'Usage: bash submit.sh [--quick]'; exit 2; }
    args+=(--export=ALL,PQC_SAMPLE_SIZE=30,PQC_ITERATIONS=2,PQC_WARMUPS=1)
fi
[[ -z ${PQC_SLURM_ACCOUNT:-} ]] || args+=(--account="$PQC_SLURM_ACCOUNT")
[[ -z ${PQC_SLURM_QOS:-} ]] || args+=(--qos="$PQC_SLURM_QOS")
sbatch "${args[@]}" run_pilot.slurm
