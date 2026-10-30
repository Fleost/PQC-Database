#!/usr/bin/env bash
#SBATCH --job-name=pqc-pilot
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=pqc-pilot-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the package directory}"
module load apptainer/1.5.4
apptainer version
[[ $(apptainer version) == 1.5.4* ]] || { echo 'Expected Apptainer 1.5.4'; exit 1; }
[[ $(uname -m) == x86_64 ]] || { echo 'This package requires x86_64'; exit 1; }
sha256sum -c pilot.sif.sha256
hostname
uname -srmo
command -v lscpu >/dev/null && lscpu
scratch_root=${PQC_SCRATCH_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}
work=$(mktemp -d "$scratch_root/pqc-${SLURM_JOB_ID}-XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
results_root=${PQC_RESULTS_ROOT:-$PWD/results}
mkdir -p "$results_root"
results=$(mktemp -d "$results_root/job-${SLURM_JOB_ID}-XXXXXXXX")
echo "RESULTS=$results"
df -h "$work" "$results"
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export APPTAINERENV_SLURM_NODELIST="${SLURM_NODELIST:-unknown}"
apptainer exec --cleanenv --bind "$work:/job,$results:/results" pilot.sif \
    bash /opt/pilot/container_entry.sh --sample-size "${PQC_SAMPLE_SIZE:-1000}" \
    --iterations "${PQC_ITERATIONS:-2}" --warmups "${PQC_WARMUPS:-2}"
echo "Pilot finished. Inspect $results/pilot/status.json"
