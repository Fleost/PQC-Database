# Paired latency and storage study (v2)

Tools for the second experimental suite on FAERS: a paired, interleaved latency and storage study of
hybrid X25519 + ML-KEM envelopes signed with ML-DSA, against a classical X25519 + ECDSA P-256 baseline,
with size-threshold and content-flag elevation policies (NIST Category 3 → Category 5).

This folder is self-contained: it has its own copy of `app/`, the Docker setup and every script the
suite uses, so it runs without the files in the repository root. Data and results are not included;
the steps below rebuild them.

## What is new or changed compared with the repository root

| File | Status | Purpose |
|---|---|---|
| `scripts/build_faers_records.py` | new | Builds the dataset: one JSON document per FAERS report, from the FDA quarterly ASCII extracts |
| `app/paired_study.py` | new | The latency harness: paired, interleaved (ABBA) measurement of every record under every profile |
| `app/elevation.py` | new | Content-flag rules (death, fatal / life-threatening, severe, special category, …) |
| `analyze_paired.py` | new | Statistics: per-record medians, bootstrap CIs, Wilcoxon + Holm, TOST, policy and storage tables |
| `scripts/make_figure_package.py` | new | Figures and tables from the analysis output |
| `scripts/check_classical.py` | new | Correctness checks: classical round trip, tamper and wrong-key rejection, policy decisions |
| `run_paired.ps1` | new | Runs one experiment in the benchmark container (resumable) |
| `run_suite_v2.ps1` | new | Runs the whole v2 suite (E3, E5–E8) in order |
| `app/feasibility.py`, `run_feasibility.ps1` | new | Early feasibility probe (not used for the v2 results) |
| `run_suite.ps1` | new | The first (v1) suite, on the earlier unstratified sample (superseded by v2) |
| `app/stage2.py` | modified | Adds the ECDSA P-256 signature backend for the classical baseline |
| `Dockerfile`, `requirements.txt` | modified | liboqs 0.14.0 → 0.16.0 and liboqs-python 0.14.1 → 0.16.0.2 (0.14.1 is no longer published) |
| `docker-compose.yml` | modified | Removes the obsolete `version:` key |

Every other file in `app/` and `scripts/` is identical to the repository root.

## Requirements

- Docker Desktop (the benchmark and PostgreSQL 16 run in containers).
- Python 3.12 or later on the host for the dataset build, analysis and figures, with
  `pip install -r requirements-analysis.txt`.
- About 2 GB of disk for the FAERS extracts, plus about 1 GB for raw results.

## Reproducing the study

Run everything from this folder (PowerShell shown; the Python steps work the same on any OS).

### 1. Get the FAERS data

Download the **FAERS Quarterly Data Extract Files** (ASCII format) for **2025 Q4** and **2026 Q1** from
the FDA and unzip each into its own folder. Each folder must contain the `ASCII/` and `Deleted/`
subfolders from the FDA archive.

### 2. Build the dataset

```powershell
python scripts/build_faers_records.py --quarter <path>\Q42025 --quarter <path>\Q12026 `
    --out Data/faers_q25k --per-quarter 25000 --seed 20260101
```

This writes 50,000 JSON reports (25,000 per quarter) to `Data/faers_q25k/`, a manifest
(`Data/faers_q25k_manifest.csv`, with size, quarter and content flags per report) and a single-file
bundle (`Data/faers_q25k.tar`) that the benchmark reads. The sample is deterministic: the same
extracts and seed give byte-identical files.

### 3. Build the container and check correctness

```powershell
docker compose build benchmark
docker compose up -d db
docker compose run --rm benchmark python /app/scripts/check_classical.py
```

### 4. Run the suite

```powershell
.\run_suite_v2.ps1
```

This runs, in order, with results in `results/paired_v2/<experiment>/` and a log in
`results/paired_v2/suite.log` (about 4.8 hours in total on an 8-core desktop):

| Experiment | Records | Profiles | Purpose |
|---|---|---|---|
| E6 r1, r2, r3 | 5,000 | classical, Cat. 3, Cat. 5 | Stability replicates at the start, middle and end |
| E3 | 50,000 | classical, Cat. 3, Cat. 5 | Main study |
| E5 | 5,000 | Cat. 3, Cat. 5 + selected policies | Runs policies directly to validate the derived policy costs |
| E7 | 5,000 | classical, Cat. 3, Cat. 5 | No database (cryptographic cost only) |
| E8 | 5,000 | classical, Cat. 3, Cat. 5 | PostgreSQL `synchronous_commit = off` |

The 5,000-record subsets take 2,500 reports per quarter and are the same in every experiment. Each
step is resumable: re-running skips completed chunks. Keep the machine otherwise idle while it runs.

To validate the size-threshold sweep directly (the second E5 session):

```powershell
.\run_paired.ps1 e5_direct_sweep -Root paired_v2 --records 5000 --stratify-quarter `
    --profiles baseline,strong,adaptive:1332,adaptive:2067,adaptive:3558,adaptive:5083,adaptive:10901 `
    --dataset faers_q25k --iterations 20 --warmup 3
```

### 5. Analyse

```powershell
python analyze_paired.py --results results/paired_v2 --out results/analysis_v2 `
    --manifest Data/faers_q25k_manifest.csv
```

This writes the policy, storage, coverage, quarter and tier tables (CSV and LaTeX) and `summary.json`
to `results/analysis_v2/`. It takes about 15 minutes.

### 6. Figures

```powershell
python scripts/make_figure_package.py
```

This writes PDF and PNG figures and copies the tables to `analysis_package/`.

## Policies

- **Uniform:** every record at Category 3 (ML-KEM-768 / ML-DSA-65) or at Category 5
  (ML-KEM-1024 / ML-DSA-87).
- **Size threshold** (`adaptive:<bytes>`): Category 5 when the plaintext exceeds the threshold. The
  sweep elevates the largest 50 / 25 / 10 / 5 / 1 % of records: 1,332 / 2,067 / 3,558 / 5,083 /
  10,901 B on this sample (set in `analyze_paired.py`, `SWEEP`).
- **Content flag** (`flag:<rule>`): Category 5 when the report matches a rule in `app/elevation.py`.

Policy latency and storage are derived from the per-record Category 3 and Category 5 measurements of
E3; E5 runs policies directly to check that derivation.
