# Babbage FAERS pilot package

This delivery branch is independent of master. It contains a prebuilt x86_64
Apptainer image split into GitHub-sized chunks, source code, tests, checksums,
and two frozen 50,000-document dataset archives. The image embeds only the
1,000-document diagnostic subset from each quarter. No package installation,
image conversion, database administrator, or internet access is needed inside
the Slurm job.

## Download directly on Babbage

After logging in as `njl2z`, clone the delivery branch specified in the final
delivery instructions, change into its `cluster` directory, and run:

```bash
bash install.sh
sbatch --test-only run_pilot.slurm
bash submit.sh --quick
```

The quick job uses 30 documents per quarter, two measured repetitions and one
warmup repetition. Check `squeue -u "$USER"` and the job's output file. The
`RESULTS=` line identifies the output directory. Its `pilot/status.json` must
contain `"status": "passed"`, a passed security selftest, and 24 reports.
After it passes, run `bash submit.sh` for the 1,000-document-per-quarter pilot,
two measured and two warmup repetitions (48,000 measured roundtrips).

The script requests `research-cpu`, one node, one task, four CPU cores, 8 GiB
RAM, and two hours. It loads `apptainer/1.5.4`. These match the supplied Babbage
module/partition listing; node-level execution still requires the quick job.
No GPU is used. Keep at least 5 GiB free for checkout, assembled image and
results. Optional site requirements can be supplied through
`PQC_SLURM_ACCOUNT` and `PQC_SLURM_QOS`. Scratch defaults to `SLURM_TMPDIR`,
then `TMPDIR`, then `/tmp`; override with `PQC_SCRATCH_ROOT`. Results default
to `cluster/results`; override with an absolute `PQC_RESULTS_ROOT`.

## Data and policy

The rebuilt samples use latest numeric case version within each quarter,
independent seed-42 random samples without replacement, 50,000 cases each.
The preprocessing script preserves strings, blanks and duplicate child rows
from all seven ASCII tables. It groups child rows by primaryid and checks
caseid consistency, then writes compact, sorted UTF-8 JSON. Raw ASCII files
are not altered. Outcomes DE or LT set an external `elevation_flag`; absence
of either is not evidence that the report is clinically safe.

Each JSON has demographics plus drugs, reactions, indications, outcomes,
report_sources and therapies. Manifest entries contain identifiers, quarter,
payload bytes, SHA-256, DE/LT booleans and `faers-de-lt-v1` rule. The loader
checks exact membership, hashes and flags against the JSON outcomes.

**These are newly rebuilt manifests.** Do not reuse numerical summaries or
manifest hashes from the lost workspace without checking them against these
files. Sampling and serialization are recorded in `metadata/preparation.json`.
The image subset covers available importance/size strata and minimum/maximum
sizes, then fills randomly. It is deliberately diagnostic, not representative
of the original 50,000 records or of FAERS as a whole. The frozen full samples
are available in `dataset-archives/` for later main experiments.

For each quarter the pilot runs 12 configurations: uniform baseline
(ML-KEM-768/ML-DSA-65), uniform strong (ML-KEM-1024/ML-DSA-87), size-only at
10/25/50/100/250 KiB, and size-or-importance at the same thresholds. A size
strictly greater than threshold elevates. The importance mode also elevates
every DE/LT-flagged document. Uniform controls ignore flags. Every write uses
hybrid X25519+ML-KEM, AES-256-GCM, and an ML-DSA signature. Reads require all
signature fields, check the expected algorithm, verify before decrypting,
and compare decrypted plaintext with the input bytes.

## Measurements and validity

`app.hpc_pilot` is the runner for this rebuilt scope. The legacy
`app.benchmark` CLI and old plotting scripts are retained for reference;
they do not consume the new reports and are not used by this package.

Each configuration uses a fresh database in a private PostgreSQL 16 instance
with a private Unix socket and no listening TCP port. Keys are generated
outside measured operations. Configuration order is seeded and shuffled;
within-quarter payload order matches across configurations and is checked
by hash. Warmup operations use the same database before measurement; this
is warmup-conditioned execution, not a guaranteed fully warm OS cache.

Each measured put/read roundtrip is retained in `samples.jsonl` with paired
encrypt, sign, DB insert, put-total, DB fetch, verify, decrypt, get-total and
roundtrip times in milliseconds. Writes include the original audit operation.
Raw samples exclude warmups. Each report computes count, mean, sample standard
deviation, minimum, maximum, median, p95 and p99 from actual pooled observations,
using linearly interpolated quantiles. Overall, flagged/unflagged, tier, and
size-bin statistics are separate. Repetitions are not independent documents.

Coverage counts each input document once: strong-record fraction, strong-byte
fraction, flagged strong coverage and elevation reasons. Logical envelope
storage sums ciphertext, nonce, tag, wrapped DEK, ephemeral key, salt, HKDF
info, AAD, PQ ciphertext and signature once per unique document. It excludes
PostgreSQL pages, indexes, audit records, labels, and key material. Repeated
writes and warmups are not multiplied into this logical cost. The actual DB
will occupy additional disk space.

Outputs include source/environment versions, selected manifests, schedule,
per-configuration reports/raw samples and status. Any integrity, signature,
roundtrip or measurement-count failure stops the pilot and retains a failed
status. The server is stopped on exit and only this job's private scratch is
removed. An abrupt cluster kill can prevent status finalization: a missing or
running status is not a pass. This diagnostic pilot is not a final statistical
study; freeze main-run repetitions/block design only after reviewing feasibility.

## Reproducibility

The image base is pinned by digest in `Containerfile`; native liboqs and its
Python wrapper are both 0.14.0. The original requirement 0.14.1 was unavailable.
Native builds use `OQS_DIST_BUILD=ON` to avoid tuning to the build host CPU.
The package includes pinned source archives and the runtime dependency listing.
The container runs as the ordinary HPC user and does not require root.
The compatibility report distinguishes local tests from pending Babbage tests.
