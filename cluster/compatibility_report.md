# Compatibility and delivery checks

Target: MTSU Babbage, `x86_64`, `research-cpu`, Apptainer module `1.5.4`.
The submitted specification is one node/task, four CPUs, 8 GiB RAM and two
hours. The supplied partition listing allows three days on research-cpu.
No GPU is required.

## Verified locally

- Python unit suite: 154 passed, including the corrected 50 KiB boundary,
  importance assignment, signature-stripping rejection, source-faithful
  reconstruction, missing chunks, manifest corruption, warmup exclusion,
  logical storage counted once, and installer corruption checks.
- Both 50,000-document manifests passed payload/size/identifier/DE-LT checks.
  The seven source tables were read with strict field-width/header checks.
  Frozen full samples and provenance are included in `dataset-archives/`.
- The same application image runs as an ordinary user, with networking
  disabled, four CPUs and 8 GiB enforced, and its own PostgreSQL instance.
  The full pilot acceptance result and independently reconciled counts are
  recorded in `validation.json`.
- Signature selftests exercise all three ML-KEM and ML-DSA parameter sets,
  authenticated roundtrips, changed AAD and stripped signature fields.
- Official Apptainer 1.5.4 built the SIF from the tested Docker image and
  inspected its `amd64` SquashFS partition. See `sif-build.log`,
  `sif-inspection.txt` and `image_metadata.json`.
- A fresh shallow clone of the GitHub delivery branch successfully
  reassembled the SIF and matched its SHA-256. No Codex artifact service is
  used for delivery.

## Portability choices

Python, PostgreSQL 16, native liboqs, its matching 0.14.0 wrapper and Python
dependencies are inside the image. Host `psql`, `initdb`, `pg_ctl`, pip,
Docker and compilers are unnecessary. Native builds use CPU dispatch
(`OQS_DIST_BUILD=ON`) rather than tuning to the build host. Image inputs,
native source commits, frozen source and runtime versions are included.
Image assembly requires only Bash, coreutils and ordinary disk space.

The job uses a private 0700 scratch directory, a private Unix database
socket with TCP disabled, a new database for each configuration, fresh
result directories and explicit server cleanup. PostgreSQL data stays in
job scratch; logical storage metrics deliberately exclude physical DB
pages and audit/index space. The job needs no network access or credentials.

## Still requires Babbage validation

This cloud does not permit the user-namespace mappings needed to execute
Apptainer here (`setgroups` is read-only). Therefore a native SIF runtime
test, the site's actual Slurm accounting/QoS rules, compute-node module
initialization, available scratch, and the site's container security
configuration remain cluster checks. We do not claim that local Docker
execution is a native Babbage execution test.

First run `sbatch --test-only run_pilot.slurm`, then `bash submit.sh --quick`.
The latter is the actual compute-node compatibility check. Confirm
`pilot/status.json` says `passed` and lists all 24 reports before submitting
the default pilot. Optional account/QoS variables are documented in README.
The local latency measurements validate instrumentation; use Babbage results
for the study, not these cloud acceptance timings.
