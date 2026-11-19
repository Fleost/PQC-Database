# run_paired.ps1
# Run app.paired_study inside the benchmark container. All arguments after the
# output name are passed through to the study.
#
#   .\run_paired.ps1 e3_main
#   .\run_paired.ps1 e5_direct --records 5000 --profiles adaptive:2990,adaptive:8091
#   .\run_paired.ps1 e7_nodb   --records 5000 --no-db
#
# Re-running with the same name resumes: completed chunks are skipped.
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Name,
    [string]$Root = "paired",   # results/<Root>/<Name>
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$StudyArgs
)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$outDir = "results/$Root/$Name"
New-Item -ItemType Directory -Force $outDir | Out-Null

docker compose up -d db
if ($LASTEXITCODE -ne 0) { throw "docker compose up db failed ($LASTEXITCODE)" }

$extra = ($StudyArgs -join " ")
docker compose run --rm benchmark bash -lc "/app/scripts/wait-for-postgres.sh db 5432 && python -m app.paired_study --out-dir /app/results/$Root/$Name $extra" |
    Tee-Object -FilePath "$outDir/run.log" -Append
if ($LASTEXITCODE -ne 0) { throw "paired study failed ($LASTEXITCODE); see $outDir/run.log" }
