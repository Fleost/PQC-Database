# run_feasibility.ps1
# Build the benchmark image, start PostgreSQL, and run the feasibility probe.
#
#   .\run_feasibility.ps1                       # default: 400 records x 20 iterations
#   .\run_feasibility.ps1 -Records 1000 -Iterations 30
param(
    [int]$Records = 400,
    [int]$Iterations = 20,
    [int]$Warmup = 3
)
# Native tools (docker) write progress/warnings to stderr; in Windows
# PowerShell 5.1 that must not be treated as fatal, so check exit codes instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "results/feasibility"
New-Item -ItemType Directory -Force $outDir | Out-Null

Write-Host "Building image and starting PostgreSQL..."
docker compose build benchmark
if ($LASTEXITCODE -ne 0) { throw "docker compose build failed ($LASTEXITCODE)" }
docker compose up -d db
if ($LASTEXITCODE -ne 0) { throw "docker compose up db failed ($LASTEXITCODE)" }

Write-Host "Running probe ($Records records x $Iterations iterations)..."
docker compose run --rm benchmark bash -lc "/app/scripts/wait-for-postgres.sh db 5432 && python -m app.feasibility --records $Records --iterations $Iterations --warmup $Warmup --output /app/results/feasibility/probe_$stamp.json" |
    Tee-Object -FilePath "$outDir/probe_$stamp.log"
if ($LASTEXITCODE -ne 0) { throw "probe failed ($LASTEXITCODE); see $outDir/probe_$stamp.log" }

Write-Host "Results: $outDir/probe_$stamp.json"
