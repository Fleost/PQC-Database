# run_suite.ps1
# Full policy-latency experimental suite (E3, E5-E8), run sequentially.
# Each step is resumable: re-running this script skips completed chunks.
#
# Order is chosen so the three E6 stability replicates are spread across the
# session (start, middle, end), giving "different times" without extra effort.
#
#   E6 r1  stability replicate 1      5,000 records  baseline,strong            ~12 min
#   E3     main paired study         50,000 records  baseline,strong            ~2.0 h
#   E6 r2  stability replicate 2      5,000 records  baseline,strong            ~12 min
#   E5     direct adaptive check      5,000 records  + adaptive:1458,4196       ~24 min
#   E7     crypto only (no DB)        5,000 records  baseline,strong            ~3 min
#   E8     synchronous_commit=off     5,000 records  baseline,strong            ~7 min
#   E6 r3  stability replicate 3      5,000 records  baseline,strong            ~12 min
#
# All runs: 20 measured iterations + 3 warmup per (record, profile).
# The 5,000-record subset is identical across E5-E8 (same --seed).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = "results/paired/suite.log"
New-Item -ItemType Directory -Force "results/paired" | Out-Null

$common = @("--iterations", "20", "--warmup", "3")
$subset = @("--records", "5000")
$steps = @(
    @{ name = "e6_stability_r1"; args = $subset },
    @{ name = "e3_main";         args = @() },
    @{ name = "e6_stability_r2"; args = $subset },
    @{ name = "e5_direct";       args = $subset + @("--profiles", "baseline,strong,adaptive:1458,adaptive:4196") },
    @{ name = "e7_nodb";         args = $subset + @("--no-db") },
    @{ name = "e8_async_commit"; args = $subset + @("--sync-commit", "off") },
    @{ name = "e6_stability_r3"; args = $subset }
)

function Log($msg) { $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg; $line | Tee-Object -FilePath $log -Append }

Log "=== suite start ==="
foreach ($s in $steps) {
    Log "--- $($s.name) start: $($s.args + $common -join ' ')"
    $t0 = Get-Date
    try {
        & "$PSScriptRoot\run_paired.ps1" $s.name @($s.args + $common) 2>$null | Out-Null
        Log ("--- {0} OK ({1:N1} min)" -f $s.name, ((Get-Date) - $t0).TotalMinutes)
    } catch {
        Log ("--- {0} FAILED after {1:N1} min: {2}" -f $s.name, ((Get-Date) - $t0).TotalMinutes, $_)
    }
}
Log "=== suite end ==="
