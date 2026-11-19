# run_suite_v2.ps1
# Second experimental suite: per-quarter dataset (2 x 25,000), classical
# baseline, and selective (content-flag) elevation.  Results go to
# results/paired_v2/; the v1 suite in results/paired/ is untouched.
# Each step is resumable: re-running this script skips completed chunks.
#
# Dataset: Data/faers_q25k (25,000 reports per quarter, 2025 Q4 + 2026 Q1).
# Subsets (5,000) are stratified: 2,500 per quarter, identical across steps.
#
#   E6 r1  stability replicate 1     5,000  classical,baseline,strong                 ~17 min
#   E3     main paired study        50,000  classical,baseline,strong                 ~2.9 h
#   E6 r2  stability replicate 2     5,000  classical,baseline,strong                 ~17 min
#   E5     direct policy check       5,000  baseline,strong + 2 size, 3 flag, 1 combo  ~45 min
#   E7     crypto only (no DB)       5,000  classical,baseline,strong                 ~4 min
#   E8     synchronous_commit=off    5,000  classical,baseline,strong                 ~10 min
#   E6 r3  stability replicate 3     5,000  classical,baseline,strong                 ~17 min
#
# Uniform, size-threshold and flag policies for the main tables are derived
# offline from E3 (a record's latency under a policy is its latency under the
# tier assigned); E5 runs a selection of them directly to validate that.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$root = "paired_v2"
$log = "results/$root/suite.log"
New-Item -ItemType Directory -Force "results/$root" | Out-Null

$common = @("--dataset", "faers_q25k", "--iterations", "20", "--warmup", "3")
$tiers = @("--profiles", "classical,baseline,strong")
$subset = @("--records", "5000", "--stratify-quarter")
$e5 = "baseline,strong,adaptive:1458,adaptive:4196,flag:death,flag:sensitive,flag:severe,flagsize:death:4196"
$steps = @(
    @{ name = "e6_stability_r1"; args = $subset + $tiers },
    @{ name = "e3_main";         args = $tiers },
    @{ name = "e6_stability_r2"; args = $subset + $tiers },
    @{ name = "e5_direct";       args = $subset + @("--profiles", $e5) },
    @{ name = "e7_nodb";         args = $subset + $tiers + @("--no-db") },
    @{ name = "e8_async_commit"; args = $subset + $tiers + @("--sync-commit", "off") },
    @{ name = "e6_stability_r3"; args = $subset + $tiers }
)

function Log($msg) { $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg; $line | Tee-Object -FilePath $log -Append }

Log "=== suite v2 start ==="
foreach ($s in $steps) {
    Log "--- $($s.name) start: $($s.args + $common -join ' ')"
    $t0 = Get-Date
    try {
        & "$PSScriptRoot\run_paired.ps1" $s.name -Root $root @($s.args + $common) 2>$null | Out-Null
        Log ("--- {0} OK ({1:N1} min)" -f $s.name, ((Get-Date) - $t0).TotalMinutes)
    } catch {
        Log ("--- {0} FAILED after {1:N1} min: {2}" -f $s.name, ((Get-Date) - $t0).TotalMinutes, $_)
    }
}
Log "=== suite v2 end ==="
