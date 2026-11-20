# Re-runs the open review of the 2022 held-out ranking in order (CPU only, about 30 minutes). From the worktree root:
#   powershell -File scripts/research/open_review_2022/run_all.ps1
$here = $PSScriptRoot
$wt = Join-Path (Split-Path (Split-Path (Split-Path $here -Parent) -Parent) -Parent) 'scripts\wt.ps1'
foreach ($s in 's01_reproduce', 's02_slices', 's03_markets', 's04_market_affinity', 's05_anyflow', 's06_loo_supply',
        's07_model_supply', 's08_artefact', 's09_affinity_variants', 's10_more', 's11_cases', 's12_structure', 's13_demand',
        's14_combined', 's15_access', 's16_blocked', 's17_products', 's18_affinity_robustness',
        's19_reach_plus_dollar', 's20_floor', 's21_floor_fair', 's22_large_markets') {
    # Windows PowerShell's `*>` writes UTF-16; Out-File -Encoding utf8 keeps the logs readable by other tools.
    & $wt python "scripts/research/open_review_2022/$s.py" 2>&1 | Out-File -Encoding utf8 (Join-Path $here "out\$s.txt")
    if ($LASTEXITCODE) { throw "$s failed" }
}
