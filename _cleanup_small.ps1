$ErrorActionPreference = 'Continue'
$dest = 'archive/merge-2026-07'

$dirs = @(
    'quant_opt_20260615_trae','quant_opt_20260616','quant_opt_20260616_core',
    'quant_opt_20260617','quant_opt_20260617_r2','quant_opt_20260618','quant_opt_20260618_r3',
    'quant_opt_20260619','quant_opt_20260619_m3','quant_opt_20260620','quant_opt_run2_20260620',
    'quant_opt_20260621','quant_opt_20260623','quant_opt_20260623_r2','quant_opt_20260624',
    'quant_opt_experiments','quant_opt','optimizations_20260616','optimizations_20260621_r2',
    'optimizations_20260622_v2','optimizations_20260624','optimizations','optimization','opt_20260618',
    'reports_20260617_agent_m3','reports_20260618','reports','research_20260617','research',
    'tests_20260624','docs_20260624','skills_backtest_opt_20260624','skills_quant_opt_20260618',
    'experiments','jingni-trader','jingni-trader-experiments','quant_skills','validation'
)

$files = @(
    '3.1 StockBullStrategyV210.py','branch_merge_evaluation_report.md','consolidate_branches.py',
    'CONSOLIDATED_INDEX.md','feat-quant-opt-20260617.patch','integrate_to_main.py',
    'optimization_report.md','optimization_report_lowercase.md','QUANT_OPT_BENCHMARK_20260615.json',
    'QUANT_OPT_REPORT_20260615.md','REPORT_2026-06-17.md','scheduled_task_content.txt'
)

$moved = 0
foreach ($d in $dirs) {
    if (Test-Path $d) {
        & git mv -- "$d" "$dest/$d" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $moved++ } else { Write-Host "DIR FAIL: $d" }
    }
}
foreach ($f in $files) {
    if (Test-Path $f) {
        & git mv -- "$f" "$dest/$f" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $moved++ } else { Write-Host "FILE FAIL: $f" }
    }
}
& git add -A 2>&1 | Out-Null
Write-Host "git mv done: $moved items"
