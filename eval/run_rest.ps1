# Runs the Mirage probe and extract() evaluations in order (CPU-bound; run alone).
# Usage (from the repo root): .\eval\run_rest.ps1 -Val <val2017 dir> -Ann <instances_val2017.json>
param(
    [Parameter(Mandatory = $true)][string]$Val,
    [Parameter(Mandatory = $true)][string]$Ann
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CACTUS_TELEMETRY = '0'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = ".venv\Scripts\python.exe"
foreach ($sub in "null", "stab", "ocr") {
    & $py eval\run_dir.py mirage "results\probes\$sub" "results\probe_out_mirage\$sub"
}
& $py eval\06b_extract_eval.py $Val $Ann results\coco_mirage results\extract_eval.json 30
& $py eval\06d_extract_fairness.py $Val $Ann results\extract_eval.json results\extract_fairness.json 15
& $py eval\06e_extract_order.py $Val results\extract_fairness.json results\extract_order.json
& $py eval\06f_extract_ocr.py results\probes\ocr results\extract_ocr.json
"ALL_DONE"
