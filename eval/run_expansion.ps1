# Runs the added detectors one at a time (never two jobs at once on this 8 GB laptop).
# GPU pass: COCO-300 + blank + stability probes (accuracy and GPU latency).
# CPU pass: COCO-300 for latency (OWLv2 and Florence-2: first 60 images, they are slow on CPU).
# Usage: powershell -File eval\run_expansion.ps1 [-Models rfdetr-nano,dfine-n] [-Pass gpu|cpu|both]
param([string[]]$Models = @("yolo26n", "rfdetr-nano", "rfdetr-small", "dfine-n", "dfine-s", "rtmdet-tiny", "rtmdet-s",
                            "owlv2-base", "florence2-base"),
      [string]$Pass = "both")
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
foreach ($m in $Models) {
    $py = if ($m -like "rtmdet*") { ".venv-mmdet\Scripts\python.exe" } else { ".venv-gpu\Scripts\python.exe" }
    if ($Pass -in @("gpu", "both")) {
        Write-Output "=== $m cuda $(Get-Date -Format HH:mm:ss)"
        & $py eval\12_run_model.py $m --device cuda --tasks coco,null,stab 2>&1 | Select-String -NotMatch "^\s*$" | Select-Object -Last 4
    }
    if ($Pass -in @("cpu", "both")) {
        $limit = if ($m -in @("owlv2-base", "florence2-base")) { 60 } else { 0 }
        Write-Output "=== $m cpu $(Get-Date -Format HH:mm:ss)"
        & $py eval\12_run_model.py $m --device cpu --tasks coco --limit $limit 2>&1 | Select-String -NotMatch "^\s*$" | Select-Object -Last 4
    }
}
Write-Output "=== done $(Get-Date -Format HH:mm:ss)"
