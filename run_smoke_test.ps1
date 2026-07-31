# Runs a tiny end-to-end pass of nested_cv.py (12 patients, 1 epoch, reduced
# folds/grid) to catch wiring bugs before committing real GPU time.
# Usage (PowerShell, from the repo root): .\run_smoke_test.ps1

$ErrorActionPreference = "Stop"

Write-Host "Copying a 12-patient subset into .\smoke_data ..."
New-Item -ItemType Directory -Force -Path .\smoke_data | Out-Null
$patients = "A_001","A_003","A_004","A_005","B_002","B_003","B_006","B_007","C_001","C_004","C_005","C_006"
foreach ($p in $patients) {
    Copy-Item -Recurse -Force ".\data\TrackRad\$p" ".\smoke_data\$p"
}

$env:MRI_DATA_DIR = (Resolve-Path .\smoke_data).Path
$env:MRI_CHECKPOINT_DIR = "$PWD\smoke_checkpoints"
$env:MRI_OUTPUTS_DIR = "$PWD\smoke_outputs"

Write-Host "Running smoke test..."
uv run python smoke_test.py
