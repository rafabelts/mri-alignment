# Removes everything run_smoke_test.ps1 generated. Safe to run even if the
# smoke test failed partway through. Usage: .\clean_smoke_test.ps1
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .\smoke_data, .\smoke_checkpoints, .\smoke_outputs, .\smoke_test.py
Write-Host "Smoke test artifacts removed."
