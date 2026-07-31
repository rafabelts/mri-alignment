#!/usr/bin/env bash
# Removes everything run_smoke_test.sh generated. Safe to run even if the
# smoke test failed partway through. Usage: bash clean_smoke_test.sh
rm -rf smoke_data smoke_checkpoints smoke_outputs smoke_test.py
echo "Smoke test artifacts removed."
