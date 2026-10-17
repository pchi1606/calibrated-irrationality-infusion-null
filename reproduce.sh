#!/usr/bin/env bash
# Author: Prabhu Rangaraju
# One-command reproduction of the S10 preregistered null-result analysis.
#
# Usage:  bash reproduce.sh
# Run from this directory. Requires Python 3.10+ and pip.
#
# What it does:
#   1. installs Python dependencies (requirements.txt: matplotlib, numpy)
#   2. re-runs the full analysis pipeline on the released raw generations
#      (scripts/pilot_analysis.py -> results/pilot_summary.json)
#   3. regenerates Figure 1 (scripts/make_figures.py)
#   4. verifies every headline number in the paper against expected/
#      (scripts/verify.py) and reports PASS/FAIL
#
# What it does NOT do: regenerate the 960 model generations themselves.
# That requires downloading Qwen3-1.7B-Instruct (GGUF Q4_K_M, ~1.3 GB) and
# running llama.cpp's llama-server on CPU (~2.5 h for the full battery).
# The raw generations are released here (results/pilot_results.jsonl) so
# the entire analysis is verifiable without re-running inference.
# See README.md for the full-regeneration instructions.
#
# Expected runtime: < 2 minutes on a modest CPU.
set -euo pipefail
cd "$(dirname "$0")"

echo "================================================================"
echo " STEP 1/4: dependencies"
echo "================================================================"
pip install -r requirements.txt || echo "WARNING: pip install failed; continuing with installed dependencies."

echo "================================================================"
echo " STEP 2/4: re-run analysis pipeline on released raw data"
echo "================================================================"
python3 scripts/pilot_analysis.py

echo "================================================================"
echo " STEP 3/4: regenerate Figure 1"
echo "================================================================"
python3 scripts/make_figures.py

echo "================================================================"
echo " STEP 4/4: verify headline numbers"
echo "================================================================"
python3 scripts/verify.py
