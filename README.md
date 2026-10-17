# Reproduction package — S10: A Small Language Model Does Not Reliably Exhibit Classic Decision Biases

**Author:** Prabhu Rangaraju

This package reproduces every number in the paper
*A Small Language Model Does Not Reliably Exhibit Classic Decision
Biases: A Preregistered Null Result in Qwen3-1.7B*
(`paper/paper.pdf`) from the released raw model generations.

## Quick start

```bash
bash reproduce.sh
```

Expected runtime: under 2 minutes on a modest CPU. Python 3.10+ required.
The script installs `matplotlib` and `numpy`, re-runs the analysis
(`scripts/pilot_analysis.py`), regenerates Figure 1
(`scripts/make_figures.py`), and verifies every headline number against
`expected/headline_numbers.json`, printing PASS/FAIL per check.

## What is verified

- Data completeness: exactly 960 pilot records, no missing/duplicated keys
- Preregistered gate: 2/5 biases pass (framing, anchoring); per-formulation p-values
- Bias magnitudes per lever (Table 2 of the paper)
- Calibration errors: L0 0.358 < L1 0.437, L2 0.416, L3 0.418
- Bootstrap 95% CIs for mean calibration error (20,000 replicates):
  L0 [0.338, 0.391], L1 [0.417, 0.451], L2 [0.399, 0.434], L3 [0.404, 0.437]
- Figure 1 regenerated from raw data (no mockups)

## Layout

- `results/pilot_results.jsonl` — the 960 raw model generations (released data)
- `scripts/pilot_analysis.py` — analysis pipeline (gate tests, magnitudes, calibration)
- `scripts/pilot.py` — output parser used by the analysis (extracted; the full
  generation harness is not needed to verify the analysis)
- `scripts/make_figures.py` — regenerates Figure 1 from raw data
- `scripts/verify.py` — headline-number verifier (incl. bootstrap CI checks
  via `scripts/calibration_bootstrap.py`)
- `scripts/calibration_bootstrap.py` — bootstrap 95% CIs for per-lever mean
  calibration error (20,000 replicates, fixed seed); reproduces the CIs in §4.2
- `expected/headline_numbers.json` — the paper's headline numbers
- `paper/paper.pdf` — the manuscript

## Full regeneration from scratch (optional, not run by reproduce.sh)

To regenerate the 960 generations themselves:

1. Download Qwen3-1.7B-Instruct GGUF Q4_K_M (~1.3 GB) and build llama.cpp.
2. Start `llama-server` (2 threads, context 2048).
3. Run the pilot battery driver (see the project repository for
   `code/pilot.py`; decoding: temperature 0.7, top-p 0.9, max 64 tokens,
   `/no_think` suffix, fixed per-sample seeds).

Takes ~2.5 hours on a 2-CPU machine. The analysis here runs on the
released generations, so this step is not required for verification.

## Preregistration

The study preregistration (`PREREGISTRATION.md`, deviations in
`DEVIATIONS.md`) was written before any experimental runs and is
released with the code.

## License

MIT License — see `LICENSE`. Copyright 2026 Prabhu Rangaraju.
