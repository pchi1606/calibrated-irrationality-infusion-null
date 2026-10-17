"""Verify headline numbers of the S10 negative-result paper.

Recomputes the analysis from results/pilot_results.jsonl and checks the
paper's headline claims against expected/headline_numbers.json.
Exit 0 = all checks pass, 1 = any failure.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pilot_analysis import load, analyze, bias_magnitude, gate_tests, HUMAN_TARGETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED = json.load(open(os.path.join(ROOT, "expected", "headline_numbers.json")))

failures = []


def check(name, actual, expected, tol=1e-9):
    ok = abs(actual - expected) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {actual}, expected {expected}")
    if not ok:
        failures.append(name)


def main():
    recs = load()
    cell, ok, tot = analyze(recs)

    print("== completeness ==")
    check("n_records", len(recs), EXPECTED["n_records"])
    all_ok = sum(ok[b] for b in ok)
    check("overall_parse", all_ok / len(recs), EXPECTED["overall_parse"], tol=1e-4)

    print("== gate ==")
    gt = gate_tests(cell)
    bias_pass = {}
    for bias in ("framing", "conjunction", "anchoring", "endowment", "crt"):
        rows = [t for t in gt if t[0] == bias]
        passed = sum(1 for _, _, pv, dok, _ in rows if pv < 0.05 and dok)
        bias_pass[bias] = passed
    n_pass = sum(1 for c in bias_pass.values() if c >= 2)
    check("biases_passing_gate", n_pass, EXPECTED["biases_passing_gate"])
    for bias, exp in EXPECTED["formulation_passes"].items():
        check(f"gate_{bias}_formulations", bias_pass[bias], exp)

    print("== bias magnitudes (L0) ==")
    for bias, exp in EXPECTED["l0_magnitudes"].items():
        m = bias_magnitude(bias, "L0", cell)
        check(f"l0_{bias}", m, exp, tol=1e-3)

    print("== calibration errors ==")
    for lever, exp in EXPECTED["calibration_error"].items():
        errs = []
        for bias in ("framing", "conjunction", "anchoring", "endowment", "crt"):
            m = bias_magnitude(bias, lever, cell)
            errs.append(abs(m - HUMAN_TARGETS[bias][0]))
        check(f"calerr_{lever}", sum(errs) / len(errs), exp, tol=1e-3)

    print("== bootstrap 95% CIs for mean calibration error ==")
    from calibration_bootstrap import bootstrap_cis
    cis = bootstrap_cis()
    for lever, exp in EXPECTED["bootstrap_cis"].items():
        lo, hi = cis[lever]
        check(f"bootci_{lever}_lo", lo, exp[0], tol=1e-3)
        check(f"bootci_{lever}_hi", hi, exp[1], tol=1e-3)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
