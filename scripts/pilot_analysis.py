"""Analyze pilot battery: gate decision + calibration-error per lever.

Reads results/pilot_results.jsonl. Prints:
  1. parse rates per bias
  2. gate tests on L0 (human-direction effect, one-sided alpha=0.05)
  3. bias magnitudes per lever + calibration error vs human targets
Writes results/pilot_summary.json
"""

import json
import math
import os
import re
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(BASE, "results", "pilot_results.jsonl")
OUT = os.path.join(BASE, "results", "pilot_summary.json")

sys.path.insert(0, os.path.join(BASE, "code"))
from pilot import num_parse  # noqa: E402  (reuse parser)

NUM_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def binom_one_sided_p(k, n, p0=0.5):
    # P(X >= k)
    from math import comb
    return sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
               for i in range(k, n + 1))


def mann_whitney_greater_p(xs, ys):
    # normal approx, one-sided: median(xs) > median(ys)
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return 1.0
    vals = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys])
    ranks, i = {}, 0
    # average ranks for ties
    rank = 1
    j = 0
    while j < len(vals):
        k = j
        while k < len(vals) and vals[k][0] == vals[j][0]:
            k += 1
        avg = (rank + rank + (k - j) - 1) / 2
        for t in range(j, k):
            ranks.setdefault(vals[t][1], []).append(avg)
        rank += (k - j)
        j = k
    r1 = sum(ranks[0])
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return 1.0
    z = (u1 - mu) / sd
    return 1 - norm_cdf(z)


def load():
    recs = []
    with open(IN) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def first_num(text):
    t = text.replace(",", "")
    m = NUM_RE.search(t)
    return float(m.group(1)) if m else None


def choice_letter(text):
    m = re.search(r"\b([AB])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"\b([12])\b", text)
    return m.group(1) if m else None


def analyze(recs):
    cell = defaultdict(list)  # (bias, formulation, lever, kind) -> [parsed values]
    parsed_ok = defaultdict(int)
    parsed_tot = defaultdict(int)
    for r in recs:
        bias, fi, lever, kind = r["bias"], r["formulation"], r["lever"], r["kind"]
        t = r["raw"]
        parsed_tot[bias] += 1
        v = None
        if bias == "framing":
            c = choice_letter(t)
            # certain option is A in formulations 0,1; B in formulation 2
            if c:
                certain = "A" if fi in (0, 1) else "B"
                v = 1.0 if c == certain else 0.0
        elif bias == "conjunction":
            c = choice_letter(t)
            v = 1.0 if c == "2" else (0.0 if c in ("1", "A", "B") else None)
            if c in ("A", "B"):
                v = None  # model answered with letters: unparseable here
        elif bias == "anchoring":
            v = first_num(t)
        elif bias == "endowment":
            v = first_num(t)
            if v is not None and v > 10000:
                v = None
        elif bias == "crt":
            v = first_num(t)
        if v is not None:
            parsed_ok[bias] += 1
            cell[(bias, fi, lever, kind)].append(v)
    return cell, parsed_ok, parsed_tot


CRT_INTUITIVE = {0: {0.10, 10.0}, 1: {100.0}, 2: {24.0}}


def gate_tests(cell):
    """Human-direction effect tests on L0, per formulation. Returns list of
    (bias, formulation, p, direction_ok)."""
    out = []
    for bias in ("framing", "conjunction", "anchoring", "endowment", "crt"):
        for fi in range(3):
            if bias == "framing":
                kinds = ["frame_gain", "frame_loss"]
                g = cell.get((bias, fi, "L0", "frame_gain"), [])
                l = cell.get((bias, fi, "L0", "frame_loss"), [])
                kg = sum(1 for x in g if x == 1.0)
                kl = sum(1 for x in l if x == 1.0)
                # Fisher-ish: use two-proportion z, one-sided
                n1, n2 = len(g), len(l)
                if n1 and n2:
                    p1, p2 = kg / n1, kl / n2
                    p = (kg + kl) / (n1 + n2)
                    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
                    z = (p1 - p2) / se if se > 0 else 0
                    pv = 1 - norm_cdf(z)
                else:
                    pv, p1, p2 = 1.0, 0, 0
                out.append((bias, fi, pv, p1 > p2, {"gain_certain": p1,
                                                   "loss_certain": p2}))
            elif bias == "conjunction":
                v = cell.get((bias, fi, "L0", "conj"), [])
                k = sum(1 for x in v if x == 1.0)
                pv = binom_one_sided_p(k, len(v)) if v else 1.0
                out.append((bias, fi, pv, k / len(v) > 0.5 if v else False,
                            {"violation_rate": k / len(v) if v else None}))
            elif bias == "anchoring":
                lo = cell.get((bias, fi, "L0", "anchor_low"), [])
                hi = cell.get((bias, fi, "L0", "anchor_high"), [])
                pv = mann_whitney_greater_p(hi, lo)
                import statistics
                out.append((bias, fi, pv,
                            (statistics.mean(hi) > statistics.mean(lo)
                             if hi and lo else False),
                            {"mean_low": statistics.mean(lo) if lo else None,
                             "mean_high": statistics.mean(hi) if hi else None}))
            elif bias == "endowment":
                wta = cell.get((bias, fi, "L0", "wta"), [])
                wtp = cell.get((bias, fi, "L0", "wtp"), [])
                pv = mann_whitney_greater_p(wta, wtp)
                import statistics
                out.append((bias, fi, pv,
                            (statistics.median(wta) > statistics.median(wtp)
                             if wta and wtp else False),
                            {"median_wta": statistics.median(wta) if wta else None,
                             "median_wtp": statistics.median(wtp) if wtp else None}))
            elif bias == "crt":
                vals = cell.get((bias, fi, "L0", "crt"), [])
                intuitive = CRT_INTUITIVE[fi]
                k = sum(1 for x in vals if x in intuitive)
                pv = binom_one_sided_p(k, len(vals)) if vals else 1.0
                out.append((bias, fi, pv, k / len(vals) > 0.5 if vals else False,
                            {"intuitive_rate": k / len(vals) if vals else None}))
    return out


HUMAN_TARGETS = {
    # bias -> (target on the [0,1]-scaled metric, description)
    "framing": (0.50, "P(certain|gain)-P(certain|loss): 0.72-0.22"),
    "conjunction": (0.85, "P(rank conjunction first)"),
    "anchoring": (0.40, "(est_high-est_low)/(anchor_high-anchor_low)"),
    "endowment": (0.50, "min(WTA/WTP,4)/4 with target ratio 2.0 -> 0.5"),
    "crt": (0.60, "mean intuitive-error rate"),
}

ANCHOR_SPANS = [(15, 65), (120, 450), (1900, 1950)]


def bias_magnitude(bias, lever, cell):
    """Aggregate measured bias on [0,1] scale across formulations."""
    import statistics
    mags = []
    for fi in range(3):
        if bias == "framing":
            g = cell.get((bias, fi, lever, "frame_gain"), [])
            l = cell.get((bias, fi, lever, "frame_loss"), [])
            if g and l:
                mags.append(sum(g) / len(g) - sum(l) / len(l))
        elif bias == "conjunction":
            v = cell.get((bias, fi, lever, "conj"), [])
            if v:
                mags.append(sum(v) / len(v))
        elif bias == "anchoring":
            lo = cell.get((bias, fi, lever, "anchor_low"), [])
            hi = cell.get((bias, fi, lever, "anchor_high"), [])
            a_lo, a_hi = ANCHOR_SPANS[fi]
            if lo and hi:
                m = (statistics.mean(hi) - statistics.mean(lo)) / (a_hi - a_lo)
                mags.append(max(0.0, min(1.0, m)))
        elif bias == "endowment":
            wta = cell.get((bias, fi, lever, "wta"), [])
            wtp = cell.get((bias, fi, lever, "wtp"), [])
            if wta and wtp and statistics.median(wtp) > 0:
                ratio = statistics.median(wta) / statistics.median(wtp)
                mags.append(min(ratio, 4.0) / 4.0)
        elif bias == "crt":
            vals = cell.get((bias, fi, lever, "crt"), [])
            intuitive = CRT_INTUITIVE[fi]
            if vals:
                mags.append(sum(1 for x in vals if x in intuitive) / len(vals))
    return statistics.mean(mags) if mags else None


def main():
    recs = load()
    print(f"records: {len(recs)}")
    cell, ok, tot = analyze(recs)
    print("\n-- parse rates --")
    all_ok = all_tot = 0
    for b in ("framing", "conjunction", "anchoring", "endowment", "crt"):
        r = ok[b] / tot[b] if tot[b] else 0
        print(f"  {b}: {ok[b]}/{tot[b]} = {r:.3f}")
        all_ok += ok[b]
        all_tot += tot[b]
    print(f"  OVERALL: {all_ok}/{all_tot} = {all_ok / all_tot:.3f}")

    print("\n-- gate tests (L0, human-direction, one-sided a=0.05) --")
    gt = gate_tests(cell)
    bias_pass = {}
    for bias in ("framing", "conjunction", "anchoring", "endowment", "crt"):
        rows = [t for t in gt if t[0] == bias]
        passed = sum(1 for _, _, pv, dok, _ in rows if pv < 0.05 and dok)
        bias_pass[bias] = passed
        for _, fi, pv, dok, info in rows:
            print(f"  {bias} f{fi}: p={pv:.4f} dir_ok={dok} {info}")
        print(f"  => {bias}: {passed}/3 formulations pass")

    n_bias_pass = sum(1 for b, c in bias_pass.items() if c >= 2)
    saturated = []
    print("\n-- bias magnitudes per lever (L0 = uncalibrated) --")
    cal_err = {}
    for lever in ("L0", "L1", "L2", "L3"):
        errs = []
        row = []
        for bias in ("framing", "conjunction", "anchoring",
                     "endowment", "crt"):
            m = bias_magnitude(bias, lever, cell)
            tgt = HUMAN_TARGETS[bias][0]
            e = abs(m - tgt) if m is not None else None
            if e is not None:
                errs.append(e)
            row.append(f"{bias}={m:.3f}" if m is not None else f"{bias}=NA")
            if lever == "L0" and m is not None and m > 0.95:
                saturated.append(bias)
        cal_err[lever] = sum(errs) / len(errs) if errs else None
        print(f"  {lever}: " + " ".join(row) +
              f" | mean|cal err|={cal_err[lever]:.3f}")

    best = min(cal_err, key=lambda k: cal_err[k])
    print(f"\n-- calibration: best lever = {best} "
          f"(err {cal_err[best]:.3f} vs L0 {cal_err['L0']:.3f})")

    gate_ok = (n_bias_pass >= 3 and all_ok / all_tot >= 0.90
               and not saturated)
    print(f"\nGATE: biases_passing(>=2/3 forms)={n_bias_pass}/5, "
          f"parse={all_ok / all_tot:.3f}, saturated={saturated}")
    print("GATE RESULT:", "PASS - proceed to calibration" if gate_ok
          else "FAIL - stop the paper")

    summary = {
        "n_records": len(recs),
        "parse_rates": {b: ok[b] / tot[b] for b in tot},
        "overall_parse": all_ok / all_tot,
        "gate_formulation_passes": bias_pass,
        "biases_passing": n_bias_pass,
        "saturated": saturated,
        "calibration_error": cal_err,
        "best_lever": best,
        "gate_pass": gate_ok,
        "human_targets": {b: HUMAN_TARGETS[b][0] for b in HUMAN_TARGETS},
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
