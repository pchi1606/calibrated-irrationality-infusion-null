"""Bootstrap 95% CIs for per-lever mean calibration error (Section 4.2).

Resamples parsed generations with replacement within each
(bias, formulation, lever, kind) cell, recomputes bias magnitudes on the
[0,1] scale (same definitions as pilot_analysis.bias_magnitude), then the
per-lever mean absolute deviation from the preregistered human targets.
20,000 replicates; percentile 95% CI. Fixed seed for reproducibility.

Reproduces the CIs reported in the manuscript:
  L0 [0.338, 0.391], L1 [0.417, 0.451], L2 [0.399, 0.434], L3 [0.404, 0.437]
"""

import random
import os
import statistics
import importlib.util
from collections import defaultdict

SEED = 20261015
REPLICATES = 20000
BIASES = ("framing", "conjunction", "anchoring", "endowment", "crt")
LEVERS = ("L0", "L1", "L2", "L3")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pa", os.path.join(HERE, "pilot_analysis.py"))
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

TARGETS = {b: pa.HUMAN_TARGETS[b][0] for b in BIASES}


def lever_errors(draw):
    out = {}
    for lever in LEVERS:
        errs = []
        for bias in BIASES:
            mags = []
            for fi in range(3):
                if bias == "framing":
                    g = draw.get((bias, fi, lever, "frame_gain"), [])
                    l = draw.get((bias, fi, lever, "frame_loss"), [])
                    if g and l:
                        mags.append(sum(g) / len(g) - sum(l) / len(l))
                elif bias == "conjunction":
                    v = draw.get((bias, fi, lever, "conj"), [])
                    if v:
                        mags.append(sum(v) / len(v))
                elif bias == "anchoring":
                    lo = draw.get((bias, fi, lever, "anchor_low"), [])
                    hi = draw.get((bias, fi, lever, "anchor_high"), [])
                    a_lo, a_hi = pa.ANCHOR_SPANS[fi]
                    if lo and hi:
                        m = (statistics.mean(hi) - statistics.mean(lo)) / (a_hi - a_lo)
                        mags.append(max(0.0, min(1.0, m)))
                elif bias == "endowment":
                    wta = draw.get((bias, fi, lever, "wta"), [])
                    wtp = draw.get((bias, fi, lever, "wtp"), [])
                    if wta and wtp and statistics.median(wtp) > 0:
                        mags.append(min(statistics.median(wta) / statistics.median(wtp), 4.0) / 4.0)
                elif bias == "crt":
                    v = draw.get((bias, fi, lever, "crt"), [])
                    intu = pa.CRT_INTUITIVE[fi]
                    if v:
                        mags.append(sum(1 for x in v if x in intu) / len(v))
            if mags:
                errs.append(abs(statistics.mean(mags) - TARGETS[bias]))
        out[lever] = sum(errs) / len(errs)
    return out


def bootstrap_cis(replicates=REPLICATES, seed=SEED):
    """Return {lever: (lo, hi)} percentile-95% CIs for mean calibration error."""
    recs = pa.load()
    cell, _, _ = pa.analyze(recs)
    random.seed(seed)
    keys = list(cell.keys())
    boot = defaultdict(list)
    for _ in range(replicates):
        draw = {k: random.choices(cell[k], k=len(cell[k])) for k in keys}
        e = lever_errors(draw)
        for lever in LEVERS:
            boot[lever].append(e[lever])
    cis = {}
    for lever in LEVERS:
        s = sorted(boot[lever])
        n = len(s)
        cis[lever] = (s[int(0.025 * n)], s[int(0.975 * n)])
    return cis


def main():
    cis = bootstrap_cis()
    for lever in LEVERS:
        lo, hi = cis[lever]
        print(f"{lever}: 95% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
