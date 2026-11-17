#!/usr/bin/env python3
"""
Outcome reliability for the HANS heuristics, computed exactly as reported (ships the numbers in
the paper; review item 6).

Two split-half schemes over the non-entailment items of each heuristic, across all 100 models:
  - stratified random item split (what the step-3 gate in feather_eval.py used; inflated because
    items from the same template are near-duplicates);
  - template-blocked split (whole templates go to one half; the stricter number we report).
Both are given as the raw split-half r (mean of 1,000 splits, with the 2.5th percentile) and the
Spearman-Brown full-length reliability.

Usage: python reliability.py [feather_out]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


def split_half(Y, units, rng, n=1000):
    uu = np.unique(units); rs = []
    for _ in range(n):
        a = np.isin(units, rng.permutation(uu)[: len(uu) // 2])
        rs.append(pearsonr(Y[:, a].mean(1), Y[:, ~a].mean(1))[0])
    rs = np.array(rs); r = rs.mean()
    return r, np.percentile(rs, 2.5), 2 * r / (1 + r)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "feather_out")
    m = np.load(out / "meta.npz", allow_pickle=True)
    gold, heur, tmpl, sub = m["hans_gold_ent"], m["hans_heuristic"], m["hans_template"], m["hans_subcase"]
    files = sorted((out / "logits").glob("model_*.npz"))
    P = np.array([np.load(f)["hans_logits"].argmax(1) != 1 for f in files])   # predicts non-entailment
    rng = np.random.default_rng(0)
    print(f"{len(files)} models")
    for h in ["lexical_overlap", "subsequence", "constituent"]:
        mask = (heur == h) & ~gold; Y = P[:, mask]
        item_units = np.array([f"{s}:{k}" for k, s in enumerate(sub[mask])])   # random items (stratified below)
        # stratified random: split within each subcase
        rs = []
        for _ in range(1000):
            a = np.zeros(mask.sum(), bool)
            for s in np.unique(sub[mask]):
                ix = np.where(sub[mask] == s)[0]; a[rng.permutation(ix)[: len(ix) // 2]] = True
            rs.append(pearsonr(Y[:, a].mean(1), Y[:, ~a].mean(1))[0])
        r1 = np.mean(rs)
        r2, lo2, sb2 = split_half(Y, tmpl[mask], rng)
        print(f"{h:16s} stratified random split r {r1:.3f} (SB {2*r1/(1+r1):.3f})   "
              f"template-blocked r {r2:.3f} [2.5%: {lo2:.2f}] (SB {sb2:.3f}, {len(np.unique(tmpl[mask]))} templates)")
        del item_units


if __name__ == "__main__":
    main()
