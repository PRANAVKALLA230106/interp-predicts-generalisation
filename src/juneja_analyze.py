#!/usr/bin/env python3
"""
Fix 4 analysis (CPU): Juneja et al.'s exact convexity-gap (CG) metric as a baseline.

Inputs: feather_out/cg/row_*.json (from juneja_cg.py pairs) and the step-4 folder (features,
logits, split.json).

1. Reproduce Juneja et al.: spectral clustering (k=2) on CG distances over all 100 models
   (label-free), spectral embedding, f(w) = ||w - c1|| - ||w - c2|| (c1 = the larger cluster);
   in-sample Pearson r(f, HANS-LO accuracy). They report r = 0.674.
2. Out-of-sample: does the metric predict models it was not fitted on (repeated CV, n=100)?
3. Increment: does it add information beyond the strongest existing baselines (and vice versa)?
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feather_step4 as s4  # noqa: E402


def load_cg(cg_dir, n=100):
    """Reads the block-pair files of the fast run (blk_*.json) and the older per-row files."""
    C = np.full((n, n), np.nan); np.fill_diagonal(C, 0.0)
    for f in Path(cg_dir).glob("row_*.json"):
        i = int(f.stem.split("_")[1])
        for j, v in json.loads(f.read_text()).items():
            C[i, int(j)] = C[int(j), i] = v["cg"]
    for f in Path(cg_dir).glob("blk_*.json"):
        for key, v in json.loads(f.read_text())["pairs"].items():
            i, j = map(int, key.split("-"))
            C[i, j] = C[j, i] = v["cg"]
    return C


def juneja_feature(C, seed=0, bw=1.0):
    """bw scales the affinity bandwidth (median off-diagonal CG). The trivial first eigenvector is
    dropped (amended 2026-11-14 after review, before any CG result was seen)."""
    from sklearn.cluster import SpectralClustering
    from sklearn.manifold import spectral_embedding
    off = C[np.triu_indices(len(C), 1)]
    A = np.exp(-C / (bw * np.median(off)))
    lab = SpectralClustering(2, affinity="precomputed", random_state=seed).fit_predict(A)
    E = spectral_embedding(A, n_components=2, random_state=seed, drop_first=True)
    big = np.argmax(np.bincount(lab))
    c1, c2 = E[lab == big].mean(0), E[lab != big].mean(0)
    f = np.linalg.norm(E - c1, axis=1) - np.linalg.norm(E - c2, axis=1)
    return A, f, lab, big


def main():
    from scipy.stats import pearsonr
    import stats_extra as sx
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "feather_out")
    C = load_cg(out / "cg")
    miss = int(np.isnan(C).sum() // 2)
    if miss:
        sys.exit(f"{miss} model pairs missing from the CG matrix; finish `juneja_cg.py pairs` first.")
    A, f, lab, big = juneja_feature(C)
    split = json.loads((out / "split.json").read_text())
    dev, test = split["dev"], split["test"]
    ids = dev + test; order = np.array(ids)
    blocks, y, _, _ = s4.build_blocks(out, ids)
    blocks["Juneja-CG"] = s4.fixed_kernel_block(A[np.ix_(order, order)])
    blocks["Juneja-f"] = s4.linear_kernel_block(f[order][:, None])
    yy = np.empty(100); yy[order] = y
    meta = np.load(out / "meta.npz", allow_pickle=True)
    lo_all = meta["hans_heuristic"] == "lexical_overlap"
    acc_all = np.array([(((np.load(out / "logits" / f"model_{i:03d}.npz")["hans_logits"].argmax(1) == s4.fe.ENT)
                          == meta["hans_gold_ent"])[lo_all]).mean() for i in range(100)])
    off = C[np.triu_indices(100, 1)]
    L = ["FIX 4: Juneja et al.'s convexity-gap (CG) metric -- three separate evaluations", "",
         "All three are post hoc: the 30 test models were already scored in step 4, so nothing here is",
         "confirmatory. Out-of-sample numbers come from repeated 10-fold CV over all 100 models.", "",
         f"CG over 4,950 pairs: median {np.median(off):.4f}, 90th pct {np.percentile(off, 90):.4f}, max {off.max():.4f}",
         f"Spectral clusters (k=2, label-free): sizes {np.bincount(lab).tolist()}", ""]
    # E1 reproduction
    r_nonent, p1 = pearsonr(f, yy); r_all, p2 = pearsonr(f, acc_all)
    L += ["E1  REPRODUCTION of Juneja et al. (in-sample, all 100 models, their pipeline)",
          f"    |r|(f, HANS-LO non-entailment acc) = {abs(r_nonent):.3f} (p = {p1:.1g})",
          f"    |r|(f, HANS-LO accuracy, all LO items) = {abs(r_all):.3f} (p = {p2:.1g})",
          "    Juneja et al. report r = 0.674 (MNLI validation-loss CG). Sign depends on cluster labelling.",
          "    Sensitivity to the affinity bandwidth (x median CG): " + "  ".join(
              f"{bw:g}x: |r| {abs(pearsonr(juneja_feature(C, bw=bw)[1], yy)[0]):.3f}" for bw in (0.25, 0.5, 1, 2, 4)), ""]
    # E2 out-of-sample
    all_idx = np.arange(100)
    oos = {}
    for k, names in [("Juneja-CG kernel", ["Juneja-CG"]), ("Juneja f", ["Juneja-f"]),
                     ("Weight-geometry proxy", ["Weight-geometry"]), ("ID-overlap-items (best baseline)", ["ID-overlap-items"]),
                     ("ALL BASELINES (step 4)", s4.BASE), ("INTERNAL", s4.INTERNAL)]:
        oos[k], r2s = s4.cv_oof(blocks, names, all_idx, y, 10, 0)
        L.append(f"E2  out-of-sample CV R2, {k:34s} {r2s.mean():.3f}  [{r2s.min():.3f}, {r2s.max():.3f}]")
    L += ["    (E2 asks whether the metric predicts models it was not fitted on; Juneja et al. report only",
          "     in-sample fits. Same population: this is out-of-sample generalisation, not cross-population transfer.)", ""]
    # E3 incremental
    jj = oos["Juneja-CG kernel"]
    L += ["E3  INCREMENT over the strongest existing baselines (partial F-test on CV predictions)",
          "    " + sx.fmt("Juneja over ID-overlap-items", sx.partial_increment(y, oos["ID-overlap-items (best baseline)"], jj)),
          "    " + sx.fmt("Juneja over ALL BASELINES   ", sx.partial_increment(y, oos["ALL BASELINES (step 4)"], jj)),
          "    " + sx.fmt("ID-overlap over Juneja      ", sx.partial_increment(y, jj, oos["ID-overlap-items (best baseline)"])),
          "    " + sx.fmt("INTERNAL over baselines+Juneja", sx.partial_increment(
              y, s4.cv_oof(blocks, s4.BASE + ["Juneja-CG", "Juneja-f"], all_idx, y, 10, 0)[0], oos["INTERNAL"])),
          "    (partial R2 CIs from the bootstrap are bounded below by 0 and are not evidence of a",
          "     non-zero increment; read the F-test p and the partial r.)"]
    rel = sorted((out).glob("cg_seed*"))
    if rel:
        C2 = load_cg(rel[0]); m = ~np.isnan(C2) & ~np.eye(100, dtype=bool) & np.triu(np.ones((100, 100), bool), 1)
        L += ["", f"CG RELIABILITY: {m.sum()} pairs recomputed on an independently drawn 512-item sample (about 5% item overlap expected) ({rel[0].name}): "
                  f"Pearson r = {pearsonr(C[m], C2[m])[0]:.3f}, Spearman = {__import__('scipy.stats').stats.spearmanr(C[m], C2[m]).correlation:.3f}"]
    rep = "\n".join(L); print(rep)
    (out / "fix4_report.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
