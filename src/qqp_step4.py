#!/usr/bin/env python3
"""
Fix 1, stage B: the step-4 test, unchanged, on the QQP population (run only if stage A's gate
named a primary outcome).

Everything is step 4's design, transplanted: a fresh 70/30 split (seed 2026), the same feature
families computed from QQP validation inputs only (never PAWS), the same models and the same
pre-registered decision rule. Task-specific changes, fixed in advance:
  - the "shortcut-prone" label is duplicate (index 1), playing entailment's role;
  - log-odds are 2-class: logit[duplicate] - logit[not_duplicate];
  - the probe: 256 high-overlap + 256 other QQP validation pairs (same overlap measure);
  - outcome: --outcome PAWS-QQP or PAWS-Wiki (non-paraphrase accuracy); each is a DISTINCT
    evaluation setting with its own verdict, allowed only if it passed stage A's gate.

PRE-REGISTERED AMENDMENT (2026-11-14, after an independent review of the MNLI study, before any
QQP feature or test result exists). The MNLI primary contrast was degenerate: the stacker gave the
internal blocks zero weight, so dR2 was exactly 0 with a [0, 0] "CI". For QQP the primary test is:
  b = out-of-sample prediction from ALL BASELINES (fitted on the 70 dev models),
  i = out-of-sample prediction from INTERNAL (fitted on the 70 dev models);
  on the 30 test models, the partial correlation r_p of y with i controlling for b.
  Inference: one-sided permutation test (10,000 permutations of i), alpha = 0.025 per PAWS setting
  (Bonferroni over the two settings). Interval: 95% Fisher-z CI for r_p (df = n - 4).
  GO: p < 0.025 AND r_p >= 0.30 (an increment of at least ~9% of the variance baselines leave).
  NULL-CONSISTENT: otherwise. A claim of "no meaningful increment" is made only if the 90% CI of
  r_p lies inside [-0.30, 0.30]; with n = 30 this is not reachable, which is stated in advance.
  The stacked dR2 of step 4 is still reported, labelled as descriptive.
  Ridge grid extended to 1e-8 .. 1e4 (the MNLI fits hit the lower boundary of 1e-4).
  Exploratory companion: repeated 10-fold CV over all 100 QQP models, same statistic.

Subcommands: split | extract (GPU, resumable, --purge) | analyze [--final]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feather_eval as fe     # noqa: E402
import feather_step4 as s4    # noqa: E402
import fix1_qqp as q          # noqa: E402  (sets fe.REPO_TMPL to the QQP models)

fe.ENT = q.DUP


def logodds_2(logits):
    import torch
    t = torch.as_tensor(logits)
    return t[..., 1] - t[..., 0]


s4.ent_logodds = logodds_2     # extract_one looks this up at call time
s4.ALPHAS = np.logspace(-8, 4, 37)   # amended grid (MNLI fits hit the old lower boundary)
R_P_MIN, ALPHA_SETTING = 0.30, 0.025


def partial_r_test(y, b, i, n_perm=10000, seed=0):
    """Partial correlation of y and i given b; one-sided permutation p; 95% and 90% Fisher-z CIs."""
    def pr(yv, iv):
        res = lambda v: v - np.poly1d(np.polyfit(b, v, 1))(b)  # noqa: E731
        ry, ri = res(yv), res(iv)
        return float(ry @ ri / np.sqrt((ry @ ry) * (ri @ ri)))
    r = pr(y, i); rng = np.random.default_rng(seed)
    perm = np.array([pr(y, rng.permutation(i)) for _ in range(n_perm)])
    p = float((np.sum(perm >= r) + 1) / (n_perm + 1))
    z, se = np.arctanh(r), 1 / np.sqrt(len(y) - 4)
    ci95 = np.tanh([z - 1.96 * se, z + 1.96 * se]); ci90 = np.tanh([z - 1.645 * se, z + 1.645 * se])
    return r, p, ci95, ci90


def cmd_split(args):
    s4.cmd_split(args)


def cmd_extract(args):
    import torch
    from transformers import BertModel
    device = fe.pick_device(args.device)
    out = Path(args.out_dir); fdir = out / "features"; fdir.mkdir(parents=True, exist_ok=True)
    meta = np.load(out / "meta.npz")
    qqp = q.load_qqp()
    assert np.array_equal(qqp["label"], meta["qqp_label"]), "QQP subsample differs from stage A"
    probe = s4.build_probe(qqp, out)
    pt = ([qqp["premise"][j] for j in probe["idx"]], [qqp["hypothesis"][j] for j in probe["idx"]])
    pre = {k: v.detach().cpu() for k, v in BertModel.from_pretrained(args.pretrained).state_dict().items()}
    models = fe.parse_models(args.models)
    for k, i in enumerate(models):
        f = fdir / f"feat_{i:03d}.npz"
        if f.exists():
            print(f"[{k+1}/{len(models)}] model {i}: exists, skipping"); continue
        t0 = time.time()
        try:
            res = s4.extract_one(i, args, device, pt, probe["is_ho"], pre)
        except Exception as e:  # noqa: BLE001
            print(f"[{k+1}/{len(models)}] model {i}: FAILED {type(e).__name__}: {str(e)[:300]}")
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        tmp = f.with_suffix(".tmp.npz"); np.savez(tmp, **res); tmp.replace(f)
        print(f"[{k+1}/{len(models)}] model {i}: ok  max|abl_ho|={np.abs(res['abl_ho']).max():.3f}  "
              f"fp16 err={res['amp_max_err']:.1e}  ({time.time()-t0:.0f}s)")
        if args.purge:
            fe.purge_from_cache(i)


def build_blocks_qqp(out, ids, prim):
    meta = np.load(out / "meta.npz")
    gate = json.loads((out / "gate_summary.json").read_text())
    assert prim in gate["passing"], f"{prim} did not pass stage A's gate; it cannot be analysed"
    key, lab = {"PAWS-QQP": ("pawsq_logits", meta["pawsq_label"]),
                "PAWS-Wiki": ("pawsw_logits", meta["pawsw_label"])}[prim]
    fdir = out / "features"
    ho = np.load(fdir / "mnli_overlap.npy") >= s4.HO_THRESH     # overlap on the QQP subsample
    ql = meta["qqp_label"]
    y, idsum, iditems, wts, sk, attn, abl, reps, acc = [], [], [], [], [], [], [], [], []
    for i in ids:
        z = np.load(out / "logits" / f"model_{i:03d}.npz")
        y.append((z[key].argmax(1)[lab == 0] != q.DUP).mean())
        lg = z["qqp_logits"]; pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        corr = lg.argmax(1) == ql
        hn, he = ho & (ql == 0), ho & (ql == 1)
        acc.append(corr.mean())
        idsum.append([corr.mean(), corr[ql == 0].mean(), corr[ql == 1].mean(), pr.max(1).mean(),
                      pr[:, 1].mean(), corr[ho].mean(), corr[hn].mean(), pr[hn, 1].mean(), pr[he, 1].mean()])
        iditems.append(lg[ho, 1] - lg[ho, 0])
        f = np.load(fdir / f"feat_{i:03d}.npz")
        wts.append(np.concatenate([f["delta_norm"], f["delta_rel"], f["sketch"], f["clf"]])); sk.append(f["sketch"])
        attn.append(np.concatenate([f["attn_match"].ravel(), f["attn_prem"].ravel()]))
        abl.append(np.concatenate([f["abl_ho"].ravel(), f["abl_rand"].ravel()]))
        reps.append(f["rep"])
    reps = np.array(reps)
    cka = np.mean([s4.linear_cka_kernel(reps[:, l]) for l in range(reps.shape[1])], 0)
    blocks = {"MNLI-acc": s4.linear_kernel_block(np.array(acc)[:, None]),   # = QQP accuracy here
              "ID-summary": s4.linear_kernel_block(idsum),
              "ID-overlap-items": s4.linear_kernel_block(iditems),
              "Weights": s4.linear_kernel_block(wts),
              "Weight-geometry": s4.rbf_distance_block(sk),
              "Repr-CKA": s4.fixed_kernel_block(cka),
              "Attn-stats": s4.linear_kernel_block(attn),
              "Head-ablation": s4.linear_kernel_block(abl)}
    return blocks, np.array(y), prim, int(ho.sum())


def cmd_analyze(args):
    out = Path(args.out_dir)
    split = json.loads((out / "split.json").read_text())
    have = {int(p.stem.split("_")[1]) for p in (out / "features").glob("feat_*.npz")}
    dev = [i for i in split["dev"] if i in have]; test = [i for i in split["test"] if i in have]
    ids = dev + test
    blocks, y, prim, n_ho = build_blocks_qqp(out, ids, args.outcome)
    dev_idx, test_idx = np.arange(len(dev)), np.arange(len(dev), len(ids))
    L = [f"QQP REPLICATION OF STEP 4 ({'FINAL: test models' if args.final else 'development CV'})",
         f"models: {len(dev)} dev, {len(test)} test; outcome: {prim} non-paraphrase accuracy; "
         f"high-overlap QQP items: {n_ho}", "('MNLI-acc only' = QQP accuracy only in this population)", ""]
    preds = {}
    for sname, names in s4.SETS.items():
        if args.final:
            preds[sname] = s4.predict_set(blocks, names, dev_idx, test_idx, y)
            L.append(f"{sname:32s} test R2 {s4.r2(y[test_idx], preds[sname]):7.3f}")
        else:
            preds[sname], r2s = s4.cv_oof(blocks, names, dev_idx, y, args.n_rep, args.seed)
            L.append(f"{sname:32s} CV R2 {r2s.mean():7.3f}  [{r2s.min():.3f}, {r2s.max():.3f}]")
    yy = y[test_idx] if args.final else y[dev_idx]
    A, B = "ALL BASELINES + INTERNAL", "ALL BASELINES"
    d = s4.r2(yy, preds[A]) - s4.r2(yy, preds[B])
    r, p, ci95, ci90 = partial_r_test(yy, preds[B], preds["INTERNAL (attn+ablation)"])
    go = p < ALPHA_SETTING and r >= R_P_MIN
    equiv = ci90[0] > -R_P_MIN and ci90[1] < R_P_MIN
    L += ["", f"PRIMARY (amended): partial r of outcome with INTERNAL given ALL BASELINES = {r:+.3f}  "
              f"95% CI [{ci95[0]:+.3f}, {ci95[1]:+.3f}]  permutation p (one-sided) = {p:.4f}",
          f"  -> {'GO' if go else 'NULL-CONSISTENT'} at alpha {ALPHA_SETTING} and r_p >= {R_P_MIN}"
          f"{'; equivalence (90% CI within +/-0.30) shown' if equiv else '; equivalence NOT shown (90% CI exceeds +/-0.30)'}",
          f"  descriptive: stacked dR2 = {d:+.3f} (can be exactly 0 by construction; not used for the verdict)",
          f"VERDICT ({'binding' if args.final else 'provisional, development CV'}): {'GO' if go else 'NULL-CONSISTENT'}"]
    rep = "\n".join(L); print(rep)
    (out / f"qqp_step4_{args.outcome}_{'final' if args.final else 'dev'}.txt").write_text(rep + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["split", "extract", "analyze"]:
        p = sp.add_parser(name); p.add_argument("--out_dir", default="qqp_out"); p.add_argument("--seed", type=int, default=0)
        if name == "split":
            p.add_argument("--force", action="store_true")
        if name == "extract":
            p.add_argument("--models", nargs="+", default=["all"]); p.add_argument("--device", default="auto")
            p.add_argument("--batch_size", type=int, default=64); p.add_argument("--local_tmpl", default=None)
            p.add_argument("--pretrained", default=s4.PRETRAINED); p.add_argument("--purge", action="store_true")
            p.add_argument("--fp32_ablation", action="store_true")
        if name == "analyze":
            p.add_argument("--final", action="store_true"); p.add_argument("--n_rep", type=int, default=20)
            p.add_argument("--outcome", required=True, choices=["PAWS-QQP", "PAWS-Wiki"])
    args = ap.parse_args()
    {"split": cmd_split, "extract": cmd_extract, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
