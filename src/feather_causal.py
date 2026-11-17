#!/usr/bin/env python3
"""
Step 5 (exploratory, decided after the binding step-4 PIVOT): are the heads whose ablation
profile PREDICTS HANS generalisation also CAUSALLY involved in it?

Head sets were chosen on the 70 development models only (feather_out/exploratory.json) and are
fixed below. The test is run on the 30 held-out test models, which played no part in choosing them.

Conditions per model (mean ablation of each head's context vector; means from the 512-item MNLI
probe set used in step 4):
  clean
  pred3, pred6   top-3 / top-6 heads by |corr(ablation effect, HANS-LO accuracy)| on dev models
  match3         top-3 heads by |corr(attention to repeated words, HANS-LO accuracy)|
  rand6_0..4     five fixed random sets of 6 heads, excluding the predictive heads

Measured: HANS lexical-overlap non-entailment accuracy (the outcome), HANS lexical-overlap
ENTAILMENT accuracy (to catch ablations that merely suppress "entailment" everywhere), and MNLI
accuracy on 2,000 fixed matched-dev items.

Hypotheses (stated before running):
  H1  |change in HANS-LO non-ent accuracy| is larger for pred6 than for the mean of the rand6 sets
      (paired Wilcoxon over the 30 test models).
  H2  pred6 changes HANS-LO non-ent accuracy more than it changes MNLI accuracy, and more than
      it changes LO-entailment accuracy in the opposite direction (i.e. it is not just a
      global shift away from "entailment").
  H3  (descriptive) the size of the change correlates with the model's clean HANS-LO accuracy.

Usage:  python feather_causal.py run      (GPU; resumable)
        python feather_causal.py analyze  (CPU)
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feather_eval as fe     # noqa: E402
import feather_step4 as s4    # noqa: E402

TEST_MODELS = [2, 3, 10, 12, 14, 15, 16, 17, 19, 25, 29, 31, 35, 44, 46, 47, 48, 51, 55, 64,
               70, 73, 74, 75, 76, 84, 85, 88, 91, 97]
PRED_HEADS = [(7, 8), (8, 4), (5, 5), (9, 8), (4, 4), (7, 0)]   # ranked, from dev models only
MATCH_HEADS = [(8, 1), (8, 4), (10, 1)]
PROBE_MD5 = "1ec2d80f86436efc94f21576c911c351"
N_MNLI = 2000


def head_sets(L=12, NH=12):
    rng = np.random.default_rng(12345)
    excl = set(PRED_HEADS) | set(MATCH_HEADS)
    pool = [(l, h) for l in range(L) for h in range(NH) if (l, h) not in excl]
    sets = {"clean": [], "pred3": PRED_HEADS[:3], "pred6": PRED_HEADS[:6], "match3": MATCH_HEADS}
    for r in range(5):
        pick = rng.choice(len(pool), 6, replace=False)
        sets[f"rand6_{r}"] = [pool[j] for j in sorted(pick)]
    return sets


def cmd_run(args):
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    device = fe.pick_device(args.device)
    out = Path(args.out_dir); rdir = out / "causal"; rdir.mkdir(parents=True, exist_ok=True)
    mnli = fe.load_mnli(args.mnli_tsv)
    probe = s4.build_probe(mnli, out)
    md5 = hashlib.md5(json.dumps(probe["idx"]).encode()).hexdigest()
    if not args.local_tmpl and md5 != PROBE_MD5:
        sys.exit(f"Probe set differs from step 4 ({md5} != {PROBE_MD5}); MNLI data changed?")
    hans = fe.load_hans(args.hans_path)
    lo = hans["heuristic"] == "lexical_overlap"
    lo_n = np.where(lo & ~hans["gold_ent"])[0]; lo_e = np.where(lo & hans["gold_ent"])[0]
    rng = np.random.default_rng(1)
    mi = np.sort(rng.choice(len(mnli["label"]), min(N_MNLI, len(mnli["label"])), replace=False))
    sets = head_sets(args.n_layers, args.n_heads)
    (rdir / "head_sets.json").write_text(json.dumps(sets))
    pidx = probe["idx"]
    models = TEST_MODELS if args.models == ["test"] else fe.parse_models(args.models)
    for k, i in enumerate(models):
        f = rdir / f"model_{i:03d}.json"
        if f.exists():
            print(f"[{k+1}/{len(models)}] model {i}: exists, skipping"); continue
        t0 = time.time()
        try:
            path, sha = fe.model_source(i, args)
            tok = AutoTokenizer.from_pretrained(path)
            model = BertForSequenceClassification.from_pretrained(path).to(device).eval()
            layers = model.bert.encoder.layer
            L, NH, H = (model.config.num_hidden_layers, model.config.num_attention_heads,
                        model.config.hidden_size)
            dh = H // NH
            # per-layer mean context vector over the probe set (same as step 4)
            pb = s4.encode_batches(tok, [mnli["premise"][j] for j in pidx],
                                   [mnli["hypothesis"][j] for j in pidx], device, args.batch_size)
            csum = torch.zeros(L, H, device=device, dtype=torch.float64); nt = 0; cur = {}

            def cap(l):
                def hk(mod, inp):
                    m = cur["m"].unsqueeze(-1).to(inp[0].dtype)
                    csum[l] += (inp[0] * m).sum((0, 1)).double()
                return hk
            hs = [layers[l].attention.output.dense.register_forward_pre_hook(cap(l)) for l in range(L)]
            with torch.inference_mode():
                for _, enc in pb:
                    cur["m"] = enc["attention_mask"]; nt += int(enc["attention_mask"].sum()); model(**enc)
            for h_ in hs:
                h_.remove()
            cmean = (csum / nt).float()

            evals = {
                "lo_nonent": s4.encode_batches(tok, [hans["premise"][j] for j in lo_n],
                                               [hans["hypothesis"][j] for j in lo_n], device, args.batch_size),
                "lo_ent": s4.encode_batches(tok, [hans["premise"][j] for j in lo_e],
                                            [hans["hypothesis"][j] for j in lo_e], device, args.batch_size),
                "mnli": s4.encode_batches(tok, [mnli["premise"][j] for j in mi],
                                          [mnli["hypothesis"][j] for j in mi], device, args.batch_size),
            }
            mlab = mnli["label"][mi]
            res = {"model": i, "sha": sha}
            for cname, heads in sets.items():
                by_layer = {}
                for (l, h) in heads:
                    by_layer.setdefault(l, []).append(h)
                hooks = []
                for l, hh in by_layer.items():
                    def hk(mod, inp, l=l, hh=hh):
                        x = inp[0].clone()
                        for h in hh:
                            x[..., h * dh:(h + 1) * dh] = cmean[l, h * dh:(h + 1) * dh].to(x.dtype)
                        return (x,)
                    hooks.append(layers[l].attention.output.dense.register_forward_pre_hook(hk))
                r = {}
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                             enabled=(device == "cuda" and not args.fp32)):
                    for ename, batches in evals.items():
                        n = sum(len(ix) for ix, _ in batches); pred = np.empty(n, np.int64)
                        for ix, enc in batches:
                            pred[ix] = model(**enc).logits.float().argmax(-1).cpu().numpy()
                        if ename == "lo_nonent":
                            r[ename] = float((pred != fe.ENT).mean())
                        elif ename == "lo_ent":
                            r[ename] = float((pred == fe.ENT).mean())
                        else:
                            r[ename] = float((pred == mlab).mean())
                for h_ in hooks:
                    h_.remove()
                res[cname] = r
            f.write_text(json.dumps(res))
            c = res["clean"]; p = res["pred6"]
            print(f"[{k+1}/{len(models)}] model {i}: clean LO-nonent {c['lo_nonent']:.3f} -> pred6 "
                  f"{p['lo_nonent']:.3f}; LO-ent {c['lo_ent']:.3f} -> {p['lo_ent']:.3f}; "
                  f"MNLI {c['mnli']:.3f} -> {p['mnli']:.3f}  ({time.time()-t0:.0f}s)")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            if args.purge and not args.local_tmpl:
                fe.purge_from_cache(i)
        except Exception as e:
            print(f"[{k+1}/{len(models)}] model {i}: FAILED {type(e).__name__}: {str(e)[:300]}")


def cmd_analyze(args):
    from scipy.stats import spearmanr, wilcoxon
    rdir = Path(args.out_dir) / "causal"
    rows = [json.loads(p.read_text()) for p in sorted(rdir.glob("model_*.json"))]
    if len(rows) < 5:
        sys.exit(f"only {len(rows)} models done")
    conds = [c for c in rows[0] if c not in ("model", "sha")]
    clean = {m: np.array([r["clean"][m] for r in rows]) for m in ("lo_nonent", "lo_ent", "mnli")}
    d = {c: {m: np.array([r[c][m] for r in rows]) - clean[m] for m in clean} for c in conds}
    rand = [c for c in conds if c.startswith("rand6")]
    rand_abs = np.mean([np.abs(d[c]["lo_nonent"]) for c in rand], 0)
    L = [f"CAUSAL CHECK ({len(rows)} held-out test models; head sets chosen on dev models)", "",
         f"{'condition':10s} {'dLO-nonent':>11s} {'|d|':>7s} {'dLO-ent':>9s} {'dMNLI':>8s}   (means over models)"]
    for c in conds:
        L.append(f"{c:10s} {d[c]['lo_nonent'].mean():+11.3f} {np.abs(d[c]['lo_nonent']).mean():7.3f} "
                 f"{d[c]['lo_ent'].mean():+9.3f} {d[c]['mnli'].mean():+8.3f}")
    p6 = np.abs(d["pred6"]["lo_nonent"])
    w = wilcoxon(p6, rand_abs)
    L += ["", f"H1  |dLO-nonent| pred6 {p6.mean():.3f} vs random-6 mean {rand_abs.mean():.3f}: "
              f"Wilcoxon p = {w.pvalue:.2g}  -> {'SUPPORTED' if (w.pvalue < 0.05 and p6.mean() > rand_abs.mean()) else 'not supported'}"]
    ratio_mnli = np.abs(d["pred6"]["lo_nonent"]).mean() / max(np.abs(d["pred6"]["mnli"]).mean(), 1e-6)
    net = d["pred6"]["lo_nonent"] + d["pred6"]["lo_ent"]   # >0: better at both; ~0: pure label shift
    L += [f"H2  |dLO-nonent| / |dMNLI| for pred6 = {ratio_mnli:.1f};  dLO-nonent + dLO-ent = "
          f"{net.mean():+.3f} (≈0 would mean a pure shift away from 'entailment')"]
    rho = spearmanr(clean["lo_nonent"], d["pred6"]["lo_nonent"])
    L += [f"H3  Spearman(clean LO-nonent, dLO-nonent under pred6) = {rho.correlation:+.2f} (p = {rho.pvalue:.2g})"]
    rep = "\n".join(L); print(rep)
    (Path(args.out_dir) / "causal_report.txt").write_text(rep + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["run", "analyze"]:
        p = sp.add_parser(name)
        p.add_argument("--out_dir", default="feather_out")
        if name == "run":
            p.add_argument("--models", nargs="+", default=["test"])
            p.add_argument("--device", default="auto")
            p.add_argument("--batch_size", type=int, default=128)
            p.add_argument("--mnli_tsv", default=None)
            p.add_argument("--hans_path", default="data/heuristics_evaluation_set.txt")
            p.add_argument("--local_tmpl", default=None)
            p.add_argument("--purge", action="store_true")
            p.add_argument("--fp32", action="store_true")
            p.add_argument("--n_layers", type=int, default=12)
            p.add_argument("--n_heads", type=int, default=12)
    args = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
