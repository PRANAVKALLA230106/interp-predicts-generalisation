#!/usr/bin/env python3
"""
Causal check v2: a stronger test of the first causal check (feather_causal.py).

Why: v1 compared the 6 predictive heads with only 5 random head sets, used only mean ablation,
and did not control for depth (the predictive heads sit in layers 4-9). Written and fixed BEFORE
running; the head sets are the same as v1 (chosen on the 70 development models only), and the
test is again run on the 30 held-out test models.

Conditions per model
  clean
  pred6, pred3, match3          as in v1 (mean ablation)
  rand6_00..39                  40 random 6-head sets (mean ablation), predictive/match heads excluded
  lm6_00..19                    20 LAYER-MATCHED random sets: one random head from each layer that
                                holds a predictive head (layers 4,5,7,7,8,9), predictive heads excluded
  z_pred6, z_rand6_00..09       zero ablation of pred6 and of the first 10 random sets

Items (fixed random subsets, same for every model and condition): 2,000 HANS lexical-overlap
non-entailment, 1,000 lexical-overlap entailment, 1,000 MNLI matched dev.

Pre-stated hypotheses (as AMENDED before any v2 result was seen; see cmd_analyze):
  H1  pred6's effect exceeds the random sets' at the SET level (p_set < 0.05 among 40 sets).
  H2  Same against the 20 layer-matched sets.
  H3  Zero ablation, pred6 vs 10 zero-ablated random sets (descriptive).
  Plus a magnitude control and a floor check (see cmd_analyze).

Usage:  python feather_causal2.py run      (GPU; resumable per model)
        python feather_causal2.py analyze  (CPU)
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
from feather_causal import (TEST_MODELS, PRED_HEADS, MATCH_HEADS, PROBE_MD5)  # noqa: E402

N_LO_N, N_LO_E, N_MNLI = 2000, 1000, 1000
N_RAND, N_LM, N_ZERO = 40, 20, 10
OUT_SUB = "causal2"


def head_sets(L=12, NH=12):
    """name -> (list of (layer, head), mode) with mode 'mean' or 'zero'."""
    rng = np.random.default_rng(2027)
    excl = set(PRED_HEADS) | set(MATCH_HEADS)
    pool = [(l, h) for l in range(L) for h in range(NH) if (l, h) not in excl]
    sets = {"clean": ([], "mean"), "pred6": (PRED_HEADS[:6], "mean"),
            "pred3": (PRED_HEADS[:3], "mean"), "match3": (MATCH_HEADS, "mean")}
    for r in range(N_RAND):
        pick = rng.choice(len(pool), 6, replace=False)
        sets[f"rand6_{r:02d}"] = ([pool[j] for j in sorted(pick)], "mean")
    layers = [l for l, _ in PRED_HEADS[:6]]
    for r in range(N_LM):
        chosen = []
        for l in layers:
            opts = [(l, h) for h in range(NH) if (l, h) not in excl and (l, h) not in chosen]
            chosen.append(opts[int(rng.integers(len(opts)))])
        sets[f"lm6_{r:02d}"] = (sorted(chosen), "mean")
    sets["z_pred6"] = (PRED_HEADS[:6], "zero")
    for r in range(N_ZERO):
        sets[f"z_rand6_{r:02d}"] = (sets[f"rand6_{r:02d}"][0], "zero")
    return sets


def cmd_run(args):
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    device = fe.pick_device(args.device)
    out = Path(args.out_dir); rdir = out / OUT_SUB; rdir.mkdir(parents=True, exist_ok=True)
    mnli = fe.load_mnli(args.mnli_tsv)
    probe = s4.build_probe(mnli, out)
    md5 = hashlib.md5(json.dumps(probe["idx"]).encode()).hexdigest()
    if not args.local_tmpl and md5 != PROBE_MD5:
        sys.exit(f"Probe set differs from step 4 ({md5} != {PROBE_MD5}); MNLI data changed?")
    hans = fe.load_hans(args.hans_path)
    lo = hans["heuristic"] == "lexical_overlap"
    rng = np.random.default_rng(7)
    pick = lambda ix, n: np.sort(rng.choice(ix, min(n, len(ix)), replace=False))  # noqa: E731
    lo_n = pick(np.where(lo & ~hans["gold_ent"])[0], N_LO_N)
    lo_e = pick(np.where(lo & hans["gold_ent"])[0], N_LO_E)
    mi = pick(np.arange(len(mnli["label"])), N_MNLI)
    sets = head_sets(args.n_layers, args.n_heads)
    (rdir / "head_sets.json").write_text(json.dumps({k: [v[0], v[1]] for k, v in sets.items()}))
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
            czero = torch.zeros_like(cmean)

            def enc(ix, P, Hy):
                return s4.encode_batches(tok, [P[j] for j in ix], [Hy[j] for j in ix], device, args.batch_size)
            evals = {"lo_nonent": enc(lo_n, hans["premise"], hans["hypothesis"]),
                     "lo_ent": enc(lo_e, hans["premise"], hans["hypothesis"]),
                     "mnli": enc(mi, mnli["premise"], mnli["hypothesis"])}
            mlab = mnli["label"][mi]
            res = {"model": i, "sha": sha}
            for cname, (heads, mode) in sets.items():
                src = cmean if mode == "mean" else czero
                by_layer = {}
                for (l, h) in heads:
                    by_layer.setdefault(l, []).append(h)
                hooks = []
                for l, hh in by_layer.items():
                    def hk(mod, inp, l=l, hh=hh, src=src):
                        x = inp[0].clone()
                        for h in hh:
                            x[..., h * dh:(h + 1) * dh] = src[l, h * dh:(h + 1) * dh].to(x.dtype)
                        return (x,)
                    hooks.append(layers[l].attention.output.dense.register_forward_pre_hook(hk))
                r = {}
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                             enabled=(device == "cuda" and not args.fp32)):
                    for ename, batches in evals.items():
                        n = sum(len(ix) for ix, _ in batches); pred = np.empty(n, np.int64)
                        for ix, e in batches:
                            pred[ix] = model(**e).logits.float().argmax(-1).cpu().numpy()
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
            rd = np.mean([res[f"rand6_{q:02d}"]["lo_nonent"] for q in range(N_RAND)])
            print(f"[{k+1}/{len(models)}] model {i}: LO-nonent clean {c['lo_nonent']:.3f}  pred6 "
                  f"{p['lo_nonent']:.3f}  random-mean {rd:.3f}  MNLI {c['mnli']:.3f}->{p['mnli']:.3f}  "
                  f"({time.time()-t0:.0f}s)")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            if args.purge and not args.local_tmpl:
                fe.purge_from_cache(i)
        except Exception as e:
            print(f"[{k+1}/{len(models)}] model {i}: FAILED {type(e).__name__}: {str(e)[:300]}")


def cmd_analyze(args):
    """AMENDED 2026-11-14, after an independent review and BEFORE any v2 result was seen:
    inference is at the level of head SETS (the unit that is randomised), not models.
      - Effect of a set = its mean over the 30 models of the drop in LO-non-entailment accuracy,
        on the accuracy scale and on the logit scale (less bound by floor effects).
      - H1/H2/H3: pred6's rank among the random / layer-matched / zero-ablated random sets;
        p_set = (1 + #sets with an effect >= pred6's) / (1 + #sets).  SUPPORTED if p_set < 0.05
        (resolvable for the 40 random and 20 layer-matched sets only).
      - Magnitude control: each set's expected size = the summed mean |ablation effect| of its heads
        on the high-overlap MNLI probe, from the 70 DEVELOPMENT models (step-4 features). pred6's
        effect is compared with the random/layer-matched sets via its residual from a regression of
        set effect on expected size (rank among the residuals).
      - Floor: model-level regression of the logit-scale drop on clean accuracy for pred6 vs the
        random-set mean (no claim from raw-accuracy correlations with clean accuracy)."""
    rdir = Path(args.out_dir) / OUT_SUB
    rows = [json.loads(p.read_text()) for p in sorted(rdir.glob("model_*.json"))]
    if len(rows) < 5:
        sys.exit(f"only {len(rows)} models done")
    names = [c for c in rows[0] if c not in ("model", "sha")]
    lg = lambda v: np.log(np.clip(v, .005, .995) / (1 - np.clip(v, .005, .995)))  # noqa: E731
    clean = np.array([r["clean"]["lo_nonent"] for r in rows])
    acc = {c: np.array([r[c]["lo_nonent"] for r in rows]) for c in names}
    eff = {c: float((clean - acc[c]).mean()) for c in names}                 # accuracy drop
    effl = {c: float((lg(clean) - lg(acc[c])).mean()) for c in names}        # logit drop
    dm = {c: float(np.mean([r[c]["mnli"] - r["clean"]["mnli"] for r in rows])) for c in names}
    de = {c: float(np.mean([r[c]["lo_ent"] - r["clean"]["lo_ent"] for r in rows])) for c in names}
    grp = lambda p: [c for c in names if c.startswith(p)]  # noqa: E731
    rand, lm, zr = grp("rand6_"), grp("lm6_"), grp("z_rand6_")
    pset = lambda t, pool, E: (1 + sum(E[c] >= E[t] for c in pool)) / (1 + len(pool))  # noqa: E731
    L = [f"CAUSAL CHECK v2 ({len(rows)} held-out models; head sets chosen on dev models; set-level inference)", "",
         f"{'condition':26s} {'acc drop':>9s} {'logit drop':>10s} {'dLO-ent':>8s} {'dMNLI':>7s}"]
    for lab, cs in [("pred6", ["pred6"]), ("pred3", ["pred3"]), ("match3", ["match3"]),
                    (f"random 6, mean of {len(rand)}", rand), (f"layer-matched, mean of {len(lm)}", lm),
                    ("zero-ablated pred6", ["z_pred6"]), (f"zero-abl. random, mean of {len(zr)}", zr)]:
        L.append(f"{lab:26s} {np.mean([eff[c] for c in cs]):+9.3f} {np.mean([effl[c] for c in cs]):+10.3f} "
                 f"{np.mean([de[c] for c in cs]):+8.3f} {np.mean([dm[c] for c in cs]):+7.3f}")
    L.append("")
    for tag, t, pool, thr in [("H1 vs random", "pred6", rand, 0.05), ("H2 vs layer-matched", "pred6", lm, 0.05),
                              ("H3 zero ablation vs random", "z_pred6", zr, None)]:
        pa, pl = pset(t, pool, eff), pset(t, pool, effl)
        verdict = ("SUPPORTED" if max(pa, pl) < thr else "not supported") if thr else "descriptive (10 sets)"
        L.append(f"{tag:28s} p_set acc {pa:.3f}, logit {pl:.3f}  -> {verdict}")
    L.append(f"random-set acc drops: min {min(eff[c] for c in rand):+.3f} median "
             f"{np.median([eff[c] for c in rand]):+.3f} max {max(eff[c] for c in rand):+.3f}; pred6 {eff['pred6']:+.3f}")
    if args.features_dir:
        split = json.loads((Path(args.features_dir).parent / "split.json").read_text())
        mag = np.mean([np.abs(np.load(Path(args.features_dir) / f"feat_{i:03d}.npz")["abl_ho"]) for i in split["dev"]], 0)
        hs = json.loads((rdir / "head_sets.json").read_text())
        size = {c: float(sum(mag[l, h] for l, h in hs[c][0])) for c in names}
        pool = rand + lm
        X = np.array([size[c] for c in pool]); Y = np.array([effl[c] for c in pool])
        k, b0 = np.polyfit(X, Y, 1)
        res = {c: effl[c] - (k * size[c] + b0) for c in pool + ["pred6"]}
        pr = (1 + sum(res[c] >= res["pred6"] for c in pool)) / (1 + len(pool))
        L += ["", f"Magnitude control (dev-model ablation size): pred6 expected size {size['pred6']:.3f} vs "
                  f"random/layer-matched median {np.median(X):.3f} (max {X.max():.3f}); slope {k:+.2f} logit per unit",
              f"  pred6 logit-drop residual {res['pred6']:+.3f}; p_set among {len(pool)} sets = {pr:.3f}  -> "
              f"{'beyond what its ablation size predicts' if pr < 0.05 else 'explained by its ablation size'}"]
    cl = lg(clean); dp = lg(clean) - lg(acc["pred6"]); dr = np.mean([lg(clean) - lg(acc[c]) for c in rand], 0)
    kp = np.polyfit(cl, dp, 1)[0]; kr = np.polyfit(cl, dr, 1)[0]
    L += ["", f"Floor check (logit scale): slope of drop on clean logit-accuracy: pred6 {kp:+.2f}, random mean {kr:+.2f}",
          "  (no claim about 'best generalisers lose most' is made from raw-accuracy drops, which are floor-bound)"]
    rep = "\n".join(L); print(rep)
    (Path(args.out_dir) / "causal2_report.txt").write_text(rep + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["run", "analyze"]:
        p = sp.add_parser(name)
        p.add_argument("--out_dir", default="feather_out")
        if name == "analyze":
            p.add_argument("--features_dir", default=None,
                           help="step-4 features (feather_out/features) for the magnitude control")
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
