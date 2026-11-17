#!/usr/bin/env python3
"""
Step 1 of the final plan: an interpretability measure that is NOT read off the model's outputs and
uses data that is neither MNLI nor HANS (answers the review's construct-validity objection).

Syntactic-role probes on Universal Dependencies (English EWT, train split, from GitHub):
  - 2,000 training and 1,000 test sentences (5-40 words), fixed by seed, the same for every model.
  - Each dependency edge is represented by [h(head word) ; h(dependent word)] from a hidden layer
    (first word-piece of each word; sentences fed alone, as single segments).
  - ARG probe: classify the relation among {nsubj, obj, nsubj:pass, iobj, obl}   (who did what to whom)
  - GEN probe: classify the base relation among the 12 most frequent relations   (general syntax)
  - CONTROL tasks (Hewitt & Liang 2019): the same probes trained on labels assigned at random per
    dependent word type, so selectivity = accuracy - control accuracy measures what the representation
    encodes rather than what the probe can memorise.
  - Probe = multinomial logistic regression (standardised inputs, L2), identical settings everywhere.
  Features per model: selectivity of ARG and GEN at each of the 13 hidden layers (26 numbers).

PRE-REGISTERED (2026-11-14, before any probe result exists):
  Primary: does probe selectivity add information about HANS-LO non-entailment accuracy beyond all
  step-4 baselines plus Juneja et al.'s convexity-gap metric? Partial correlation of the outcome with
  the probe block's out-of-fold prediction, given the out-of-fold prediction of baselines + Juneja,
  repeated 10-fold CV over all 100 models (exploratory by necessity: the 30 test models were already
  scored in step 4). Significance by permuting the probe features across models (200 permutations,
  baselines and outcome fixed; stats_extra.perm_cv_partial). PASS if one-sided p < 0.05 AND partial
  r >= 0.20. (Amended before any probe result existed: the parametric F-test was found to be invalid
  on CV predictions; see the calibration check in perm_incremental_report.txt.)
  Also reported: the probe block alone (CV R2), increment over the best single baseline, and the
  30-test-model number for consistency with step 4 (descriptive).

Usage:  python probe_ud.py run [--models all] [--purge]     (GPU)
        python probe_ud.py analyze --step4_dir <feather_out with features, logits, split, cg>   (CPU)
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feather_eval as fe  # noqa: E402

UD_URLS = ["https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu",
           "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/dev/en_ewt-ud-train.conllu"]
ARG = ["nsubj", "obj", "nsubj:pass", "iobj", "obl"]
N_TRAIN, N_TEST, SEED = 2000, 1000, 2026
N_LAYERS = 13


def read_conllu(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur); cur = []
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        cur.append((int(c[0]), c[1], int(c[6]) if c[6] != "_" else 0, c[7]))
    if cur:
        sents.append(cur)
    return sents


def build_data(out):
    f = out / "ud_probe_data.json"
    if f.exists():
        return json.loads(f.read_text())
    raw = out / "en_ewt-ud-train.conllu"
    if not raw.exists():
        for u in UD_URLS:
            try:
                urllib.request.urlretrieve(u, raw); break
            except Exception as e:  # noqa: BLE001
                err = e
        else:
            raise err
    sents = [s for s in read_conllu(raw) if 5 <= len(s) <= 40]
    rng = np.random.default_rng(SEED)
    pick = rng.permutation(len(sents))[: N_TRAIN + N_TEST]
    chosen = [sents[k] for k in pick]
    base = lambda d: d.split(":")[0]  # noqa: E731
    gen_counts = {}
    for s in chosen[:N_TRAIN]:
        for (_, _, h, d) in s:
            if h > 0:
                gen_counts[base(d)] = gen_counts.get(base(d), 0) + 1
    GEN = [k for k, _ in sorted(gen_counts.items(), key=lambda kv: -kv[1])[:12]]
    argmap = lambda d: "obl" if d.startswith("obl") else d  # noqa: E731
    vocab = sorted({w.lower() for s in chosen for (_, w, _, _) in s})
    crng = np.random.default_rng(SEED + 1)
    ctrl_arg = {w: int(crng.integers(len(ARG))) for w in vocab}
    ctrl_gen = {w: int(crng.integers(len(GEN))) for w in vocab}
    data = {"words": [], "edges": [], "split": []}
    for k, s in enumerate(chosen):
        words = [w for (_, w, _, _) in s]
        edges = []
        for (i, w, h, d) in s:
            if h <= 0:
                continue
            a = argmap(d); g = base(d)
            edges.append([h - 1, i - 1, ARG.index(a) if a in ARG else -1, GEN.index(g) if g in GEN else -1,
                          ctrl_arg[w.lower()], ctrl_gen[w.lower()]])
        data["words"].append(words); data["edges"].append(edges)
        data["split"].append(0 if k < N_TRAIN else 1)
    data["GEN"] = GEN; data["ARG"] = ARG
    f.write_text(json.dumps(data))
    return data


def word_reps(tok, model, words_list, device, bs=64):
    """Per sentence: (13, n_words, H) hidden states at each word's first word-piece (None if cut off)."""
    import torch
    out = []
    with torch.inference_mode():
        for s in range(0, len(words_list), bs):
            chunk = words_list[s:s + bs]
            enc = tok(chunk, is_split_into_words=True, truncation=True, max_length=fe.MAX_LEN,
                      padding=True, return_tensors="pt")
            hs = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
            H = torch.stack(hs, 1).half()   # (B, 13, T, H)
            for b in range(len(chunk)):
                wid = enc.word_ids(b); first = {}
                for t, w in enumerate(wid):
                    if w is not None and w not in first:
                        first[w] = t
                idx = [first.get(w, -1) for w in range(len(chunk[b]))]
                out.append((H[b], idx))
    return out


def fit_probe(Xtr, ytr, Xte, yte, n_cls, seed, steps=300, wd=1e-3):
    import torch
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    torch.manual_seed(seed)
    W = torch.zeros(Xtr.shape[1], n_cls, device=Xtr.device, requires_grad=True)
    b = torch.zeros(n_cls, device=Xtr.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.01, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        return float(((Xte @ W + b).argmax(1) == yte).float().mean())


def probe_model(tok, model, data, device):
    import torch
    reps = word_reps(tok, model, data["words"], device)
    feats = {sp: [] for sp in (0, 1)}; labs = {sp: [] for sp in (0, 1)}
    for (H, idx), edges, sp in zip(reps, data["edges"], data["split"]):
        for (h, d, a, g, ca, cg) in edges:
            if idx[h] < 0 or idx[d] < 0:
                continue
            feats[sp].append(torch.cat([H[:, idx[h]], H[:, idx[d]]], -1))   # (13, 2H)
            labs[sp].append((a, g, ca, cg))
    X = {sp: torch.stack(feats[sp]).float() for sp in (0, 1)}               # (E, 13, 2H)
    Y = {sp: torch.tensor(labs[sp], device=device) for sp in (0, 1)}
    acc = np.zeros((4, N_LAYERS), np.float32)   # rows: ARG, ARG-control, GEN, GEN-control
    for l in range(N_LAYERS):
        for row, (col, keep_col, ncls) in enumerate([(0, 0, len(ARG)), (2, 0, len(ARG)),
                                                     (1, 1, len(data["GEN"])), (3, 1, len(data["GEN"]))]):
            mtr = Y[0][:, keep_col] >= 0; mte = Y[1][:, keep_col] >= 0
            acc[row, l] = fit_probe(X[0][mtr, l], Y[0][mtr, col], X[1][mte, l], Y[1][mte, col], ncls, seed=l * 10 + row)
    n_edges = {k: int((Y[0][:, c] >= 0).sum()) for k, c in (("arg_train", 0), ("gen_train", 1))}
    return acc, n_edges


def cmd_run(args):
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    device = fe.pick_device(args.device)
    out = Path(args.out_dir); (out / "probe").mkdir(parents=True, exist_ok=True)
    data = build_data(out)
    print(f"UD probe data: {sum(1 for s in data['split'] if s == 0)} train / "
          f"{sum(1 for s in data['split'] if s == 1)} test sentences; GEN labels {data['GEN']}", flush=True)
    for k, i in enumerate(fe.parse_models(args.models)):
        f = out / "probe" / f"probe_{i:03d}.npz"
        if f.exists():
            print(f"[{k+1}] model {i}: exists, skipping"); continue
        t0 = time.time()
        try:
            path, sha = fe.model_source(i, args)
            tok = AutoTokenizer.from_pretrained(path)
            model = BertForSequenceClassification.from_pretrained(path).to(device).eval()
            acc, ne = probe_model(tok, model, data, device)
        except Exception as e:  # noqa: BLE001
            print(f"[{k+1}] model {i}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True); continue
        np.savez(f, acc=acc, sha=sha, **ne)
        sel_a = acc[0] - acc[1]; sel_g = acc[2] - acc[3]
        print(f"[{k+1}] model {i}: ARG acc max {acc[0].max():.3f} (control {acc[1].max():.3f}), "
              f"best-layer selectivity ARG {sel_a.max():.3f} GEN {sel_g.max():.3f}  ({time.time()-t0:.0f}s)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.purge and not args.local_tmpl:
            fe.purge_from_cache(i)


def cmd_analyze(args):
    import feather_step4 as s4
    import juneja_analyze as ja
    import stats_extra as sx
    from scipy.stats import pearsonr
    out = Path(args.out_dir); s4dir = Path(args.step4_dir)
    split = json.loads((s4dir / "split.json").read_text())
    dev, test = split["dev"], split["test"]; ids = dev + test; order = np.array(ids)
    blocks, y, _, _ = s4.build_blocks(s4dir, ids)
    C = ja.load_cg(s4dir / "cg"); A, fj, _, _ = ja.juneja_feature(C)
    blocks["Juneja-CG"] = s4.fixed_kernel_block(A[np.ix_(order, order)])
    blocks["Juneja-f"] = s4.linear_kernel_block(fj[order][:, None])
    P = np.array([np.load(out / "probe" / f"probe_{i:03d}.npz")["acc"] for i in ids])   # models x 4 x 13
    sel = np.concatenate([P[:, 0] - P[:, 1], P[:, 2] - P[:, 3]], 1)                       # models x 26
    blocks["Probe"] = s4.linear_kernel_block(sel)
    BJ = s4.BASE + ["Juneja-CG", "Juneja-f"]
    all_idx = np.arange(len(ids))
    oof = {k: s4.cv_oof(blocks, v, all_idx, y, 10, 0) for k, v in
           [("probe", ["Probe"]), ("bj", BJ), ("best", ["ID-overlap-items"]), ("internal", s4.INTERNAL)]}
    L = ["PROBE (UD syntactic roles) as an interpretability signal not read off outputs", "",
         f"Probe accuracy, ARG task: best layer mean over models {P[:, 0].max(1).mean():.3f} "
         f"(control {P[:, 1].max(1).mean():.3f}); GEN task {P[:, 2].max(1).mean():.3f} (control {P[:, 3].max(1).mean():.3f})",
         "Correlation of ARG selectivity with HANS-LO accuracy, by layer: " +
         " ".join(f"L{l}:{pearsonr(sel[:, l], y)[0]:+.2f}" for l in range(N_LAYERS)), "",
         f"CV R2 (n=100), probe block alone: {oof['probe'][1].mean():.3f}   "
         f"(baselines+Juneja {oof['bj'][1].mean():.3f}; internal {oof['internal'][1].mean():.3f}; "
         f"best single baseline {oof['best'][1].mean():.3f})", ""]
    r, p, q95, m = sx.perm_cv_partial(s4, blocks, BJ, ["Probe"], y, n_rep=5, n_perm=args.n_perm)
    passed = p < 0.05 and r >= 0.20
    L += [f"PRIMARY (pre-registered): probe over baselines + Juneja: partial r {r:+.3f}, permutation p = {p:.3f} "
          f"(null mean {m:+.3f}, 95th pct {q95:+.3f})",
          f"  -> {'PASS' if passed else 'NOT PASSED'} (needs p < 0.05 and partial r >= 0.20)"]
    for lab, base in [("best single baseline", ["ID-overlap-items"]), ("output-based internal", s4.INTERNAL)]:
        r2_, p2_, _, _ = sx.perm_cv_partial(s4, blocks, base, ["Probe"], y, n_rep=5, n_perm=args.n_perm)
        L.append(f"secondary: probe over {lab}: partial r {r2_:+.3f}, permutation p = {p2_:.3f}")
    rep = "\n".join(L); print(rep)
    (out / "probe_report.txt").write_text(rep + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["run", "analyze"]:
        p = sp.add_parser(name); p.add_argument("--out_dir", default="probe_out")
        if name == "run":
            p.add_argument("--models", nargs="+", default=["all"]); p.add_argument("--device", default="auto")
            p.add_argument("--local_tmpl", default=None); p.add_argument("--purge", action="store_true")
        else:
            p.add_argument("--step4_dir", required=True)
            p.add_argument("--n_perm", type=int, default=200)
    args = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
