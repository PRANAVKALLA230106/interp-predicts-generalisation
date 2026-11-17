#!/usr/bin/env python3
"""
Fix 1, stage A: is there a second population with a REAL out-of-distribution outcome?

Population: the 100 BERT-base models Juneja et al. (2023) fine-tuned on QQP (paraphrase detection),
hosted as connectivity/bert_ft_qqp-0 ... -99. Label order (from each repo's eval files):
0 = not_duplicate, 1 = duplicate. The shortcut analogous to HANS lexical overlap: "the two
questions share their words, so they are duplicates". PAWS is built from exactly such traps.

This stage only runs the models and checks the outcome, as step 3 did for HANS:
  run      GPU. For each model: QQP validation (fixed 10k subsample), PAWS-QQP (the 677 dev+test
           items stored in each repo's paws_eval json, sentences included) and PAWS-Wiki
           (labeled_final test, 8,000 items). Saves logits; resumable; --purge frees disk.
  gate     CPU. Reproduction check against the stored PAWS-QQP logits, and the reliability gate
           for each candidate outcome.

  groups   CPU + internet. Writes item groups for the split-half (items sharing a sentence are
           one group), so reliability is not inflated by near-duplicate items.

Pre-stated (written before running; revised 2026-11-14 BEFORE any stage-A result was seen):
  Two DISTINCT evaluation settings, never pooled and never substituted for one another:
    PAWS-QQP   adversarial pairs built from Quora questions (same source domain as training);
    PAWS-Wiki  adversarial pairs built from Wikipedia (adversarial AND a domain shift).
  Outcome in each: accuracy on NON-paraphrase (label 0) items.
  Gate per setting, thresholds as in step 3 (fixed before step 3 was run):
    - group-blocked split-half r >= 0.80 (items sharing a sentence never straddle the halves).
      Rationale: the Spearman-Brown full-length reliability is then >= 0.89, so at most ~11% of
      between-model variance is measurement noise and an increment of dR2 = 0.10 is resolvable;
      0.80 is the conventional floor for score reliability.
    - between-model SD >= 3x the binomial sampling SD (sampling variance <= 1/9 of the observed
      variance, consistent with the reliability floor).
    - range >= 0.10 (smaller differences between models are too small to matter in practice).
  Each passing setting gets its own step-4 test and its own verdict, at alpha = 0.025 each
  (Bonferroni over the two settings). If neither passes, the QQP population cannot test the
  hypothesis, reported as such.
  Amendment 2026-11-14 (after an independent review, before any stage-A result was seen):
  groups also merge items whose sentences share a multiset of words (PAWS word-swap families).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feather_eval as fe  # noqa: E402

fe.REPO_TMPL = "connectivity/bert_ft_qqp-{i}"   # model_source / purge_from_cache read this
QQP_LABELS = {"not_duplicate": 0, "duplicate": 1}
DUP = 1
N_QQP = 10000
PAWSQ_FILE = "paws_eval@34110steps_bert_ft_qqp-{i}.json"
GATE_R, GATE_SD, GATE_RANGE = 0.80, 3.0, 0.10


def load_qqp(n=N_QQP):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    idx = np.sort(np.random.default_rng(0).choice(len(ds), n, replace=False))
    sub = ds.select(idx.tolist())
    return {"premise": sub["question1"], "hypothesis": sub["question2"], "label": np.array(sub["label"])}


def load_paws_wiki():
    from datasets import load_dataset
    for name in ("google-research-datasets/paws", "paws"):
        try:
            ds = load_dataset(name, "labeled_final", split="test"); break
        except Exception as e:  # noqa: BLE001
            err = e
    else:
        raise err
    return {"premise": ds["sentence1"], "hypothesis": ds["sentence2"], "label": np.array(ds["label"])}


def load_pawsq(i):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(fe.REPO_TMPL.format(i=i), PAWSQ_FILE.format(i=i))
    d = json.load(open(p))
    assert {k.lower(): int(v) for k, v in d["label_dict"].items()} == QQP_LABELS, d["label_dict"]
    prem, hyp, lab, logit, keep = [], [], [], [], []
    for s in d["samples"]:
        pr = fe.split_stored_sentence(s["sentence"])
        if pr is None:
            continue
        prem.append(pr[0]); hyp.append(pr[1]); lab.append(int(s["label"])); logit.append(s["logits"])
    return {"premise": prem, "hypothesis": hyp, "label": np.array(lab),
            "stored": np.array(logit, np.float32)}


def run(tok, model, P, H, device, bs=256):
    import torch
    n = len(P); order = np.argsort([len(a) + len(b) for a, b in zip(P, H)])
    out = np.empty((n, model.config.num_labels), np.float32)
    with torch.inference_mode():
        for s in range(0, n, bs):
            ix = order[s:s + bs]
            enc = tok([P[j] for j in ix], [H[j] for j in ix], truncation=True, max_length=fe.MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            out[ix] = model(**enc).logits.float().cpu().numpy()
    return out


def cmd_run(args):
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    device = fe.pick_device(args.device)
    out = Path(args.out_dir); (out / "logits").mkdir(parents=True, exist_ok=True)
    qqp = load_qqp(); pw = load_paws_wiki()
    ref = load_pawsq(0)
    np.savez(out / "meta.npz", qqp_label=qqp["label"], pawsw_label=pw["label"], pawsq_label=ref["label"])
    print(f"device={device}  QQP n={len(qqp['label'])}  PAWS-QQP n={len(ref['label'])} "
          f"(non-dup {(ref['label'] == 0).sum()})  PAWS-Wiki n={len(pw['label'])} (non-dup {(pw['label'] == 0).sum()})")
    for k, i in enumerate(fe.parse_models(args.models)):
        f = out / "logits" / f"model_{i:03d}.npz"
        if f.exists():
            print(f"[{k+1}] model {i}: exists, skipping"); continue
        t0 = time.time()
        try:
            pq = load_pawsq(i)
            same = pq["premise"] == ref["premise"] and pq["hypothesis"] == ref["hypothesis"]
            path, sha = fe.model_source(i, args)
            tok = AutoTokenizer.from_pretrained(path)
            model = BertForSequenceClassification.from_pretrained(path).to(device).eval()
            assert model.config.num_labels == 2
            ql = run(tok, model, qqp["premise"], qqp["hypothesis"], device)
            pql = run(tok, model, ref["premise"], ref["hypothesis"], device)
            pwl = run(tok, model, pw["premise"], pw["hypothesis"], device)
        except Exception as e:  # noqa: BLE001
            print(f"[{k+1}] model {i}: FAILED {type(e).__name__}: {str(e)[:300]}"); continue
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, qqp_logits=ql, pawsq_logits=pql, pawsw_logits=pwl,
                 pawsq_stored=pq["stored"] if same else np.zeros(0), same_items=same, sha=sha)
        tmp.replace(f)
        agree = (pql.argmax(1) == pq["stored"].argmax(1)).mean() if same else float("nan")
        print(f"[{k+1}] model {i}: QQP acc {(ql.argmax(1) == qqp['label']).mean():.4f}  "
              f"PAWS-QQP non-dup {(pql.argmax(1)[ref['label'] == 0] != DUP).mean():.3f}  "
              f"PAWS-Wiki non-dup {(pwl.argmax(1)[pw['label'] == 0] != DUP).mean():.3f}  "
              f"stored-agree {agree:.4f}  ({time.time()-t0:.0f}s)")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.purge:
            fe.purge_from_cache(i)


def sentence_groups(P, H):
    """Union-find: items that share a sentence, OR a sentence with the same multiset of words (PAWS
    builds pairs by swapping words, so families share a bag of words), form one group."""
    import re
    parent = list(range(len(P))); first = {}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for k, (p, h) in enumerate(zip(P, H)):
        for s in (p, h):
            toks = re.findall(r"\w+", s.lower())
            for key in (" ".join(toks), "BAG:" + " ".join(sorted(toks))):
                if key in first:
                    parent[find(k)] = find(first[key])
                else:
                    first[key] = k
    return np.array([find(k) for k in range(len(P))])


def cmd_groups(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pw = load_paws_wiki(); pq = load_pawsq(0)
    np.savez(out / "paws_groups.npz", pawsw=sentence_groups(pw["premise"], pw["hypothesis"]),
             pawsq=sentence_groups(pq["premise"], pq["hypothesis"]))
    g = np.load(out / "paws_groups.npz")
    print(f"PAWS-Wiki: {len(g['pawsw'])} items in {len(set(g['pawsw']))} groups; "
          f"PAWS-QQP: {len(g['pawsq'])} items in {len(set(g['pawsq']))} groups")


def gate_one(C, rng, groups, n_splits=1000):
    """C: models x items correctness; groups: group id per item (None = random item split)."""
    from scipy.stats import pearsonr
    y = C.mean(1); p = y.mean(); n = C.shape[1]
    rs = []
    ug = np.unique(groups) if groups is not None else None
    for _ in range(n_splits):
        if ug is None:
            a = np.zeros(n, bool); a[rng.permutation(n)[: n // 2]] = True
        else:
            a = np.isin(groups, rng.permutation(ug)[: len(ug) // 2])
        rs.append(pearsonr(C[:, a].mean(1), C[:, ~a].mean(1))[0])
    sd, bs = y.std(ddof=1), np.sqrt(p * (1 - p) / n)
    r = float(np.mean(rs))
    ok = r >= GATE_R and sd >= GATE_SD * bs and y.max() - y.min() >= GATE_RANGE
    return dict(n_items=int(n), n_groups=int(len(ug)) if ug is not None else int(n),
                mean=float(p), sd=float(sd), sd_ratio=float(sd / bs),
                min=float(y.min()), max=float(y.max()), split_half_r=r,
                r_lo=float(np.percentile(rs, 2.5)), PASS=bool(ok))


def cmd_gate(args):
    from scipy.stats import pearsonr
    out = Path(args.out_dir); meta = np.load(out / "meta.npz")
    files = sorted((out / "logits").glob("model_*.npz"))
    Z = [np.load(f) for f in files]
    ids = [int(f.stem.split("_")[1]) for f in files]
    qacc = np.array([(z["qqp_logits"].argmax(1) == meta["qqp_label"]).mean() for z in Z])
    agree = [float((z["pawsq_logits"].argmax(1) == z["pawsq_stored"].argmax(1)).mean())
             for z in Z if z["pawsq_stored"].size]
    gf = out / "paws_groups.npz"
    if not gf.exists():
        sys.exit("paws_groups.npz missing: run `python fix1_qqp.py groups` first (needs internet).")
    G = np.load(gf)
    rng = np.random.default_rng(0)
    res = {}
    for name, key, lab, gk in [("PAWS-QQP", "pawsq_logits", meta["pawsq_label"], "pawsq"),
                               ("PAWS-Wiki", "pawsw_logits", meta["pawsw_label"], "pawsw")]:
        C = np.array([z[key].argmax(1)[lab == 0] != DUP for z in Z])
        res[name] = gate_one(C, rng, G[gk][lab == 0])
        res[name]["random_split_r"] = gate_one(C, rng, None, 200)["split_half_r"]
        res[name]["r_with_qqp_acc"] = float(pearsonr(qacc, C.mean(1))[0])
    passing = [k for k in ("PAWS-QQP", "PAWS-Wiki") if res[k]["PASS"]]
    L = [f"FIX 1 STAGE A: QQP population ({len(ids)} models)", "",
         f"QQP validation accuracy (10k subsample): {qacc.min():.4f} - {qacc.max():.4f}",
         f"Reproduction of stored PAWS-QQP predictions: argmax agreement min {min(agree):.4f}, "
         f"median {np.median(agree):.4f} over {len(agree)} models" if agree else
         "Reproduction: stored items differ across repos (not compared)", ""]
    for k, v in res.items():
        L.append(f"{k:10s} non-paraphrase accuracy: n={v['n_items']} in {v['n_groups']} groups  mean {v['mean']:.3f}  "
                 f"range {v['min']:.3f}-{v['max']:.3f}  SD {v['sd']:.4f} (x{v['sd_ratio']:.1f} sampling)  "
                 f"group split-half r {v['split_half_r']:.3f} [{v['r_lo']:.2f}] (random-item split {v['random_split_r']:.3f})  "
                 f"r(QQP acc) {v['r_with_qqp_acc']:+.2f}  -> {'PASS' if v['PASS'] else 'FAIL'}")
    L += ["", f"SETTINGS THAT PASS (each tested separately): {', '.join(passing)}" if passing else
          "VERDICT: no reliable OOD outcome in the QQP population; it cannot test the hypothesis."]
    rep = "\n".join(L); print(rep)
    (out / "gate_report.txt").write_text(rep + "\n")
    (out / "gate_summary.json").write_text(json.dumps({"results": res, "passing": passing}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["run", "gate", "groups"]:
        p = sp.add_parser(name); p.add_argument("--out_dir", default="qqp_out")
        if name == "run":
            p.add_argument("--models", nargs="+", default=["all"])
            p.add_argument("--device", default="auto")
            p.add_argument("--local_tmpl", default=None)
            p.add_argument("--purge", action="store_true")
    args = ap.parse_args()
    {"run": cmd_run, "gate": cmd_gate, "groups": cmd_groups}[args.cmd](args)


if __name__ == "__main__":
    main()
