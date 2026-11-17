#!/usr/bin/env python3
"""
Steps 2-3 of the "mechanistic vs representational predictors of OOD behaviour" project.

Population: the 100 BERT-base models McCoy et al. (2020) fine-tuned on MNLI, hosted as
    connectivity/feather_berts_0 ... connectivity/feather_berts_99   (Juneja et al. 2023)

Subcommands
-----------
  check    Step 2. Load a few models; verify (a) the label order, (b) that our forward pass
           reproduces the logits stored in each repo's mnli_eval_bert_{i}.json, and (c) that
           MNLI matched-dev accuracy falls in McCoy et al.'s reported 83.6-84.8% range.
  eval     Step 3a. Run every model on MNLI matched-dev and all 30k HANS items and save
           per-example logits. Resumable: models with an existing output file are skipped.
  analyze  Step 3b. Is the OOD outcome real? Split-half reliability and between-model spread
           of HANS lexical-overlap non-entailment accuracy, plus diagnostics. Prints PASS/FAIL
           against pre-set gates.

Typical use (on a GPU machine with Hugging Face access)
-------------------------------------------------------
  python feather_eval.py check   --models 0 1 2 3 4
  python feather_eval.py eval    --models all --purge
  python feather_eval.py analyze

Label conventions (verified against the repos' own eval files):
  model output index : 0 = contradiction, 1 = entailment, 2 = neutral   (Google BERT MnliProcessor)
  GLUE/HF MNLI label : 0 = entailment,    1 = neutral,    2 = contradiction
  HANS               : prediction is "entailment" iff argmax == 1; neutral + contradiction
                       collapse to "non-entailment" (as in McCoy et al.).
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- constants

REPO_TMPL = "connectivity/feather_berts_{i}"
N_MODELS = 100
MODEL_LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}
ENT = MODEL_LABELS["entailment"]
GLUE_TO_MODEL = np.array([1, 2, 0])  # GLUE ent/neu/con -> model index
HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
MCCOY_MNLI_RANGE = (0.836, 0.848)  # reported MNLI matched-dev range across the 100 models
MAX_LEN = 128                       # max_seq_length used by McCoy et al. (Google BERT default)

# Pre-registered gates for step 3 (set before seeing data; do not tune).
GATE_SPLIT_HALF_R = 0.80   # raw (not Spearman-Brown) stratified split-half correlation
GATE_SD_RATIO = 3.0        # between-model SD / binomial sampling SD
GATE_RANGE = 0.10          # max - min accuracy across models


# ----------------------------------------------------------------------------- data

def load_hans(path):
    import pandas as pd
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading HANS to {path} ...")
        urllib.request.urlretrieve(HANS_URL, path)
    d = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    assert len(d) == 30000, f"HANS should have 30000 rows, got {len(d)}"
    return {
        "premise": d["sentence1"].tolist(),
        "hypothesis": d["sentence2"].tolist(),
        "gold_ent": (d["gold_label"] == "entailment").to_numpy(),
        "heuristic": d["heuristic"].to_numpy(),
        "subcase": d["subcase"].to_numpy(),
        "template": d["template"].to_numpy(),
        "pairID": d["pairID"].to_numpy(),
    }


def load_mnli(tsv=None):
    """MNLI matched dev. Labels returned in MODEL index space."""
    if tsv:  # GLUE dev_matched.tsv
        prem, hyp, lab = [], [], []
        name2model = MODEL_LABELS
        with open(tsv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in r:
                if row["gold_label"] not in name2model:
                    continue
                prem.append(row["sentence1"]); hyp.append(row["sentence2"])
                lab.append(name2model[row["gold_label"]])
        return {"premise": prem, "hypothesis": hyp, "label": np.array(lab)}
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    lab = np.array(ds["label"])
    keep = lab >= 0
    return {
        "premise": [p for p, k in zip(ds["premise"], keep) if k],
        "hypothesis": [h for h, k in zip(ds["hypothesis"], keep) if k],
        "label": GLUE_TO_MODEL[lab[keep]],
    }


# ----------------------------------------------------------------------------- models

def model_source(i, args):
    """Return (local_dir, revision_sha). Downloads from the Hub unless --local_tmpl is set."""
    if args.local_tmpl:
        return Path(args.local_tmpl.format(i=i)), "local"
    from huggingface_hub import snapshot_download
    repo = REPO_TMPL.format(i=i)
    path = snapshot_download(
        repo,
        allow_patterns=["config.json", "pytorch_model.bin", "*.safetensors", "tokenizer*",
                        "vocab.txt", "special_tokens_map.json"],
    )
    return Path(path), Path(path).name  # snapshot dir name is the commit sha


def purge_from_cache(i):
    from huggingface_hub import scan_cache_dir
    repo = REPO_TMPL.format(i=i)
    info = scan_cache_dir()
    for r in info.repos:
        if r.repo_id == repo:
            info.delete_revisions(*[rev.commit_hash for rev in r.revisions]).execute()


def load_model(path, device):
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path).to(device).eval()
    # Sanity: uncased tokenizer and 3-way head.
    if tok("Hello World")["input_ids"] != tok("hello world")["input_ids"]:
        print("  WARNING: tokenizer is not lower-casing; McCoy models are bert-base-uncased.")
    assert model.config.num_labels == 3, model.config.num_labels
    return tok, model


def run_model(tok, model, premises, hypotheses, device, batch_size=256):
    """Logits (float32, original order). Batches are formed over length-sorted inputs."""
    import torch
    n = len(premises)
    order = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses)])
    out = np.empty((n, 3), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            enc = tok([premises[j] for j in idx], [hypotheses[j] for j in idx],
                      truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out[idx] = model(**enc).logits.float().cpu().numpy()
    return out


def pick_device(arg):
    import torch
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_models(spec):
    if spec == ["all"]:
        return list(range(N_MODELS))
    return [int(x) for x in spec]


# ----------------------------------------------------------------------------- step 2: check

def read_stored_eval(path_dir, i, args):
    """The repo's own mnli_eval_bert_{i}.json: label_dict + samples with logits and label."""
    fname = f"mnli_eval_bert_{i}.json"
    if args.local_tmpl:
        p = Path(path_dir) / fname
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO_TMPL.format(i=i), fname)
    with open(p) as f:
        d = json.load(f)
    return d["label_dict"], d["samples"]


def split_stored_sentence(s):
    s = s.strip()
    if s.startswith("[CLS]"):
        s = s[len("[CLS]"):]
    parts = [p.strip() for p in s.split("[SEP]")]
    parts = [p for p in parts if p]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def cmd_check(args):
    import torch
    device = pick_device(args.device)
    print(f"device={device}  torch={torch.__version__}")
    mnli = load_mnli(args.mnli_tsv)
    print(f"MNLI matched dev: {len(mnli['label'])} labelled examples")
    rows, all_ok = [], True
    for i in parse_models(args.models):
        t0 = time.time()
        path, sha = model_source(i, args)
        tok, model = load_model(path, device)

        # (a) label order
        label_dict, samples = read_stored_eval(path, i, args)
        label_ok = {k.lower(): int(v) for k, v in label_dict.items()} == MODEL_LABELS

        # (b) reproduce stored logits on the first N stored examples
        pairs, stored = [], []
        for smp in samples[: args.n_reproduce]:
            pr = split_stored_sentence(smp["sentence"])
            if pr is not None:
                pairs.append(pr); stored.append(smp["logits"])
        stored = np.array(stored, dtype=np.float32)
        ours = run_model(tok, model, [p for p, _ in pairs], [h for _, h in pairs], device,
                         args.batch_size)
        absdiff = np.abs(ours - stored)
        argmax_agree = float((ours.argmax(1) == stored.argmax(1)).mean())

        # accuracy implied by the stored file itself (all samples)
        st_logits = np.array([s["logits"] for s in samples], dtype=np.float32)
        st_labels = np.array([s["label"] for s in samples])
        stored_acc = float((st_logits.argmax(1) == st_labels).mean())

        # (c) our accuracy on MNLI matched dev with our label mapping
        logits = run_model(tok, model, mnli["premise"], mnli["hypothesis"], device, args.batch_size)
        acc = float((logits.argmax(1) == mnli["label"]).mean())

        lo, hi = MCCOY_MNLI_RANGE
        ok = (label_ok and argmax_agree >= 0.995 and float(np.median(absdiff)) < 1e-2
              and lo - 0.005 <= acc <= hi + 0.005 and abs(acc - stored_acc) < 0.005)
        all_ok &= ok
        rows.append(dict(model=i, sha=sha, label_order_ok=label_ok, n_stored=len(samples),
                         n_reproduced=len(pairs), median_absdiff=float(np.median(absdiff)),
                         max_absdiff=float(absdiff.max()), argmax_agree=argmax_agree,
                         stored_file_acc=stored_acc, our_mnli_acc=acc,
                         seconds=round(time.time() - t0, 1), PASS=ok))
        r = rows[-1]
        print(f"[model {i:>2}] label_order={'ok' if label_ok else 'MISMATCH'}  "
              f"reproduce: argmax_agree={argmax_agree:.4f} median|dlogit|={r['median_absdiff']:.2e} "
              f"max|dlogit|={r['max_absdiff']:.2e}  stored_acc={stored_acc:.4f}  "
              f"our_acc={acc:.4f}  ({r['seconds']}s)  {'PASS' if ok else 'FAIL'}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "step2_check.json", "w") as f:
        json.dump({"rows": rows, "env": env_info(), "all_pass": all_ok}, f, indent=2)
    print("\nSTEP 2:", "PASS - population is usable; run `eval`." if all_ok else
          "FAIL - inspect step2_check.json before continuing.")
    print("Interpretation: our_acc ~= stored_file_acc and argmax_agree ~= 1 means weights, "
          "tokenisation and label mapping all match the original evaluation.")


# ----------------------------------------------------------------------------- step 3a: eval

def cmd_eval(args):
    import torch
    device = pick_device(args.device)
    out = Path(args.out_dir); (out / "logits").mkdir(parents=True, exist_ok=True)
    mnli = load_mnli(args.mnli_tsv)
    hans = load_hans(args.hans_path)
    # Save aligned metadata once so `analyze` never depends on loader order at analysis time.
    np.savez(out / "meta.npz", mnli_label=mnli["label"], hans_gold_ent=hans["gold_ent"],
             hans_heuristic=hans["heuristic"], hans_subcase=hans["subcase"],
             hans_template=hans["template"], hans_pairID=hans["pairID"])
    models = parse_models(args.models)
    print(f"device={device}  models={len(models)}  MNLI n={len(mnli['label'])}  HANS n=30000")
    failed = {}
    for k, i in enumerate(models):
        f = out / "logits" / f"model_{i:03d}.npz"
        if f.exists():
            print(f"[{k+1}/{len(models)}] model {i}: exists, skipping"); continue
        t0 = time.time()
        try:
            path, sha = model_source(i, args)
            tok, model = load_model(path, device)
            ml = run_model(tok, model, mnli["premise"], mnli["hypothesis"], device, args.batch_size)
            hl = run_model(tok, model, hans["premise"], hans["hypothesis"], device, args.batch_size)
        except Exception as e:  # one bad repo must not kill a 100-model run; rerun later to retry
            failed[i] = f"{type(e).__name__}: {e}"
            print(f"[{k+1}/{len(models)}] model {i}: FAILED ({failed[i][:200]})")
            continue
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, mnli_logits=ml, hans_logits=hl, sha=sha)
        os.replace(tmp, f)  # atomic: a crash never leaves a half-written file
        mnli_acc = (ml.argmax(1) == mnli["label"]).mean()
        lo = (hans["heuristic"] == "lexical_overlap") & ~hans["gold_ent"]
        lo_acc = (hl[lo].argmax(1) != ENT).mean()
        print(f"[{k+1}/{len(models)}] model {i}: MNLI={mnli_acc:.4f}  HANS-LO-nonent={lo_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.purge and not args.local_tmpl:
            purge_from_cache(i)
    with open(out / "eval_env.json", "w") as fh:
        json.dump({**env_info(), "failed": failed}, fh, indent=2)
    done = len(list((out / "logits").glob("model_*.npz")))
    print(f"\nDone: {done} models have logits. Failed this run: {sorted(failed) or 'none'}"
          + ("  (rerun the same command to retry; finished models are skipped)" if failed else ""))


# ----------------------------------------------------------------------------- step 3b: analyze

def bimodality_coefficient(x):
    from scipy.stats import kurtosis, skew
    n = len(x)
    g, k = skew(x, bias=False), kurtosis(x, bias=False)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def cmd_analyze(args):
    import itertools
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    out = Path(args.out_dir)
    meta = np.load(out / "meta.npz", allow_pickle=True)
    files = sorted((out / "logits").glob("model_*.npz"))
    if len(files) < 10:
        sys.exit(f"Only {len(files)} models evaluated; run `eval` first.")
    ids = [int(f.stem.split("_")[1]) for f in files]
    gold_ent, heur, sub = meta["hans_gold_ent"], meta["hans_heuristic"], meta["hans_subcase"]

    # correctness matrices: models x items
    mnli_correct, hans_correct = [], []
    for f in files:
        z = np.load(f)
        mnli_correct.append(z["mnli_logits"].argmax(1) == meta["mnli_label"])
        pred_ent = z["hans_logits"].argmax(1) == ENT
        hans_correct.append(pred_ent == gold_ent)
    mnli_correct = np.array(mnli_correct); hans_correct = np.array(hans_correct)

    # per-model table
    table = pd.DataFrame({"model": ids, "mnli_acc": mnli_correct.mean(1),
                          "hans_acc": hans_correct.mean(1)})
    for h in ["lexical_overlap", "subsequence", "constituent"]:
        for g, gname in [(True, "ent"), (False, "nonent")]:
            m = (heur == h) & (gold_ent == g)
            table[f"{h}_{gname}"] = hans_correct[:, m].mean(1)
    for s in sorted(set(sub)):
        table[f"sub_{s}"] = hans_correct[:, sub == s].mean(1)
    table.to_csv(out / "per_model_accuracy.csv", index=False)

    # ---- primary outcome: lexical-overlap non-entailment accuracy
    lo_mask = (heur == "lexical_overlap") & ~gold_ent
    Y = hans_correct[:, lo_mask]            # models x 5000
    lo_sub = sub[lo_mask]
    y = Y.mean(1)
    n_items = Y.shape[1]
    rng = np.random.default_rng(args.seed)

    # (1) spread vs. binomial sampling noise
    p = y.mean()
    samp_sd = np.sqrt(p * (1 - p) / n_items)
    between_sd = y.std(ddof=1)
    rng_ = y.max() - y.min()

    # (2) stratified random split-half reliability (each subcase split in half)
    rs = []
    sub_idx = {s: np.where(lo_sub == s)[0] for s in np.unique(lo_sub)}
    for _ in range(args.n_splits):
        a = np.zeros(n_items, bool)
        for s, ix in sub_idx.items():
            perm = rng.permutation(ix)
            a[perm[: len(ix) // 2]] = True
        rs.append(pearsonr(Y[:, a].mean(1), Y[:, ~a].mean(1))[0])
    rs = np.array(rs)
    r_half = float(np.mean(rs))
    r_full = 2 * r_half / (1 + r_half)    # Spearman-Brown: reliability of the full 5000-item score

    # (3) stricter: subcase split (do constructions agree about which models generalise?)
    subs = sorted(sub_idx)
    sub_acc = np.stack([Y[:, sub_idx[s]].mean(1) for s in subs], 1)
    sub_corr = pd.DataFrame(np.corrcoef(sub_acc.T), index=subs, columns=subs)
    part_r = []
    for k in (2,):
        for A in itertools.combinations(range(len(subs)), k):
            B = [j for j in range(len(subs)) if j not in A]
            part_r.append(pearsonr(sub_acc[:, list(A)].mean(1), sub_acc[:, B].mean(1))[0])

    # (4) diagnostics for later steps
    mnli_acc = mnli_correct.mean(1)
    r_mnli = pearsonr(mnli_acc, y)[0]
    rho_mnli = spearmanr(mnli_acc, y)[0]
    bc = bimodality_coefficient(y)

    g_rel = r_half >= GATE_SPLIT_HALF_R
    g_spread = (between_sd >= GATE_SD_RATIO * samp_sd) and (rng_ >= GATE_RANGE)
    lo_, hi_ = MCCOY_MNLI_RANGE

    lines = [
        f"STEP 3 ANALYSIS  ({len(ids)} models)",
        "",
        f"MNLI matched-dev accuracy: min {mnli_acc.min():.4f}  max {mnli_acc.max():.4f}  "
        f"(McCoy et al. report {lo_:.3f}-{hi_:.3f})",
        "",
        f"Primary outcome: HANS lexical-overlap non-entailment accuracy (n={n_items} items/model)",
        f"  mean {y.mean():.4f}  SD {between_sd:.4f}  min {y.min():.4f}  max {y.max():.4f}  "
        f"IQR {np.percentile(y,25):.4f}-{np.percentile(y,75):.4f}",
        f"  binomial sampling SD at this mean: {samp_sd:.4f}  ->  between-model/sampling SD ratio "
        f"{between_sd/samp_sd:.1f}",
        "",
        f"Reliability (stratified random split-half, {args.n_splits} splits):",
        f"  raw split-half r  mean {r_half:.3f}  [2.5%-97.5%: {np.percentile(rs,2.5):.3f}-"
        f"{np.percentile(rs,97.5):.3f}]",
        f"  Spearman-Brown full-length reliability {r_full:.3f}",
        f"  -> ceiling for any predictor: R^2 <= {r_full:.3f}, |r| <= {np.sqrt(r_full):.3f}",
        "",
        "Stricter: agreement across constructions (2-vs-3 subcase partitions, r):",
        "  " + "  ".join(f"{v:.2f}" for v in part_r) + f"   (mean {np.mean(part_r):.3f})",
        "  Between-subcase correlation matrix:",
        sub_corr.round(2).to_string(),
        "",
        "Diagnostics for step 4:",
        f"  MNLI accuracy vs outcome: Pearson {r_mnli:.3f}, Spearman {rho_mnli:.3f}  "
        f"(R^2 = {r_mnli**2:.3f}; share of reliable variance = {r_mnli**2 / r_full:.2f})",
        f"  bimodality coefficient {bc:.3f}  (>0.555 suggests two clusters, cf. Juneja's two basins)",
        "",
        "GATES (pre-registered):",
        f"  G1 reliability: raw split-half r >= {GATE_SPLIT_HALF_R}  ->  {'PASS' if g_rel else 'FAIL'}",
        f"  G2 spread: SD ratio >= {GATE_SD_RATIO} and range >= {GATE_RANGE}  ->  "
        f"{'PASS' if g_spread else 'FAIL'}",
        "",
        ("VERDICT: outcome is real; proceed to step 4 (baselines)." if (g_rel and g_spread) else
         "VERDICT: outcome fails a gate; OOD differences are mostly noise or too small. Stop or "
         "change the outcome before building predictors."),
    ]
    if r_mnli ** 2 / r_full > 0.5:
        lines.append("NOTE: MNLI accuracy alone already explains >50% of the reliable variance; "
                     "step 4 headroom may be small.")
    report = "\n".join(lines)
    print(report)
    (out / "step3_report.txt").write_text(report + "\n")
    with open(out / "step3_summary.json", "w") as f:
        json.dump(dict(n_models=len(ids), outcome_mean=float(y.mean()), outcome_sd=float(between_sd),
                       outcome_min=float(y.min()), outcome_max=float(y.max()),
                       sampling_sd=float(samp_sd), split_half_r=r_half, spearman_brown=r_full,
                       subcase_partition_r=[float(v) for v in part_r], mnli_pearson=float(r_mnli),
                       bimodality_coef=float(bc), gate_reliability=bool(g_rel),
                       gate_spread=bool(g_spread)), f, indent=2)


# ----------------------------------------------------------------------------- misc

def env_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["torch", "transformers", "numpy", "scipy", "huggingface_hub", "datasets"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["check", "eval", "analyze"]:
        p = sp.add_parser(name)
        p.add_argument("--out_dir", default="feather_out")
        p.add_argument("--seed", type=int, default=0)
        if name in ("check", "eval"):
            p.add_argument("--models", nargs="+", default=["0", "1", "2", "3", "4"] if name == "check" else ["all"])
            p.add_argument("--device", default="auto")
            p.add_argument("--batch_size", type=int, default=256)
            p.add_argument("--mnli_tsv", default=None, help="GLUE dev_matched.tsv (else HF datasets)")
            p.add_argument("--local_tmpl", default=None,
                           help="Local model dirs, e.g. /models/feather_berts_{i} (skips the Hub)")
        if name == "check":
            p.add_argument("--n_reproduce", type=int, default=2000,
                           help="stored examples to re-run for the logit comparison")
        if name == "eval":
            p.add_argument("--hans_path", default="data/heuristics_evaluation_set.txt")
            p.add_argument("--purge", action="store_true",
                           help="delete each model from the HF cache after use (saves ~44 GB)")
        if name == "analyze":
            p.add_argument("--n_splits", type=int, default=1000)
    args = ap.parse_args()
    {"check": cmd_check, "eval": cmd_eval, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
