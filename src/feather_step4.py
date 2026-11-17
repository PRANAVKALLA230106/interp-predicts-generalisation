#!/usr/bin/env python3
"""
Step 4: do internal (mechanistic) signals predict HANS lexical-overlap generalisation better than
in-distribution behaviour, weight-space and representational baselines?

Needs feather_eval.py (steps 2-3) in the same folder and its output folder (feather_out/ with
meta.npz and logits/).

Subcommands
-----------
  split    Freeze a 70/30 split of the 100 models (seed fixed). Do this ONCE, before anything else.
           The 30 test models are not looked at until `analyze --final`.
  extract  GPU. For every model, compute features from MNLI matched-dev inputs ONLY (never HANS):
             - representations (mean-pooled hidden states at 4 depths) on a 512-item probe set
             - attention statistics: attention from hypothesis tokens to identical premise tokens
             - head-ablation profile: mean-ablate each head, change in entailment log-odds on
               high-overlap and on ordinary probe items
             - weights: per-block distance from pretrained BERT, a 4096-d CountSketch of the
               fine-tuning delta, and the classifier weights
           Resumable; --purge deletes each model from the HF cache afterwards.
  analyze  CPU. Kernel ridge regression from each feature block to the outcome (several blocks: stacked
           with non-negative weights fitted on leave-one-out predictions). Development
           models: repeated 10-fold cross-validation. With --final: fit on all 70 development
           models and score once on the 30 test models.

Outcome: HANS lexical-overlap non-entailment accuracy (as in step 3). Secondary: same without the
ln_passive subcase.

Pre-registered decision (primary contrast, on the 30 test models):
  dR2 = R2(baselines + internal) - R2(baselines) >= 0.10 and its 95% bootstrap CI excludes 0
  -> GO.  Internal signals predict but dR2 < 0.10 or CI includes 0 -> PIVOT.  Nothing beats the
  MNLI-accuracy-only model -> STOP.
"""

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feather_eval as fe  # noqa: E402

N_PROBE_HO = 256        # high-overlap probe items
N_PROBE_RAND = 256      # ordinary probe items
HO_THRESH = 0.8         # share of hypothesis words that also occur in the premise
SKETCH_D = 4096
SPLIT_SEED = 2026
N_TEST = 30
PRETRAINED = "google-bert/bert-base-uncased"
MIN_DELTA_R2 = 0.10     # pre-registered GO threshold


# ----------------------------------------------------------------------------- helpers

def word_overlap(p, h):
    hw = re.findall(r"\w+", h.lower())
    pw = set(re.findall(r"\w+", p.lower()))
    return float(np.mean([w in pw for w in hw])) if hw else 0.0


def ent_logodds(logits):
    """log P(entailment) - log P(non-entailment), from 3-way logits (numpy or torch)."""
    import torch
    t = torch.as_tensor(logits)
    others = [k for k in range(3) if k != fe.ENT]
    return (t[..., fe.ENT] - torch.logsumexp(t[..., others], -1))


# ----------------------------------------------------------------------------- split

def cmd_split(args):
    out = Path(args.out_dir)
    f = out / "split.json"
    if f.exists() and not args.force:
        print(f"{f} already exists; refusing to overwrite (that would leak the test set).")
        print(json.loads(f.read_text()))
        return
    perm = np.random.default_rng(SPLIT_SEED).permutation(fe.N_MODELS)
    split = {"seed": SPLIT_SEED, "test": sorted(int(i) for i in perm[:N_TEST]),
             "dev": sorted(int(i) for i in perm[N_TEST:]),
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(split, indent=1))
    print(f"Wrote {f}: {len(split['dev'])} dev, {len(split['test'])} test models.")


# ----------------------------------------------------------------------------- probe set

def build_probe(mnli, out):
    """Fixed probe set from MNLI matched dev: high-overlap items + ordinary items."""
    fdir = out / "features"; fdir.mkdir(parents=True, exist_ok=True)
    pf = fdir / "probe.json"
    ov = np.array([word_overlap(p, h) for p, h in zip(mnli["premise"], mnli["hypothesis"])])
    np.save(fdir / "mnli_overlap.npy", ov)
    if pf.exists():
        probe = json.loads(pf.read_text())
    else:
        rng = np.random.default_rng(0)
        ho = np.where(ov >= HO_THRESH)[0]; rest = np.where(ov < HO_THRESH)[0]
        ho_sel = np.sort(rng.choice(ho, min(N_PROBE_HO, len(ho)), replace=False))
        rd_sel = np.sort(rng.choice(rest, min(N_PROBE_RAND, len(rest)), replace=False))
        probe = {"idx": [int(i) for i in np.concatenate([ho_sel, rd_sel])],
                 "is_ho": [True] * len(ho_sel) + [False] * len(rd_sel),
                 "n_ho_in_dev": int(len(ho)),
                 "n_ho_nonent_in_dev": int(((ov >= HO_THRESH) & (mnli["label"] != fe.ENT)).sum())}
        pf.write_text(json.dumps(probe))
    print(f"MNLI dev: {probe['n_ho_in_dev']} high-overlap items "
          f"({probe['n_ho_nonent_in_dev']} non-entailment); probe set = {len(probe['idx'])} items")
    return probe


# ----------------------------------------------------------------------------- extract

def get_layers(model):
    return model.bert.encoder.layer


def encode_batches(tok, prem, hyp, device, bs):
    order = np.argsort([len(p) + len(h) for p, h in zip(prem, hyp)])
    batches = []
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prem[j] for j in idx], [hyp[j] for j in idx], truncation=True,
                  max_length=fe.MAX_LEN, padding=True, return_tensors="pt")
        batches.append((idx, {k: v.to(device) for k, v in enc.items()}))
    return batches


def extract_one(i, args, device, probe_text, is_ho, pre_sd):
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    path, sha = fe.model_source(i, args)
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = BertForSequenceClassification.from_pretrained(path, attn_implementation="eager")
    except (TypeError, ValueError):
        model = BertForSequenceClassification.from_pretrained(path)
    model = model.to(device).eval()
    cfg = model.config
    L, NH, H = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    dh = H // NH
    layers = get_layers(model)
    rep_layers = [L // 4, L // 2, (3 * L) // 4, L]
    N = len(probe_text[0])
    batches = encode_batches(tok, probe_text[0], probe_text[1], device, args.batch_size)
    is_ho_t = torch.tensor(is_ho, device=device)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id

    # ---- clean pass: logits, representations, attention stats, per-layer context means
    ctx_sum = torch.zeros(L, H, device=device, dtype=torch.float64); ntok = 0
    hooks, cur_mask = [], {}

    def make_capture(l):
        def hook(mod, inp):
            x = inp[0]; m = cur_mask["m"].unsqueeze(-1).to(x.dtype)
            ctx_sum[l] += (x * m).sum((0, 1)).double()
        return hook

    for l in range(L):
        hooks.append(layers[l].attention.output.dense.register_forward_pre_hook(make_capture(l)))
    clean_lo = torch.empty(N, device=device)
    rep = torch.empty(len(rep_layers), N, H, device=device)
    match_sum = torch.zeros(L, NH, device=device, dtype=torch.float64); match_n = 0
    prem_sum = torch.zeros(L, NH, device=device, dtype=torch.float64); prem_n = 0
    with torch.inference_mode():
        for idx, enc in batches:
            am = enc["attention_mask"]; cur_mask["m"] = am; ntok += int(am.sum())
            o = model(**enc, output_hidden_states=True, output_attentions=True)
            it = torch.as_tensor(idx, device=device)
            clean_lo[it] = ent_logodds(o.logits.float())
            amf = am.unsqueeze(-1).float()
            for r, l in enumerate(rep_layers):
                hs = o.hidden_states[l].float()
                rep[r, it] = (hs * amf).sum(1) / amf.sum(1)
            # attention statistics on high-overlap items only
            ids, tt = enc["input_ids"], enc["token_type_ids"]
            special = (ids == cls_id) | (ids == sep_id) | (am == 0)
            if tok.unk_token_id is not None:
                special = special | (ids == tok.unk_token_id)
            hypq = (tt == 1) & ~special
            premk = (tt == 0) & ~special
            match = (ids.unsqueeze(2) == ids.unsqueeze(1)) & hypq.unsqueeze(2) & premk.unsqueeze(1)
            sel = is_ho_t[it]
            q_has = match.any(-1) & sel.unsqueeze(1)                 # (B,T) queries with a match
            q_hyp = hypq & sel.unsqueeze(1)
            att = torch.stack([a.float() for a in o.attentions], 1)  # (B,L,NH,T,T)
            mass_match = (att * match[:, None, None].float()).sum(-1)  # (B,L,NH,T)
            mass_prem = (att * premk[:, None, None, None, :].float()).sum(-1)
            match_sum += (mass_match * q_has[:, None, None].float()).sum((0, 3)).double()
            match_n += int(q_has.sum())
            prem_sum += (mass_prem * q_hyp[:, None, None].float()).sum((0, 3)).double()
            prem_n += int(q_hyp.sum())
            del o, att
    for hk in hooks:
        hk.remove()
    ctx_mean = (ctx_sum / ntok).float()
    attn_match = (match_sum / max(match_n, 1)).float().cpu().numpy()
    attn_prem = (prem_sum / max(prem_n, 1)).float().cpu().numpy()

    # ---- head-ablation profile (mean ablation of each head's context vector)
    abl_ho = np.zeros((L, NH), np.float32); abl_rd = np.zeros((L, NH), np.float32)
    ho_np = np.array(is_ho)
    base = clean_lo.cpu().numpy()
    use_amp = device == "cuda" and not args.fp32_ablation

    def run_lo():
        lo = np.empty(N, np.float32)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            for idx, enc in batches:
                lo[idx] = ent_logodds(model(**enc).logits.float()).cpu().numpy()
        return lo

    with torch.inference_mode():
        base_abl = run_lo()  # clean reference at the same precision as the ablated runs
        amp_err = float(np.abs(base_abl - base).max())
        for l in range(L):
            for h in range(NH):
                sl = slice(h * dh, (h + 1) * dh)

                def hook(mod, inp, sl=sl, l=l):
                    x = inp[0].clone(); x[..., sl] = ctx_mean[l, sl].to(x.dtype)
                    return (x,)
                hk = layers[l].attention.output.dense.register_forward_pre_hook(hook)
                lo = run_lo()
                hk.remove()
                d = lo - base_abl
                abl_ho[l, h] = d[ho_np].mean() if ho_np.any() else 0.0
                abl_rd[l, h] = d[~ho_np].mean()

    # ---- weights: distance from pretrained, CountSketch of the delta, classifier
    ft = model.bert.state_dict()
    groups = ["embeddings"] + [f"layer.{k}" for k in range(L)] + ["pooler"]
    gsq = dict.fromkeys(groups, 0.0); gsq0 = dict.fromkeys(groups, 0.0)
    sketch = torch.zeros(SKETCH_D, device=device, dtype=torch.float64)
    with torch.inference_mode():
        for key in sorted(ft):
            w = ft[key]
            if key not in pre_sd or not torch.is_floating_point(w) or w.shape != pre_sd[key].shape:
                continue
            w0 = pre_sd[key].to(device=device, dtype=torch.float32)
            d = (w.float() - w0).flatten()
            g = next((gr for gr in groups if f"{gr}." in key or key.startswith(gr)), None)
            if g is not None:
                gsq[g] += float((d.double() ** 2).sum()); gsq0[g] += float((w0.double() ** 2).sum())
            gen = torch.Generator(device=device).manual_seed(zlib.crc32(key.encode()))
            b = torch.randint(0, SKETCH_D, (d.numel(),), generator=gen, device=device)
            s = torch.randint(0, 2, (d.numel(),), generator=gen, device=device).float() * 2 - 1
            sketch.index_add_(0, b, (d * s).double())
    dnorm = np.array([np.sqrt(gsq[g]) for g in groups], np.float32)
    drel = np.array([np.sqrt(gsq[g] / gsq0[g]) if gsq0[g] > 0 else 0 for g in groups], np.float32)
    clf = torch.cat([model.classifier.weight.flatten(), model.classifier.bias]).detach().float().cpu().numpy()

    res = dict(rep=rep.cpu().numpy().astype(np.float16), rep_layers=np.array(rep_layers),
               attn_match=attn_match, attn_prem=attn_prem, abl_ho=abl_ho, abl_rand=abl_rd,
               clean_lo=base, delta_norm=dnorm, delta_rel=drel, groups=np.array(groups),
               sketch=sketch.float().cpu().numpy(), clf=clf, sha=sha, device=device,
               amp_max_err=amp_err)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def cmd_extract(args):
    import torch
    from transformers import BertModel
    device = fe.pick_device(args.device)
    out = Path(args.out_dir); fdir = out / "features"; fdir.mkdir(parents=True, exist_ok=True)
    meta = np.load(out / "meta.npz", allow_pickle=True)
    mnli = fe.load_mnli(args.mnli_tsv)
    assert np.array_equal(mnli["label"], meta["mnli_label"]), \
        "MNLI order/labels differ from step 3 - cannot align with saved logits"
    probe = build_probe(mnli, out)
    pidx = probe["idx"]
    probe_text = ([mnli["premise"][j] for j in pidx], [mnli["hypothesis"][j] for j in pidx])
    print(f"Loading pretrained reference {args.pretrained} ...")
    pre_sd = {k: v.detach().cpu() for k, v in BertModel.from_pretrained(args.pretrained).state_dict().items()}
    models = fe.parse_models(args.models)
    print(f"device={device} models={len(models)}")
    failed = {}
    for k, i in enumerate(models):
        f = fdir / f"feat_{i:03d}.npz"
        if f.exists():
            print(f"[{k+1}/{len(models)}] model {i}: exists, skipping"); continue
        t0 = time.time()
        try:
            res = extract_one(i, args, device, probe_text, probe["is_ho"], pre_sd)
        except Exception as e:
            failed[i] = f"{type(e).__name__}: {e}"
            print(f"[{k+1}/{len(models)}] model {i}: FAILED ({failed[i][:300]})")
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        tmp = f.with_suffix(".tmp.npz"); np.savez(tmp, **res); tmp.replace(f)
        print(f"[{k+1}/{len(models)}] model {i}: ok  max|abl_ho|={np.abs(res['abl_ho']).max():.3f}  "
              f"fp16 err={res['amp_max_err']:.1e}  "
              f"({time.time()-t0:.0f}s)")
        if args.purge and not args.local_tmpl:
            fe.purge_from_cache(i)
    (out / "step4_extract_env.json").write_text(json.dumps({**fe.env_info(), "failed": failed}, indent=2))
    done = len(list(fdir.glob("feat_*.npz")))
    print(f"\nDone: {done} models have features. Failed this run: {sorted(failed) or 'none'}")


# ----------------------------------------------------------------------------- analyze

def center_kernels(Ktr, Kte):
    n = Ktr.shape[0]
    mr = Ktr.mean(0); ma = Ktr.mean()
    Ktr_c = Ktr - mr[None] - mr[:, None] + ma
    Kte_c = Kte - mr[None] - Kte.mean(1, keepdims=True) + ma
    return Ktr_c, Kte_c


ALPHAS = np.logspace(-4, 3, 22)


def krr(Ktr, Kte, ytr):
    """Kernel ridge with intercept; alpha by exact leave-one-out on the training models."""
    Ktr_c, Kte_c = center_kernels(Ktr, Kte)
    yc = ytr - ytr.mean()
    w, V = np.linalg.eigh(Ktr_c)
    w = np.clip(w, 0, None)
    best = (np.inf, None, None)
    Vy = V.T @ yc
    for a in ALPHAS:
        Hdiag = (V ** 2) @ (w / (w + a)) + 1.0 / len(ytr)  # + the intercept's share of the hat matrix
        fit = V @ (w / (w + a) * Vy)
        res = (yc - fit) / np.clip(1 - Hdiag, 1e-6, None)   # exact leave-one-out residuals
        if (res ** 2).mean() < best[0]:
            best = ((res ** 2).mean(), a, ytr - res)
    a = best[1]
    coef = V @ (Vy / (w + a))
    return ytr.mean() + Kte_c @ coef, a, best[2]


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def linear_kernel_block(X):
    """Returns f(tr, rows) -> kernel using train-fold standardisation."""
    X = np.asarray(X, np.float64)

    def f(tr):
        mu = X[tr].mean(0); sd = X[tr].std(0); keep = sd > 1e-12
        Z = (X[:, keep] - mu[keep]) / sd[keep]
        K = Z @ Z.T / max(keep.sum(), 1)
        return K / np.mean(np.diag(K)[tr])
    return f


def rbf_distance_block(X):
    """Juneja-style weight geometry: Gaussian kernel on Euclidean distances between models'
    weights (the CountSketch of the fine-tuning delta preserves these distances). Juneja et al.
    report Euclidean distance behaves similarly to their convexity-gap metric. Bandwidth =
    median distance among the training models only."""
    X = np.asarray(X, np.float64)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)

    def f(tr):
        d = D2[np.ix_(tr, tr)][np.triu_indices(len(tr), 1)]
        return np.exp(-D2 / np.median(d))
    return f


def fixed_kernel_block(K):
    def f(tr):
        return K / np.mean(np.diag(K)[tr])
    return f


def linear_cka_kernel(reps):
    """reps: models x N x H -> models x models linear CKA."""
    F = []
    for X in reps:
        X = X.astype(np.float64); X = X - X.mean(0)
        G = X @ X.T
        F.append((G / np.linalg.norm(G)).ravel())
    F = np.array(F)
    return F @ F.T


def predict_set(blocks, names, tr, te, y):
    """One block: kernel ridge. Several blocks: stacking - a kernel ridge per block, then
    non-negative weights fitted to the blocks' leave-one-out predictions on the training models,
    so one strong block is not diluted by weak ones."""
    if len(names) == 1:
        K = blocks[names[0]](tr)
        return krr(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], y[tr])[0]
    from scipy.optimize import nnls
    P_loo, P_te = [], []
    for n in names:
        K = blocks[n](tr)
        pte, _, ploo = krr(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], y[tr])
        P_loo.append(ploo); P_te.append(pte)
    P_loo = np.array(P_loo).T; P_te = np.array(P_te).T
    mu = P_loo.mean(0); ym = y[tr].mean()
    wts, _ = nnls(P_loo - mu, y[tr] - ym)
    return ym + (P_te - mu) @ wts


def cv_oof(blocks, names, idx, y, n_rep, seed):
    """Repeated 10-fold CV on the model indices `idx`; returns mean OOF prediction and R2s."""
    rng = np.random.default_rng(seed)
    preds = np.zeros((n_rep, len(idx)))
    for r in range(n_rep):
        folds = np.array_split(rng.permutation(len(idx)), 10)
        for fo in folds:
            te = idx[fo]; tr = idx[np.setdiff1d(np.arange(len(idx)), fo)]
            preds[r, fo] = predict_set(blocks, names, tr, te, y)
    r2s = np.array([r2(y[idx], p) for p in preds])
    return preds.mean(0), r2s


def boot_delta(y, pa, pb, n=4000, seed=0):
    rng = np.random.default_rng(seed); m = len(y); d = []
    for _ in range(n):
        b = rng.integers(0, m, m)
        if np.var(y[b]) == 0:
            continue
        d.append(r2(y[b], pa[b]) - r2(y[b], pb[b]))
    return np.percentile(d, [2.5, 97.5])


def build_blocks(out, ids):
    meta = np.load(out / "meta.npz", allow_pickle=True)
    fdir = out / "features"
    ov = np.load(fdir / "mnli_overlap.npy")
    lab = meta["mnli_label"]
    gold, heur, sub = meta["hans_gold_ent"], meta["hans_heuristic"], meta["hans_subcase"]
    lo_mask = (heur == "lexical_overlap") & ~gold
    lo_np = lo_mask & (sub != "ln_passive")
    ho = ov >= HO_THRESH
    y, y_np, idsum, iditems, wts, attn, abl, reps, mnli_acc = [], [], [], [], [], [], [], [], []
    sketch = []
    for i in ids:
        z = np.load(out / "logits" / f"model_{i:03d}.npz")
        hl = z["hans_logits"]; ml = z["mnli_logits"]
        y.append((hl[lo_mask].argmax(1) != fe.ENT).mean())
        y_np.append((hl[lo_np].argmax(1) != fe.ENT).mean())
        pr = np.exp(ml - ml.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        pred = ml.argmax(1); corr = pred == lab
        rec = [corr[lab == c].mean() for c in range(3)]
        ent = -(pr * np.log(pr + 1e-12)).sum(1)
        hn = ho & (lab != fe.ENT); he = ho & (lab == fe.ENT)
        mnli_acc.append(corr.mean())
        idsum.append([corr.mean(), *rec, pr.max(1).mean(), ent.mean(), pr[:, fe.ENT].mean(),
                      corr[ho].mean(), corr[hn].mean(), pr[hn, fe.ENT].mean(), pr[he, fe.ENT].mean()])
        iditems.append(np.log(pr[ho, fe.ENT] + 1e-9) - np.log(1 - pr[ho, fe.ENT] + 1e-9))
        f = np.load(fdir / f"feat_{i:03d}.npz")
        wts.append(np.concatenate([f["delta_norm"], f["delta_rel"], f["sketch"], f["clf"]]))
        sketch.append(f["sketch"])
        attn.append(np.concatenate([f["attn_match"].ravel(), f["attn_prem"].ravel()]))
        abl.append(np.concatenate([f["abl_ho"].ravel(), f["abl_rand"].ravel()]))
        reps.append(f["rep"])
    reps = np.array(reps)  # models x layers x N x H
    cka = np.mean([linear_cka_kernel(reps[:, l]) for l in range(reps.shape[1])], 0)
    blocks = {
        "MNLI-acc": linear_kernel_block(np.array(mnli_acc)[:, None]),
        "ID-summary": linear_kernel_block(idsum),
        "ID-overlap-items": linear_kernel_block(iditems),
        "Weights": linear_kernel_block(wts),
        "Repr-CKA": fixed_kernel_block(cka),
        "Weight-geometry": rbf_distance_block(sketch),
        "Attn-stats": linear_kernel_block(attn),
        "Head-ablation": linear_kernel_block(abl),
    }
    extra = dict(n_ho=int(ho.sum()), n_ho_nonent=int((ho & (lab != fe.ENT)).sum()))
    return blocks, np.array(y), np.array(y_np), extra


BASE = ["ID-summary", "ID-overlap-items", "Weights", "Weight-geometry", "Repr-CKA"]
INTERNAL = ["Attn-stats", "Head-ablation"]
SETS = {
    "MNLI-acc only": ["MNLI-acc"],
    "ID-summary": ["ID-summary"],
    "ID-overlap-items": ["ID-overlap-items"],
    "Weights": ["Weights"],
    "Weight-geometry (Juneja-style)": ["Weight-geometry"],
    "Repr-CKA": ["Repr-CKA"],
    "Attn-stats": ["Attn-stats"],
    "Head-ablation": ["Head-ablation"],
    "INTERNAL (attn+ablation)": INTERNAL,
    "ALL BASELINES": BASE,
    "ALL BASELINES + INTERNAL": BASE + INTERNAL,
}


def cmd_analyze(args):
    out = Path(args.out_dir)
    split = json.loads((out / "split.json").read_text())
    have = {int(p.stem.split("_")[1]) for p in (out / "features").glob("feat_*.npz")}
    dev = [i for i in split["dev"] if i in have]; test = [i for i in split["test"] if i in have]
    miss = sorted(set(split["dev"] + split["test"]) - have)
    ids = dev + test
    blocks, y_all, ynp_all, extra = build_blocks(out, ids)
    dev_idx = np.arange(len(dev)); test_idx = np.arange(len(dev), len(ids))
    lines = [f"STEP 4 ANALYSIS  ({'FINAL: test models' if args.final else 'development models, CV'})",
             f"models: {len(dev)} dev, {len(test)} test" + (f"; MISSING features: {miss}" if miss else ""),
             f"MNLI high-overlap items (overlap >= {HO_THRESH}): {extra['n_ho']} "
             f"({extra['n_ho_nonent']} non-entailment)", ""]
    summary = {}
    for oname, y in [("HANS-LO non-entailment (primary)", y_all),
                     ("HANS-LO non-entailment without ln_passive (secondary)", ynp_all)]:
        lines.append(f"== Outcome: {oname}")
        preds = {}
        if not args.final:
            lines.append(f"{'feature set':32s} {'CV R2 mean':>10s} {'[min, max over repeats]':>24s}")
            for sname, names in SETS.items():
                p, r2s = cv_oof(blocks, names, dev_idx, y, args.n_rep, args.seed)
                preds[sname] = p
                lines.append(f"{sname:32s} {r2s.mean():10.3f}   [{r2s.min():.3f}, {r2s.max():.3f}]")
            yy = y[dev_idx]
        else:
            lines.append(f"{'feature set':32s} {'test R2':>10s}")
            for sname, names in SETS.items():
                preds[sname] = predict_set(blocks, names, dev_idx, test_idx, y)
                lines.append(f"{sname:32s} {r2(y[test_idx], preds[sname]):10.3f}")
            yy = y[test_idx]
        A, B = "ALL BASELINES + INTERNAL", "ALL BASELINES"
        d = r2(yy, preds[A]) - r2(yy, preds[B]); ci = boot_delta(yy, preds[A], preds[B])
        d2 = r2(yy, preds["INTERNAL (attn+ablation)"]) - r2(yy, preds["ID-overlap-items"])
        ci2 = boot_delta(yy, preds["INTERNAL (attn+ablation)"], preds["ID-overlap-items"])
        lines += ["",
                  f"PRIMARY CONTRAST  dR2 = R2(baselines+internal) - R2(baselines) = {d:+.3f}  "
                  f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
                  f"secondary         R2(internal) - R2(ID-overlap-items)          = {d2:+.3f}  "
                  f"95% CI [{ci2[0]:+.3f}, {ci2[1]:+.3f}]"]
        r2_int = r2(yy, preds["INTERNAL (attn+ablation)"])
        r2_acc = r2(yy, preds["MNLI-acc only"])
        if d >= MIN_DELTA_R2 and ci[0] > 0:
            verdict = "GO - internal signals add predictive information beyond all baselines."
        elif r2_int >= max(0.05, r2_acc + 0.05):
            verdict = ("PIVOT - internal signals predict the outcome but do not clear the "
                       f"pre-registered bar over baselines (need dR2 >= {MIN_DELTA_R2} with CI > 0).")
        else:
            verdict = ("STOP - internal signals predict no better than MNLI accuracy alone "
                       "(check whether a baseline does: that would be a different paper).")
        lines += [f"VERDICT ({'binding' if args.final else 'provisional, dev CV only'}): {verdict}", ""]
        summary[oname] = dict(delta_r2=float(d), ci=[float(c) for c in ci],
                              r2={k: float(r2(yy, p)) for k, p in preds.items()}, verdict=verdict)
    if args.final:
        lk = out / "step4_test_unlocked.json"
        prev = json.loads(lk.read_text()) if lk.exists() else []
        if prev:
            lines.insert(1, f"WARNING: test set already scored {len(prev)} time(s) before; "
                            "only the first look is confirmatory.")
        prev.append(time.strftime("%Y-%m-%d %H:%M:%S")); lk.write_text(json.dumps(prev))
    rep = "\n".join(lines); print(rep)
    tag = "final" if args.final else "dev"
    (out / f"step4_report_{tag}.txt").write_text(rep + "\n")
    (out / f"step4_summary_{tag}.json").write_text(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["split", "extract", "analyze"]:
        p = sp.add_parser(name)
        p.add_argument("--out_dir", default="feather_out")
        p.add_argument("--seed", type=int, default=0)
        if name == "split":
            p.add_argument("--force", action="store_true")
        if name == "extract":
            p.add_argument("--models", nargs="+", default=["all"])
            p.add_argument("--device", default="auto")
            p.add_argument("--batch_size", type=int, default=64)
            p.add_argument("--mnli_tsv", default=None)
            p.add_argument("--local_tmpl", default=None)
            p.add_argument("--pretrained", default=PRETRAINED)
            p.add_argument("--purge", action="store_true")
            p.add_argument("--fp32_ablation", action="store_true",
                           help="run the 144 ablation passes in fp32 (slower; default fp16 on GPU)")
        if name == "analyze":
            p.add_argument("--final", action="store_true",
                           help="score on the 30 held-out test models (do this once)")
            p.add_argument("--n_rep", type=int, default=20)
    args = ap.parse_args()
    {"split": cmd_split, "extract": cmd_extract, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
