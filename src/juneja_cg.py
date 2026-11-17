#!/usr/bin/env python3
"""
Fix 4: Juneja et al. (2023)'s exact weight-geometry metric, the convexity gap (CG), for the 100
MNLI BERTs, so step 4's weight-geometry proxy can be replaced by the real thing.

Protocol (Juneja et al., Sec. 4 and App. F.2):
  - 512 random MNLI matched-validation examples, the same for every interpolation;
  - for each pair of models, the loss of the linear interpolation at alpha = 0, 0.1, ..., 1;
  - CG = the largest barrier height over any sub-segment of that path:
        max over a < k < b of  L_k - [(b-k) L_a + (k-a) L_b] / (b-a)
  - then spectral clustering on CG distances; predictor = ||w - c1|| - ||w - c2|| in the
    spectral embedding (done later, on CPU, by juneja_analyze.py).

Stages
  prepare  download each model once; store its fine-tuning delta from pretrained BERT in fp16
           (plus the classifier in fp32) on local disk (~220 MB/model, ~22 GB total: use a disk
           with room, e.g. /tmp on Kaggle). --purge frees the HF cache.
  pairs    GPU. For every pair i<j, 9 interior interpolation points (+ the 100 endpoints once).
           Resumable: one file per i. Roughly 1-2 h on a T4.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feather_eval as fe  # noqa: E402

N_SAMPLES = 512
ALPHAS = [k / 10 for k in range(11)]


def sample_mnli(args):
    mnli = fe.load_mnli(args.mnli_tsv)
    idx = np.sort(np.random.default_rng(getattr(args, "sample_seed", 512)).choice(len(mnli["label"]), min(N_SAMPLES, len(mnli["label"])), replace=False))
    return ([mnli["premise"][j] for j in idx], [mnli["hypothesis"][j] for j in idx],
            mnli["label"][idx], idx)


def cmd_prepare(args):
    import torch
    from transformers import BertForSequenceClassification, BertModel
    ddir = Path(args.delta_dir); ddir.mkdir(parents=True, exist_ok=True)
    pre = {k: v.float() for k, v in BertModel.from_pretrained(args.pretrained).state_dict().items()}
    for k, i in enumerate(fe.parse_models(args.models)):
        f = ddir / f"delta_{i:03d}.npy"
        if f.exists():
            continue
        t0 = time.time()
        try:
            path, sha = fe.model_source(i, args)
            m = BertForSequenceClassification.from_pretrained(path)
            sd = m.bert.state_dict()
            names = [n for n, _ in m.bert.named_parameters()]   # fixed order = the flat layout
            flat = torch.cat([(sd[n].float() - pre[n]).flatten() for n in names]).half().numpy()
            np.savez(ddir / f"clf_{i:03d}.npz", weight=m.classifier.weight.detach().float().numpy(),
                     bias=m.classifier.bias.detach().float().numpy(), sha=sha)
            tmp = ddir / f"delta_{i:03d}.tmp.npy"; np.save(tmp, flat); tmp.replace(f)
            print(f"[{k+1}] model {i}: delta saved ({time.time()-t0:.0f}s)")
            if args.purge and not args.local_tmpl:
                fe.purge_from_cache(i)
        except Exception as e:  # noqa: BLE001
            print(f"[{k+1}] model {i}: FAILED {type(e).__name__}: {str(e)[:300]}")


def convexity_gap(L):
    """L: losses at evenly spaced alphas (endpoints included)."""
    L = np.asarray(L, np.float64); n = len(L); best = 0.0
    for a in range(n):
        for b in range(a + 2, n):
            k = np.arange(a + 1, b)
            chord = ((b - k) * L[a] + (k - a) * L[b]) / (b - a)
            best = max(best, float((L[k] - chord).max()))
    return best


def cmd_pairs(args):
    """Fast version: deltas are flat fp16 vectors; models are processed in BLOCKS held on the GPU,
    so each model is read from disk a few times instead of ~100 times. Work is split into
    block-pairs; --worker k --n_workers W runs every W-th block-pair (one process per GPU).
    Output: one JSON per block-pair (resumable)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, BertForSequenceClassification
    device = fe.pick_device(args.device)
    ddir = Path(args.delta_dir)
    out = Path(args.out_dir) / ("cg" if args.sample_seed == 512 else f"cg_seed{args.sample_seed}")
    out.mkdir(parents=True, exist_ok=True)
    P, H, y, idx = sample_mnli(args)
    (out / "sample_idx.json").write_text(json.dumps([int(v) for v in idx]))
    ids = sorted(int(p.stem.split("_")[1]) for p in ddir.glob("delta_*.npy") if ".tmp" not in p.name)
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    model = BertForSequenceClassification.from_pretrained(args.pretrained, num_labels=3).to(device).eval()
    params = dict(model.bert.named_parameters()); names = list(params)
    pre = {n: params[n].detach().clone() for n in names}
    spans, off = {}, 0
    for n in names:
        k = params[n].numel(); spans[n] = (off, off + k); off += k
    enc = tok(P, H, truncation=True, max_length=fe.MAX_LEN, padding=True, return_tensors="pt").to(device)
    yt = torch.as_tensor(y, device=device)
    amp = device == "cuda" and not args.fp32
    blocks = [ids[s:s + args.block] for s in range(0, len(ids), args.block)]
    work = [(a, b) for a in range(len(blocks)) for b in range(a, len(blocks))]
    mine = [w for k, w in enumerate(work) if k % args.n_workers == args.worker]
    keep = None
    if args.n_pairs:
        allp = [(i, j) for i in ids for j in ids if j > i]
        sel = np.random.default_rng(99).choice(len(allp), min(args.n_pairs, len(allp)), replace=False)
        keep = {allp[k] for k in sel}
    print(f"device={device} models={len(ids)} blocks={len(blocks)} block-pairs for this worker: {len(mine)}", flush=True)

    def load_block(bl):
        D = {i: torch.from_numpy(np.load(ddir / f"delta_{i:03d}.npy")).to(device) for i in bl}
        C = {}
        for i in bl:
            z = np.load(ddir / f"clf_{i:03d}.npz")
            C[i] = (torch.as_tensor(z["weight"], device=device), torch.as_tensor(z["bias"], device=device))
        return D, C

    @torch.inference_mode()
    def loss_at(Di, Ci, Dj, Cj, a):
        for n in names:
            s0, s1 = spans[n]
            dj = Dj[s0:s1].float()
            params[n].copy_(pre[n] + (dj + a * (Di[s0:s1].float() - dj)).view_as(pre[n]))
        model.classifier.weight.copy_(Cj[0] + a * (Ci[0] - Cj[0]))
        model.classifier.bias.copy_(Cj[1] + a * (Ci[1] - Cj[1]))
        tot = 0.0
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            for s in range(0, len(y), args.batch_size):
                sl = slice(s, s + args.batch_size)
                logits = model(**{k: v[sl] for k, v in enc.items()}).logits.float()
                tot += float(F.cross_entropy(logits, yt[sl], reduction="sum"))
        return tot / len(y)

    for (a, b) in mine:
        f = out / f"blk_{a:02d}_{b:02d}.json"
        if f.exists():
            print(f"block-pair {a},{b}: exists, skipping", flush=True); continue
        t0 = time.time()
        DA, CA = load_block(blocks[a])
        DB, CB = (DA, CA) if b == a else load_block(blocks[b])
        endp = {i: loss_at(D[i], C[i], D[i], C[i], 0.0) for D, C in ((DA, CA), (DB, CB)) for i in D}
        res = {}
        for i in blocks[a]:
            for j in blocks[b]:
                if j <= i or (keep is not None and (i, j) not in keep):
                    continue
                curve = [endp[j]] + [loss_at(DA[i], CA[i], DB[j], CB[j], al) for al in ALPHAS[1:-1]] + [endp[i]]
                res[f"{i}-{j}"] = {"curve": curve, "cg": convexity_gap(curve)}
                if len(res) % 50 == 0:
                    el = time.time() - t0
                    print(f"  block-pair {a},{b}: {len(res)} pairs, {el / len(res):.2f}s per pair", flush=True)
        f.write_text(json.dumps({"endpoints": {str(k): v for k, v in endp.items()}, "pairs": res}))
        print(f"block-pair {a},{b}: {len(res)} pairs ({time.time()-t0:.0f}s)", flush=True)
        del DA, DB
        if device == "cuda":
            torch.cuda.empty_cache()
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "pairs"]:
        p = sp.add_parser(name)
        p.add_argument("--out_dir", default="feather_out")
        p.add_argument("--delta_dir", default="/tmp/deltas")
        p.add_argument("--pretrained", default="google-bert/bert-base-uncased")
        p.add_argument("--device", default="auto")
        p.add_argument("--local_tmpl", default=None)
        p.add_argument("--mnli_tsv", default=None)
        if name == "prepare":
            p.add_argument("--models", nargs="+", default=["all"])
            p.add_argument("--purge", action="store_true")
        else:
            p.add_argument("--batch_size", type=int, default=256)
            p.add_argument("--fp32", action="store_true")
            p.add_argument("--sample_seed", type=int, default=512,
                           help="seed of the 512-item MNLI sample; 512 = the main run")
            p.add_argument("--n_pairs", type=int, default=0,
                           help="only this many random pairs (for a CG reliability run)")
            p.add_argument("--block", type=int, default=25, help="models held on the GPU per block")
            p.add_argument("--worker", type=int, default=0)
            p.add_argument("--n_workers", type=int, default=1)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "pairs": cmd_pairs}[args.cmd](args)


if __name__ == "__main__":
    main()
