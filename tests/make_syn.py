"""Fully synthetic 100-model populations with a planted answer, for testing `analyze`."""
import sys, json, numpy as np
from pathlib import Path
scen, out = sys.argv[1], Path(sys.argv[2])
rng = np.random.default_rng({"signal": 1, "null": 2, "baseline": 3}[scen])
real = np.load(sys.argv[3] if len(sys.argv) > 3 else "feather_out/meta.npz", allow_pickle=True)  # needs step-3 meta.npz
lab = real["mnli_label"]; n_m = len(lab)
(out / "logits").mkdir(parents=True, exist_ok=True); (out / "features").mkdir(exist_ok=True)
np.savez(out / "meta.npz", **{k: real[k] for k in real})
ov = rng.random(n_m); np.save(out / "features/mnli_overlap.npy", ov); ho = ov >= 0.8
gold, heur = real["hans_gold_ent"], real["hans_heuristic"]
lo = (heur == "lexical_overlap") & ~gold
y = rng.uniform(0.05, 0.55, 100)
u = rng.normal(size=288)
for i in range(100):
    ml = rng.normal(size=(n_m, 3)); ml[np.arange(n_m), lab] += 2.2
    if scen == "baseline":
        ml[ho, 1] += -4 * (y[i] - 0.3) + rng.normal(0, 0.3, ho.sum())  # ID overlap items carry y
    hl = rng.normal(size=(30000, 3))
    wrong = rng.random(30000) > np.where(lo, y[i], 0.5)
    hl[:, 1] = np.where(wrong == (lo | gold) , 5.0, -5.0) if False else hl[:, 1]
    # entailment predicted iff (gold_ent and correct) or (not gold_ent and wrong)
    p_ent = np.where(gold, rng.random(30000) < 0.9, wrong)
    hl[:, 1] = np.where(p_ent, 5.0, -5.0)
    np.savez(out / f"logits/model_{i:03d}.npz", mnli_logits=ml, hans_logits=hl, sha="syn")
    noise = rng.normal(size=288)
    abl = (3 * (y[i] - 0.3) * u + 0.5 * noise) if scen in ("signal", "baseline") else noise
    np.savez(out / f"features/feat_{i:03d}.npz",
             rep=rng.normal(size=(4, 128, 16)).astype(np.float16),
             attn_match=rng.normal(size=(12, 12)), attn_prem=rng.normal(size=(12, 12)),
             abl_ho=abl[:144].reshape(12, 12), abl_rand=abl[144:].reshape(12, 12),
             delta_norm=rng.normal(size=14), delta_rel=rng.normal(size=14),
             sketch=rng.normal(size=4096), clf=rng.normal(size=2307))
perm = np.random.default_rng(2026).permutation(100)
(out / "split.json").write_text(json.dumps({"test": sorted(map(int, perm[:30])), "dev": sorted(map(int, perm[30:]))}))
