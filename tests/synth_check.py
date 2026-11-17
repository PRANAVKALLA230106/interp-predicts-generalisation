"""Known-answer test for `analyze`: signal population must PASS, pure-noise population must FAIL."""
import shutil, subprocess, numpy as np
from pathlib import Path
src = np.load("test/out/meta.npz", allow_pickle=True)
gold, heur = src["hans_gold_ent"], src["hans_heuristic"]
lo = (heur == "lexical_overlap") & ~gold
mnli_label = src["mnli_label"]
def make(name, theta):
    out = Path(f"test/synth_{name}"); shutil.rmtree(out, ignore_errors=True); (out/"logits").mkdir(parents=True)
    shutil.copy("test/out/meta.npz", out/"meta.npz")
    rng = np.random.default_rng(1)
    for m, t in enumerate(theta):
        hl = rng.normal(size=(len(gold), 3)).astype(np.float32)
        correct = rng.random(lo.sum()) < t                    # independent items, true accuracy t
        pred = np.where(correct, 0, 1)                         # non-entailment correct -> argmax 0, else 1
        sub = hl[lo]; sub[np.arange(len(sub)), pred] = 10.0; hl[lo] = sub
        ml = rng.normal(size=(len(mnli_label), 3)).astype(np.float32)
        ml[np.arange(len(ml)), mnli_label] += 1.6             # ~84% MNLI accuracy, unrelated to t
        np.savez(out/"logits"/f"model_{m:03d}.npz", mnli_logits=ml, hans_logits=hl, sha="synth")
    r = subprocess.run(["python3","feather_eval.py","analyze","--out_dir",str(out),"--n_splits","300"],capture_output=True,text=True)
    keep = [l for l in r.stdout.splitlines() if any(k in l for k in ("mean 0","raw split-half","Spearman-Brown","G1","G2","VERDICT","partitions"))]
    print(f"--- {name} ---"); print("\n".join(keep)); return r.stdout
rng = np.random.default_rng(0)
sig = make("signal", rng.uniform(0.05, 0.50, 100))
expected = 0.45**2/12 / (0.45**2/12 + 0.275*0.725/2500)
print(f"(expected raw split-half r for signal case ~ {expected:.3f})")
null = make("noise", np.full(100, 0.25))
print("(expected raw split-half r for noise case ~ 0)")
assert "G1 reliability: raw split-half r >= 0.8  ->  PASS" in sig and "G2 spread" in sig and "VERDICT: outcome is real" in sig
assert "G1 reliability: raw split-half r >= 0.8  ->  FAIL" in null and "G2 spread: SD ratio >= 3.0 and range >= 0.1  ->  FAIL" in null
print("\nKNOWN-ANSWER TEST PASSED")
