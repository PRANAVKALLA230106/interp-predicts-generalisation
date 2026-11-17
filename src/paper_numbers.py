"""Numbers and figure data for the paper. Run from the repo root with feather_out/ (step-4 outputs) present."""
import sys, json, glob, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import feather_step4 as s4, juneja_analyze as ja
from pathlib import Path
out = Path('feather_out'); sp = json.loads((out/'split.json').read_text())
ids = sp['dev'] + sp['test']; order = np.array(ids)
blocks, y, _, _ = s4.build_blocks(out, ids)
C = ja.load_cg(out/'cg'); A, fj, _, _ = ja.juneja_feature(C)
blocks['Juneja-CG'] = s4.fixed_kernel_block(A[np.ix_(order, order)])
P = np.array([np.load(f'results/data/probe/probe_{i:03d}.npz')['acc'] for i in ids])
blocks['Probe'] = s4.linear_kernel_block(np.concatenate([P[:, 0]-P[:, 1], P[:, 2]-P[:, 3]], 1))
idx = np.arange(100); res = {}
for name, b in [("MNLI accuracy", ["MNLI-acc"]), ("ID behaviour summary", ["ID-summary"]),
                ("ID high-overlap items", ["ID-overlap-items"]), ("Weight deltas", ["Weights"]),
                ("Weight geometry (Euclidean)", ["Weight-geometry"]), ("Convexity-gap kernel (Juneja)", ["Juneja-CG"]),
                ("Representations (CKA)", ["Repr-CKA"]), ("Attention to repeated words", ["Attn-stats"]),
                ("Head-ablation profile", ["Head-ablation"]), ("Syntactic-role probe", ["Probe"]),
                ("All baselines", s4.BASE)]:
    _, r = s4.cv_oof(blocks, b, idx, y, 10, 0)
    res[name] = [float(r.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    print(f"{name:32s} {r.mean():.3f} [{r.min():.3f},{r.max():.3f}]", flush=True)
json.dump(res, open('paper/cv100.json', 'w'), indent=1)
# fragility
R = [json.load(open(f)) for f in sorted(glob.glob('results/data/causal2/model_*.json'))]
names = [k for k in R[0] if k.startswith('rand6_') or k.startswith('lm6_')]
lg = lambda v: np.log(np.clip(v, .005, .995)/(1-np.clip(v, .005, .995)))
c = np.array([r['clean']['lo_nonent'] for r in R])
d = np.array([(lg(c)-lg(np.array([r[k]['lo_nonent'] for r in R]))).mean() for k in names])
da = np.array([(c-np.array([r[k]['lo_nonent'] for r in R])).mean() for k in names])
dm = np.array([np.mean([r[k]['mnli']-r['clean']['mnli'] for r in R]) for k in names])
print('sets lowering (logit):', (d > 0).sum(), '/', len(d), ' (acc):', (da > 0).sum(), ' mean acc drop %.3f' % da.mean(), ' max|dMNLI| %.4f mean dMNLI %+.4f' % (np.abs(dm).max(), dm.mean()))
json.dump({'n_sets': len(d), 'lower_logit': int((d > 0).sum()), 'lower_acc': int((da > 0).sum()), 'acc_drops': da.tolist(), 'mnli': dm.tolist(), 'names': names}, open('paper/fragility.json', 'w'))
