"""Final incremental-value tests (nested CV, corrected t-test). Usage: python src/incr_v3.py {calib|main} [feather_out] [probe_dir]"""
import sys, json, time, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import feather_step4 as s4, juneja_analyze as ja, stats_extra as sx
from pathlib import Path
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'feather_out')  # step-4 folder (features, logits, split.json, cg)
PROBE = Path(sys.argv[3] if len(sys.argv) > 3 else 'results/data/probe'); sp = json.loads((out/'split.json').read_text())
ids = sp['dev'] + sp['test']; order = np.array(ids)
blocks, y, _, _ = s4.build_blocks(out, ids)
C = ja.load_cg(out/'cg'); A, fj, _, _ = ja.juneja_feature(C)
blocks['Juneja-CG'] = s4.fixed_kernel_block(A[np.ix_(order, order)]); blocks['Juneja-f'] = s4.linear_kernel_block(fj[order][:, None])
P = np.array([np.load(PROBE / f'probe_{i:03d}.npz')['acc'] for i in ids])
blocks['Probe'] = s4.linear_kernel_block(np.concatenate([P[:, 0] - P[:, 1], P[:, 2] - P[:, 3]], 1))
blocks['ID-dup'] = blocks['ID-overlap-items']; blocks['CKA-dup'] = blocks['Repr-CKA']
rng = np.random.default_rng(7)
for k in range(5): blocks[f'NOISE{k}'] = s4.linear_kernel_block(rng.normal(size=(100, 26)))
B, BJ = s4.BASE, s4.BASE + ['Juneja-CG']
which = sys.argv[1]
if which == 'calib':
    tests = [(f"CALIB noise{k} over BASE+J", BJ, [f'NOISE{k}']) for k in range(5)] + [
             ("CALIB redundant: ID-overlap copy over BASE", B, ['ID-dup']),
             ("CALIB redundant: CKA copy over BASE", B, ['CKA-dup']),
             ("CALIB redundant: ID-overlap copy over ID-overlap", ['ID-overlap-items'], ['ID-dup'])]
else:
    tests = [("F1 INTERNAL over BASE", B, s4.INTERNAL),
             ("F2 INTERNAL over BASE+Juneja-CG", BJ, s4.INTERNAL),
             ("F3 Probe over BASE+Juneja-CG", BJ, ['Probe']),
             ("F4 Juneja-CG over BASE", B, ['Juneja-CG']),
             ("F5 Juneja-CG over ID-overlap-items", ['ID-overlap-items'], ['Juneja-CG']),
             ("    (descr.) Juneja-f over BASE", B, ['Juneja-f']),
             ("    (descr.) INTERNAL over ID-overlap-items", ['ID-overlap-items'], s4.INTERNAL)]
for name, base, new in tests:
    t0 = time.time(); r = sx.nested_cv_increment(s4, blocks, base, new, y)
    print(f"{name:50s} CV R2 gain {r['gain_r2']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  t {r['t']:+.2f}  p {r['p']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
