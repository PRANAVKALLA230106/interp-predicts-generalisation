"""SUPERSEDED (kept for the audit trail): see AUDIT.md and src/incr_v3.py."""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import feather_step4 as s4, juneja_analyze as ja, stats_extra as sx
out = __import__('pathlib').Path('feather_out'); sp = json.loads((out/'split.json').read_text())
ids = sp['dev'] + sp['test']; order = np.array(ids)
blocks, y, _, _ = s4.build_blocks(out, ids)
C = ja.load_cg(out/'cg'); A, fj, _, _ = ja.juneja_feature(C)
blocks['Juneja-CG'] = s4.fixed_kernel_block(A[np.ix_(order, order)]); blocks['Juneja-f'] = s4.linear_kernel_block(fj[order][:, None])
rng = np.random.default_rng(0)
for k in range(3): blocks[f'NOISE{k}'] = s4.linear_kernel_block(rng.normal(size=(100, 26)))
BJ = s4.BASE + ['Juneja-CG', 'Juneja-f']
tests = [(f"NOISE{k} over baselines+Juneja (calibration)", BJ, [f'NOISE{k}']) for k in range(3)] + [
         ("Juneja-CG over best baseline (ID-overlap-items)", ['ID-overlap-items'], ['Juneja-CG']),
         ("Juneja-CG over all step-4 baselines", s4.BASE, ['Juneja-CG']),
         ("ID-overlap-items over Juneja-CG", ['Juneja-CG'], ['ID-overlap-items']),
         ("INTERNAL over all step-4 baselines", s4.BASE, s4.INTERNAL),
         ("INTERNAL over best baseline", ['ID-overlap-items'], s4.INTERNAL),
         ("INTERNAL over baselines+Juneja", BJ, s4.INTERNAL)]
L = ["Incremental tests with a permutation null over the whole CV pipeline (n=100, 5x10-fold, 200 permutations; one-sided)"]
for name, base, new in tests:
    t0 = time.time(); r, p, q95, m = sx.perm_cv_partial(s4, blocks, base, new, y, n_rep=5, n_perm=200)
    L.append(f"{name:50s} partial r {r:+.3f}  p = {p:.3f}  (null mean {m:+.3f}, 95th pct {q95:+.3f})  [{time.time()-t0:.0f}s]")
    print(L[-1], flush=True)
open(out/'perm_incremental_report.txt','w').write("\n".join(L) + "\n")
