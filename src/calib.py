"""SUPERSEDED (kept for the audit trail): see AUDIT.md and src/incr_v3.py."""
import sys, json, numpy as np
sys.path.insert(0, '.')
import feather_step4 as s4, juneja_analyze as ja, stats_extra as sx
out = __import__('pathlib').Path('feather_out'); sp = json.loads((out/'split.json').read_text())
ids = sp['dev'] + sp['test']; order = np.array(ids)
blocks, y, _, _ = s4.build_blocks(out, ids)
C = ja.load_cg(out/'cg'); A, fj, _, _ = ja.juneja_feature(C)
blocks['Juneja-CG'] = s4.fixed_kernel_block(A[np.ix_(order, order)]); blocks['Juneja-f'] = s4.linear_kernel_block(fj[order][:, None])
BJ = s4.BASE + ['Juneja-CG', 'Juneja-f']
rng = np.random.default_rng(123); ps = []; rs = []
for k in range(20):
    blocks['N'] = s4.linear_kernel_block(rng.normal(size=(100, 26)))
    r, p, _, _ = sx.perm_cv_partial(s4, blocks, BJ, ['N'], y, n_rep=5, n_perm=100, seed=0)
    ps.append(p); rs.append(r); print(f"noise {k}: r {r:+.3f} p {p:.3f}", flush=True)
ps = np.array(ps)
print("share p<0.05:", (ps < 0.05).mean(), " share p<0.2:", (ps < 0.2).mean(), " mean p:", ps.mean().round(3))
