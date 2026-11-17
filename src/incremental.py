"""SUPERSEDED (kept for the audit trail): see AUDIT.md and src/incr_v3.py."""
"""Incremental-information tests for internal signals over the step-4 baselines (exploratory,
post hoc; the binding step-4 result is unchanged).
(a) 30 test models: out-of-sample baseline prediction b and internal prediction i (fitted on the 70
    dev models); partial R2 of i given b by OLS on the test models, partial F-test, bootstrap CI.
(b) all 100 models: repeated 10-fold CV out-of-fold predictions, same test (more power, but the
    test models have been seen, so this is not confirmatory).
(c) power of the partial F-test to detect a true incremental R2 of 0.02/0.05/0.10."""
import json, numpy as np, sys
from pathlib import Path
from scipy import stats
sys.path.insert(0, '.')
import feather_step4 as s4
out = Path('feather_out'); sp = json.loads((out/'split.json').read_text())
dev, test = sp['dev'], sp['test']; ids = dev + test
blocks, y, _, _ = s4.build_blocks(out, ids)
dv, te = np.arange(70), np.arange(70, 100)

def partial(yv, b, i, nboot=4000, seed=0):
    def fit(yv, b, i):
        X1 = np.c_[np.ones_like(b), b]; X2 = np.c_[X1, i]
        r1 = yv - X1 @ np.linalg.lstsq(X1, yv, rcond=None)[0]; r2 = yv - X2 @ np.linalg.lstsq(X2, yv, rcond=None)[0]
        sst = ((yv - yv.mean())**2).sum()
        return (r1 @ r1 - r2 @ r2) / sst, r1 @ r1, r2 @ r2
    d, s1, s2 = fit(yv, b, i); n = len(yv)
    F = (s1 - s2) / (s2 / (n - 3)); p = stats.f.sf(F, 1, n - 3)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        k = rng.integers(0, n, n); bs.append(fit(yv[k], b[k], i[k])[0])
    pr = stats.pearsonr(yv - np.poly1d(np.polyfit(b, yv, 1))(b), i - np.poly1d(np.polyfit(b, i, 1))(b))[0]
    return d, np.percentile(bs, [2.5, 97.5]), p, pr

B, I = s4.BASE, s4.INTERNAL
best = ["ID-overlap-items"]
L = ["INCREMENTAL INFORMATION OF INTERNAL SIGNALS (exploratory)", ""]
for lab, base in [("all baselines", B), ("best single baseline (ID-overlap-items)", best)]:
    b = s4.predict_set(blocks, base, dv, te, y); i = s4.predict_set(blocks, I, dv, te, y)
    d, ci, p, pr = partial(y[te], b, i)
    L.append(f"(a) test, n=30, over {lab}: partial R2 {d:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  F-test p = {p:.2f}  partial r {pr:+.2f}")
all_idx = np.arange(100)
for lab, base in [("all baselines", B), ("best single baseline (ID-overlap-items)", best)]:
    b, _ = s4.cv_oof(blocks, base, all_idx, y, 10, 0); i, _ = s4.cv_oof(blocks, I, all_idx, y, 10, 0)
    d, ci, p, pr = partial(y, b, i)
    L.append(f"(b) CV, n=100, over {lab}: partial R2 {d:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  F-test p = {p:.2f}  partial r {pr:+.2f}")
L.append("")
r2b = 0.69
for n in (30, 100):
    row = []
    for dlt in (0.02, 0.05, 0.10):
        f2 = dlt / (1 - r2b - dlt); lam = f2 * n; crit = stats.f.ppf(0.95, 1, n - 3)
        row.append(f"dR2={dlt:.2f}: {stats.ncf.sf(crit, 1, n - 3, lam):.2f}")
    L.append(f"(c) power of partial F-test (alpha 0.05, baseline R2 0.69), n={n}: " + "  ".join(row))
rep = "\n".join(L); print(rep); (out/'incremental_report.txt').write_text(rep + "\n")
