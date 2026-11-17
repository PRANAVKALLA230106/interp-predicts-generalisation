"""Incremental-validity statistics shared by the post-hoc analyses (not used by the binding step 4)."""
import numpy as np
from scipy import stats


def partial_increment(y, b, i, nboot=4000, seed=0):
    """Does prediction i carry information about y beyond prediction b?
    OLS of y on [1, b] vs [1, b, i] over the given models. Returns the partial R2 (share of total
    variance), a bootstrap 95% CI, the partial F-test p-value and the partial correlation."""
    def fit(yv, bv, iv):
        X1 = np.c_[np.ones_like(bv), bv]; X2 = np.c_[X1, iv]
        r1 = yv - X1 @ np.linalg.lstsq(X1, yv, rcond=None)[0]
        r2 = yv - X2 @ np.linalg.lstsq(X2, yv, rcond=None)[0]
        return (r1 @ r1 - r2 @ r2) / ((yv - yv.mean()) ** 2).sum(), r1 @ r1, r2 @ r2
    d, s1, s2 = fit(y, b, i); n = len(y)
    p = stats.f.sf((s1 - s2) / (s2 / (n - 3)), 1, n - 3)
    rng = np.random.default_rng(seed)
    bs = [fit(y[k], b[k], i[k])[0] for k in (rng.integers(0, n, n) for _ in range(nboot))]
    res = lambda v: v - np.poly1d(np.polyfit(b, v, 1))(b)  # noqa: E731
    pr = stats.pearsonr(res(y), res(i))[0]
    return dict(partial_r2=float(d), ci=[float(c) for c in np.percentile(bs, [2.5, 97.5])],
                p=float(p), partial_r=float(pr), n=int(n))


def power_partial_f(n, delta, base_r2, alpha=0.05):
    f2 = delta / (1 - base_r2 - delta)
    crit = stats.f.ppf(1 - alpha, 1, n - 3)
    return float(stats.ncf.sf(crit, 1, n - 3, f2 * n))


def fmt(tag, r):
    return (f"{tag}: partial R2 {r['partial_r2']:.3f}  95% CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
            f"F-test p = {r['p']:.3f}  partial r {r['partial_r']:+.2f}  (n={r['n']})")


def perm_cv_partial(s4, blocks, base_names, new_names, y, n_rep=5, n_perm=200, seed=0):
    """Test of incremental information under cross-validation.
    Statistic: partial correlation of y with the out-of-fold (OOF) prediction of `new_names`, given
    the OOF prediction of `base_names` (repeated 10-fold CV over all models).
    Null: the new feature blocks are permuted ACROSS MODELS (rows and columns of their kernels), with
    y and the baselines left intact, and the new block's CV is re-run. This keeps the baseline-outcome
    relation and the CV fold-mean artefact, and breaks only the link between the new features and the
    models. One-sided p for a positive increment. Returns (partial r, p, null 95th pct, null mean)."""
    idx = np.arange(len(y))
    b, _ = s4.cv_oof(blocks, base_names, idx, y, n_rep, seed)
    res = lambda v: v - np.poly1d(np.polyfit(b, v, 1))(b)  # noqa: E731
    ry = res(y)

    def stat(bl):
        i, _ = s4.cv_oof(bl, new_names, idx, y, n_rep, seed)
        ri = res(i)
        return float(ry @ ri / np.sqrt((ry @ ry) * (ri @ ri)))
    obs = stat(blocks)
    rng = np.random.default_rng(seed + 1); null = []
    for _ in range(n_perm):
        pi = rng.permutation(len(y))
        pb = dict(blocks)
        for n in new_names:
            f = blocks[n]
            pb[n] = (lambda f, pi: (lambda tr: f(pi[tr])[np.ix_(pi, pi)]))(f, pi)
        null.append(stat(pb))
    null = np.array(null)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p, float(np.percentile(null, 95)), float(np.mean(null))


def nested_cv_increment(s4, blocks, base_names, new_names, y, k=10, reps=20, seed=0):
    """Does adding `new_names` to `base_names` improve held-out prediction?
    Within each fold of repeated k-fold CV, the base model and the joint model (base + new, stacked
    as in step 4) are both fitted on the training models only and scored on the held-out models.
    Statistic: mean per-fold reduction in squared error (base - joint), reported as a CV R2 gain.
    Inference: corrected repeated k-fold t-test (Bouckaert & Frank 2004; Nadeau & Bengio 2003),
    variance inflated by (1/(k*reps) + n_test/n_train) for overlapping training sets; one-sided.
    A block that is redundant with the base cannot make the joint model better on average, so this
    tests incremental PREDICTIVE information, unlike permuting the new block."""
    from scipy import stats as st
    n = len(y); idx = np.arange(n); rng = np.random.default_rng(seed)
    d, gains = [], []
    sst = ((y - y.mean()) ** 2).sum()
    for r in range(reps):
        folds = np.array_split(rng.permutation(n), k)
        pb = np.zeros(n); pj = np.zeros(n)
        for fo in folds:
            tr = np.setdiff1d(idx, fo)
            pb[fo] = s4.predict_set(blocks, base_names, tr, fo, y)
            pj[fo] = s4.predict_set(blocks, base_names + new_names, tr, fo, y)
            d.append(((y[fo] - pb[fo]) ** 2).mean() - ((y[fo] - pj[fo]) ** 2).mean())
        gains.append((((y - pb) ** 2).sum() - ((y - pj) ** 2).sum()) / sst)
    d = np.array(d); J = len(d); n_te = n / k; n_tr = n - n_te
    se = np.sqrt((1 / J + n_te / n_tr) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else 0.0
    p = float(st.t.sf(t, J - 1))
    g = float(np.mean(gains)); scale = g / d.mean() if d.mean() != 0 else 0.0
    ci = (g - 1.96 * se * abs(scale), g + 1.96 * se * abs(scale)) if d.mean() != 0 else (0.0, 0.0)
    return dict(gain_r2=g, ci=[float(ci[0]), float(ci[1])], t=float(t), p=p)
