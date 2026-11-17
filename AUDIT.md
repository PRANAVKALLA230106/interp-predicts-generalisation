# Audit: what touched the 30 held-out test models, and when

Test models (split seed 2026): 2, 3, 10, 12, 14, 15, 16, 17, 19, 25, 29, 31, 35, 44, 46, 47, 48,
51, 55, 64, 70, 73, 74, 75, 76, 84, 85, 88, 91, 97.

## Before the split existed (step 3, all 100 models)

Outcome-level statistics were computed on all 100 models before the split was drawn:

- reliability;
- spread vs sampling noise;
- the MNLI-accuracy correlation (r = 0.49);
- bimodality;
- the subcase correlation matrix.

These used no features. They informed two design choices:

- the secondary "without ln_passive" outcome;
- the expectation that there was headroom beyond MNLI accuracy.

No feature, threshold or model choice was tuned on them.

## Fixed before any feature data existed

| Decision | Where fixed |
|---|---|
| Step-3 gates (split-half r ≥ 0.80, SD ratio ≥ 3, range ≥ 0.10) | `feather_eval.py` constants, before step 3 ran |
| 70/30 split, seed 2026 | `feather_step4.py split`, run before `extract` |
| Probe set (256 + 256), overlap threshold 0.8, feature definitions | `feather_step4.py`, before `extract` |
| Kernel ridge + exact LOO ridge choice; GO threshold ΔR² ≥ 0.10 with CI > 0 | `feather_step4.py`, before `extract` |
| Weight-geometry proxy and stacking of combined sets | Added after the notebook was sent, **before** the user had run it; no feature data existed |
| LOO intercept bug fix | Found on synthetic data, before real data |

Feature families were motivated by the literature (Bhaskar et al. 2024; Juneja et al. 2023), not by data.

## Use of the test models after the split

| When | Analysis | Status |
|---|---|---|
| 2026-11-13 (logged in `step4_test_unlocked.json`) | `analyze --final`, run **once**, after the development verdict (PIVOT) and with no change in between | **Confirmatory, binding** |
| After the final | Variance partitioning: CV over all 100 models | Exploratory |
| After the final | Head ranking for the causal check: **development models only** | Exploratory selection |
| After the final | Causal checks v1 and v2: ablations on the test models, with heads chosen on development models | Exploratory |
| After the final | Incremental-validity tests and power (`incremental.py`) | Exploratory |
| After the final | Fix 4 (Juneja CG): label-free clustering on all 100; evaluation by CV over all 100 | Exploratory |
| While testing analysis code | `juneja_analyze.py` run on synthetic CG data printed test-set R² values already known from the final | No new information |

**Consequence:** only the step-4 primary contrast is confirmatory in this population. Every later
claim about the MNLI population is exploratory, whatever data split it uses. The QQP population
(fix 1) has a fresh, untouched split.

## Post-hoc revisions to planned analyses (all made before the relevant results were seen)

- **PAWS reliability gate.** The random-item split-half was replaced by a sentence-group-blocked
  split-half: items sharing a sentence never straddle halves. PAWS-QQP and PAWS-Wiki became two
  separate settings, each with its own verdict; the earlier "primary = PAWS-QQP, else PAWS-Wiki"
  rule was dropped. Made 2026-11-14, before any stage-A result was seen.
- **Fix-4 analysis.** Split into three separate evaluations: reproduction, out-of-sample
  prediction and increment over the best baselines. Made before any CG result was seen.

## Amendments after the independent review (2026-11-14)

All of these were made **before** the results of causal v2, fix 1 and fix 4 were seen.

- **QQP primary test (`qqp_step4.py`).** The stacked ΔR² was replaced by a test that cannot be
  degenerate: a one-sided permutation test of the partial correlation r_p (outcome with
  internal-signal predictions, given baseline predictions) on the 30 QQP test models.
  - GO needs p < 0.025 (Bonferroni over the two PAWS settings) and r_p ≥ 0.30.
  - "No meaningful increment" needs the 90% CI inside ±0.30, which is stated in advance to be
    unreachable at n = 30.
  - Ridge grid extended to 1e-8–1e4.
- **PAWS grouping.** Groups also merge sentences that share a multiset of words (word-swap
  families).
- **Causal v2 analysis.**
  - Inference is at the head-set level, not the model level.
  - Effects are reported on the logit scale as well as accuracy.
  - A magnitude control uses each head's ablation size on development models.
  - A floor check is added.
- **Fix 4.** The trivial eigenvector is dropped from the spectral embedding; affinity-bandwidth
  sensitivity is added; and an optional convexity-gap reliability run on a second 512-item sample
  is added.
- **Reliability numbers.** Now produced by `reliability.py`. The step-3 gate itself was passed on
  the stratified random split (r = 0.998). The template-blocked r = 0.961 is the stricter figure
  we report.

## Claims withdrawn or corrected (MNLI population)

- **"ΔR² = 0.00 / adds nothing" is withdrawn.** The development-fitted stacker gave the internal
  blocks zero weight; the [0, 0] interval is an artefact of that.
- **Test-set R² values are not validity estimates.** The test split is atypical: r(MNLI accuracy,
  outcome) = 0.70 in the test models against 0.35 in the development models, which happens in 2.9%
  of random 30-model splits. Lead with repeated CV over all 100 models, labelled exploratory.
- **Causal v1's p = 6×10⁻⁸ is withdrawn.** The unit of inference is the head set. Pred6 ranks
  first of 6 sets (p ≈ 0.17). The "fragility" and "best generalisers lose most" claims are
  floor-bound and await v2's controls.
- **Variance partitioning is dropped.** It was built on CV R², where combinations of feature sets
  scored below single sets.
- **"A third of reliable variance unexplained" is dropped.** It mixed estimation loss with
  unexplained signal.

## Statistical method correction (2026-11-14)

- **The parametric partial F-test on cross-validated predictions is invalid.** Useless predictors
  come out *anti*-correlated with the outcome out of fold (a fold-mean artefact). Permuting the
  outcome is also invalid, because it breaks the baselines too: a noise block came out
  "significant".
- **The valid test** (`stats_extra.perm_cv_partial`) permutes the new feature block across models,
  with the outcome and baselines fixed, and re-runs that block's cross-validation. Calibrated on 20
  pure-noise blocks: false-positive rate 5% at α = 0.05, mean p = 0.46.
- **Fix-4 and incremental results recomputed this way** (`perm_incremental_report.txt`). These
  supersede the F-test p-values in `incremental_report.txt` and `fix4_report.txt`.
- **The probe experiment's pre-registered test** was switched to this method before any probe
  result existed.

## Probe experiment (2026-11-14)

- Run on all 100 models.
- Analysed once with the pre-registered test (permutation p, after the method correction).
- Primary result: NOT PASSED (partial r +0.12, p = 0.050).
- No changes were made after seeing the result.
- **Experiments frozen from here.**

## Final statistical method for incremental claims (2026-11-14, after the second independent review)

The feature-permutation test (`perm_cv_partial`) tests "unrelated to the outcome", not "adds
nothing beyond the baselines". A redundant copy of a baseline block passed it (partial r ≈ 0.20).
It is replaced by `stats_extra.nested_cv_increment`:

- **Method.** The base and joint (base + new) models are both fitted inside each fold of 20×10-fold
  CV, and their held-out errors are compared with the corrected repeated k-fold t-test (Bouckaert
  & Frank 2004).
- **Calibration** (`increment_v3_report.txt`). Pure-noise blocks give p = 0.62–0.84; the test is
  slightly conservative, because the stacker pays a small penalty for useless blocks. Redundant
  copies give p = 0.45–0.50.
- **Results.** No block shows a positive increment. Interpretability signals: −0.012 CV R², 95% CI
  [−0.035, +0.011]. Syntactic probe: −0.004 [−0.019, +0.012]. Convexity-gap geometry: +0.014
  [−0.059, +0.087], p = 0.35.
- **Withdrawn:** the earlier claim that "Juneja geometry adds beyond behaviour". It rested on the
  invalid tests.
- **Probe majority-class baselines:** ARG 0.418, GEN 0.151 (test edges). The control-task
  accuracies (0.67 and 0.76) are higher because the control labels are a deterministic function of
  word identity (Hewitt & Liang).
- **Reliability figures, reconciled.** 0.961 is the template-blocked split-half r of the outcome
  (`reliability.py`). 0.932 (`step3_report.txt`) is a different statistic: the mean agreement
  between 2-vs-3 subcase partitions, i.e. agreement across constructions.
- **Bandwidth.** The convexity-gap affinity bandwidth (1× median CG) was fixed in code before any
  CG result existed. The sensitivity analysis is reported.
