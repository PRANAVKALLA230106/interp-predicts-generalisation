# Literature update (2026-11-14)

Search scope:

- papers citing or updating Li et al. 2025;
- follow-ups to Juneja et al. 2023;
- predicting OOD performance across seeds;
- behavioural baselines in mechanistic-interpretability evaluation;
- interpretability agents.

Read in full where marked; otherwise abstract or key sections only.

## Changes the framing

1. **Li et al., "Can Interpretation Predict Behavior on Unseen Data?"** (arXiv 2507.06445, v4
   Sept 2026; *BlackboxNLP 2026* per a co-author's page; read v4 key sections).
   - **Baseline comparison.** v4 compares interpretability predictions with a **hyperparameter
     baseline**. On random splits, hyperparameters alone reach AUROC 0.95, against 0.85 for the
     head rule and 0.88 for attention statistics. When a hyperparameter setting is held out,
     hyperparameters fall to 0.54, while the head rule reaches 0.73 and the attention statistics
     0.75. So the question "does interpretability beat a cheap baseline?" is already partly
     posed, with mixed answers, in a synthetic setting and against training *metadata*.
   - **"Additional predictive power."** Mostly qualitative (median comparisons and Mann–Whitney
     tests); no incremental-validity analysis.
   - **Causal tests.** Predictive power is decoupled from causal faithfulness. Ablating some
     predictive heads *raises* OOD accuracy. ID and OOD ablation effects correlate only weakly
     (ρ = 0.24).
   - **Not covered.** No BERT, no HANS, no realistic population. The limitations section calls
     for "more realistic settings".
   - **Implication.** Our novelty must be stated precisely. We are not the first to compare
     interpretability with a cheap baseline. What's new:
     - a realistic population;
     - baselines that are themselves measurements of the trained model (behaviour,
       representations, weights, loss geometry), not training metadata;
     - incremental-validity tests with confidence intervals, power analysis and a pre-registered
       held-out evaluation;
     - a causal check. Our predictive heads are *not* special once the size of their ablation
       effect is controlled for; generalisation is fragile to ablating almost any heads.

2. **Ding et al. 2021, "Grounding Representation Similarity with Statistical Testing"**
   (arXiv 2108.01661; read in full in step 1). It already uses the McCoy 100 models with HANS
   lexical-overlap non-entailment accuracy as a *functional benchmark* for representational
   similarity measures (CKA, PWCCA, Procrustes). Our CKA baseline is in this tradition, and we
   must cite it as the origin of this outcome on this population. It also builds 100 BERT-medium
   models (10 pre-training × 10 fine-tuning seeds) on MNLI with Antonymy and Numerical stress
   tests. That is a candidate second population for fix 1, **if** the checkpoints are public
   (unverified).

3. **Juneja et al. 2023 (ICLR).** No direct follow-up found that predicts held-out models'
   generalisation from convexity-gap clustering. Their r = 0.674 is in-sample, on MNLI
   validation-loss CG (Sec. 4.1); the training-loss variant is in an appendix.

## Supports the motivation

- **Position: "Don't just fix it in post"** (arXiv 2606.06533, 2026). Argues that interpretability
  claims should be evaluated by whether they predict behaviour under controlled distribution
  shift, citing McCoy et al. 2019.
- **"Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety"**
  (arXiv 2611.08438, Nov 2026). Lists "missing baselines" as a recurring weakness: claimed
  interpretability gains are often not compared against behavioural methods. Our study is a
  direct instance of the evaluation it calls for.

## Related but distinct

- **Huang et al. 2025** (arXiv 2505.11770): internal causal mechanisms predict *per-input*
  correctness OOD, not *per-model* generalisation.
- **Bhaskar et al. 2024 (ACL):** heuristic core heads in pruned BERT subnetworks; descriptive.
- **Brown et al. 2023** (arXiv 2310.14993): McCoy models; CKA and stitching locate the
  generalising/heuristic divergence in late layers; descriptive.
- **Andriushchenko et al. 2023; da Silva et al. 2025:** sharpness measures on McCoy models (weak or
  modest correlations). Possible extra baselines.
- **Interpretability agents** (InterpAgent 2605.01555; HYVE, EMNLP 2026 Findings; Mueller et al.
  2603.20101 on evaluation pitfalls): not about prediction across models; out of scope.

## Still to do before submission

- Check that the Ding et al. BERT-medium checkpoints are public (a possible second population).
- Re-run this search within two weeks of the deadline (the area moves fast).
- Read in full: Li et al. v4 (Appendix H/I), and Ding et al. Table 1 values for this population.
