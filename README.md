# Do interpretability signals add predictive information about generalisation?

A pre-registered test on the 100 MNLI BERTs of McCoy et al. (2020). Do signals from inside a
model, computed on in-distribution data only, predict which models fall for the lexical-overlap
heuristic on HANS **better than cheap behavioural, weight-space and representational
measurements do**?

Paper draft: [`paper/paper.tex`](paper/paper.tex). Every contact with the held-out models, every
amendment and every withdrawn claim: [`AUDIT.md`](AUDIT.md).

## Findings

All results come from one population. Only the step-4 held-out test is confirmatory; the rest is
exploratory.

- **Each signal predicts generalisation on its own.** Cross-validated R² over 100 models:

  | Signal | CV R² |
  |---|---|
  | Behaviour on high-overlap MNLI items (best single predictor) | 0.61 |
  | Representations (CKA) | 0.56 |
  | Head-ablation profile | 0.53 |
  | Weight deltas | 0.53 |
  | Convexity-gap geometry (Juneja et al.) | 0.51 |
  | Attention to repeated words | 0.49 |
  | Syntactic-role probe | 0.27 |
  | MNLI accuracy | 0.21 |

- **None improves held-out prediction over the baselines.** Gain in R², from nested 20×10-fold CV
  with a corrected t-test:
  - head ablation + attention: **−0.012** [−0.035, +0.011];
  - syntactic probe: −0.004 [−0.019, +0.012];
  - convexity gap: +0.014 [−0.059, +0.087] (inconclusive).
- **The pre-registered held-out test (30 models, scored once) gave PIVOT.** Its stacked statistic
  turned out to be degenerate, and the test split is atypical; both are documented in the audit.
- **Generalisation is fragile to head ablation.** 56 of 60 random or depth-matched six-head
  ablations lower HANS accuracy, while MNLI accuracy moves by under 0.75 points. The heads that best
  predict generalisation are not special once their ablation magnitude is controlled for.
- **A second population failed its gate.** The 100 QQP BERTs, tested on PAWS, failed the
  pre-registered reliability gate, so no replication was possible.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | All scripts. They import each other, so run them from `src/` or with `src/` on the path. |
| `notebooks/` | Self-contained Kaggle and Colab notebooks (scripts embedded) for every GPU stage; each runs on a free T4. |
| `results/reports/` | Every report quoted in the paper. |
| `results/data/` | Small result data: the 70/30 split, per-model accuracies, convexity gaps for all 4,950 pairs (plus the reliability rerun), probe accuracies, per-model causal ablations. |
| `paper/` | Paper source, figures, and the script and data that draw them. |
| `tests/` | Synthetic populations with planted answers, used to validate the analysis code before real data. |

The large intermediate files (about 350 MB of per-model logits and features) are not included;
the notebooks regenerate them.

## Pipeline

| Stage | Script | GPU time (T4) |
|---|---|---|
| Reproduce stored predictions; evaluate all models on MNLI and HANS (steps 2–3) | `feather_eval.py check / eval / analyze` | ~1–2 h |
| Outcome reliability | `reliability.py` | CPU |
| Frozen split, MNLI-only features, pre-registered test (step 4) | `feather_step4.py split / extract / analyze [--final]` | ~2–3 h |
| Causal check (set-level inference, magnitude control) | `feather_causal2.py run / analyze` | ~1–2 h |
| Juneja et al.'s convexity gap | `juneja_cg.py prepare / pairs`, `juneja_analyze.py` | ~6 h on T4 x2 |
| Second population (QQP, PAWS) | `fix1_qqp.py run / groups / gate`; `qqp_step4.py` was not run because the gate failed | ~1–2 h |
| Syntactic-role probes (Universal Dependencies) | `probe_ud.py run / analyze` | ~1–1.5 h |
| **Final incremental tests** | `incr_v3.py calib / main` (uses `stats_extra.nested_cv_increment`) | CPU |
| Paper numbers and figures | `paper_numbers.py`, `paper/figs.py` | CPU |

`feather_causal.py`, `incremental.py`, `permcheck.py` and `calib.py` are **superseded** analyses,
kept for the audit trail; `AUDIT.md` explains why each was replaced.

## Data and models

- Models: `connectivity/feather_berts_{0..99}` (MNLI) and `connectivity/bert_ft_qqp-{0..99}` (QQP)
  on the Hugging Face Hub (Juneja et al., 2023; originally McCoy et al., 2020).
- HANS: <https://github.com/tommccoy1/hans>.
- MNLI and QQP: GLUE via `datasets`.
- PAWS: `google-research-datasets/paws`.
- Universal Dependencies English EWT: <https://github.com/UniversalDependencies/UD_English-EWT>.

## Requirements

Python 3.10+, `torch>=2.6`, `transformers>=4.40`, `huggingface_hub`, `datasets`, `numpy`, `scipy`,
`pandas`, `scikit-learn` (see `requirements.txt`).

## AI assistance

Developed with substantial assistance from an AI assistant (Claude, Anthropic): code, statistical
analysis, literature search and drafting. Independent AI reviewers audited the design, and they
found several of the errors recorded in `AUDIT.md`. All design decisions and claims are the
author's responsibility.

## Licence

MIT (see `LICENSE`).
