# Lead-Scoring Model: Experimental Results

Produced by `python -m leads.train_ml`. Reproduce with:

```bash
python -m leads.features                    # real feature matrix   (964 rows)
python -m leads.rubric                      # rule-based scores
python -m leads.personas --n 1000 --seed 42 # synthetic sessions
python -m leads.train_ml --cv 5 --cv-repeat 2 --tag _rf_full
python -m leads.train_ml --no-engagement    --cv 5 --cv-repeat 2 --tag _rf_noeng
python -m leads.train_ml --no-synthetic     --cv 5 --cv-repeat 2 --tag _rf_realonly
python -m leads.train_ml --ablate-english   --cv 5             --tag _rf_ablate_eng
python -m leads.train_ml --model dummy                         --tag _dummy
```

Artifacts land in `artifacts/metrics_<tag>.json`.

## 1. Headline results

| Variant | Features | Headline macro F1 | 5-fold x2 CV macro F1 | Generalization delta |
|---|---|---|---|---|
| `_rf_full` | all | **0.8958** | **0.9078 +/- 0.0118** | **-0.0450** |
| `_rf_noeng` | no engagement | 0.6172 | 0.6241 +/- 0.0251 | **-0.1569** |
| `_rf_realonly` | all (real only) | 0.6209 | 0.6376 +/- 0.0297 | n/a |
| `_rf_ablate_eng` | no english | 0.8952 | 0.9026 +/- 0.0132 | -0.0904 |
| `_logreg_full` | all (logistic) | 0.8376 | 0.8360 +/- 0.0099 | **-0.0531** |
| `_dummy` | most-frequent | 0.2104 | n/a | +0.0000 |

Per-class CV F1 (95% CI over 10 folds):

| Variant | Cold | Warm | Hot |
|---|---|---|---|
| `_rf_full` | 0.880 (0.865-0.895) | 0.905 (0.897-0.912) | 0.938 (0.933-0.944) |
| `_rf_noeng` | 0.691 (0.667-0.716) | 0.501 (0.483-0.519) | 0.680 (0.665-0.695) |
| `_rf_realonly` | 0.853 (0.832-0.874) | 0.692 (0.670-0.714) | **0.368 (0.336-0.400)** |

## 2. Two findings that invalidate the naive H2 reading

### Finding 1 - the 0.90 headline is a confound, not skill

The combined model's top features are engagement signals that exist **only in
the synthetic data**:

```
session_word_count         0.254
avg_delay_s                0.167
message_count              0.140     <- 56% of total importance
funding_clarity            0.100
question_category_entropy  0.099
```

Engagement coverage: **real 0.00, synthetic 1.00** for all six columns. The
real rows are median-imputed to a constant, which in the synthetic corpus
corresponds to "Cold". The model therefore learns *"low engagement => Cold"*
and, because every real row has engagement 0, it predicts almost no Hot leads
on real data. Generalization confusion matrix for the augmented model:

```
            pred Cold  pred Warm  pred Hot
true Cold        67         8          0
true Warm         9        77          0
true Hot          1        11          0     <- zero Hot predicted
```

Dropping engagement (`_rf_noeng`) removes the artifact and the headline falls to
**0.617**. Conclusion: **0.90 is not a valid estimate of real-world performance.**

### Finding 2 - synthetic augmentation *hurts* real-world performance

Evaluated on a held-out set of **real** leads (172 rows, 80/20 stratified split
of the 861 labelled rows):

| Features | real-train only | real + synthetic | delta |
|---|---|---|---|
| all | 0.6209 | 0.5759 | **-0.0450** |
| no engagement | 0.6157 | 0.4589 | **-0.1569** |
| no english | 0.6419 | 0.5515 | **-0.0904** |

Every configuration is negative. Adding 1000 synthetic sessions makes the model
**worse** at ranking real leads.

## 3. Root cause: the two label functions disagree

`leads.features` labels come from the **counsellor's rating** (`Cold`/`Good`/
`Excellent`). `leads.personas` labels come from the **persona** (`Cold`/`Warm`/
`Hot`), which encodes the *rubric*. These are different constructs.

Mean profile by label - real vs synthetic:

| label | real `funding_clarity` | synth | real `destination_uk` | synth |
|---|---|---|---|---|
| 0 | **1.955** | 0.821 | **0.816** | 0.250 |
| 1 | **2.901** | 1.729 | 0.946 | 0.749 |
| 2 | 2.934 | 2.924 | 1.000 | 0.984 |

Real "Cold" leads look like synthetic "Warm"; real "Good" leads look like
synthetic "Hot". Only label 2 agrees. The counsellor x rubric cross-tab on the
real corpus confirms it - of 374 real *Cold* rows only 26 (7%) are rubric-Cold,
while the 426 real *Good* rows contain 353 rubric-Hot.

Learnability of each label set from profile features alone:

```
REAL       n= 861  profile-only CV macro F1 = 0.662 +/- 0.055
SYNTHETIC  n=1000  profile-only CV macro F1 = 0.805 +/- 0.018
```

Synthetic labels are far more separable because they were *generated from* the
features. Augmentation therefore teaches the model the generator's definition of
a lead, which conflicts with the counsellor's.

## 4. What this means for the dissertation

* **H2 (>0.80 macro F1) is met only under the synthetic-confound configuration.**
  The defensible real-data number is `_rf_realonly`: macro F1 **0.62 / 0.64**,
  with Hot F1 **0.37 +/- 0.05** - the thin minority class (n=61 real) is the
  binding constraint, exactly as predicted.
* **H3-style trends still hold**: personas separate monotonically by rubric score
  and by engagement (see `leads/personas.py`), but that is a property of the
  generator, not evidence about real leads.
* **The negative result is itself a contribution**: synthetic persona
  augmentation improved no real-world metric here. That is a genuine, reportable
  finding for resource-constrained EdTech, and it is only visible because the
  generalization experiment was run.

## 5. Recommended fixes (priority order)

1. **Invert the generator's dependency.** Sample *features first*, then assign the
   label from the empirical real conditional `P(counsellor_label | features)`
   estimated from the 861 real rows. Synthetic labels then draw from the real
   labelling function instead of the rubric.
2. **Exclude engagement features from any cross-source claim.** They are absent
   for real leads (coverage 0.00); use `--no-engagement` for all H2/H3 numbers.
3. **Fill engagement features for real leads** before they can be used - requires
   chat/session logs, which do not exist yet. Until then they are synthetic-only.
4. **Collect more real `Excellent` examples** (n=61). Hot F1 CI 0.336-0.400 is the
   single biggest limiter; roughly 200 more would materially tighten it.
5. **Re-run after (1)** and report the delta again. A positive delta would be the
   positive version of this contribution.

## 6. Environment

Validated on Python 3.14.6 with numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1,
joblib 1.6.0. Note `LogisticRegression(multi_class=...)` no longer exists in
scikit-learn 1.9; the multinomial default is used.


