# RL results review — verified evidence, not a final submission

The supplied CartPole recovery and original Pong backups were inspected without retraining.
The review checked ZIP integrity, loaded the selected checkpoints with `weights_only=True`,
recomputed evaluation statistics from raw episode files, checked checkpoint selection against
the saved validation results, recomputed gradient-variance statistics, and visually inspected
the learning curves and all four final histograms. The original archives were not modified.

## Verified main results

Each final evaluation contains 500 episodes of a frozen stochastic policy.

| Experiment | Selected episode | Mean reward | Sample SD | Approximate 95% CI for mean |
| --- | ---: | ---: | ---: | --- |
| CartPole recovery, no baseline | 600 | 141.990 | 43.0976 | [138.2123, 145.7677] |
| CartPole recovery, time baseline | 400 | 344.440 | 75.6734 | [337.8069, 351.0731] |
| Original Pong, no baseline | 400 | -20.406 | 0.7735 | [-20.4738, -20.3382] |
| Original Pong, time baseline | 600 | -20.552 | 0.6294 | [-20.6072, -20.4968] |

CartPole recovery is a **new experiment**, not recovery of the missing original weights.
The initialization seed 42 is reused, but the training, validation and evaluation seed
protocol differs. Do not attach historical means 191.86/376.36 to the new raw files or plots.
Keep the original notebook as historical evidence and label the recovery explicitly.

Both recovery variants contain exactly 1,000 ordered training episodes and validation at
episodes 200, 400, 600, 800 and 1,000. The recorded selections agree with highest validation
mean reward, with discounted return as tie-break and earlier episode on an exact tie.
The selected policies have 898 parameters for CartPole and 819,586 for Pong.

The recovery no-baseline policy reached 500 steps in 0/500 evaluations; the baseline policy
did so in 6/500 (1.2%). Both improved beyond the old random-reference mean 20.44, but neither
reliably completes the 500-step task. That historical random reference is not a newly matched
500-episode control. The baseline model also has a larger reward SD; this does not contradict
the measured reduction in gradient-estimator variance.

Paired final-score differences, baseline minus no baseline, use matched environment and
action seed identifiers. CartPole: 202.45, approximate 95% CI [195.5131, 209.3869]. Pong:
-0.146, CI [-0.2336, -0.0584]. These intervals are conditional on the selected policies;
they do not measure uncertainty over independently trained seeds or justify a general
causal claim about every baseline-trained policy.

## Gradient variance and analytical tests

The covariance-trace statistic was recomputed from the saved full gradient arrays for
CartPole and from the saved online second moments for Pong, each over 100 trajectories.

| Frozen policy | Without baseline | With baseline | Estimated reduction |
| --- | ---: | ---: | ---: |
| Recovery CartPole baseline checkpoint 400 | 18895.833738 | 2.545579 | 99.986528% |
| Original Pong baseline checkpoint 600 | 626.465462 | 88.160239 | 85.927358% |

These estimates compare paired gradient estimators on the same policy and trajectories,
with the baseline fixed and no parameter updates. They are not reward-variance reductions,
not convergence guarantees, and do not include confidence intervals on the variance ratio.

The uploaded validation log reports 17 passing methods and 20 exact-gradient comparisons,
with maximum absolute gradient error 4.112720453086638e-08. The same suite was rerun against
the uploaded source in the review runtime and also passed. This remains 17 distinct test
methods, not 34. The cases include closed-form gradient enumeration, a finite-difference
cross-check, intentional incorrect-estimator controls, small-step objective improvement,
baseline properties, episode endings, action mapping, and independent shuttle checks.
Unit assertions alone do not establish sufficient policy tuning.

## Newly executed frozen-policy Pong diagnostic

The review executed the completion package's `pong_diagnostics.py` on 20 fresh episodes for
each of four predeclared policies (80 total). There was **no training, model selection, or
parameter update**. Models were the seed-42 untrained network, no-baseline checkpoint 400,
baseline checkpoint 600, and baseline checkpoint 1,000. Environment seeds are 10000000+i;
action seeds are 100000+i. The required final-test results above are unchanged.

| Policy | Mean full-game score | Mean discounted return | First point won | Mean policy entropy |
| --- | ---: | ---: | ---: | ---: |
| Untrained | -20.55 | -1.408737 | 1/20 | 0.692492 |
| No baseline, 400 | -20.55 | -1.232344 | 1/20 | 0.610750 |
| Time baseline, 600 | -20.45 | -0.965021 | 2/20 | 0.632002 |
| Time baseline, 1,000 | -20.45 | -0.867826 | 5/20 | 0.555252 |

This suggests some early-game learning without effective full-game play. It does **not**
prove that delaying early losses is the sole cause: the last baseline model also won the
first point more often in this small diagnostic sample. First-reward timing is not monotonic
across checkpoints. A simple near-deterministic action collapse is not supported by the
observed entropy and confidence measures: the first three policies never gave an action
probability above 0.99 in the measured states; the last did so in about 2.65% of states
(mean of per-episode fractions).

Discounting strongly attenuates late events: 0.99^500 = 0.00657048. Mean scores after step 500
were approximately -10.65 to -12.30, while their mean contribution to the starting discounted
return was only about -0.015 to -0.017. This illustrates a mismatch between improving the
specified discounted objective and substantially improving the reported full-game score.
The diagnostic is observational and small; it is not a causal ablation or a replacement for
the 500-episode final tests. The next useful controlled tests are representation and
batch-size/budget ablations, with the objective unchanged and fresh validation/test seeds.

The new probe ran in a different software runtime from the original Colab training. Exact
versions, source and checkpoint hashes are in `review_provenance.json`. Do not imply that
these 80 episodes were run by the student in Colab; reproduce them there if course rules
require student execution. The recorded model-training evidence is from the uploaded ZIPs.

## Metadata issue to correct in the report, not silently in the archive

The original Pong time-baseline `config.json` incorrectly repeats the no-baseline loss
description `-sum_t gamma^t * G_t * log_prob_t`. The actual original baseline code passes the
lagged baseline into the loss and uses `(G_t-b_t)`. Keep the archive intact; document this
metadata-only correction and use the correct formula in the report.

## Files and reproduction

- `audit.json`: recomputed main statistics and validation selections.
- `pong_diagnostics/`: all 80 episode measurements, summary, and first-ten-step/scoring traces.
- `independent_checks/`: the local rerun of the 17 tests and exact-gradient evidence.
- `colab_checks/`: the validation evidence uploaded in the CartPole backup.
- `source/`: code used for the frozen diagnostic and its reviewed notebook snapshot.
- `audit_uploaded_rl.py`: the read/check/extract program for the supplied archives.
- `review_provenance.json`: software versions and hashes linking the review to its inputs.

To reproduce the archive audit, put the two supplied ZIPs in an `upload/` directory alongside
`audit_uploaded_rl.py`, then run this from the extracted review-bundle directory:

```bash
python audit_uploaded_rl.py audit_rerun
```

Use a new output directory; the audit refuses to overwrite previously extracted directories.
The audit needs NumPy and PyTorch. In the existing project environment with Gymnasium and
ALE installed, reproduce the additional diagnostic with:

```bash
python source/pong_diagnostics.py --root audit_rerun/pong --output probe_rerun --episodes 20
```

To rerun the exact-gradient checks, use the source in the CartPole recovery archive:

```bash
python audit_rerun/cartpole/source/run_checks.py --output checks_rerun
```

## Submission still to finish

The four game result sets, main required plots, and saved model files are now available.
The final report must distinguish historical CartPole results from the recovery and include
the MDP assumptions, edge cases, validation rationale and numerical evidence, reproduction
commands, baseline explanation, failed-Pong discussion and three-month future plan.
Supply the actual author identity and GitHub username/repository/link, publish all source,
check grader access, and review the final PDF before submitting. This bundle is not the
final PDF or complete GitHub repository.

The additional multi-initialization CartPole experiments and controlled Pong batch-size
training in the completion notebook have not been provided or run in this review. Do not
claim they are complete. Pong performance and limited tuning remain grading risks;
passing tests and complete files cannot guarantee full marks.
