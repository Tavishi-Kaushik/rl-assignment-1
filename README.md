# Assignment 1: Reinforcement Learning

Author: Tavishi Kaushik  
GitHub username: Tavishi-Kaushik  
Repository: [rl-assignment-1](https://github.com/Tavishi-Kaushik/rl-assignment-1)

## Ready for final review

All reported training and final evaluations are complete. The updated report includes
the original shuttle study, recovered CartPole, initial seed-42 Pong, and completed
seed-7 DeepDish Pong. No more training is required to regenerate the report.

1. Review and submit `Assignment1_RL_Report.pdf` on Canvas.
2. Push the extracted code, data and figures to GitHub, not just a ZIP.
3. Confirm grader access to the private repository. A valid URL alone is not access.

The report has 20 pages, 16 labeled figures and the actual GitHub identity on page 1.
No script publishes commits, invites graders or submits to Canvas.
Passing tests does not guarantee a grade or establish successful Pong learning.

## Results: 500 held-out stochastic-policy episodes per row

| Experiment | Training episodes | Selected episode | Mean | Sample SD |
| --- | ---: | ---: | ---: | ---: |
| CartPole recovery, no baseline | 1,000 | 600 | 141.990 | 43.0976 |
| CartPole recovery, time baseline | 1,000 | 400 | 344.440 | 75.6734 |
| Initial Pong seed 42, no baseline | 1,000 | 400 | -20.406 | 0.7735 |
| Initial Pong seed 42, time baseline | 1,000 | 600 | -20.552 | 0.6294 |
| DeepDish Pong seed 7, no baseline | 4,000 | 4,000 | -20.286 | 0.8659 |
| DeepDish Pong seed 7, time baseline | 4,000 | 4,000 | -18.926 | 1.6871 |

Within the new pair, the baseline mean advantage is +1.360 (approximate 95% paired
CI [1.195, 1.525]), conditional on these trained models. Both lose all 500 final games.
Across initial/new protocols, seed, batch, budget and hardware all changed:
do not attribute improvements to one cross-protocol change.

Gradient-variance reductions of 99.9865% (CartPole) and 85.9274% (initial Pong)
apply ONLY to those original selected checkpoints. No new gradient-variance audit
was performed on the DeepDish models. Reward SD and gradient variance are distinct.

## Folder guide

- `recovery/source/Assignment1_RL_original.ipynb`: original shuttle and game code/outputs.
- `recovery/source/`: loader, training runner, 17-test mathematical/component suite,
  analytical-gradient oracle and CartPole variance experiment.
- `recovery/additional_runs/`: both completed CartPole recovery runs and checkpoints.
- `recovery/historical_evidence/`: shuttle plots and preserved original notebook outputs.
- `pong_original/`: initial Pong logs, figures, final scores, variance moments and
  needed checkpoints (no-baseline 400; baseline 600 and 1,000).
- `deepdish_pong.py`: unchanged adapter actually used for the new single-GPU study.
- `deepdish_runs/paired_seed7_b8_e4000/`: both new runs; each has 4,000 training rows,
  ten 30-episode validation sets, 500 final scores, selected/latest models and source/hardware protocol.
- `review/`: original audit, executed Colab/review tests, 80-episode failure probe and provenance.
- `review/deepdish_audit/`: new numerical/source-integrity audit and regenerated figures.
- `audit_deepdish.py`: raw-data audit and optional safe PyTorch checkpoint checks.
- `verify_submission.py`: original and DeepDish evidence verification.
- `build_submission_pdf.py`, `finalize_submission.py`: report generation; no training/publishing.
- `START_HERE.ipynb`: optional Colab validation/report workflow, not GPU training.

All reported source code is included. Extra original intermediate Pong snapshots remain in
`pong_results_backup.zip`; restore it to reevaluate every historical checkpoint without
retraining. Partial temporary checkpoint copies are excluded from new complete bundles.
The included original experiment files and adapter are unchanged.

## Verification: no training

Use the existing DeepDish environment or a fresh environment with `requirements.txt`.

```bash
python verify_submission.py
```

Expected: 17 test methods pass; 20 exact-gradient comparisons have error below 2e-6;
all SIX final means/SDs match; selected checkpoints load with weights_only=True.
For raw-evidence checking without PyTorch:

```bash
python verify_submission.py --results-only
python audit_deepdish.py --make-plots
```

The new audit checks ordered records, seed separation, every validation set, selection,
500 raw rewards per method, paired intervals, source hashes and checkpoint ZIP CRCs.
Default mode does not deserialize tensors. For model shapes, finite weights/baseline,
configuration, selected/latest agreement and 500 saved Adam steps:

```bash
python audit_deepdish.py --load-checkpoints
```

The report-packaging runtime ran the NumPy-only audit and source syntax checks.
It has no PyTorch, so tensor loading and the mathematical suite were not rerun there.
Previously executed 17-test logs are preserved separately and identified as prior evidence.

## What validates the algorithms, beyond file-format checks

`python recovery/source/run_checks.py --output checks_rerun` tests:

- Shuttle: independent two-period contingent-plan enumeration, analytic parameter limits,
  accumulated probability mass at the queue cap, Bellman bounds.
- Gradients: independent exact expected-gradient oracle over four trajectories, both
  losses, five discounts, baseline/no baseline; central finite differences.
- Negative controls: wrong update direction, missing outer discount and action-dependent
  baseline must disagree with the analytical objective.
- Task behavior: termination/truncation, actions 2/3 only, fresh frame history, immutable
  preprocessing, variable-length baseline updates, chunk/full and batch-gradient agreement.

The notebook also records the 5,000-trajectory shuttle Monte Carlo check, cap sensitivity,
random-policy controls and first-ten-step trace. The report provides predictions, measured
results, limitations and commands; a pass flag alone is not the scientific justification.

## Reproduce the DeepDish study (optional; not needed for submission)

Recorded stack: Python 3.11.17; PyTorch 2.7.1+cu126; NumPy 2.1.3;
Gymnasium 1.4.0; ALE 0.12.1; Matplotlib 3.10.0; pandas 2.2.3.
One GTX 1080 was visible; CPU rollouts/action sampling and CUDA gradients.
GPU visibility is not reservation: follow allocation rules before using a shared GPU.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python deepdish_pong.py --check
python deepdish_pong.py --train --seeds 7 \
  --variants no_baseline time_baseline \
  --episodes 4000 --batch-size 8 --lr 0.001 \
  --checkpoint-every 400 --validation-episodes 30 \
  --output fresh_deepdish/paired_seed7_b8_e4000
```

The check compares CPU/CUDA gradients, paired real actions/rewards/frames and unchanged
weights. User-pasted executed output reports relative errors 1.114e-7, 1.100e-7 and
1.136e-7, with an 824-step matched episode. This verifies the adapter, not game mastery.
Training requires a successful check with matching source/hardware/software protocol.
Both variants train sequentially and each automatically receives 500 final episodes.

Resume with exactly the same arguments/output directory. Changed configuration or
protocol requires a fresh directory. Never train over the submitted evidence.

## Reproduce the other experiments

Run original notebook setup/shuttle/Pong sections in a fresh Colab folder. Full
top-to-bottom execution also reruns historical CartPole, distinct from the recovery here.

```bash
python recovery/source/train_additional.py --game cartpole \
  --seeds 42 --episodes 1000 --batch-size 8 --lr 0.01 \
  --checkpoint-every 200 --validation-episodes 50 \
  --output fresh_cartpole/additional_runs

python review/source/pong_diagnostics.py --root pong_original \
  --output probe_rerun --episodes 20
```

The probe runs 80 frozen-policy episodes, no training/model selection.
Original Colab stack: Python 3.13.16, NumPy 2.1.3, PyTorch 2.11.0+cpu,
Gymnasium 1.4.0, ALE 0.12.1, Matplotlib 3.10.0.
Different versions or seed schedules are not exact replications.

## Regenerate the report

```bash
python finalize_submission.py \
  --repo https://github.com/Tavishi-Kaushik/rl-assignment-1
```

This writes the root PDF and GitHub_Upload.zip, not a GitHub commit or Canvas submission.
Review the regenerated PDF and preserve the currently reviewed copy before changing it.

## Assistance and limitations

ChatGPT assisted with implementation, debugging, validation, interpretation and reporting.
The student executed the Colab and DeepDish training/evaluations. The extra 80-episode
Pong probe and an independent test rerun used the assistant review runtime.
Follow the course's AI policy and understand the submitted work.

CartPole does not reliably reach 500; all Pong final games are losses. No multi-seed
replication under a fixed protocol, controlled batch-size ablation, or new DeepDish
gradient-variance audit is claimed. The report gives a three-month plan for these gaps.

## Recorded submission identity

GitHub username: Tavishi-Kaushik

Repository name: rl-assignment-1

URL: https://github.com/Tavishi-Kaushik/rl-assignment-1
