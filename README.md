# Assignment 1: Reinforcement Learning

Author: Tavishi Kaushik

## Finish these two submission steps

1. **Canvas:** submit `Assignment1_RL_Report.pdf` after completing its three required GitHub fields on page 1. The fields are interactive. Alternatively, use the finalization command below to generate a clean PDF with the actual link.
2. **GitHub:** upload the contents of this folder, not this ZIP as a single file. Include the notebook, scripts, tests, figures, and raw result files. Check that the grader can access the repository. This package does not create or publish a repository and cannot verify access.

The report and experiments are prepared. The repository details were supplied during finalization and are printed in the regenerated report. Do not submit the PDF until they are completed. Pong's poor performance remains a scientific limitation, not a packaging problem.

## Fast start in Colab

Open `START_HERE.ipynb` in Colab and follow its cells. Upload `Assignment1_RL_Submission.zip` when prompted. The default workflow verifies existing evidence and generates the report; it **does not train** either game. Enter your actual repository URL in the marked cell to finish the PDF.

## Folder contents

- `Assignment1_RL_Report.pdf`: 16-page report, 12 labeled figures, all four 500-episode summaries, edge cases, quantitative validation, execution directions and three-month plan.
- `recovery/source/Assignment1_RL_original.ipynb`: original implementation and preserved outputs, including all shuttle and original Pong training code. Historical CartPole outputs are labeled separately from recovery results.
- `recovery/source/`: training runner, original-definition loader, independent mathematical/component tests and CartPole variance audit.
- `recovery/additional_runs/`: the two completed new CartPole runs, including selected/latest checkpoints, raw training/validation/evaluation CSVs, plots and gradient arrays.
- `recovery/historical_evidence/`: original shuttle figures and notebook outputs; historical game figures are not the primary recovered CartPole results.
- `pong_original/`: original Pong logs, figures, configs, final evaluations, gradient moments, and the checkpoints actually needed for the selected-policy evaluation and completed diagnostic (no-baseline 400; baseline 600 and 1000).
- `review/`: recomputed audit, 17-test rerun, uploaded Colab checks, 80-episode frozen-policy Pong diagnostic, source and provenance.
- `build_submission_pdf.py`: regenerates the report directly from these folders.
- `verify_submission.py`: checks source compilation, the 17-test suite, checkpoint/data agreement, raw summary statistics and gradient-variance calculations.
- `finalize_submission.py`: inserts an actual GitHub URL and regenerates the PDF and upload ZIP; does not publish anything.
- `requirements.txt`: general dependencies. The recorded training and review versions are preserved separately, not silently treated as identical.

Intermediate Pong optimizer snapshots from the original 227.52 MB backup are not duplicated here. The original `pong_results_backup.zip` retains them. All source code and recorded validation scores are included. If rerunning validation at every old checkpoint without retraining, restore those additional snapshots from that backup. The selected models and every model used in the new diagnostic are included here.

## Installation and quick verification

Use a fresh virtual environment or Colab runtime. For matching training trajectories, use the recorded training stack: Python 3.13.16, NumPy 2.1.3, torch 2.11.0+cpu, Gymnasium 1.4.0, ALE 0.12.1, Matplotlib 3.10.0. A changed dependency version or RNG protocol may change a new training run.

```bash
python -m pip install -r requirements.txt
python verify_submission.py
```

Expected: 17 test methods pass; 20 exact-gradient comparisons have maximum absolute error below 2e-6; all four recomputed 500-episode means/SDs match their summaries. No training occurs. The command reports missing repository metadata as an explicit submission warning.

## Finish the report with your repository URL

From this folder, using your real URL:

```bash
python finalize_submission.py --repo https://github.com/YOUR_USERNAME/YOUR_REPOSITORY
```

The example URL above is instructional, not a supplied repository. The command replaces the local report with a regenerated copy, records the actual identity and creates `GitHub_Upload.zip`. It does not retrain, create a repository, push files, or submit to Canvas. Review the resulting PDF and upload the **extracted folder contents** to GitHub. Keep the URL identical in the report and repository.

## Reproduction without changing saved experiments

The submitted folder already contains extracted `recovery/`, `pong_original/`, and `review/`; the PDF's extraction commands are only needed when starting from the original separate backups.

Mathematical and component checks:

```bash
python recovery/source/run_checks.py --output checks_rerun
```

Fresh CartPole recovery-style training (both variants, not needed for report generation):

```bash
python recovery/source/train_additional.py --game cartpole \
  --seeds 42 --episodes 1000 --batch-size 8 --lr 0.01 \
  --checkpoint-every 200 --validation-episodes 50 \
  --output fresh_cartpole/additional_runs
```

Fresh frozen-policy Pong diagnostic, using included checkpoints:

```bash
python review/source/pong_diagnostics.py --root pong_original \
  --output probe_rerun --episodes 20
```

For shuttle and original Pong, open the original notebook in a fresh Colab runtime and run its setup and relevant sections. Running the entire notebook reproduces the original sequence, including historical CartPole experiments; the new recovery measurements require the separate runner above. Original run folders have overwrite protections, so do not blindly rerun training over the submitted folders.

## Main results and interpretation

| Experiment | Selected episode | Evaluation episodes | Mean | Sample SD |
| --- | ---: | ---: | ---: | ---: |
| CartPole recovery, no baseline | 600 | 500 | 141.990 | 43.0976 |
| CartPole recovery, time baseline | 400 | 500 | 344.440 | 75.6734 |
| Original Pong, no baseline | 400 | 500 | -20.406 | 0.7735 |
| Original Pong, time baseline | 600 | 500 | -20.552 | 0.6294 |

The CartPole recovery reuses initialization seed 42 but changes environment/action seed protocols. Do not replace these measurements with the historical 191.86/376.36 means. Frozen-policy gradient variance reductions are 99.9865% (recovery CartPole) and 85.9274% (Pong); these are not reward-SD reductions and do not guarantee convergence. No multi-initialization study or controlled Pong batch-size experiment is claimed to be completed.

## Assistance and review

ChatGPT assisted with implementation, debugging, validation and reporting. Main training/evaluation occurred in the student Colab workflow. The additional 80-episode Pong probe and independent test rerun were performed in the assistant review runtime and are labeled in the report. Follow course disclosure rules. Check code, conclusions, author identity, PDF fields and grader repository access before submission.

## Recorded submission identity

GitHub username: Tavishi-Kaushik

Repository name: rl-assignment-1

URL: https://github.com/Tavishi-Kaushik/rl-assignment-1
