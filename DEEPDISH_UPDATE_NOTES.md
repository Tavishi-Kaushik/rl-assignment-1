# Completed DeepDish results update

This update belongs in the existing `Tavishi-Kaushik/rl-assignment-1` repository.
It is not a new training project. Merge its contents; do not delete the original
`recovery/`, `pong_original/`, `review/`, or `deepdish_checks/` folders.

## What changed

- The report now has 20 pages and 16 labeled figures, including both new learning
  curves, both new 500-game histograms, all validation scores and hardware details.
- The actual GitHub username, repository name and URL are printed on page 1.
- Both raw 4,000-episode runs, selected/latest checkpoints and all validation/final
  evaluation files are included under `deepdish_runs/paired_seed7_b8_e4000/`.
- `audit_deepdish.py` checks the new raw data, selection, source hashes and integrity.
- `verify_submission.py`, `README.md`, report builder and the optional Colab
  finish notebook now recognize the new evidence.
- The adapter and the original training code/results are unchanged.

## Scientific conclusions

No baseline: mean -20.286, sample SD 0.865854, N=500.
Time baseline: mean -18.926, sample SD 1.687073, N=500.
Both selected checkpoints are episode 4,000. The paired mean difference is +1.360
(approximate 95% interval [1.195, 1.525]). All 1,000 final games are losses.
These intervals are conditional on the trained models, not across training seeds.
Original gradient-variance results remain labeled with their original checkpoints;
no new seed-7 gradient-variance measurement is claimed.

## What remains for the student

1. Review the revised PDF and understand the methods/limitations.
2. Merge and push the update to the existing repository.
3. Confirm grader access to the private repository.
4. Submit the revised PDF on Canvas.

No more training is required. Optional non-training final tensor validation on the
installed DeepDish environment: `python audit_deepdish.py --load-checkpoints`.
This packaging runtime did not perform that step because PyTorch is unavailable.

The existing mathematical/component suite has 17 previously passed methods. Its
evidence remains in the package; raw numerical results were audited again for this
update. Passing checks does not guarantee full marks or establish Pong mastery.
