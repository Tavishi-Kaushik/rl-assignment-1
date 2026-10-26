"""Validate the packaged evidence without training or changing saved models."""
import argparse
import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(root, skip_suite=False, results_only=False):
    root = Path(root).resolve()
    for folder in [root/'recovery/source', root/'review/source']:
        for path in folder.glob('*.py'):
            ast.parse(path.read_text())
    if not results_only:
        import torch
    if not skip_suite and not results_only:
        subprocess.run([sys.executable, str(root/'recovery/source/run_checks.py'),
                        '--output', str(root/'checks_rerun')], check=True)
    print('\nRAW RESULT AND CHECKPOINT AUDIT')
    for game in ['cartpole', 'pong']:
        for variant in ['no_baseline', 'time_baseline']:
            if game == 'cartpole':
                folder=root/'recovery/additional_runs'/f'cartpole_{variant}_seed42'
                scores=folder/'final_scores.csv'; summary=folder/'final_summary.json'
                checkpoint=folder/'checkpoint_selected.pt'; sdkey='sample_sd'
                expected_episode=600 if variant=='no_baseline' else 400
            else:
                folder=root/'pong_original/pong_runs'/f'{variant}_seed42_lr0001'
                expected_episode=400 if variant=='no_baseline' else 600
                final=folder/f'final_evaluation_checkpoint{expected_episode:05d}'
                scores=final/'episode_scores.csv'; summary=final/'summary.json'
                checkpoint=folder/f'checkpoint_{expected_episode:05d}.pt'
                sdkey='sample_standard_deviation'
            data=read_rows(scores)
            if len(data)!=500: raise AssertionError(f'{scores}: not 500 episodes')
            rewards=np.array([float(r['reward']) for r in data])
            saved=json.loads(summary.read_text())
            np.testing.assert_allclose([rewards.mean(),rewards.std(ddof=1)],
                [saved['mean_reward'],saved[sdkey]],atol=1e-10,rtol=0)
            log=read_rows(folder/'training.csv')
            assert [int(r['episode']) for r in log]==list(range(1,1001))
            assert checkpoint.is_file()
            if not results_only:
                ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
                assert ck['completed_episodes']==expected_episode
                assert all(torch.isfinite(v).all() for v in ck['policy_state_dict'].values())
            print(f'{game:8s} {variant:13s}: N=500, mean={rewards.mean():.3f}, '
                  f'sample SD={rewards.std(ddof=1):.6f}, checkpoint={expected_episode}')
    cp=root/'recovery/additional_runs/cartpole_time_baseline_seed42/gradient_variance'
    with np.load(cp/'paired_episode_gradients.npz',allow_pickle=False) as z:
        values=[z[k].var(axis=0,ddof=1).sum() for k in ['without_baseline','with_baseline']]
    s=json.loads((cp/'summary.json').read_text())
    np.testing.assert_allclose(values,[s['without_baseline'],s['with_baseline']],rtol=1e-12)
    pg=root/'pong_original/pong_runs/time_baseline_seed42_lr0001/gradient_variance'
    with np.load(pg/'gradient_moments.npz',allow_pickle=False) as z:
        assert int(z['count'])==100
        values=[z[k].sum()/99 for k in ['m2_without','m2_with']]
    s=json.loads((pg/'summary.json').read_text())
    np.testing.assert_allclose(values,[s['without_baseline'],s['with_baseline']],rtol=1e-12)
    assert len(read_rows(root/'review/pong_diagnostics/episodes.csv'))==80
    print('Gradient moments/arrays and the 80-episode diagnostic checked.')
    if results_only:
        print('RESULTS-ONLY MODE: mathematical test suite and checkpoint tensor loading were not rerun.')
    identity=root/'submission_identity.json'
    if not identity.exists():
        print('SUBMISSION WARNING: actual GitHub URL is not recorded. Complete report page 1 '
              'or run finalize_submission.py before submitting.')
    print('Verification complete. No training or parameter updates performed.')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(Path(__file__).resolve().parent))
    ap.add_argument('--skip-suite',action='store_true')
    ap.add_argument('--results-only',action='store_true',
                    help='Check raw numerical evidence without PyTorch or game training.')
    args=ap.parse_args();main(args.root,args.skip_suite,args.results_only)
