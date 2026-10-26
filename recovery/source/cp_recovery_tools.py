"""Recovery-run provenance and frozen-policy diagnostics. Never reconstruct weights from outputs."""
from pathlib import Path
import base64
import csv
import hashlib
import json
import platform
import zipfile
import importlib.metadata


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def prepare(root):
    """Prevent an interrupted run from resuming silently under changed software."""
    root = Path(root)
    versions = {'python': platform.python_version()}
    for name in ['numpy', 'torch', 'gymnasium', 'matplotlib']:
        versions[name] = importlib.metadata.version(name)
    path = root / 'results' / 'cartpole_recovery_versions.json'
    if path.exists() and json.loads(path.read_text()) != versions:
        raise RuntimeError(
            'Software versions changed since this recovery started. Do not mix runs. '
            'Restore the recorded versions or ask for a separate recovery directory.'
        )
    write_json(path, versions)
    source = Path(__file__).with_name('Assignment1_RL_original.ipynb')
    manifest = dict(
        purpose='New CartPole runs after original checkpoint/CSV backup was unavailable',
        historical_source_file='Assignment1_RL (3).ipynb',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_weights_recovered_from_notebook=False,
        historical_outputs_preserved=True,
        new_runs_are_not_the_original_seed42_experiment=True,
        environment='CartPole-v1', gamma=0.95, optimizer='Adam', learning_rate=0.01,
        episodes_per_variant=1000, batch_size=8, initialization_seed=42,
        variants=['no_baseline', 'time_baseline'],
        baseline='lagged time-dependent EMA, alpha=0.1; updated after actor step',
        checkpoint_selection='Highest validation mean reward; discounted return tie-break; earlier exact tie',
        validation_episodes=50, final_evaluation_episodes_per_variant=500,
        training_environment_seed_start=24200000,
        validation_environment_seed_start=30042000,
        final_environment_seed_start=40042000,
        note='The initial parameter seed is reused, but environment/action seed protocols differ from historical runs. '
             'Keep new measurements and historical results separate. No Pong training is performed.'
    )
    write_json(root / 'results' / 'cartpole_recovery_manifest.json', manifest)
    historical = root / 'historical_evidence'
    historical.mkdir(exist_ok=True)
    notebook = json.loads(source.read_text())
    texts = []
    images = 0
    for index, cell in enumerate(notebook['cells']):
        for number, output in enumerate(cell.get('outputs', [])):
            if output.get('output_type') == 'stream':
                value = output.get('text', '')
                texts.append(f'CELL {index}\n' + (''.join(value) if isinstance(value, list) else value))
            data = output.get('data', {}).get('image/png')
            if data:
                (historical / f'cell_{index:02d}_output_{number:02d}.png').write_bytes(
                    base64.b64decode(''.join(data) if isinstance(data, list) else data)
                )
                images += 1
    (historical / 'original_saved_output.txt').write_text('\n\n'.join(texts), encoding='utf-8')
    (historical / 'README.txt').write_text(
        'These images and printed outputs were extracted from the uploaded notebook.\n'
        'They are historical evidence, NOT regenerated checkpoints or per-episode CSVs.\n'
        'New recovery measurements are in additional_runs/. Do not relabel them as the old run.\n',
        encoding='utf-8'
    )
    print('Historical plot images preserved:', images)
    print('Persistent results directory:', root)
    return versions


def frozen_gradient_variance(root, n=100, seed=42):
    """Paired gradient estimates on new episodes; frozen actor and frozen old baseline."""
    import gymnasium as gym
    import numpy as np
    import torch
    from project_code import load_project
    root = Path(root)
    directory = root / 'additional_runs' / f'cartpole_time_baseline_seed{seed}'
    checkpoint_path = directory / 'checkpoint_selected.pt'
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    p = load_project()
    p['gym'] = gym
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        model = p['CartPolePolicy']()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['policy_state_dict'])
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    baseline = checkpoint['time_baseline'].numpy().copy()
    gradients_no, gradients_with, rows = [], [], []
    env = gym.make('CartPole-v1')
    try:
        with torch.random.fork_rng(devices=[]):
            for i in range(n):
                torch.manual_seed(500000 + seed * 1000 + i)
                episode = p['collect_cartpole_episode'](env, model, 50000000 + seed * 1000 + i)
                loss_no = p['reinforce_loss'](episode['log_probs'], episode['rewards'], .95)
                grad_no = torch.autograd.grad(loss_no, tuple(model.parameters()), retain_graph=True)
                loss_with = p['reinforce_loss_with_baseline'](
                    episode['log_probs'], episode['rewards'], .95, baseline
                )
                grad_with = torch.autograd.grad(loss_with, tuple(model.parameters()))
                gradients_no.append(torch.cat([g.reshape(-1) for g in grad_no]).numpy().astype(np.float64))
                gradients_with.append(torch.cat([g.reshape(-1) for g in grad_with]).numpy().astype(np.float64))
                rows.append(dict(episode=i+1, environment_seed=50000000+seed*1000+i,
                                 action_seed=500000+seed*1000+i, reward=float(sum(episode['rewards']))))
                if (i+1) % 25 == 0:
                    print(f'Frozen-policy gradient measurements: {i+1}/{n}', flush=True)
    finally:
        env.close()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    a, b = np.asarray(gradients_no), np.asarray(gradients_with)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Nonfinite gradients; do not interpret the variance estimate.')
    va = float(a.var(axis=0, ddof=1).sum())
    vb = float(b.var(axis=0, ddof=1).sum())
    ratio = vb/va if va > 0 else None
    summary = dict(trajectories=n, checkpoint_episode=int(checkpoint['completed_episodes']),
                   checkpoint_sha256=checkpoint_hash, without_baseline=va, with_baseline=vb,
                   variance_ratio=ratio, reduction_percent=None if ratio is None else 100*(1-ratio),
                   estimator='Sum of componentwise sample variances, ddof=1, paired trajectories',
                   baseline_frozen=True, parameter_updates=0,
                   limitation='Conditional on one frozen policy; not between-training-seed variability.')
    output = directory / 'gradient_variance'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/'paired_episode_gradients.npz', without_baseline=a, with_baseline=b)
    with (output/'episode_diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


def results_table(root):
    import numpy as np
    root = Path(root)
    rows = []
    for variant in ['no_baseline', 'time_baseline']:
        directory = root/'additional_runs'/f'cartpole_{variant}_seed42'
        summary = json.loads((directory/'final_summary.json').read_text())
        with (directory/'final_scores.csv').open() as f:
            scores = np.array([float(r['reward']) for r in csv.DictReader(f)])
        if len(scores) != 500:
            raise ValueError('Expected 500 held-out episodes per variant.')
        rows.append(dict(method=variant, selected_episode=summary['selected_episode'],
                         evaluation_episodes=len(scores), mean_reward=float(scores.mean()),
                         sample_sd=float(scores.std(ddof=1)),
                         reached_500_percent=float(100*np.mean(scores == 500))))
    write_json(root/'results'/'cartpole_recovery_comparison.json', rows)
    print('\nNEW CARTPOLE RECOVERY RESULTS (not the original experiment)')
    print('Method              | Checkpoint | Episodes | Mean reward | Sample SD | Reached 500')
    for r in rows:
        print(f"{r['method']:19s} | {r['selected_episode']:10d} | {r['evaluation_episodes']:8d} | "
              f"{r['mean_reward']:11.2f} | {r['sample_sd']:9.2f} | {r['reached_500_percent']:10.1f}%")
    return rows


def make_backup(root):
    root = Path(root)
    archive = root/'cartpole_recovery_results.zip'
    temporary = archive.with_suffix('.zip.tmp')
    members = []
    for folder in ['source', 'historical_evidence', 'results', 'additional_runs', 'validation_results']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.tmp'}:
                members.append(path)
    required = [root/'additional_runs'/f'cartpole_{v}_seed42'/'final_summary.json'
                for v in ['no_baseline', 'time_baseline']]
    if not all(p.is_file() for p in required):
        raise FileNotFoundError('Finish both CartPole variants before making the final recovery ZIP.')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in members:
            z.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive verification failed; keep the result folders.')
    temporary.replace(archive)
    print(f'Created verified backup: {archive} ({archive.stat().st_size/1024**2:.2f} MB)')
    return archive
