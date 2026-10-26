"""Audit supplied RL results without retraining; keep originals unchanged."""
import argparse
import csv
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import torch

def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))

def unpack(archive, destination, allowed):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            r = PurePosixPath(member.filename)
            assert not r.is_absolute() and '..' not in r.parts and '\\' not in member.filename
            assert r.parts and r.parts[0] in allowed and not stat.S_ISLNK(member.external_attr >> 16)
            if member.is_dir():
                continue
            target = destination.joinpath(*r.parts)
            assert target.resolve().is_relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(z.read(member))

def stats(path):
    data = rows(path)
    values = np.asarray([float(r['reward']) for r in data])
    n = len(values)
    sd = values.std(ddof=1)
    se = sd/np.sqrt(n)
    return dict(n=n,mean=float(values.mean()),sample_sd=float(sd),
                ci95=[float(values.mean()-1.96*se),float(values.mean()+1.96*se)],
                minimum=float(values.min()),maximum=float(values.max()),
                reaching_500=int((values==500).sum()),
                truncated=sum(r['truncated'].lower()=='true' for r in data)), data

def main(base):
    base = Path(base).resolve()
    inputs = Path(__file__).resolve().parent/'upload'
    cp = base/'cartpole'
    pong = base/'pong'
    unpack(inputs/'cartpole_recovery_results.zip',cp,
           {'source','historical_evidence','results','additional_runs','validation_results'})
    unpack(inputs/'pong_results_backup.zip',pong,{'pong_runs','pong_results'})
    report = dict(archives_crc_valid=True,training_performed=False,experiments={},paired_comparisons={})
    data_by_game = {'cartpole':{},'pong':{}}
    for game in ['cartpole','pong']:
        for method in ['no_baseline','time_baseline']:
            if game=='cartpole':
                folder = cp/'additional_runs'/f'cartpole_{method}_seed42'
                summary = json.loads((folder/'final_summary.json').read_text())
                recorded_episode = summary['selected_episode']
                evaluation = folder/'final_scores.csv'
                checkpoint = folder/'checkpoint_selected.pt'
                sdkey = 'sample_sd'
                validation = []
                for p in sorted(folder.glob('validation_*.csv')):
                    vr = rows(p)
                    validation.append((int(p.stem.split('_')[1]),float(np.mean([float(r['reward']) for r in vr])),
                                       float(np.mean([float(r['discounted_return']) for r in vr]))))
            else:
                folder = pong/'pong_runs'/f'{method}_seed42_lr0001'
                validation = [(int(r['checkpoint_episode']),float(r['mean_score']),float(r['mean_discounted_return']))
                              for r in rows(pong/'pong_results/checkpoint_validation/checkpoint_summary.csv')
                              if r['method']==method]
                recorded_episode = 400 if method=='no_baseline' else 600
                final_dir = folder/f'final_evaluation_checkpoint{recorded_episode:05d}'
                evaluation = final_dir/'episode_scores.csv'
                summary = json.loads((final_dir/'summary.json').read_text())
                checkpoint = folder/f'checkpoint_{recorded_episode:05d}.pt'
                sdkey = 'sample_standard_deviation'
            computed, data = stats(evaluation)
            assert computed['n']==500
            np.testing.assert_allclose([computed['mean'],computed['sample_sd']],
                                       [summary['mean_reward'],summary[sdkey]],atol=1e-12,rtol=0)
            selected = max(validation, key=lambda v:(v[1],v[2],-v[0]))[0]
            assert selected==recorded_episode
            ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert ck['completed_episodes']==recorded_episode
            assert all(torch.isfinite(t).all() for t in ck['policy_state_dict'].values())
            training = rows(folder/'training.csv')
            assert [int(r['episode']) for r in training]==list(range(1,1001))
            rewards = np.array([float(r['reward']) for r in training])
            ma = np.convolve(rewards,np.ones(100)/100,'valid')
            train_seeds = {int(r['environment_seed']) for r in training}
            final_seeds = {int(r['environment_seed']) for r in data}
            assert train_seeds.isdisjoint(final_seeds)
            if game=='cartpole':
                assert all(float(r['reward'])==int(r['length']) for r in data)
                error=max(abs(float(r['discounted_return'])-(1-.95**int(r['length']))/(1-.95)) for r in data)
                assert error<1e-11
            computed.update(checkpoint=selected,checkpoint_selection_matches=True,
                            checkpoint_parameters=sum(t.numel() for t in ck['policy_state_dict'].values()),
                            first100=float(rewards[:100].mean()),last100=float(rewards[-100:].mean()),
                            highest100=float(ma.max()),highest100_window_end=int(ma.argmax()+100),
                            training_episodes=1000,validation=validation)
            report['experiments'][game+'_'+method]=computed
            data_by_game[game][method]=data
        a=data_by_game[game]['no_baseline'];b=data_by_game[game]['time_baseline']
        assert [(r['environment_seed'],r['action_seed']) for r in a]==[(r['environment_seed'],r['action_seed']) for r in b]
        delta=np.array([float(y['reward'])-float(x['reward']) for x,y in zip(a,b)])
        se=delta.std(ddof=1)/np.sqrt(len(delta))
        report['paired_comparisons'][game]=dict(mean_baseline_minus_without=float(delta.mean()),
            ci95=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],
            scope='Conditional on these two selected policies, not a training-seed population effect')
    with np.load(cp/'additional_runs/cartpole_time_baseline_seed42/gradient_variance/paired_episode_gradients.npz',
                 allow_pickle=False) as z:
        va=float(z['without_baseline'].var(axis=0,ddof=1).sum())
        vb=float(z['with_baseline'].var(axis=0,ddof=1).sum())
        assert z['without_baseline'].shape==z['with_baseline'].shape==(100,898)
    with np.load(pong/'pong_runs/time_baseline_seed42_lr0001/gradient_variance/gradient_moments.npz',
                 allow_pickle=False) as z:
        assert int(z['count'])==100
        pa=float(z['m2_without'].sum()/(int(z['count'])-1))
        pb=float(z['m2_with'].sum()/(int(z['count'])-1))
    for label,without,with_,file in [
        ('cartpole',va,vb,cp/'additional_runs/cartpole_time_baseline_seed42/gradient_variance/summary.json'),
        ('pong',pa,pb,pong/'pong_runs/time_baseline_seed42_lr0001/gradient_variance/summary.json')]:
        saved=json.loads(file.read_text())
        np.testing.assert_allclose([without,with_],[saved['without_baseline'],saved['with_baseline']],rtol=1e-12)
        report[label+'_gradient_variance']=dict(without_baseline=without,with_baseline=with_,
                                               reduction_percent=100*(1-with_/without),trajectories=100)
    report['saved_validation']=json.loads((cp/'validation_results/summary.json').read_text())
    report['provenance_note']='CartPole recovery is new; original CartPole raw files remain unavailable. Pong is original.'
    report['pong_metadata_issue']='Baseline config loss string omits b_t, although the original implementation subtracts the lagged baseline.'
    (base/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output')
    main(parser.parse_args().output)
