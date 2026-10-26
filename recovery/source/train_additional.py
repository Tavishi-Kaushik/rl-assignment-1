"""Fresh matched-seed REINFORCE experiments, validation selection, held-out tests.

Separate directories protect the original experiments. Default: no training on import.
Resume only at completed batch boundaries; an interrupted CSV suffix is backed up.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.distributions import Categorical
from project_code import load_project

def save_checkpoint(path,model,optimizer,baseline,completed,config,best_score,best_return):
    temp=path.with_suffix('.tmp')
    torch.save(dict(policy_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        time_baseline=torch.tensor(baseline.copy()),completed_episodes=completed,config=config,
        torch_rng_state=torch.get_rng_state(),best_score=best_score,best_return=best_return),temp)
    temp.replace(path)

@torch.no_grad()
def evaluate(p,game,model,n,seed_base,action_base):
    env=p['gym'].make('CartPole-v1') if game=='cartpole' else p['make_pong_env']()
    collect=p['collect_cartpole_episode'] if game=='cartpole' else p['collect_pong_episode']
    rows=[]
    try:
        with torch.random.fork_rng(devices=[]):
            for i in range(n):
                torch.manual_seed(action_base+i)
                e=collect(env,model,seed_base+i)
                rows.append(dict(episode=i+1,environment_seed=seed_base+i,action_seed=action_base+i,
                    reward=float(np.sum(e['rewards'])),discounted_return=float(p['discounted_returns'](
                        e['rewards'],.95 if game=='cartpole' else .99)[0]),
                    length=len(e['rewards']),terminated=bool(e['terminated']),truncated=bool(e['truncated'])))
    finally: env.close()
    return rows

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def plot_curve(rows,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores=np.array([float(r['reward']) for r in rows]); x=np.arange(1,len(scores)+1)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(x,scores,alpha=.3,label='Episode reward')
    if len(scores)>=100: ax.plot(x[99:],np.convolve(scores,np.ones(100)/100,'valid'),label='100-episode mean')
    ax.set(xlabel='Training episode',ylabel='Undiscounted episode reward',title=title)
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

def run_one(args,p,seed,variant):
    game=args.game; gamma=.95 if game=='cartpole' else .99
    directory=Path(args.output)/f'{game}_{variant}_seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    config=dict(game=game,environment='CartPole-v1' if game=='cartpole' else 'ALE/Pong-v5',
        gamma=gamma,learning_rate=args.lr,episodes=args.episodes,batch_size=args.batch_size,
        seed=seed,baseline=variant,baseline_alpha=.1,checkpoint_every=args.checkpoint_every,
        validation_episodes=args.validation_episodes,final_episodes=500,
        loss='-sum_t gamma^t * (G_t - b_t) * log_prob_t' if variant=='time_baseline'
             else '-sum_t gamma^t * G_t * log_prob_t',
        return_normalization=False,point_boundary_return_reset=False,
        source_sha256=p['source_sha256'])
    cfgpath=directory/'config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=config:
        raise ValueError('Configuration changed: use a different --output directory.')
    cfgpath.write_text(json.dumps(config,indent=2))
    latest=directory/'checkpoint_latest.pt'; best=directory/'checkpoint_selected.pt'
    torch.manual_seed(seed); model=p['CartPolePolicy' if game=='cartpole' else 'PongPolicy']()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    baseline=np.zeros(500 if game=='cartpole' else 1); start=0
    best_score=-math.inf; best_return=-math.inf
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=True)
        model.load_state_dict(ck['policy_state_dict']);opt.load_state_dict(ck['optimizer_state_dict'])
        baseline=ck['time_baseline'].numpy();start=ck['completed_episodes']
        best_score=ck['best_score'];best_return=ck['best_return']
        torch.set_rng_state(ck['torch_rng_state'])
    else:
        save_checkpoint(latest,model,opt,baseline,0,config,best_score,best_return)
    logpath=directory/'training.csv'; rows=[]
    if logpath.exists():
        with logpath.open() as f: rows=list(csv.DictReader(f))
        if len(rows)<start: raise RuntimeError('Checkpoint has more episodes than the log.')
        if len(rows)>start:
            shutil.copy2(logpath,directory/f'training_interrupted_{time.time_ns()}.csv')
            rows=rows[:start]
            if rows: write_csv(logpath,rows)
            else: logpath.write_text('')
    if start and not rows: raise RuntimeError('Training log is missing. Restore the whole run folder.')
    env=p['gym'].make('CartPole-v1') if game=='cartpole' else p['make_pong_env']()
    collect=p['collect_cartpole_episode'] if game=='cartpole' else p['collect_pong_episode']
    starttime=time.time()
    try:
        for first in range(start,args.episodes,args.batch_size):
            end=min(first+args.batch_size,args.episodes)
            batch=[collect(env,model,20_000_000+seed*100_000+i) for i in range(first,end)]
            if variant=='time_baseline':
                baseline=np.pad(baseline,(0,max(0,max(len(e['rewards']) for e in batch)-len(baseline))))
            opt.zero_grad(set_to_none=True)
            for e in batch:
                b=baseline if variant=='time_baseline' else None
                if game=='pong':
                    p['accumulate_pong_gradient'](model,e,gamma,b,episode_divisor=len(batch),chunk_size=256)
                else:
                    loss=p['reinforce_loss'](e['log_probs'],e['rewards'],gamma) if b is None else p[
                        'reinforce_loss_with_baseline'](e['log_probs'],e['rewards'],gamma,b)
                    (loss/len(batch)).backward()
            norm=float(torch.sqrt(sum(v.grad.square().sum() for v in model.parameters())))
            if not np.isfinite(norm): raise RuntimeError('Nonfinite gradient; inspect this experiment.')
            opt.step()
            if variant=='time_baseline':
                baseline=p['update_time_baseline'](baseline,[e['rewards'] for e in batch],gamma,.1)
            newrows=[dict(episode=first+j+1,environment_seed=20_000_000+seed*100_000+first+j,
                reward=float(np.sum(e['rewards'])),discounted_return=float(p['discounted_returns'](e['rewards'],gamma)[0]),
                length=len(e['rewards']),terminated=bool(e['terminated']),truncated=bool(e['truncated']),
                batch_gradient_norm=norm) for j,e in enumerate(batch)]
            with logpath.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=newrows[0].keys())
                if f.tell()==0:w.writeheader()
                w.writerows(newrows); f.flush()
            rows.extend(newrows)
            # Crossing a checkpoint boundary also works for nondivisible batch sizes.
            if end//args.checkpoint_every>first//args.checkpoint_every or end==args.episodes:
                val=evaluate(p,game,model,args.validation_episodes,30_000_000+seed*1000,300_000+seed*1000)
                write_csv(directory/f'validation_{end:05d}.csv',val)
                score=float(np.mean([e['reward'] for e in val])); ret=float(np.mean([e['discounted_return'] for e in val]))
                if (score,ret)>(best_score,best_return):
                    best_score,best_return=score,ret
                    save_checkpoint(best,model,opt,baseline,end,config,best_score,best_return)
                print(game,variant,'seed',seed,'episode',end,'train100',
                      round(np.mean([float(r['reward']) for r in rows[-100:]]),2),
                      'validation',round(score,2),'minutes',round((time.time()-starttime)/60,1),flush=True)
            save_checkpoint(latest,model,opt,baseline,end,config,best_score,best_return)
            del batch
    finally:env.close()
    plot_curve(rows,directory/'learning_curve.png',f'{game}: {variant}, seed {seed}')
    ck=torch.load(best,map_location='cpu',weights_only=True); model.load_state_dict(ck['policy_state_dict']);model.eval()
    scorepath=directory/'final_scores.csv'
    if scorepath.exists():
        with scorepath.open() as f: final=list(csv.DictReader(f))
        if len(final)!=500:raise RuntimeError('Incomplete final evaluation file.')
    else:
        # New holdout, distinct from every original experiment and validation set.
        final=evaluate(p,game,model,500,40_000_000+seed*1000,400_000+seed*1000)
        write_csv(scorepath,final)
    scores=np.array([float(r['reward']) for r in final])
    summary=dict(**config,selected_episode=ck['completed_episodes'],mean_reward=float(scores.mean()),
                 sample_sd=float(scores.std(ddof=1)),se=float(scores.std(ddof=1)/np.sqrt(500)))
    (directory/'final_summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4)); bins=np.arange(-21.5,22.5) if game=='pong' else np.arange(-.5,510,10)
    ax.hist(scores,bins=bins);ax.set(title=f'{game}: {variant}, seed {seed} (500 episodes)',
                                  xlabel='Undiscounted reward',ylabel='Episodes')
    fig.tight_layout();fig.savefig(directory/'reward_histogram.png',dpi=180);plt.close(fig)
    print('Final:',summary,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--game',choices=['cartpole','pong'],required=True)
    a.add_argument('--seeds',type=int,nargs='+',default=[7,123]);a.add_argument('--episodes',type=int,default=2000)
    a.add_argument('--batch-size',type=int,default=16);a.add_argument('--lr',type=float,default=.001)
    a.add_argument('--checkpoint-every',type=int,default=200);a.add_argument('--validation-episodes',type=int,default=30)
    a.add_argument('--output',default='additional_runs');a.add_argument('--variants',nargs='+',
        choices=['no_baseline','time_baseline'],default=['no_baseline','time_baseline'])
    args=a.parse_args()
    if min(args.episodes,args.batch_size,args.checkpoint_every,args.validation_episodes)<=0 or args.lr<=0:
        a.error('Budgets and learning rate must be positive.')
    import gymnasium as gym
    if args.game=='pong':
        import ale_py
        gym.register_envs(ale_py)
    p=load_project();p['gym']=gym;torch.set_num_threads(1)
    for seed in args.seeds:
        for variant in args.variants:run_one(args,p,seed,variant)
