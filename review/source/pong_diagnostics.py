"""Frozen-policy diagnostic for discount timing and exploration; no tuning/test reuse."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from project_code import load_project

@torch.no_grad()
def episode(env,model,p,seed,action_seed):
    torch.manual_seed(action_seed)
    frame,_=env.reset(seed=seed); previous=None
    rows=[]; rewards=[]
    while True:
        current=p['preprocess_pong'](frame)
        x=np.zeros(6400,dtype=np.float32) if previous is None else (current-previous).ravel()
        previous=current
        dist=Categorical(logits=model(torch.tensor(x,dtype=torch.float32)))
        action=int(dist.sample()); probability=dist.probs.cpu().numpy()
        frame,r,terminated,truncated,_=env.step((2,3)[action])
        rows.append(dict(t=len(rewards),action=action+2,p_right=float(probability[0]),
                         entropy=float(dist.entropy()),confidence=float(probability.max()),
                         changed_pixels=int(np.count_nonzero(x)),reward=float(r)))
        rewards.append(float(r))
        if terminated or truncated: break
    r=np.asarray(rewards); times=np.arange(len(r)); events=np.flatnonzero(r)
    discount=.99**times
    return dict(score=float(r.sum()),discounted_return=float(discount@r),length=len(r),
                first_reward_step=int(events[0]) if len(events) else None,
                first_reward=float(r[events[0]]) if len(events) else None,
                mean_entropy=float(np.mean([v['entropy'] for v in rows])),
                confident_state_fraction=float(np.mean([v['confidence']>.99 for v in rows])),
                discounted_contribution_t500plus=float((discount*r)[500:].sum()),
                score_t500plus=float(r[500:].sum()),
                terminated=terminated,truncated=truncated),rows

def main(root,output,n):
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    p=load_project(); p['gym']=gym
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    candidates={
        'untrained':None,
        'no_baseline_400':root/'pong_runs/no_baseline_seed42_lr0001/checkpoint_00400.pt',
        'baseline_600':root/'pong_runs/time_baseline_seed42_lr0001/checkpoint_00600.pt',
        'baseline_1000':root/'pong_runs/time_baseline_seed42_lr0001/checkpoint_01000.pt',
    }
    missing=[str(v) for v in candidates.values() if v and not v.exists()]
    if missing: raise FileNotFoundError('Restore pong_results_backup.zip first. Missing: '+str(missing))
    all_rows=[]; traces=[]; summary=[]
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        for name,path in candidates.items():
            torch.manual_seed(42); model=p['PongPolicy']()
            if path:
                ck=torch.load(path,map_location='cpu',weights_only=True)
                model.load_state_dict(ck['policy_state_dict'])
            model.eval(); env=p['make_pong_env'](); candidate_rows=[]
            try:
                for i in range(n):
                    stats,trace=episode(env,model,p,10_000_000+i,100_000+i)
                    stats.update(method=name,episode=i+1,environment_seed=10_000_000+i,action_seed=100_000+i)
                    candidate_rows.append(stats); all_rows.append(stats)
                    if i==0:
                        traces.extend(dict(method=name,**r) for r in trace if r['t']<10 or r['reward']!=0)
                    if (i+1)%10==0: print(name,i+1,'/',n,flush=True)
            finally: env.close()
            s={'method':name,'episodes':n}
            for key in ['score','discounted_return','length','first_reward_step','mean_entropy',
                        'confident_state_fraction','score_t500plus','discounted_contribution_t500plus']:
                vals=[r[key] for r in candidate_rows if r[key] is not None]
                s['mean_'+key]=float(np.mean(vals)) if vals else None
            summary.append(s); print(json.dumps(s,indent=2),flush=True)
    for filename,rows in [('episodes.csv',all_rows),('first_episode_traces.csv',traces),('summary.csv',summary)]:
        with (output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (output/'protocol.json').write_text(json.dumps(dict(
        gamma=.99,parameter_updates=0,episodes_per_candidate=n,
        interpretation='Exploratory timing/exploration diagnostic, not a causal proof or a new final test.',
        hypothesis='Higher discounted return may reflect delaying early losses without increasing full-game score.'),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.')
    ap.add_argument('--output',default='validation_results/pong_diagnostics')
    ap.add_argument('--episodes',type=int,default=20)
    args=ap.parse_args(); main(args.root,args.output,args.episodes)
