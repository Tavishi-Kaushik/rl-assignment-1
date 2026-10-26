"""Load the submitted definitions without executing training or notebook outputs."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

SOURCE = Path(__file__).with_name('Assignment1_RL_original.ipynb')
NAMES = {
    'passengers_remaining', 'immediate_cost', 'next_state', 'solve_shuttle_vi',
    'CartPolePolicy', 'discounted_returns', 'collect_cartpole_episode',
    'reinforce_loss', 'reinforce_loss_with_baseline', 'update_time_baseline',
    'PongPolicy', 'preprocess_pong', 'collect_pong_episode',
    'accumulate_pong_gradient', 'make_pong_env',
}

def load_project(path=SOURCE):
    namespace = dict(np=np, torch=torch, nn=nn, Categorical=Categorical,
                     K=15, C_F=100, C_H=2, GAMMA=.95, MAX_QUEUE=200,
                     STATES=np.arange(201), ACTIONS=np.array([0, 1]),
                     ARRIVALS=np.arange(1, 6), ARRIVAL_PROBS=np.full(5, .2),
                     BASELINE_ALPHA=.1, CP_GAMMA=.95, PONG_GAMMA=.99,
                     PONG_ACTIONS=(2, 3), cp_device=torch.device('cpu'),
                     pong_device=torch.device('cpu'))
    notebook = json.loads(Path(path).read_text())
    found = set()
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.lstrip().startswith(('%', '!')):
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES:
                if node.name in found:
                    raise ValueError(f'Duplicate definition: {node.name}')
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             str(path), 'exec'), namespace)
                found.add(node.name)
    if found != NAMES:
        raise ValueError(f'Missing definitions: {NAMES - found}')
    namespace['source_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return namespace

def finite_horizon_from_notebook(horizon=2):
    """Execute the original finite-horizon cell; suppress only print expressions."""
    namespace = load_project()
    namespace['T'] = horizon
    notebook = json.loads(SOURCE.read_text())
    matches = [c for c in notebook['cells'] if c['cell_type'] == 'code'
               and 'V_finite = np.zeros' in ''.join(c['source'])]
    if len(matches) != 1:
        raise ValueError('Cannot identify finite-horizon algorithm uniquely.')
    tree = ast.parse(''.join(matches[0]['source']))
    tree.body = [node for node in tree.body if not isinstance(node, ast.Expr)]
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    return namespace['V_finite'], namespace['policy_finite']

