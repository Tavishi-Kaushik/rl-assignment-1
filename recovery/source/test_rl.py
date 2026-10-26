"""Unit tests AND independent, quantitative RL validation experiments."""
import itertools
import unittest
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from project_code import load_project, finite_horizon_from_notebook

P = load_project()
torch.set_num_threads(1)

class TwoStepPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(
            [np.log(.4/.6), np.log(.7/.3)], dtype=torch.float64))

    def forward(self, x):
        z = x.to(self.theta.dtype) @ self.theta
        return torch.stack((torch.zeros_like(z), z), dim=-1)

def objective(theta, gamma):
    p, q = 1 / (1 + np.exp(-np.asarray(theta)))
    return 1 - 2*p + gamma*(3*p + 2*q + 4*p*q)

def exact_gradient(theta, gamma):
    p, q = 1 / (1 + np.exp(-np.asarray(theta)))
    return np.array([p*(1-p)*(-2 + gamma*(3+4*q)),
                     gamma*q*(1-q)*(2+4*p)])

def expected_loss_gradient(kind, gamma, baseline=None, broken=None):
    model = TwoStepPolicy()
    p, q = torch.sigmoid(model.theta).detach().numpy()
    expected = np.zeros(2)
    for a, b in itertools.product((0, 1), repeat=2):
        # Enumerate every trajectory with its exact probability, not Monte Carlo.
        probability = (p if a else 1-p) * (q if b else 1-q)
        rewards = np.array([1 - 2*a, 3*a + 2*b + 4*a*b], dtype=float)
        actions = torch.tensor([a, b])
        model.zero_grad(set_to_none=True)
        if kind == 'pong':
            P['accumulate_pong_gradient'](
                model, {'observations': np.eye(2, dtype=np.float32),
                        'actions': actions.numpy(), 'rewards': rewards},
                gamma, baseline=baseline, chunk_size=1)
        else:
            lp = Categorical(logits=model(torch.eye(2))).log_prob(actions)
            if broken == 'omit_outer_discount':
                # Deliberate error: our test must catch this.
                weights = torch.tensor([rewards[0]+gamma*rewards[1], rewards[1]])
                loss = -(weights*lp).sum()
            elif broken == 'action_dependent_baseline':
                weights = torch.tensor([rewards[0]+gamma*rewards[1]-a,
                                        gamma*(rewards[1]-b)])
                loss = -(weights*lp).sum()
            elif baseline is None:
                loss = P['reinforce_loss'](lp, rewards, gamma)
            else:
                loss = P['reinforce_loss_with_baseline'](lp, rewards, gamma, baseline)
            loss.backward()
        expected += probability * model.theta.grad.detach().numpy()
    return expected

def analytical_rows():
    rows = []
    theta = TwoStepPolicy().theta.detach().numpy()
    for kind in ('cartpole', 'pong'):
        for gamma in (0., .5, .95, .99, 1.):
            for baseline in (None, np.array([1.2, -.7])):
                actual = expected_loss_gradient(kind, gamma, baseline)
                truth = -exact_gradient(theta, gamma)
                rows.append(dict(implementation=kind, gamma=gamma,
                    baseline='none' if baseline is None else 'fixed_time',
                    expected_0=float(truth[0]), expected_1=float(truth[1]),
                    measured_0=float(actual[0]), measured_1=float(actual[1]),
                    max_absolute_error=float(np.max(abs(actual-truth)))))
    return rows

class AnalyticalPolicyGradientTests(unittest.TestCase):
    def test_expected_gradient_matches_closed_form_both_implementations(self):
        for row in analytical_rows():
            with self.subTest(**row):
                self.assertLess(row['max_absolute_error'], 2e-6)

    def test_closed_form_matches_finite_difference_of_true_objective(self):
        theta = TwoStepPolicy().theta.detach().numpy()
        h = 1e-5
        fd = np.array([(objective(theta + h*np.eye(2)[j], .5)
                        - objective(theta - h*np.eye(2)[j], .5))/(2*h)
                       for j in range(2)])
        np.testing.assert_allclose(fd, exact_gradient(theta, .5), atol=1e-9)

    def test_baseline_changes_sample_gradients_but_not_expectation(self):
        for kind in ('cartpole', 'pong'):
            g0 = expected_loss_gradient(kind, .5)
            gb = expected_loss_gradient(kind, .5, np.array([1.2, -.7]))
            np.testing.assert_allclose(g0, gb, atol=2e-6)

    def test_negative_controls_detect_missing_discount_and_baseline_bias(self):
        truth = -exact_gradient(TwoStepPolicy().theta.detach().numpy(), .5)
        for bug in ('omit_outer_discount', 'action_dependent_baseline'):
            bad = expected_loss_gradient('cartpole', .5, broken=bug)
            self.assertGreater(np.linalg.norm(bad-truth), .05)

    def test_gradient_descent_on_loss_increases_true_expected_reward(self):
        theta = TwoStepPolicy().theta.detach().numpy()
        for kind in ('cartpole', 'pong'):
            g = expected_loss_gradient(kind, .5)
            self.assertGreater(objective(theta-.05*g, .5), objective(theta, .5))
            self.assertLess(objective(theta+.05*g, .5), objective(theta, .5))

class ReturnAndBaselineTests(unittest.TestCase):
    def test_sparse_signed_rewards_and_discount_edges(self):
        for gamma in (0., .5, .95, .99, 1.):
            rewards = [0., -1., 0., 1., 0.]
            truth = [sum(gamma**(k-t)*rewards[k] for k in range(t, 5))
                     for t in range(5)]
            np.testing.assert_allclose(P['discounted_returns'](rewards, gamma), truth)
        np.testing.assert_array_equal(P['discounted_returns']([], .99), [])

    def test_gamma_zero_removes_later_decision_gradient(self):
        g = expected_loss_gradient('pong', 0.)
        np.testing.assert_allclose(g, [.48, 0.], atol=2e-6)

    def test_baseline_uses_only_visited_times_and_preserves_old_array(self):
        old = np.array([0., 0., 0., 7.])
        updated = P['update_time_baseline'](old, [[1.,2.,3.], [4.]], .5, .1)
        np.testing.assert_allclose(updated, [.3375, .35, .3, 7.])
        np.testing.assert_array_equal(old, [0., 0., 0., 7.])

    def test_zero_baseline_same_loss_and_gradient(self):
        for baseline in (None, np.zeros(2)):
            np.testing.assert_allclose(expected_loss_gradient('cartpole', .95, baseline),
                                       expected_loss_gradient('pong', .95, baseline), atol=2e-6)

    def test_chunking_and_batch_averaging(self):
        m = TwoStepPolicy()
        e = {'observations': np.eye(2), 'actions': np.array([0,1]), 'rewards': [1.,2.]}
        gradients = []
        for chunk, divisor in [(1,1), (2,1), (1,4)]:
            m.zero_grad(set_to_none=True)
            P['accumulate_pong_gradient'](m,e,.99,episode_divisor=divisor,chunk_size=chunk)
            gradients.append(m.theta.grad.detach().numpy().copy())
        np.testing.assert_allclose(gradients[0], gradients[1], atol=1e-7)
        np.testing.assert_allclose(gradients[0]/4, gradients[2], atol=1e-7)

class ShuttleTests(unittest.TestCase):
    def test_costs_and_empty_partial_full_queues(self):
        self.assertEqual(P['immediate_cost'](20, 1),110)
        self.assertEqual(P['immediate_cost'](20, 0),40)
        for s in (0,1,14,15):
            self.assertEqual(P['passengers_remaining'](s,1),0)
        self.assertEqual(P['immediate_cost'](0,1),100)

    def test_capped_probability_mass(self):
        successors = [P['next_state'](198,0,a) for a in range(1,6)]
        self.assertEqual(successors,[199,200,200,200,200])
        self.assertEqual(sum(s==200 for s in successors)/5,.8)

    def test_two_period_dp_against_exhaustive_contingent_policies(self):
        values, policies = finite_horizon_from_notebook(2)
        # 2 initial actions x 2^5 decisions contingent on the arrival outcome.
        for s in (0,13,15,20,198,200):
            candidates=[]
            for a0 in (0,1):
                remaining=max(s-15*a0,0)
                for followups in itertools.product((0,1),repeat=5):
                    future=0.
                    for arrivals,a1 in zip(range(1,6),followups):
                        s1=min(200,remaining+arrivals)
                        future += (100*a1+2*max(s1-15*a1,0))/5
                    candidates.append(100*a0+2*remaining+.95*future)
            self.assertAlmostEqual(values[0,s], min(candidates), places=9)
        self.assertAlmostEqual(values[0,20],83.7)
        self.assertTrue(np.all(policies==0))

    def test_analytical_parameter_limits(self):
        solve=P['solve_shuttle_vi']
        v,p=solve(gamma=0)
        np.testing.assert_allclose(v,2*np.arange(201)); self.assertTrue(np.all(p==0))
        v,p=solve(holding_cost=0)
        self.assertEqual(np.max(abs(v)),0); self.assertTrue(np.all(p==0))
        v,p=solve(capacity=0)
        self.assertAlmostEqual(v[200],8000,places=5); self.assertTrue(np.all(p==0))
        v,p=solve(dispatch_cost=0)
        self.assertEqual(v[0],0); self.assertTrue(np.all(p[1:]==1))

    def test_independent_bellman_residual_and_policy_evaluation(self):
        values,policy=P['solve_shuttle_vi']()
        # Scalar transition construction, independent of vectorized VI.
        transition=np.zeros((201,201)); costs=np.zeros(201); q=np.zeros((201,2))
        for s in range(201):
            for a in (0,1):
                r=max(s-15*a,0); c=100*a+2*r
                q[s,a]=c+.95*sum(values[min(200,r+x)]/5 for x in range(1,6))
                if a==policy[s]:
                    costs[s]=c
                    for x in range(1,6): transition[s,min(200,r+x)]+=.2
        exact=np.linalg.solve(np.eye(201)-.95*transition,costs)
        residual=np.max(abs(q.min(1)-values))
        self.assertLess(residual,1e-8)
        self.assertLessEqual(np.max(abs(exact-values)),residual/.05+1e-9)
        np.testing.assert_array_equal(policy, (np.arange(201)>=13).astype(int))

class FakeEnv:
    def __init__(self, pong=False, truncate=False):
        self.pong=pong; self.truncate=truncate; self.actions=[]

    def observation(self):
        if not self.pong: return np.array([self.t,0,0,0],dtype=np.float32)
        frame=np.full((210,160,3),144,dtype=np.uint8)
        frame[55,40+2*self.t,:]=255
        return frame

    def reset(self, seed=None):
        self.t=0; self.actions=[]
        return self.observation(),{}

    def step(self,action):
        if self.t>=3: raise AssertionError('Stepped after episode ended')
        self.actions.append(action); self.t+=1
        return self.observation(),1.,self.t==3 and not self.truncate,self.t==3 and self.truncate,{}

class FixedPolicy(nn.Module):
    def __init__(self, action=0):
        super().__init__()
        self.logits=nn.Parameter(torch.tensor([50.,-50.] if action==0 else [-50.,50.]))

    def forward(self,x):
        return self.logits

class RolloutTests(unittest.TestCase):
    def test_stops_on_termination_or_truncation_and_maps_actions(self):
        for pong,truncate,action in itertools.product((False,True),repeat=3):
            env=FakeEnv(pong,truncate)
            model=FixedPolicy(int(action))
            before=model.logits.detach().clone()
            name='collect_pong_episode' if pong else 'collect_cartpole_episode'
            e=P[name](env,model,123)
            self.assertEqual(len(e['rewards']),3)
            self.assertEqual(env.actions,[int(action)+2 if pong else int(action)]*3)
            self.assertEqual(e['truncated'],truncate)
            self.assertEqual(e['terminated'],not truncate)
            self.assertTrue(torch.equal(before,model.logits))
            if pong:
                self.assertTrue(np.all(e['observations'][0]==0))
                self.assertEqual(set(np.unique(e['observations'][1])),{-1,0,1})
                again=P[name](env,model,124)
                self.assertTrue(np.all(again['observations'][0]==0))

    def test_preprocessing_preserves_input_and_removes_both_backgrounds(self):
        frame=np.full((210,160,3),144,dtype=np.uint8)
        frame[35,0,:]=109; frame[37,2,:]=255
        copy=frame.copy(); output=P['preprocess_pong'](frame)
        np.testing.assert_array_equal(frame,copy)
        self.assertEqual(output[0,0],0); self.assertEqual(output[1,1],1)
        self.assertEqual(output.sum(),1)

if __name__=='__main__': unittest.main(verbosity=2)
