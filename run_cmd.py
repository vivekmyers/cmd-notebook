import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import logging
import signal
import copy
import haiku as hk
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import tqdm
import optax
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import os
import wandb
import argparse
import pickle
from simple_pytree import Pytree, static_field


print("JAX Devices: ", jax.devices())

parser = argparse.ArgumentParser(description="Run CMD")
parser.add_argument("--repr_dim", type=int, default=256, help="repr dim")
parser.add_argument("--margin", type=float, default=0.05, help="margin")
parser.add_argument("--lam0", type=float, default=1.0, help="lam0")
parser.add_argument("--gamma", type=float, default=0.8, help="gamma")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--bc_coef", type=float, default=1e-2, help="bc coef")
parser.add_argument("--tau", type=float, default=0.99, help="tau")
parser.add_argument("--lr", type=float, default=1e-3, help="lr")
parser.add_argument("--dual_lr", type=float, default=0.01, help="dual lr")
parser.add_argument("--traj_avg_len", type=int, default=24, help="traj avg len")
parser.add_argument("--num_train_traj", type=int, default=2000, help="num train traj")
parser.add_argument("--traj_max_len", type=int, default=200, help="traj max len")
parser.add_argument("--traj_min_len", type=int, default=5, help="traj min len")
parser.add_argument("--eval_steps", type=int, default=2000, help="eval steps")
parser.add_argument("--step_noise", type=float, default=1.0, help="step noise")
parser.add_argument("--step_speed", type=float, default=1.5, help="step speed")
parser.add_argument("--obs_noise", type=float, default=0.2, help="obs noise")
parser.add_argument("--succ_thresh", type=float, default=2.0, help="succ thresh")
parser.add_argument(
    "--metric",
    type=str,
    default="mrn",
    help="metric",
    choices=["mrn", "iqe", "max", "soft"],
)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name', type=str, default='cmd')
parser.add_argument('--train_steps', type=int, default=500)
args = parser.parse_args()

wandb.init(project="cmd", config=args, name=args.name)




def parse_traj(traj, act, n=1000):
    k = len(traj)
    traj = jnp.concatenate([traj, jnp.zeros((n - k, 2))], axis=0)
    act = jnp.concatenate([act, jnp.zeros((n - k, 2))], axis=0)
    # return _parse_traj(traj, act, k)[:k]
    res = _parse_traj(traj, act, k)
    return [x[:k] for x in res]


@jax.jit
def _parse_traj(traj, act, k):
    def scan_fn(i):
        x = traj[i]
        a = act[i]
        return (x, k - i - 1, i == k - 1, k, a)

    res = jax.vmap(scan_fn)(jnp.arange(len(traj)))
    return res



eps = 1e-6


# pointmaze that constrains actions to be along the paths in the dataset
# env selects next state from set pre-selected above closest to the normalized action vector
class Pointmaze(Pytree, mutable=True):
    def __init__(self, key, states, action_idx, nearby_idx):
        self.key = key
        self.ds = states
        self.actions = action_idx
        self.start = self.goal = self.state = None
        self.nearby = nearby_idx

    def reset(self, key=None, start=None, goal=None):
        if key is not None:
            self.key = key
        xpos = self.ds[:, 0]
        ypos = self.ds[:, 1]
        self.key, key = jax.random.split(self.key)
        self.start = self.state = jax.random.choice(
            key,
            len(self.ds),  # p=mask_start / mask_start.sum()
        )

        self.key, key = jax.random.split(key)
        prop_goals = jax.random.choice(
            key,
            len(self.ds),  # p=mask_goal / mask_goal.sum()
            (10,)
        )
        dists = jnp.linalg.norm(self.ds[prop_goals] - self.ds[self.start], axis=-1)
        self.goal = prop_goals[jnp.argmax(dists)]

        if start is not None:
            self.start = self.state = jnp.argmin(
                jnp.linalg.norm(self.ds - start, axis=-1)
            )
        if goal is not None:
            self.goal = jnp.argmin(jnp.linalg.norm(self.ds - goal, axis=-1))
        return self.ds[self.state], self.ds[self.goal]

    def step(self, action):
        self.key, key = jax.random.split(self.key)
        action = normalize(action)
        self.key, key = jax.random.split(self.key)
        noise = args.step_noise * jax.random.normal(key, (2,))

        start = self.ds[self.state]
        s_hat = self.ds[self.state] + action
        prop_act = self.actions[self.state]
        act_idx = jnp.argmin(jnp.linalg.norm(self.ds[prop_act] - s_hat, axis=-1))
        proj_idx = prop_act[act_idx]
        nearby = self.nearby[self.state]
        next_mean = self.ds[proj_idx] + noise
        next_mean = normalize(next_mean - start) * args.step_speed + start
        next_idx = jnp.argmin(jnp.linalg.norm(self.ds[nearby] - next_mean, axis=-1))
        self.state = nearby[next_idx]

        done = (
            jnp.linalg.norm(self.ds[self.state] - self.ds[self.goal]) < args.succ_thresh
        )
        reward = jnp.array(done, int)
        obs = self.ds[self.state]
        self.key, key = jax.random.split(self.key)
        noise = args.obs_noise * jax.random.normal(key, (2,))
        obs = obs + noise
        return obs, reward, done, {}

    def get_task(self):
        return self.ds[self.start], self.ds[self.goal]

    def get_obs(self):
        return self.ds[self.state]


def normalize(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


@partial(jax.jit, static_argnames="steps")
def random_rollout(key, env, steps):
    key, subkey = jax.random.split(key)
    obs, goal = env.reset(key=subkey)
    key, subkey = jax.random.split(key)
    returns = 0
    time = 0
    ended = False
    path = [obs]
    base_ac = jax.random.normal(subkey, (2,))

    def scan_fn(carry, i):
        env, obs, returns, ended, key, time = carry
        key, subkey = jax.random.split(key)
        action = base_ac + jax.random.normal(subkey, (2,))
        action = normalize(action)
        next_obs, reward, done, info = env.step(action)
        returns += (1 - ended) * reward
        ended |= done
        time = time + (1 - ended)
        return (env, next_obs, returns, ended, key, time), (obs, action)

    carry, path = jax.lax.scan(
        scan_fn, (env, obs, returns, ended, key, time), None, length=steps
    )
    _, _, returns, _, key, time = carry
    obs, acts = path

    return returns, time, obs, acts


@partial(jax.jit, static_argnames="steps")
def rollout_policy(params, key, s, g, steps):
    env = Pointmaze(key, ds_sampled, sampled_actions, sampled_nearby)
    obs, goal = env.reset(start=s, goal=g)
    succ = 0
    rets = 0
    ended = False
    path = [obs]

    def scan_fn(carry, i):
        env, obs, succ, rets, ended = carry
        action = policy_fn.apply(params, obs, goal)
        next_obs, reward, done, info = env.step(action)
        succ += (1 - ended) * reward
        ended |= done
        rets += ended
        return (env, next_obs, succ, rets, ended), obs

    carry, path = jax.lax.scan(
        scan_fn, (env, obs, succ, rets, ended), None, length=steps
    )
    _, _, succ, rets, _ = carry

    return succ, rets, path


def evaluation_rollout(params, key, steps):
    env = Pointmaze(key, ds_sampled, sampled_actions, sampled_nearby)
    start, goal = env.reset()
    succ, rewards, path = rollout_policy(params, key, start, goal, steps)
    return succ, rewards


# evaluate policy with given params across n rollouts in random PointMaze envs
@partial(jax.jit, static_argnames=("n", "steps"))
def evaluate(params, seed=0, n=1500, steps=args.eval_steps):
    key = jax.random.key(seed)
    keys = jax.random.split(key, n)
    succ, returns = jax.vmap(evaluation_rollout, in_axes=(None, 0, None))(
        params, keys, steps
    )
    return jnp.mean(succ), jnp.mean(returns)



### load data

with open("cache.pkl", "rb") as f:
    data = pickle.load(f)
    ds_sampled = data["ds_sampled"]
    sampled_actions = data["sampled_actions"]
    sampled_nearby = data["sampled_nearby"]
    dataset_unstacked = data["dataset_unstacked"]
    ds_unstack = data["ds_unstack"]
    dataset = data["dataset"]
    ds_obs = data["ds_obs"]
    ds_last = data["ds_last"]
    ds_acts = data["ds_acts"]

### resample trajectories with env args

key = jax.random.PRNGKey(args.seed)
env = Pointmaze(key, ds_sampled, sampled_actions, sampled_nearby)

min_len = args.traj_min_len
max_len = args.traj_max_len
avg_len = args.traj_avg_len
num_train_traj = args.num_train_traj
keys = jax.random.split(key, num_train_traj)

rets, times, trajs, acts = jax.vmap(random_rollout, in_axes=(0, None, None))(
    keys, env, max_len
)
lens = jax.random.geometric(key, 1 / avg_len, shape=(num_train_traj,))

obs, left, done, horizon, acts = jax.vmap(_parse_traj)(trajs, acts, lens)
dataset_unstacked = (obs, left, done, horizon, acts)

ds_unstack = tuple([x[:k] for x, k in zip(dat, lens)] for dat in tqdm.tqdm(dataset_unstacked))
dataset = tuple(jnp.concatenate(x) for x in ds_unstack)

print("random rollout score:", np.mean(rets))

ds_obs = dataset[0]
ds_last = dataset[1] + jnp.arange(len(dataset[1]))
ds_acts = dataset[4]


### functions for training



def featurize_state(z):
    X = jnp.linspace(-60, 60, 20)
    Y = jnp.linspace(-60, 60, 20)
    x, y = z[..., 0], z[..., 1]
    x_ = x[..., None] + X
    y_ = y[..., None] + Y
    z_ = jnp.concatenate([x_, y_], axis=-1)
    z_ = jnp.tanh(z_ / 5)
    return z_


def add_stats(data, stat):
    return jnp.append(data, stat, axis=-1)


@hk.without_apply_rng
@hk.transform
def repr_fn(x, a):
    x = featurize_state(x)
    a = normalize(a)
    a = jnp.repeat(a, x.shape[-1], axis=-1)
    x = jax.nn.swish(hk.Linear(64)(x))
    psi_fn = hk.nets.MLP(output_sizes=[64, 64, 64, repr_dim], activation=jax.nn.swish)
    phi_fn = hk.nets.MLP(output_sizes=[64, 64, 64, repr_dim], activation=jax.nn.swish)

    xa = jnp.concatenate([x, a], axis=-1)

    phi = phi_fn(xa)
    psi = psi_fn(x)

    return phi, psi


@hk.without_apply_rng
@hk.transform
def policy_fn(s, g):
    s = featurize_state(s)
    g = featurize_state(g)
    h_fn = hk.nets.MLP(output_sizes=[64, 64, 64, x_dim], activation=jax.nn.swish)
    h = h_fn(jnp.concatenate([s, g], axis=-1))
    h = normalize(h)
    return h


@hk.without_apply_rng
@hk.transform
def critic_fn(s, a, g):
    s = featurize_state(s)
    g = featurize_state(g)
    a = normalize(a)
    critic_fn = hk.nets.MLP(output_sizes=[64, 64, 64, 1], activation=jax.nn.swish)
    q_vals = critic_fn(jnp.concatenate([s, a, g], axis=-1))
    return q_vals


@hk.without_apply_rng
@hk.transform
def scalar_fn(x):
    x = featurize_state(x)
    scalar_fn = hk.nets.MLP(output_sizes=[64, 64, 64, 1], activation=jax.nn.swish)
    vals = scalar_fn(x).squeeze(axis=-1)
    return vals


def max_metric(x, y):
    h_fn = hk.nets.MLP(output_sizes=[32, 32, repr_dim], activation=jax.nn.swish)
    h1 = h_fn(x)
    h2 = h_fn(y)
    diffs = jax.nn.relu(h1 - h2)
    return diffs.max(axis=-1)


def soft_metric(x, y):
    h_fn = hk.nets.MLP(output_sizes=[32, 32, repr_dim], activation=jax.nn.swish)
    h1 = h_fn(x)
    h2 = h_fn(y)
    diffs = jax.nn.relu(h1 - h2)
    return jnp.sum(diffs * jax.nn.softmax(diffs, axis=-1), axis=-1)


def mrn_metric(x, y):
    h_fn = hk.nets.MLP(output_sizes=[64, 64, 64, repr_dim], activation=jax.nn.swish)
    g_fn = hk.nets.MLP(output_sizes=[64, 64, 64, repr_dim], activation=jax.nn.swish)
    h1 = h_fn(x)
    h2 = h_fn(y)
    g1 = g_fn(x)
    g2 = g_fn(y)
    diffs = jax.nn.relu(h1 - h2)
    d_asym = diffs.max(axis=-1)
    d_sym = jnp.sqrt(jnp.sum(jnp.square(g1 - g2) + eps, axis=-1) + eps)
    return d_asym + d_sym


def iqe_metric(x, y):
    h_fn = hk.nets.MLP(output_sizes=[64, 64, 64, 256], activation=jax.nn.swish)
    alpha_raw = hk.get_parameter("alpha", shape=(1,), init=jnp.zeros)
    alpha = jax.nn.sigmoid(alpha_raw)
    x = h_fn(x)
    y = h_fn(y)
    reshape = (32, 8)
    x = jnp.reshape(x, (*x.shape[:-1], *reshape))
    y = jnp.reshape(y, (*y.shape[:-1], *reshape))
    valid = x < y
    D = x.shape[-1]
    xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
    ixy = xy.argsort(axis=-1)
    sxy = jnp.take_along_axis(xy, ixy, axis=-1)
    neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(
        ixy < D, -1, 1
    )
    neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
    neg_f = (neg_inp_copies < 0) * (-1.0)
    neg_incf = jnp.concatenate(
        [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1
    )
    components = (sxy * neg_incf).sum(-1)
    result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
    return result


# loss functions

stopgrad = jax.lax.stop_gradient


# dot prod dissimilarity for CRL
def rep_diff(phi, psi):
    return jnp.mean(phi * psi, axis=-1)


# metric distillation (ours) loss
# uses contrastive features learned by crl_loss
def cmd2_loss(params, x0, x1, xT, key, a0, delta, t, aT):
    phi, _ = repr_fn.apply(params["crl_repr"], x0, a0)
    phi_, psi = repr_fn.apply(params["crl_repr"], xT, a0)

    perm_idx = jax.random.permutation(key, jnp.arange(batch_size))
    ap_perm = aT[perm_idx]
    xT_perm = xT[perm_idx]

    pred_actions = policy_fn.apply(params["cmd2_policy"], x0, xT_perm)

    value = metric_fn.apply(
        stopgrad(params["cmd2_metric"]), x0, pred_actions, xT_perm, aT
    ) / repr_dim
    l_policy = jnp.mean(value)

    true_pred_actions = policy_fn.apply(params["cmd2_policy"], x0, xT)
    bc_loss = jnp.mean(jnp.square(a0 - true_pred_actions).sum(axis=-1))

    pdist = rep_diff(phi[:, None], psi[None])
    stabilizer = rep_diff(phi_, psi)
    assert pdist.shape == (batch_size, batch_size)
    assert stabilizer.shape == (batch_size,)

    htarget = stopgrad(pdist - stabilizer)
    hdist = metric_fn.apply(
        params["cmd2_metric"], x0[:, None], a0[:, None], xT[None], ap_perm[None]
    ) / repr_dim
    assert hdist.shape == (batch_size, batch_size)
    assert htarget.shape == (batch_size, batch_size)

    I = jnp.eye(batch_size)
    hinge_pos = optax.huber_loss(jnp.maximum(0, jnp.diag(hdist - htarget))).mean()
    hinge_neg = optax.huber_loss(jnp.maximum(0, (1 - I) * (htarget - hdist))).mean()
    lam = jax.nn.softplus(params["cmd2_lam"])
    # ps = stopgrad(lam)
    # ns = 1
    ps = stopgrad(jnp.sqrt(lam))
    ns = 1 / stopgrad(jnp.sqrt(lam))
    h_loss = hinge_pos * ps + hinge_neg * ns
    dual_loss = lam * stopgrad(margin - hinge_pos)

    cmd2_loss = h_loss + dual_loss + (1 - bc_coef) * l_policy + bc_coef * bc_loss

    return cmd2_loss, dict(
        cmd2_h_loss=h_loss,
        cmd2_dual_loss=dual_loss,
        cmd2_policy_loss=l_policy,
        cmd2_lam=lam,
        cmd2_hdist=jnp.mean(hdist),
        cmd2_htarget=jnp.mean(htarget),
        cmd2_stabilizer=jnp.mean(stabilizer),
        cmd2_hinge_pos=hinge_pos,
        cmd2_hinge_neg=hinge_neg,
        cmd2_l_combined=cmd2_loss,
        cmd2_bc_loss=bc_loss,
        cmd2_delta=stopgrad(margin - hinge_pos),
    )


def cmd1_loss(params, x0, x1, xT, key, a0, delta, t, aT):
    perm_idx = jax.random.permutation(key, jnp.arange(batch_size))
    aT_perm = aT[perm_idx]
    xT_perm = xT[perm_idx]

    pred_actions = policy_fn.apply(params["cmd1_policy"], x0, xT_perm)

    value = jnp.maximum(
        metric_fn.apply(stopgrad(params["cmd1_metric"]), x0, pred_actions, xT_perm, aT),
        metric_fn.apply(stopgrad(params["cmd1_metric2"]), x0, pred_actions, xT_perm, aT),
    )
    l_policy = jnp.mean(value) 

    true_pred_actions = policy_fn.apply(params["cmd1_policy"], x0, xT)
    bc_loss = jnp.mean(jnp.square(a0 - true_pred_actions).sum(axis=-1))

    z = jnp.zeros_like(x0)
    potential = scalar_fn.apply(params["cmd1_pot"], xT)

    hdist = jnp.stack([
        metric_fn.apply(
            params["cmd1_metric"], x0[:, None], a0[:, None], xT[None], aT_perm[None]
        ),
        metric_fn.apply(
            params["cmd1_metric2"], x0[:, None], a0[:, None], xT[None], aT_perm[None]
        ),
    ]) / repr_dim
    pred = hdist - potential
    # pred = pred / repr_dim
    assert hdist.shape == (2, batch_size, batch_size)
    assert potential.shape == (batch_size,)
    I = jnp.eye(batch_size)
    adj = I * 0

    l_align = jnp.diagonal(pred, axis1=-2, axis2=-1).mean()
    l_unif = (
        jnp.mean(jax.nn.logsumexp(-pred - adj, axis=1) + jax.nn.logsumexp(-jnp.moveaxis(pred, -1, -2) - adj, axis=1))
        / 2.0
    )

    accuracy = jnp.mean(jnp.argmin(pred, axis=-1) == jnp.arange(batch_size))
    cmd1_loss = (1 - bc_coef) * l_policy + l_align + l_unif + bc_coef * bc_loss

    return cmd1_loss, dict(
        cmd1_policy_loss=l_policy,
        cmd1_metric_loss=l_align + l_unif,
        cmd1_loss=cmd1_loss,
        cmd1_align=l_align,
        cmd1_unif=l_unif,
        cmd1_accuracy=accuracy,
        cmd1_bc_loss=bc_loss,
    )


def crl_loss(params, x0, x1, xT, key, actions, delta, t, aT):
    I = jnp.eye(batch_size)
    phi, _ = repr_fn.apply(params["crl_repr"], x0, actions)
    _, psi = repr_fn.apply(params["crl_repr"], xT, aT)
    pdist = rep_diff(phi[:, None], psi[None])
    l_align = jnp.diag(pdist).mean()
    l_unif = (
        jnp.mean(jax.nn.logsumexp(-pdist, axis=1) + jax.nn.logsumexp(-pdist.T, axis=1))
        / 2.0
    )
    xT_perm = jax.random.permutation(key, xT)
    crl_feature_loss = l_align + l_unif
    accuracy = jnp.mean(jnp.argmin(pdist, axis=1) == jnp.arange(batch_size))

    pred_actions = policy_fn.apply(params["crl_policy"], x0, xT_perm)

    phi_pred, _ = repr_fn.apply(stopgrad(params["crl_repr"]), x0, pred_actions)
    _, psi_pred = repr_fn.apply(stopgrad(params["crl_repr"]), xT_perm, aT)
    value = rep_diff(phi_pred, psi_pred).mean()

    crl_actor_loss = jnp.mean(value)

    true_pred_actions = policy_fn.apply(params["crl_policy"], x0, xT)
    bc_loss = jnp.mean(jnp.square(actions - true_pred_actions).sum(axis=-1))
    loss = (1 - bc_coef) * crl_actor_loss + crl_feature_loss + bc_coef * bc_loss

    return loss, dict(
        crl_actor_loss=crl_actor_loss,
        crl_feature_loss=crl_feature_loss,
        accuracy=accuracy,
        crl_bc_loss=bc_loss,
    )


def td_loss(params, x0, x1, xT, key, a0, delta, t, aT):
    xT_perm = jax.random.permutation(key, xT)
    xTperm_xT = jnp.vstack([xT_perm, xT])
    x0x0 = jnp.vstack([x0, x0])
    a0a0 = jnp.vstack([a0, a0])
    x1x1 = jnp.vstack([x1, x1])

    rewards = jnp.vstack([jnp.zeros((batch_size, 1)), delta[:, None] <= 1])
    td_actions = policy_fn.apply(params["td_policy"], x0x0, xTperm_xT)
    td_actions = td_actions
    qval = critic_fn.apply(params["td_critic"], x0x0, a0a0, xTperm_xT)
    baseline = critic_fn.apply(
        params["td_critic"], x0x0, stopgrad(td_actions), xTperm_xT
    )
    a1a1 = policy_fn.apply(params["td_policy"], x1x1, xTperm_xT)
    qtarget = critic_fn.apply(
        params["td_target_critic"], x1x1, stopgrad(a1a1), xTperm_xT
    )
    notdone = rewards > 0
    assert (
        notdone.shape == qtarget.shape == qval.shape == baseline.shape == rewards.shape
    )
    critic_loss = jnp.mean(
        jnp.square(qval - stopgrad(notdone * qtarget * gamma + rewards)).sum(axis=-1)
    )
    assert qval.shape == qtarget.shape

    pred_actions = td_actions[:batch_size]
    true_pred_actions = td_actions[batch_size:]

    value = critic_fn.apply(stopgrad(params["td_critic"]), x0, pred_actions, xT_perm)
    bc_loss = jnp.mean(jnp.square(a0 - true_pred_actions).sum(axis=-1))

    td_actor_loss = bc_coef * bc_loss - (1 - bc_coef) * jnp.mean(value)

    td_loss = critic_loss + td_actor_loss

    return td_loss, dict(
        td_actor_loss=td_actor_loss,
        td_critic_loss=critic_loss,
        td_loss=td_loss,
        td_bc_loss=bc_loss,
        td_value_loss=value,
    )


def qrl_loss(params, x0, x1, xT, key, actions, delta, t, aT):
    z = jnp.zeros_like(x0)
    xT_perm = jax.random.permutation(key, xT)
    qrl_lam = jax.nn.softplus(params["qrl_lam"])

    qrl_phi = lambda x: -100 * jax.nn.softplus(5 - x / 100)
    qrl_neg = -qrl_phi(
        metric_fn.apply(params["qrl_metric"], x0, z, xT_perm, z)
    ).mean()
    qrl_pos = jnp.mean(
        jax.nn.relu(metric_fn.apply(params["qrl_metric"], x0, z, x1, z) - 1) ** 2
    )
    qrl_critic_loss = jnp.mean(
        qrl_neg + qrl_pos * stopgrad(qrl_lam)
    )
    qrl_dual = qrl_lam * (margin - stopgrad(qrl_pos))

    transition_loss = (
        0.5 * (
            metric_fn.apply(params["qrl_metric"], x0, actions, x1, z) ** 2
            + metric_fn.apply(params["qrl_metric"], x1, z, x0, actions) ** 2
        ).mean()
    )
    # adv = metric_fn.apply(params["qrl_metric"], x0, z, xT_perm, z) - metric_fn.apply(
    #     params["qrl_metric"], x1, z, xT_perm, z
    # )
    # adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + eps)
    # weight = stopgrad(jnp.exp(adv)).clip(max=1e2)
    true_pred_actions = policy_fn.apply(params["qrl_policy"], x0, xT)
    bc_loss = jnp.mean(jnp.square(true_pred_actions - actions).sum(axis=-1))

    qrl_actions = policy_fn.apply(params["qrl_policy"], x0, xT_perm)
    value = metric_fn.apply(stopgrad(params["qrl_metric"]), x0, qrl_actions, stopgrad(xT_perm), z)
    qrl_actor_loss = (1 - bc_coef) * jnp.mean(value) + bc_coef * bc_loss

    loss = qrl_critic_loss + qrl_actor_loss + qrl_dual + transition_loss

    return loss, dict(
        qrl_critic_loss=qrl_critic_loss,
        qrl_actor_loss=qrl_actor_loss,
        qrl_dual=qrl_dual,
        qrl_loss=loss,
        qrl_bc_loss=bc_loss,
        qrl_lam=qrl_lam,
        qrl_transition_loss=transition_loss,
    )


def bc_loss(params, x0, x1, xT, key, actions, delta, t, ap):
    bc_actions = policy_fn.apply(params["bc_policy"], x0, xT)
    bc_actions = bc_actions
    bc_loss = jnp.mean(jnp.square(bc_actions - actions).sum(axis=-1))
    return bc_loss, dict(bc_loss=bc_loss)


def loss_fn(params, x0, x1, xT, key, actions, ap, delta, t):
    actions = actions
    keys = hk.PRNGSequence(key)

    loss = 0.0
    metrics = {}

    loss_, metrics_ = cmd2_loss(params, x0, x1, xT, next(keys), actions, delta, t, ap)
    loss += loss_
    metrics.update(metrics_)

    loss_, metrics_ = crl_loss(params, x0, x1, xT, next(keys), actions, delta, t, ap)
    loss += loss_
    metrics.update(metrics_)

    loss_, metrics_ = td_loss(params, x0, x1, xT, next(keys), actions, delta, t, ap)
    loss += loss_
    metrics.update(metrics_)

    loss_, metrics_ = qrl_loss(params, x0, x1, xT, next(keys), actions, delta, t, ap)
    loss += loss_
    metrics.update(metrics_)

    loss_, metrics_ = bc_loss(params, x0, x1, xT, next(keys), actions, delta, t, ap)
    loss += loss_
    metrics.update(metrics_)

    loss_, metrics_ = cmd1_loss(params, x0, x1, xT, next(keys), actions, delta, t, ap)
    loss += loss_
    metrics.update(metrics_)

    metrics = {k: jnp.mean(v) for k, v in metrics.items()}
    return loss, metrics


@hk.without_apply_rng
@hk.transform
def metric_fn(s, a, g, a_):
    s = featurize_state(s)
    g = featurize_state(g)
    a = jnp.repeat(normalize(a), s.shape[-1] // 2, axis=-1)
    a_ = jnp.repeat(normalize(a_), s.shape[-1] // 2, axis=-1)
    sa = jnp.concatenate([s, a], axis=-1)
    ga = jnp.concatenate([g, a_], axis=-1)
    return _metric_fn(sa, ga)


@hk.without_apply_rng
@hk.transform
def qrl_metric_fn(s, a, g, a_):
    s = featurize_state(s)
    g = featurize_state(g)
    a = jnp.repeat(normalize(a), s.shape[-1] // 2, axis=-1)
    a_ = jnp.repeat(normalize(a_), s.shape[-1] // 2, axis=-1)
    sa = jnp.concatenate([s, a], axis=-1)
    ga = jnp.concatenate([g, a_], axis=-1)
    return iqe_metric(sa, ga)

### initialize params and optimizer
repr_dim = args.repr_dim
x_dim = 2
max_episode_steps = jnp.max(dataset[3])
eps = 1e-6

_metric_fn = dict(
    max=max_metric,
    soft=soft_metric,
    mrn=mrn_metric,
    iqe=iqe_metric,
)[args.metric]


lam0 = args.lam0
margin = args.margin
gamma = args.gamma
batch_size = args.batch_size
bc_coef = args.bc_coef
tau = args.tau
lr = args.lr
dual_lr = args.dual_lr

keys = hk.PRNGSequence(args.seed)
x = jnp.zeros((1, 2))
z = jnp.zeros((1, repr_dim))
params = {}
params["crl_repr"] = repr_fn.init(rng=next(keys), x=x, a=x)
params["cmd2_metric"] = metric_fn.init(rng=next(keys), s=x, g=x, a=x, a_=x)
params["cmd2_policy"] = policy_fn.init(rng=next(keys), s=x, g=x)
params["cmd2_lam"] = lam0
params["bc_policy"] = policy_fn.init(rng=next(keys), s=x, g=x)
params["crl_policy"] = policy_fn.init(rng=next(keys), s=x, g=x)
params["td_critic"] = critic_fn.init(rng=next(keys), s=x, a=x, g=x)
params["td_target_critic"] = critic_fn.init(rng=next(keys), s=x, a=x, g=x)
params["td_policy"] = policy_fn.init(rng=next(keys), s=x, g=x)
params["qrl_metric"] = metric_fn.init(rng=next(keys), s=x, g=x, a=x, a_=x)
params["qrl_policy"] = policy_fn.init(rng=next(keys), s=x, g=x)
params["qrl_lam"] = lam0
params["cmd1_metric"] = metric_fn.init(rng=next(keys), s=x, g=x, a=x, a_=x)
params["cmd1_metric2"] = metric_fn.init(rng=next(keys), s=x, g=x, a=x, a_=x)
params["cmd1_pot"] = scalar_fn.init(rng=next(keys), x=x)
params["cmd1_policy"] = policy_fn.init(rng=next(keys), s=x, g=x)


optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))
dual_opt = optax.sgd(learning_rate=dual_lr)

opt_state = optimizer.init(params)
opt_state_dual = dual_opt.init(params)
grad_fn = jax.value_and_grad(loss_fn, has_aux=True)


def update_targets(params, t):
    params = copy.deepcopy(params)
    params["td_target_critic"] = jax.tree_map(
        lambda x, y: tau * x + (1 - tau) * y,
        params["td_target_critic"],
        params["td_critic"],
    )
    return params


num_tran = len(ds_obs)
m = defaultdict(list)
t = 0
times = []


# utility


# our learned md distance quasimetric
@jax.jit
def dsd(s, g, params):
    z = jnp.zeros_like(s)
    dist = metric_fn.apply(params, s, z, g, z)
    return dist


# the crl feature cosine dissimilarity
@jax.jit
def p_goal(s, g, params):
    z = jnp.zeros_like(s)
    phi_s, psi_s = repr_fn.apply(params["crl_repr"], s, z)
    phi_g, psi_g = repr_fn.apply(params["crl_repr"], g, z)
    return rep_diff(phi_s, psi_g)


@jax.jit
def get_data(key):
    key, rng1, rng2 = jax.random.split(key, 3)
    i = jax.random.randint(rng1, shape=(batch_size,), minval=0, maxval=num_tran - 1)
    horizon = ds_last[i] - i
    max_horizon = max_episode_steps
    probs = gamma ** jnp.tile(jnp.arange(max_horizon)[None], (batch_size, 1))
    log_probs = jnp.log(probs)
    delta = jax.random.categorical(key=rng2, logits=log_probs)
    delta = jnp.minimum(delta, horizon)
    clip_delta = jnp.minimum(delta, 1)
    next_delta = jnp.minimum(delta + 1, max_horizon - 1)
    state = ds_obs[i]
    future_state = ds_obs[i + delta]
    next_state = ds_obs[i + clip_delta]
    action = ds_acts[i]
    future_action = ds_acts[i + delta]

    key, rng1, rng2 = jax.random.split(key, 3)
    noise1 = jax.random.normal(rng1, (batch_size, 2)) * args.obs_noise
    noise2 = jax.random.normal(rng2, (batch_size, 2)) * args.obs_noise
    state = state + noise1
    future_state = future_state + noise2
    return key, state, next_state, future_state, action, future_action, delta


@jax.jit
def step_fn(params, opt_state, opt_state_dual, key, t):
    key, x0, x1, xT, actions, ap, delta = get_data(key)
    key, subkey = jax.random.split(key)
    (loss, metrics), grad = grad_fn(params, x0, x1, xT, subkey, actions, ap, delta, t)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    dual_updates, opt_state_dual = dual_opt.update(grad, opt_state_dual, params)
    updates = {k: updates[k] if "_lam" not in k else dual_updates[k] for k in updates}
    params = optax.apply_updates(params, updates)
    params = update_targets(params, t)  # for TD q target udpate
    return loss, metrics, params, opt_state, opt_state_dual, key


class Atomic:

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt.")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


### run training

plt.ioff()

plt.style.use("default")
sns.set_style("whitegrid")
key = next(keys)


def scan_fn(carry, _):
    params, opt_state, opt_state_dual, key, t = carry
    loss, metrics, params, opt_state, opt_state_dual, key = step_fn(
        params, opt_state, opt_state_dual, key, t
    )
    t += 1
    return (params, opt_state, opt_state_dual, key, t), (loss, metrics)


for itr in tqdm.trange(int(args.train_steps)):
    # loss, metrics, params, opt_state, key = step_fn(params, opt_state, key, t)
    carry, xs = jax.lax.scan(
        scan_fn,
        (params, opt_state, opt_state_dual, key, t),
        None,
        length=1000,
    )
    with Atomic():
        params, opt_state, opt_state_dual, key, t = carry
        loss, metrics = jax.tree_map(jnp.mean, xs)

        for k in ["cmd2", "cmd1", "crl", "td", "qrl", "bc"]:
            succ, ret = evaluate(params[k + "_policy"])
            metrics[k + "_success"] = succ
            metrics[k + "_return"] = ret

        for k, v in metrics.items():
            m[k].append(v)

        times.append(t)

        wandb.log(metrics, step=t)
        wandb.log({"obj": min(m["cmd2_success"][-1], m["cmd1_success"][-1]) - max(m["qrl_success"][-1], m["bc_success"][-1])}, step=t)




### make training curves


def pplot(suffix):
    mpl.rc_file_defaults()
    plt.style.use("ggplot")
    if "texlive" not in os.environ["PATH"]:
        os.environ["PATH"] += (
            os.pathsep + "/nas/ucb/vivek/texlive/2023/bin/x86_64-linux"
        )

    random_policy = policy_fn.init(rng=next(keys), s=x, g=x)
    baseline = evaluate(random_policy)
    if suffix == "success":
        baseline = baseline[0]
    elif suffix == "return":
        baseline = baseline[1]

    rcParams = {}
    # rcParams["figure.dpi"] = 50
    rcParams["lines.linewidth"] = 3
    rcParams["lines.markersize"] = 5
    rcParams["font.size"] = 18
    rcParams["legend.fontsize"] = 18
    rcParams["figure.titlesize"] = 30
    rcParams["axes.facecolor"] = "white"
    rcParams["text.color"] = "black"
    rcParams["axes.labelcolor"] = "black"
    rcParams["xtick.color"] = "black"
    rcParams["ytick.color"] = "black"
    rcParams["ytick.labelsize"] = 18
    rcParams["xtick.labelsize"] = 18
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Palatino"

    names = {
        "cmd1_success": (r"$\textbf{CMD-1\ (ours)}$", 5),
        "cmd2_success": (r"$\textbf{CMD-2\ (ours)}$", 0),
        "td_success": ("Q-learning", 1),
        "crl_success": ("contrastive RL", 2),
        "qrl_success": ("quasimetric RL", 4),
        "bc_success": ("behavioral cloning", 3),
    }

    with plt.rc_context(rcParams):
        plt.figure(figsize=(12, 6.5))
        plt.grid(False)
        plt.gca().spines[:].set_color("black")
        sns.despine(top=True, right=True)

        plt.ylim((0, 0.9))
        plt.title("Pointmaze training curves", color="black", pad=15)
        plt.xlabel("offline steps")
        plt.ylabel(suffix)
        for k in names:
            if k.endswith(suffix):
                t0 = jnp.concatenate([np.zeros(1), np.array(times)])
                t_mask = t0 < 5e5
                v = jnp.concatenate([np.zeros(1) + baseline, np.array(m[k])])[t_mask]
                smoothed = jnp.array(gaussian_filter1d(v, 5)).at[0].set(baseline)
                l, c = names[k]
                p = plt.plot(t0[t_mask], smoothed, label=l, color="C%d" % c)
                plt.plot(t0[t_mask], v, alpha=0.1, c=p[0].get_color())
        # plt.legend(bbox_to_anchor=(1., .29), loc='upper right', facecolor='white', framealpha=0.)
        lgnd = plt.legend(
            loc="center right",
            facecolor="white",
            framealpha=0.0,
            bbox_to_anchor=(1.0, 0.0, 0.5, 1.0),
        )
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        wandb.log({f"{suffix}_plot": plt})


pplot("success")
pplot("return")


# plotting code


def waypoint_mass(s, g, w, d):
    w, g = jnp.broadcast_arrays(w, g[None])
    weight = d(w, g)
    return weight


def plot_plan(s, g, d, policy, legend=False, ax=None):
    if ax is None:
        ax = plt.gca()
    s = jnp.array(s)
    g = jnp.array(g)
    ax.tick_params(labelbottom=False, labelleft=False, color="white")

    masses = waypoint_mass(s, g, ds_sampled, d)

    key = jax.random.key(0)
    returns, _, pt = rollout_policy(params[policy], key, s, g, args.eval_steps)
    sk = 10
    for xy, xy_ in list(zip(pt[:-sk], pt[sk:]))[::sk]:
        dxy = xy_ - xy
        offset = 0.1 * dxy
        ax.arrow(
            *(xy + offset),
            *(dxy - offset),
            head_width=1,
            head_length=1,
            fc="k",
            ec="k",
            zorder=50,
            alpha=1.0,
            linewidth=2,
        )
        if jnp.linalg.norm(xy_ - g) < 2:
            break

    ax.scatter(ds_sampled[:, 0], ds_sampled[:, 1], s=1.5, c=masses, cmap="viridis_r")
    ssz = 10**2
    gsz = 16**2
    lw = 1.5
    ax.scatter(
        *s[None].T,
        marker="o",
        edgecolors="black",
        c="red",
        s=ssz,
        zorder=100,
        linewidth=lw,
    )
    ax.scatter(
        *g[None].T,
        marker="*",
        edgecolors="black",
        c="green" if returns > 0.5 else "orange",
        s=gsz,
        zorder=100,
        linewidth=lw,
    )
    if legend:
        ax.scatter(
            [],
            [],
            marker="o",
            edgecolors="black",
            c="red",
            s=ssz * 3,
            zorder=100,
            label="start",
            linewidth=lw,
        )
        ax.scatter(
            [],
            [],
            marker="*",
            edgecolors="black",
            c="green",
            s=gsz * 3,
            zorder=100,
            label="goal",
            linewidth=lw,
        )
    ax.set_aspect("equal")


def hillclimb(key, s, g, d):
    s = jnp.array(s)
    g = jnp.array(g)
    path = [s]
    key, subkey = jax.random.split(key)
    prop = jax.random.permutation(subkey, ds_obs)[:1500]
    prop = jnp.concatenate([prop, jnp.stack(path)])
    while jnp.linalg.norm(path[-1] - g) > 2 and len(path) < 50:
        key, subkey, subkey_ = jax.random.split(key, 3)
        proposals = prop[jnp.linalg.norm(prop - path[-1], axis=-1) < 2.5]
        # proposals = prop[jnp.any(jnp.linalg.norm(prop[None] - proposals[:, None], axis=-1) < 2, axis=0)]
        proposals = proposals + 0.2 * jax.random.normal(subkey, shape=proposals.shape)
        m_vec = path[-1] - (path[:1] + path)[-2]
        m_penalty = (
            jnp.linalg.norm(m_vec)
            * 1e-2
            * jnp.linalg.norm(proposals - (s + m_vec), axis=-1) ** 2
        )
        still_penalty = -jnp.linalg.norm(proposals - path[-1], axis=-1) * 1e-3
        scores = (
            jax.vmap(d, in_axes=(0, None))(proposals, g) + m_penalty + still_penalty
        )
        s = proposals[jnp.argmin(scores)]
        path.append(s)
        past = jnp.stack(path[-5:])
        pdist = jnp.linalg.norm(past[:, None] - past[None], axis=-1)
        # if len(path) > 5 and jnp.all(pdist < 2):
        #    path = path[:-4]
        #    break
    return jnp.stack(path)


### Visualize trajectories

for method in ["cmd2", "cmd1", "qrl"]:
    d = 20
    fig, ax = plt.subplots(nrows=4, ncols=4, dpi=80, figsize=(15, 15))
    axs = ax.flatten()

    metric = partial(dsd, params=params.get(f"{method}_metric", params["cmd2_metric"]))
    pol = f"{method}_policy"
    rcParams = {
        "font.size": 20,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Palatino",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "ytick.labelsize": 20,
        "xtick.labelsize": 20,
        "figure.titlesize": 30,
        "figure.dpi": 150,
        "lines.linewidth": 3,
        "lines.markersize": 5,
        "legend.fontsize": 20,
        "axes.facecolor": "white",
    }
    with plt.rc_context(rcParams):

        env = Pointmaze(key, ds_sampled, sampled_actions, sampled_nearby)

        coord = lambda: env.reset()

        plot_plan(*coord(), d=metric, ax=axs[0], policy=pol, legend=True)
        for i in tqdm.trange(1, 16):
            plot_plan(*coord(), d=metric, ax=axs[i], policy=pol)

        cbar = fig.colorbar(
            axs[0].collections[0], ax=axs[-1], cax=fig.add_axes([0.94, 0.15, 0.02, 0.7])
        )
        cbar.set_label("goal distance")
        plt.subplots_adjust(right=0.88, wspace=0.2, hspace=0.2)
        fig.suptitle("Metric distillation in pointmaze", x=0.54, y=0.94, fontsize=29)
        lgnd = fig.legend(
            framealpha=0, bbox_to_anchor=(0.7, 0.12), fontsize=30, ncol=2, handletextpad=0.0
        )
        for legend_handle in lgnd.legend_handles:
            try:
                legend_handle.set_markersize(100)
            except:
                pass
        plt.savefig("pointmaze.pdf", bbox_inches="tight")
        wandb.log({"pointmaze": plt})
        img = wandb.Image(plt)
        wandb.log({"img": img})
