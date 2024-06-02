
# imports

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
from functools import partial
from simple_pytree import Pytree, static_field
import argparse

print("JAX Devices: ", jax.devices())

# |%%--%%| <CW92zGgP1u|NBP7ZdO87J>

parser = argparse.ArgumentParser(description="Run CMD")
parser.add_argument("--repr_dim", type=int, default=256, help="repr dim")
parser.add_argument('--asym_shift', type=int, default=8, help='asym shift')
parser.add_argument("--margin", type=float, default=0.05, help="margin")
parser.add_argument("--lam0", type=float, default=1.0, help="lam0")
parser.add_argument("--gamma", type=float, default=0.9, help="gamma")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--bc_coef", type=float, default=1e-3, help="bc coef")
parser.add_argument("--tau", type=float, default=0.99, help="tau")
parser.add_argument("--lr", type=float, default=5e-4, help="lr")
parser.add_argument("--dual_lr", type=float, default=0.05, help="dual lr")
parser.add_argument("--traj_avg_len", type=int, default=30, help="traj avg len")
parser.add_argument("--num_train_traj", type=int, default=2000, help="num train traj")
parser.add_argument("--traj_max_len", type=int, default=200, help="traj max len")
parser.add_argument("--traj_min_len", type=int, default=5, help="traj min len")
parser.add_argument("--eval_steps", type=int, default=2000, help="eval steps")
parser.add_argument("--step_noise", type=float, default=1.0, help="step noise")
parser.add_argument("--step_speed", type=float, default=1.0, help="step speed")
parser.add_argument("--obs_noise", type=float, default=0.1, help="obs noise")
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
args = parser.parse_args("")

    
    
# |%%--%%| <NBP7ZdO87J|0Ua3f3Eh9C>




# code to generate paths


def on_cpu(f):
    def wrapped(*args, **kwargs):
        with jax.default_device(jax.devices("cpu")[0]):
            return f(*args, **kwargs)

    return wrapped


@jax.jit
def _step(key, x, g, speed, tempscale, precision, source, vec):
    delta = g - x
    delta = delta / jnp.linalg.norm(delta)
    delta = delta - 0.8 * (vec @ delta) * vec
    delta = delta / jnp.linalg.norm(delta)
    temp = tempscale * jnp.exp(-jnp.linalg.norm(precision * (x - source)) ** 2) + 0.02
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(2,))
    key, subkey = jax.random.split(key)
    x = x + (0.15 * delta + noise * temp) * speed
    return key, x


def path(
    key,
    start,
    goal,
    speed=1.0,
    tempscale=0.5,
    precision=0.1,
    source=(0, 0),
    xnoise=0.0,
    maxiter=1000,
):
    x = jnp.array(start)
    g = jnp.array(goal)
    key, subkey = jax.random.split(key)
    x = x + 0.1 * jax.random.normal(subkey, shape=(2,))
    key, subkey = jax.random.split(key)
    g = g + xnoise * jax.random.normal(subkey, shape=(2,))
    path = [x]
    source = jnp.array(source)
    vec = (g - x) / jnp.linalg.norm(g - x)
    g = g + vec
    for i in range(maxiter - 1):
        key, x = _step(key, x, g, speed, tempscale, precision, source, vec)
        path.append(x)
        if len(path) >= 4 and jnp.linalg.norm(path[-4] - g) <= 0.5:
            break
    return jnp.stack(path)


def multipath(key, *waypoints, maxiter=1000, **kwargs):
    pts = []
    keys = jax.random.split(key, len(waypoints) - 1)
    iters = maxiter // (len(waypoints) - 1) + 1
    for key, start, goal in zip(keys, waypoints[:-1], waypoints[1:]):
        if len(pts) > 0:
            start = pts[-1][-1]
        path_pts = path(key, start, goal, maxiter=iters, **kwargs)
        pts.append(path_pts)
    return jnp.concatenate(pts)


def smooth_path(pts, filter):
    pts = jnp.concatenate(
        [jnp.stack([pts[0]] * (filter - 1)), pts, jnp.stack([pts[-1]] * (filter - 1))],
        axis=0,
    )
    smoothed = jax.vmap(
        lambda x: jnp.convolve(x, jnp.ones(filter) / filter, mode="valid"),
        in_axes=1,
        out_axes=1,
    )(jnp.array(pts))
    return smoothed


def multicurve(key, *waypoints, filter=7, **kwargs):
    pts = []
    waypoints = list(map(jnp.array, waypoints))
    while len(waypoints) > 1:
        while jnp.linalg.norm(waypoints[0] - waypoints[1]) > 2:
            waypoints.insert(1, (waypoints[0] + waypoints[1]) / 2)
        pts.append(waypoints.pop(0))
    pts.append(waypoints.pop(0))
    pts = jnp.stack(pts)
    pts = smooth_path(pts, filter)
    pts = jnp.concatenate([pts[::3], pts[-1:]], axis=0)
    return multipath(key, *pts, **kwargs)


# |%%--%%| <0Ua3f3Eh9C|0td4ff3XvS>

# configure training data

from collections import namedtuple
import tqdm
from joblib import Parallel, delayed
import itertools
import matplotlib as mpl


data = []
all_data = []


def vis_last(cm, num):
    n = int(min(10, num))
    # n = num
    # k = int(min(10, num))
    k = 10
    if n < 5:
        n = 5
    for i, p in enumerate(data[-n:]):
        x, y = p.T
        plt.plot(x, y, c=cm((i + 1) / n))
    off = int(len(all_data[-1]) * 0.3)
    avg_start = np.mean([x[:off][-1] for x in all_data[-k:]], axis=0)
    avg_end = np.mean([x[-off:][0] for x in all_data[-k:]], axis=0)
    direction = avg_end - avg_start
    arr_start = avg_start - 0.3 * direction
    arr_delta = 1.7 * direction
    plt.arrow(
        *arr_start,
        *arr_delta,
        width=3.5,
        zorder=100,
        alpha=0.6,
        fc="gray",
        ec="gray",
        head_width=7,
        head_length=7,
        length_includes_head=True,
        linewidth=0,
    )


def parse_traj(traj, act, n=1000):
    k = len(traj)
    traj = jnp.concatenate([traj, jnp.zeros((n - k, 2))], axis=0)
    act = jnp.concatenate([act, jnp.zeros((n - k, 2))], axis=0)
    # return _parse_traj(traj, act, k)[:k]
    res = _parse_traj(traj, act, k)
    return [x[:k] for x in res]


@jax.jit
def _parse_traj(traj, act, k):
    # return [
    #     (x, len(traj) - i - 1, i == len(traj) - 1, traj, a)
    #     for i, (x, a) in enumerate(zip(traj, act))
    # ]
    def scan_fn(i):
        x = traj[i]
        a = act[i]
        return (x, k - i - 1, i == k - 1, k, a)

    res = jax.vmap(scan_fn)(jnp.arange(len(traj)))
    return res


# |%%--%%| <0td4ff3XvS|3SYUdFpup0>


key = jax.random.key(0)

thunks = []
clusters = []
repeats = 1


def connect(fn, n, cm, *args, **kwargs):
    global key, thunks, clusters
    for i in range(n * repeats):
        key, rng = jax.random.split(key, 2)
        thunk = delayed(fn)(rng, *args, **kwargs)
        thunks.append(thunk)
    clusters.append((n * repeats, cm))


prec = 0.5

connect(path, 30, plt.cm.Blues, [-20, 0], [20, 0], precision=prec)
connect(path, 30, plt.cm.Reds, [0, 20], [0, -20], precision=prec)
connect(
    multicurve,
    15,
    plt.cm.Greens,
    [0, 20],
    [-5, 30],
    [15, 30],
    [30, 15],
    [30, -5],
    [20, 0],
    filter=20,
    precision=prec,
    source=(21, 21),
)
connect(
    multicurve,
    15,
    plt.cm.Purples,
    [0, -20],
    [-30, -20],
    [-30, 20],
    [0, 20],
    filter=40,
    precision=prec,
    source=(-10, 0),
)
connect(
    multicurve,
    15,
    plt.cm.Oranges,
    [-24, 10],
    [-40, 20],
    [-30, 45],
    filter=30,
    precision=prec,
    source=(-24, 10),
)
connect(
    multicurve,
    15,
    plt.cm.Greys,
    [-30, 45],
    [-14, 38],
    [0, 20],
    filter=30,
    precision=prec,
    source=(-10, 10),
)
connect(
    multicurve,
    15,
    plt.cm.YlOrBr,
    [21.5, 22],
    [30, 45],
    [-20, 50],
    [-15, 35],
    filter=30,
    precision=prec,
    source=(21, 21),
)
connect(
    multicurve,
    15,
    plt.cm.Wistia,
    [20, 0],
    [12, -35],
    [-30, -35],
    [-20, 0.5],
    filter=30,
    precision=prec,
    source=(0, 0),
)

# |%%--%%| <3SYUdFpup0|ITnxjZMPWC>

### generate env

key = jax.random.PRNGKey(args.seed)
plt.figure(dpi=200)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
with Parallel(n_jobs=20, verbose=1) as parallel:
    all_results = parallel(thunks)

    for n, cm in clusters:
        for i in range(n):
            traj = all_results.pop(0)
            all_data.append(traj)
            data.append(traj)
        vis_last(cm, n)

plt.gca().set_aspect("equal")
plt.axis("off")
plt.savefig("dataset.png", bbox_inches="tight", pad_inches=0, dpi=400)

# |%%--%%| <ITnxjZMPWC|sQnTazVw5G>


max_len = max(len(traj) for traj in data)
parsed = [parse_traj(traj, jnp.zeros_like(traj), n=max_len) for traj in tqdm.tqdm(data)]
dataset = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *parsed)
supp_obs = dataset[0]
supp_last = dataset[1] + jnp.arange(len(dataset[1]))

# |%%--%%| <sQnTazVw5G|ZNJqaqPwoU>


# plot traj len distribution

plt.hist(dataset[3], bins=50)
plt.xlabel("Trajectory Length")
plt.ylabel("Count")

# |%%--%%| <ZNJqaqPwoU|CcB40V3pxi>

# env config

# subsample dataset and pre-select possible actions to be among 20 nearby point successors

ds_size = 20_000

idx_sampled = jax.random.choice(
    jax.random.PRNGKey(args.seed), len(supp_obs), shape=(ds_size,), replace=False
)
ds_sampled = supp_obs[idx_sampled]


def neighbor(x, ds, supp):
    return jnp.argmin(jnp.linalg.norm(ds[x] - supp, axis=-1))


# |%%--%%| <CcB40V3pxi|bBpEzs0y8b>

### sample actions


def get_actions(x, ds):
    n = 100
    dist = 2.0
    dists = jnp.linalg.norm(ds - x, axis=-1)
    dsort = dists.sort(axis=-1)[:n]
    nearby = dists.argsort(axis=-1)[:n]
    nearby_clip = jnp.where(dsort < 3.0, nearby, jnp.repeat(nearby[:1], n))
    assert nearby_clip.shape == (n,)
    nearby_next = jnp.minimum(nearby_clip + args.asym_shift, supp_last[nearby_clip])
    idx = jnp.vectorize(neighbor, signature="(),(i,k),(j,k)->()")(
        nearby_next, ds, ds_sampled
    )
    nearby_samp = jnp.vectorize(neighbor, signature="(),(i,k),(j,k)->()")(
        nearby, ds, ds_sampled
    )
    return idx, nearby_samp


# sampled_actions = jax.vmap(get_actions, in_axes=(0, None))(ds_sampled, supp_obs)
# sampled_actions = np.vectorize(get_actions, signature="(k),(j,k)->(i)")(ds_sampled, supp_obs)
# sampled_actions = jnp.stack([get_actions(x, supp_obs) for x in tqdm.tqdm(ds_sampled)])
chunks = jnp.array_split(ds_sampled, 300)
sampled_actions, sampled_nearby = jax.tree_map(
    lambda *x: jnp.concatenate(x, axis=0),
    *[
        jax.vmap(get_actions, in_axes=(0, None))(chunk, supp_obs)
        for chunk in tqdm.tqdm(chunks)
    ],
)

#|%%--%%| <bBpEzs0y8b|xa43Ry8Wcm>

plt.figure(dpi=200)
for i in tqdm.trange(9):
    plt.subplot(3, 3, i + 1)
    start = ds_sampled[i]
    next_states = ds_sampled[sampled_actions[i]]
    plt.scatter(*ds_sampled.T, c="blue", s=0.5, alpha=0.5)
    for x in next_states:
        plt.arrow(*start, *(x - start), width=0.07, facecolor="red", head_width=0.17, head_length=0.17, alpha=0.25, edgecolor="k", linewidth=0.3)

    window = 4
    plt.xlim(start[0] - window, start[0] + window)
    plt.ylim(start[1] - window, start[1] + window)

plt.tight_layout()

# |%%--%%| <xa43Ry8Wcm|HkbkYsrtYr>

# eval code

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


# |%%--%%| <HkbkYsrtYr|hoGGcH8kss>


### generate training data

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
#|%%--%%| <hoGGcH8kss|dcCzydtwAS>

cache = dict(
    ds_sampled=ds_sampled,
    sampled_actions=sampled_actions,
    sampled_nearby=sampled_nearby,
    dataset_unstacked=dataset_unstacked,
    ds_unstack=ds_unstack,
    dataset=dataset,
    ds_obs=ds_obs,
    ds_last=ds_last,
    ds_acts=ds_acts,
)

import pickle

with open("cache.pkl", "wb") as f:
    pickle.dump(cache, f)



# |%%--%%| <dcCzydtwAS|ppDCxHdZDT>


plt.figure(figsize=(10, 10))
# plt.scatter(*ds_sampled[:3000].T, s=0.1)

for i in tqdm.trange(9):
    plt.subplot(3, 3, i + 1)
    start = ds_sampled[i]
    env = Pointmaze(key, ds_sampled, sampled_actions, sampled_nearby)
    keys = jax.random.split(key, 50)

    def stepfn(a, k):
        env.reset(key=k, start=start)
        return env.step(a)[0]

    actions_samp = jax.random.normal(jax.random.key(i), (1, 2))
    actions_samp = normalize(actions_samp)
    next_states = jax.vmap(stepfn, in_axes=(None, 0))(actions_samp, keys)
    mn = jnp.mean(next_states - start, axis=0) 
    for x, a in zip(next_states, jnp.repeat(actions_samp, len(keys), axis=0)):
        plt.arrow(*start, *(x - start), width=0.01, facecolor="red", head_width=0.05, head_length=0.05, lw=0.1, edgecolor="k")
        plt.arrow(*start, *a/2, width=0.01, facecolor="green", head_width=0.05, head_length=0.05, lw=0.1, edgecolor="k")
        plt.arrow(*start, *mn, width=0.01, facecolor="purple", head_width=0.05, head_length=0.05, lw=0.1, edgecolor="k")

    plt.scatter(*ds_sampled.T, c="blue", s=1)
    window = 1
    plt.xlim(start[0] - window, start[0] + window)
    plt.ylim(start[1] - window, start[1] + window)

plt.tight_layout()

# |%%--%%| <ppDCxHdZDT|5fq8qwGyOV>


plt.figure(figsize=(10, 10))
plt.scatter(*ds_sampled[:3000].T, s=0.1)
for traj in ds_unstack[0][:50]:
    x, y = traj.T
    plt.plot(x, y, linewidth=1)
    # plt.scatter(x, y, c=jnp.arange(len(traj)), s=1)



