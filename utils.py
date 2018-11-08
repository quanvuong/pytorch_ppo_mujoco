import argparse
import os

import torch
import numpy as np
import random

from wrappers import FloatTensorFromNumpyVar, FloatTensorFromNumpy

import gym


def set_global_seeds(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.seed_all(seed)


def set_torch_num_threads():

    try:
        nt = int(os.environ['OMP_NUM_THREADS'])
        torch.set_num_threads(nt)
    except KeyError:
        torch.set_num_threads(1)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='InvertedPendulum-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-timesteps', type=int, default=int(1e6))

    parser.add_argument('--pol-fixed-std', type=bool, default=True)  # Fixed across action dimension
    parser.add_argument('--pol-init-std', type=float, default=1.0)

    parser.add_argument('--hid-size', type=int, default=64)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--adam-epsilon', type=float, default=1e-5)

    parser.add_argument('--ts-per-batch', type=int, default=2048)
    parser.add_argument('--optim-epoch', type=int, default=10)
    parser.add_argument('--optim-stepsize', type=float, default=3e-4)
    parser.add_argument('--optim-batch-size', type=int, default=64)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument('--render', type=bool, default=False)
    return parser


def make_mujoco_env(env_id, seed):
    set_global_seeds(seed)
    env = gym.make(env_id)
    env.seed(seed)
    return env


def traj_seg_gen(env, pol, val, state_running_m_std, args):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(args.ts_per_batch)])
    rews = np.zeros(args.ts_per_batch, 'float32')
    vpreds = np.zeros(args.ts_per_batch, 'float32')
    news = np.zeros(args.ts_per_batch, 'int32')
    acs = np.array([ac for _ in range(args.ts_per_batch)])

    while True:
        ob = np.clip((ob - state_running_m_std.mean) / state_running_m_std.std, -5.0, 5.0)
        t_state = FloatTensorFromNumpyVar(np.expand_dims(ob, 0), requires_grad=False)

        ac, _ = pol(t_state, args)
        ac = ac.data.numpy()
        vpred = val(t_state).data.numpy()

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % args.ts_per_batch == 0:
            yield {"obs": obs, "rews": rews, "vpreds": vpreds, "news": news,
                   "acs": acs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        i = t % args.ts_per_batch
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def weights_init(c, std):

    out = np.random.randn(c.out_features, c.in_features).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))

    c.weight.data.copy_(FloatTensorFromNumpy(out))
    c.bias.data.fill_(0)


class Dataset(object):
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        x = x.astype('float64')
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(np.maximum(new_var, 1e-2))
        self.count = new_count