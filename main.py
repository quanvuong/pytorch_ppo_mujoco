from utils import get_args_parser, make_mujoco_env, traj_seg_gen, RunningMeanStd, weights_init
from models import Policy, ValueNet
from train import one_train_iter

import torch
import torch.optim as optim

import os
import time
from collections import deque
import pickle

import matplotlib.pyplot as plt

TIMER_NUM_ITER = 1


def main():

    try:
        nt = int(os.environ['OMP_NUM_THREADS'])
        torch.set_num_threads(nt)
    except KeyError:
        torch.set_num_threads(1)

    args = get_args_parser().parse_args()
    env = make_mujoco_env(args.env, args.seed)

    # Construct policy and value network
    pol = Policy(env.observation_space, env.action_space, args)
    pol_optim = optim.Adam(pol.parameters(), lr=args.optim_stepsize, eps=args.adam_epsilon)

    old_pol = Policy(env.observation_space, env.action_space, args)
    old_pol.load_state_dict(pol.state_dict())
    # Do not calculate grad with respect to param of old pol
    for param in old_pol.parameters():
        param.requires_grad = False

    val = ValueNet(env.observation_space, args)
    val_optim = optim.Adam(val.parameters(), lr=args.optim_stepsize, eps=args.adam_epsilon)

    optims = {'pol_optim': pol_optim, 'val_optim': val_optim}

    num_train_iter = int(args.max_timesteps / args.ts_per_batch)

    start = time.time()

    # Buffer for running statistics
    eps_rets_buff = deque(maxlen=100)
    eps_rets_mean_buff = []

    state_running_m_std = RunningMeanStd(shape=env.observation_space.shape)

    # seg_gen is a generator that yields the training data points
    seg_gen = traj_seg_gen(env, pol, val, state_running_m_std, args)

    for iter_i in range(num_train_iter):
        print('\nStarting training iter', iter_i)
        one_train_iter(pol, old_pol, val, optims,
                       iter_i, eps_rets_buff, eps_rets_mean_buff, seg_gen,
                       state_running_m_std,
                       args)
        print('Finished training iter', iter_i, ' Taken: ', round(time.time() - start, 3), ' seconds so far')

        if args.timer_code and iter_i == TIMER_NUM_ITER:
            break

    if args.timer_code:
        print('Will take ', ((time.time() - start) * num_train_iter / TIMER_NUM_ITER) / (60 * 60),
              'hours to finish training ', args.max_timesteps, ' timesteps')

    file_prefix = args.env + '_eps_rets_mean_buff_' + args.new_config

    with open(file_prefix + '.pickled', 'wb+') as f:
        pickle.dump(eps_rets_mean_buff, f)

        x = [i[0] for i in eps_rets_mean_buff]
        y = [i[1] for i in eps_rets_mean_buff]

        plt.plot(x, y)
        plt.savefig(file_prefix + '.png')


if __name__ == '__main__':
    main()