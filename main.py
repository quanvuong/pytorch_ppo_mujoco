import os

from utils import get_args_parser, make_mujoco_env, traj_seg_gen, RunningMeanStd, set_torch_num_threads, set_global_seeds
from models import Policy, ValueNet
from train import one_train_iter

import torch.optim as optim

from collections import deque
import pickle

from tqdm import trange


def main():

    set_torch_num_threads()

    args = get_args_parser().parse_args()

    # Create environment and set global seeds
    env = make_mujoco_env(args.env, args.seed)
    set_global_seeds(args.seed)

    print(f'args: {args}\n')

    # Construct policy and value network
    pol = Policy(env.observation_space, env.action_space, args)
    pol_optim = optim.Adam(pol.parameters(), lr=args.optim_stepsize, eps=args.adam_epsilon)

    val = ValueNet(env.observation_space, args)
    val_optim = optim.Adam(val.parameters(), lr=args.optim_stepsize, eps=args.adam_epsilon)

    optims = {'pol_optim': pol_optim, 'val_optim': val_optim}

    num_train_iter = int(args.max_timesteps / args.ts_per_batch)

    # Buffer for running statistics
    eps_rets_buff = deque(maxlen=100)
    eps_rets_mean_buff = []

    state_running_m_std = RunningMeanStd(shape=env.observation_space.shape)

    # seg_gen is a generator that yields the training data points
    seg_gen = traj_seg_gen(env, pol, val, state_running_m_std, args)

    for iter_i in trange(num_train_iter):
        print('\nStarting training iter', iter_i)
        one_train_iter(pol, val, optims,
                       iter_i, eps_rets_buff, eps_rets_mean_buff, seg_gen,
                       state_running_m_std,
                       args)
        print()

    # Save training result
    save_dir = f'res/{args.env}/seed_{args.seed}'

    os.makedirs(save_dir, exist_ok=True)

    save_dir = f'{save_dir}/avg_episode_returns_buffer.pkl'

    with open(save_dir, 'wb+') as f:
        pickle.dump(eps_rets_mean_buff, f)


if __name__ == '__main__':
    main()
