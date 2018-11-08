import pickle

import numpy as np
import matplotlib.pyplot as plt
import os


ENVS = ["InvertedPendulum-v2", "Hopper-v2", "InvertedDoublePendulum-v2", "Reacher-v2",
        "Swimmer-v2", "Walker2d-v2", "HalfCheetah-v2",
        'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2']
SEEDS = [int(i) for i in range(5)]
SEED = 'SEED'
ENV = 'ENV'
E = 'E'
LAM = 'LAM'
MY_GRAPH_L = 'Ours'
OA_GRAPH_L = 'PPO'
X_AXIS = 'Number of timesteps'
Y_AXIS = 'Mean episode reward of the last 100 episodes'
RAND_L = 'Random Policy'
OA_COLOR = [1.0, 0.0, 0.0, 1.0]
RAND_COLOR = [0.0, 0.0, 0.0, 1.0]
MY_COLOR = [0.0, 0.0, 1.0, 1.0]


def load_pkl_f(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)


def get_x(d):
    return [i[0] for i in d]


def get_y(d):
    return [i[1] for i in d]


def get_faded_color(color):
    import copy
    f = copy.deepcopy(color)
    f[-1] = 0.08
    return f


def plot_mul_seed_res(reses, label, color):
    avgs = []
    maxs = []
    mins = []

    for seeds_res in zip(*reses):

        all(i[0] == seeds_res[0][0] for i in seeds_res)  # Assert that all reward are from the same ts

        rews = [i[1] for i in seeds_res]
        avg = np.mean(rews)

        avgs.append((seeds_res[0][0], avg))

        std = np.std(rews)

        maxs.append(avg + std)

        mins.append(avg - std)

    x = get_x(avgs)
    y = get_y(avgs)

    faded_color = get_faded_color(color)

    plt.plot(x, y, color=color, label=label)
    plt.plot(x, maxs, color=faded_color)
    plt.plot(x, mins, color=faded_color)
    plt.fill_between(x, mins, maxs, color=faded_color)


def get_oa_rand_res(dir, num_seed):
    """Get results for each seed for each environment in this dir"""

    reses = {}
    for env in ENVS:
        reses[env] = []

    for env_i, env in enumerate(ENVS):
        start_i = env_i * num_seed
        for seed_i in range(start_i, start_i + num_seed):
            path = dir + 'res_' + str(seed_i) + '.pickled'
            res = load_pkl_f(path)

            reses[env].append(res)

    return reses


def get_avg_fin_rew(seeds_res):
    avg_reses = {}
    for env in ENVS:
        final_rew = [item[-1][1] for item in seeds_res[env]]  # -1 is the last time step. 1 is the rew.

        avg_res = np.mean(final_rew)
        avg_reses[env] = avg_res

    return avg_reses


def get_pc_imp(my_res_all_env, random_res, ppo_res, env):
    rand_avg_fin_rew = get_avg_fin_rew(random_res)
    oa_avg_fin_rew = get_avg_fin_rew(ppo_res)
    my_res_fin_rew = get_avg_fin_rew(my_res_all_env)

    rand_avg_fin_rew = rand_avg_fin_rew[env]
    oa_avg_fin_rew = oa_avg_fin_rew[env]
    my_res_fin_rew = my_res_fin_rew[env]

    # Calculate percentage improve for each environment and across all envs
    oa_abs_imp = np.abs(oa_avg_fin_rew - rand_avg_fin_rew)
    my_abs_imp = np.abs(my_res_fin_rew - rand_avg_fin_rew)

    pc_imp = (my_abs_imp - oa_abs_imp) * 100.0 / oa_abs_imp

    return round(pc_imp, 2)


def get_pc_imps(reses, random_res, ppo_res, envs):
    pc_imps = {}

    avg_pc_imp = 0.0
    num_better = 0.0

    for env in envs:
        pc_imp = get_pc_imp(reses, random_res, ppo_res, env)
        pc_imps[env] = pc_imp

        avg_pc_imp += pc_imp

        if pc_imp > 0.0:
            num_better += 1

    pc_imps['avg_pc_imp'] = round(avg_pc_imp / float(len(envs)), 2)

    return pc_imps, num_better


def compare_against_baseline(random_res, ppo_res, envs):
    reses = {}

    # Only compare on envs where we have the results
    compare_envs = []

    for env in envs:
        reses[env] = []
        for seed in SEEDS:
            try:
                env_seed_res = load_pkl_f(f'res/{env}/seed_{seed}/avg_episode_returns_buffer.pkl')
                reses[env].append(env_seed_res)
                compare_envs.append(env)
            except FileNotFoundError:
                pass

    envs = compare_envs

    # Obtain percent improvement
    pc_imps, num_env_better = get_pc_imps(reses, random_res, ppo_res, envs)

    # Create folder to contain graphs
    os.makedirs('graphs/', exist_ok=True)

    # Plot comparison against OpenAI and random policy for each environment
    for env in envs:
        plt.clf()

        rand_pol_res = random_res[env]
        oa_res = ppo_res[env]

        my_res_env = reses[env]

        plot_mul_seed_res(my_res_env, label=MY_GRAPH_L, color=MY_COLOR)
        plot_mul_seed_res(rand_pol_res, label=RAND_L, color=RAND_COLOR)
        plot_mul_seed_res(oa_res, label=OA_GRAPH_L, color=OA_COLOR)

        plt.legend()
        plt.xlabel(X_AXIS)
        plt.ylabel(Y_AXIS)

        pc_imp = pc_imps[env]

        graph_title = [env, 'Improved by: ' + str(pc_imp) + '%',
                       'Averaged improvement over all envs: ' + str(pc_imps['avg_pc_imp']) + '%',
                       'Better in ' + str(num_env_better) + 'environment out of ' + str(len(envs))]

        # graph_title = [env, 'Improved by: ' + str(pc_imp) + '% over PPO']

        graph_title = '\n'.join(graph_title)
        plt.title(graph_title)

        graph_f = f'graphs/{env}'
        plt.savefig(graph_f, bbox_inches='tight')

    return pc_imps


if __name__ == '__main__':

    random_res = get_oa_rand_res('baselines/random/', len(SEEDS))
    ppo_res = get_oa_rand_res('baselines/ppo1/', len(SEEDS))

    pc_imps = compare_against_baseline(random_res, ppo_res, ENVS)

    print(pc_imps)
