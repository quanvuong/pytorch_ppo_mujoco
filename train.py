from utils import Dataset

import torch
import torch.nn.functional as F
import numpy as np


def update_params(m_b, pol, val, optims, args):

    keys = ('obs', 'acs', 'vtargs', 'atargs', 'pold')
    obs, acs, vtargs, atargs, pold = (torch.from_numpy(m_b[i]).float() for i in keys)

    vtargs = vtargs.view(-1, 1)
    atargs = atargs.view(-1, 1)

    # Calculate policy surrogate objective
    pnew = pol.prob(obs, acs)

    ratio = pnew / pold

    surr1 = ratio * atargs
    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * atargs
    pol_surr, _ = torch.min(torch.cat((surr1, surr2), dim=1), dim=1)
    pol_surr = - torch.sum(pol_surr) / obs.size()[0]

    # Calculate value function loss
    val_loss = F.mse_loss(val(obs), vtargs)

    optims['pol_optim'].zero_grad()
    optims['val_optim'].zero_grad()

    total_loss = pol_surr + val_loss
    total_loss.backward(retain_graph=True)

    optims['pol_optim'].step()
    optims['val_optim'].step()


def add_vtarg_and_adv(seg, args):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    lam = args.lam
    gamma = args.gamma

    news = np.append(seg["news"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpreds = np.append(seg["vpreds"], seg["nextvpred"])
    T = len(seg["rews"])
    seg["advs"] = gaelam = np.empty(T, 'float32')
    rews = seg["rews"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-news[t+1]
        delta = rews[t] + gamma * vpreds[t+1] * nonterminal - vpreds[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamrets"] = seg["advs"] + seg["vpreds"]


def one_train_iter(pol, val, optims,
                   iter_i, eps_rets_buff, eps_rets_mean_buff, seg_gen,
                   state_running_m_std,
                   args):

    num_ts_so_far = iter_i * args.ts_per_batch

    # Obtain training batch
    seg = seg_gen.__next__()

    # Update running mean and std of states
    state_running_m_std.update(seg['obs'])

    eps_rets_buff.extend(seg['ep_rets'])
    eps_rets_mean_buff.append((num_ts_so_far, np.mean(eps_rets_buff)))
    print('Last 100 episodes mean returns:', np.mean(eps_rets_buff))

    add_vtarg_and_adv(seg, args)

    seg['advs'] = (seg['advs'] - seg['advs'].mean()) / seg['advs'].std()
    pold = pol.prob(torch.from_numpy(seg['obs']).float(),
                    torch.from_numpy(seg['acs']).float()).data.numpy()

    batch = Dataset(dict(obs=seg['obs'], acs=seg['acs'], atargs=seg['advs'], vtargs=seg['tdlamrets'], pold=pold))

    for epoch_i in range(args.optim_epoch):
        for m_b in batch.iterate_once(args.optim_batch_size):
            update_params(m_b, pol, val, optims, args)

