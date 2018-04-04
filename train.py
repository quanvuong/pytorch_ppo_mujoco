from wrappers import FloatTensorFromNumpyVar
from utils import Dataset

import torch
import torch.nn.functional as F
import numpy as np


def c_nonzero(var):
    return np.count_nonzero(var.data.numpy())


def update_params(m_b, pol, val, optims, args):

    keys = ('obs', 'acs', 'vtargs', 'atargs', 'pold')
    obs, acs, vtargs, atargs, pold = (FloatTensorFromNumpyVar(m_b[i]) for i in keys)

    vtargs = vtargs.view(-1, 1)
    atargs = atargs.view(-1, 1)

    # Calculate policy surrogate objective
    pnew = pol.prob(obs, acs, args)

    ratio = pnew / pold

    # Detach m_advs for surr loss so we don't have to backwards pass through those.
    surr1 = ratio * atargs
    surr2 = torch.clamp(ratio, 1.0 - args.clip_param_annealed, 1.0 + args.clip_param_annealed) * atargs

    mask = (surr1 > surr2).detach()

    if c_nonzero(mask) > 0:
        surr1[mask] = 0.0

    pol_surr = - torch.sum(surr1) / obs.size()[0]

    # Calculate value function loss
    val_loss = F.mse_loss(val(obs), vtargs)

    optims['pol_optim'].zero_grad()
    optims['val_optim'].zero_grad()

    total_loss = pol_surr + val_loss
    total_loss.backward(retain_graph=True)

    optims['pol_optim'].step()
    optims['val_optim'].step()


def change_lr(optim, new_lr):
    assert new_lr > 0
    for param_group in optim.param_groups:
        param_group['lr'] = new_lr


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


def one_train_iter(pol, old_pol, val, optims,
                   iter_i, eps_rets_buff, eps_rets_mean_buff, seg_gen,
                   state_running_m_std,
                   args):

    # Anneal lr and clip param
    num_ts_so_far = iter_i * args.ts_per_batch
    lr_mult = max(1.0 - float(num_ts_so_far) / args.max_timesteps, 0)
    args.clip_param_annealed = args.clip_param * lr_mult
    change_lr(optims['pol_optim'], args.optim_stepsize * lr_mult)
    change_lr(optims['val_optim'], args.optim_stepsize * lr_mult)

    # Sync params
    old_pol.load_state_dict(pol.state_dict())

    # Obtain training batch
    seg = seg_gen.__next__()

    eps_rets_buff.extend(seg['ep_rets'])
    eps_rets_mean_buff.append((num_ts_so_far, np.mean(eps_rets_buff)))
    print('Last 100 episodes mean rets', np.mean(eps_rets_buff))
    print('Finished getting batch data')

    add_vtarg_and_adv(seg, args)
    print('Finished calculating advantages and td lambda return')

    seg['advs'] = (seg['advs'] - seg['advs'].mean()) / seg['advs'].std()
    pold = old_pol.prob(FloatTensorFromNumpyVar(seg['obs']), FloatTensorFromNumpyVar(seg['acs']), args).data.numpy()

    batch = Dataset(dict(obs=seg['obs'], acs=seg['acs'], atargs=seg['advs'], vtargs=seg['tdlamrets'], pold=pold))

    for epoch_i in range(args.optim_epoch):
        for m_b in batch.iterate_once(args.optim_batch_size):
            update_params(m_b, pol, val, optims, args)

    # Update running mean and std of states
    state_running_m_std.update(seg['obs'])
