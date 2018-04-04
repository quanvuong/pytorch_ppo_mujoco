from wrappers import FloatTensorFromNumpyVar
from utils import Dataset

import torch
import torch.nn.functional as F
import numpy as np


def rearrange(data, mask_ve, mask_ne, size):
    return torch.cat((data[mask_ve].view(-1, size), data[mask_ne].view(-1, size)))


def linearly_decay(tensor, max_val, reverse_tensor_sign=False):
    assert type(max_val) == float

    decay_f = tensor.clone().detach() - 1.0  # -1 to shift the beginning to x = 0
    if reverse_tensor_sign:
        decay_f = - decay_f # To decay value < 1.0

    decay_f = 1.0 - decay_f / max_val
    decay_f[decay_f <= 0.0] = 0.0

    return tensor * decay_f


def c_nonzero(var):
    return np.count_nonzero(var.data.numpy())


def update_params(m_b, pol, val, optims, args):

    keys = ('obs', 'acs', 'vtargs', 'atargs', 'pold')
    obs, acs, vtargs, atargs, pold = (FloatTensorFromNumpyVar(m_b[i]) for i in keys)

    vtargs = vtargs.view(-1, 1)
    atargs = atargs.view(-1, 1)

    # Calculate policy surrogate objective
    pnew = pol.prob(obs, acs, args)
    old_vals = obs, acs, args

    ratio = pnew / pold

    # Linearly decay in range (1 - beta * eps, 1 + beta * eps). Decay from 1 to either endpoint
    # Zero otherwise
    # beta is 3.0 works best out of [1.0, 2.0, 3.0, 4.0]
    # The graph in the repo is for 4.0
    beta = 3.0

    mask_ve = (ratio >= 1.0).detach()
    mask_ne = (ratio < 1.0).detach()

    # Obtained decayed ratio values
    ve_size = c_nonzero(mask_ve)
    ne_size = c_nonzero(mask_ne)

    if ve_size > 0 and ne_size > 0:
        obs = rearrange(obs, mask_ve, mask_ne, obs.shape[-1])
        acs = rearrange(acs, mask_ve, mask_ne, acs.shape[-1])
        vtargs = rearrange(vtargs, mask_ve, mask_ne, vtargs.shape[-1])
        atargs = rearrange(atargs, mask_ve, mask_ne, atargs.shape[-1])
        ratio = rearrange(ratio, mask_ve, mask_ne, ratio.shape[-1])

        ve = ratio[:ve_size]
        ve_decayed = linearly_decay(ve, max_val=beta * args.clip_param_annealed)

        ne = ratio[ve_size:]
        ne_decayed = linearly_decay(ne, max_val=beta * args.clip_param_annealed, reverse_tensor_sign=True)

        ratio = torch.cat((ve_decayed, ne_decayed))

    elif ve_size > 0:
        assert ne_size == 0

        ratio = linearly_decay(ratio, max_val=beta * args.clip_param_annealed)

    elif ne_size > 0:
        assert ve_size == 0

        ratio = linearly_decay(ratio, max_val=beta * args.clip_param_annealed, reverse_tensor_sign=True)

    else:
        print('Unrecognized branch')
        print('ratio', ratio)
        print('mask_ve', mask_ve)
        print('mask_ne', mask_ne)
        print('pnew', pnew)
        print('pold', pold)
        print('old_vals', old_vals)
        for m in pol.parameters():
            print(m)

        import sys
        sys.exit(0)

    # Ensure that no value in ratio goes out of the range
    surr1 = ratio * atargs
    pol_surr = - torch.sum(surr1) / obs.size()[0]

    # Calculate value function loss
    val_loss = F.mse_loss(val(obs), vtargs)

    optims['pol_optim'].zero_grad()
    optims['val_optim'].zero_grad()

    total_loss = pol_surr + val_loss
    total_loss.backward(retain_graph=True)

    # detect nan in grad update and clip gradient
    for m in pol.parameters():
        m.grad[m.grad != m.grad] = 0.0
        m.grad.data.clamp_(-1.0, 1.0)

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
