from utils import weights_init

import torch
import torch.nn as nn

import numpy as np


class Policy(nn.Module):

    def __init__(self, ob_space, ac_space, args):
        super().__init__()

        self.l_in = nn.Linear(ob_space.shape[0], args.hid_size)
        self.l1 = nn.Linear(args.hid_size, args.hid_size)
        self.l_out = nn.Linear(args.hid_size, ac_space.shape[0])

        self.std = nn.Parameter(torch.tensor([[args.pol_init_std] * ac_space.shape[0]], dtype=torch.float32))

        for name, c in self.named_children():
            if name == 'l_out':
                weights_init(c, 0.01)
            else:
                weights_init(c, 1.0)

    def forward(self, x):

        x = torch.tanh(self.l_in(x))
        x = torch.tanh(self.l1(x))
        mean = self.l_out(x)

        ac = torch.normal(mean, self.std.expand(mean.shape[0], -1))

        return ac, mean

    def neglogp(self, states, acs):

        _, mean = self.forward(states)

        ac_size = acs.size()[-1]

        return 0.5 * torch.sum(((acs - mean) / self.std)**2, dim=-1, keepdim=True) + \
               0.5 * np.log(2.0 * np.pi) * float(ac_size) + \
               torch.sum(torch.log(self.std), dim=-1)

    def logp(self, state, ac):
        return - self.neglogp(state, ac)

    def prob(self, state, ac):
        return torch.exp(self.logp(state, ac))


class ValueNet(nn.Module):

    def __init__(self, ob_space, args):

        super().__init__()

        self.l_in = nn.Linear(ob_space.shape[0], args.hid_size)
        self.l1 = nn.Linear(args.hid_size, args.hid_size)
        self.l_out = nn.Linear(args.hid_size, 1)

        for c in self.children():
            weights_init(c, 1.0)

    def forward(self, x):

        x = torch.tanh(self.l_in(x))
        x = torch.tanh(self.l1(x))
        x = self.l_out(x)

        return x
