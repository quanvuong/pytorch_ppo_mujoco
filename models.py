from utils import weights_init
from wrappers import FloatTensorFromNumpyVar

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
import torch.nn.functional as F

import numpy as np


class Policy(nn.Module):

    def __init__(self, ob_space, ac_space, args):
        super().__init__()

        self.l_in = nn.Linear(ob_space.shape[0], args.hid_size)
        self.l1 = nn.Linear(args.hid_size, args.hid_size)
        self.l_out = nn.Linear(args.hid_size, ac_space.shape[0])

        self.std = nn.Parameter(torch.FloatTensor([[args.pol_init_std] * ac_space.shape[0]]))

        for name, c in self.named_children():
            if name == 'l_out':
                weights_init(c, 0.01)
            else:
                weights_init(c, 1.0)

    def forward(self, x, args):

        x = F.tanh(self.l_in(x))
        x = F.tanh(self.l1(x))
        mean = self.l_out(x)

        ac = torch.normal(mean, self.std.expand(mean.shape[0], -1))

        return ac, mean

    def neglogp(self, states, acs, args):

        _, mean = self.forward(states, args)

        ac_size = acs.size()[-1]

        return 0.5 * torch.sum(((acs - mean) / self.std)**2, dim=-1, keepdim=True) + \
               0.5 * np.log(2.0 * np.pi) * float(ac_size) + \
               torch.sum(torch.log(self.std), dim=-1)

    def logp(self, state, ac, args):
        return - self.neglogp(state, ac, args)

    def prob(self, state, ac, args):
        return torch.exp(self.logp(state, ac, args))


class ValueNet(nn.Module):

    def __init__(self, ob_space, args):

        super().__init__()

        self.l_in = nn.Linear(ob_space.shape[0], args.hid_size)
        self.l1 = nn.Linear(args.hid_size, args.hid_size)
        self.l_out = nn.Linear(args.hid_size, 1)

        for c in self.children():
            weights_init(c, 1.0)

    def forward(self, x):

        x = F.tanh(self.l_in(x))
        x = F.tanh(self.l1(x))
        x = self.l_out(x)

        return x
