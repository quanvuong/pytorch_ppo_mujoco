import torch
from torch.autograd import Variable

FloatTensor = lambda x: torch.FloatTensor(x)
FloatTensorFromNumpy = lambda x: torch.from_numpy(x).float()

ZeroTensor = lambda *s: torch.zeros(*s)
ZeroTensorVar = lambda *tensor: Variable(ZeroTensor(*tensor))

ByteTensor = lambda x: torch.ByteTensor(x)
ByteTensorFromNumpy = lambda x: torch.from_numpy(x).byte()

IntTensor = lambda x: torch.from_numpy(x).int()
IntTensorVar = lambda x: Variable(IntTensor(x))


def FloatTensorVar(x, requires_grad=False):
    return Variable(FloatTensor(x), requires_grad=requires_grad)


def FloatTensorFromNumpyVar(x, requires_grad=False):
    return Variable(FloatTensorFromNumpy(x), requires_grad=requires_grad)


def ByteTensorVar(x):
    return Variable(ByteTensor(x))


def ByteTensorFromNumpyVar(x):
    return Variable(ByteTensorFromNumpy(x))