from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, dtype=torch.float32, device=device)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-20):
    y = torch.log(logits + eps) + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def hard_gumbel_softmax_sample(logits, temperature, eps=1e-20):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, eps)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

if __name__ == '__main__':
    logits = np.array([0.5, 5*0.04, 5*0.03, 5*0.02, 5*0.01], dtype=np.float32)
    for temperature in [0.01, 0.1, 1.0, 10.0]:
        samples = gumbel_softmax_sample(Variable(torch.Tensor(np.tile(logits, [10, 1]), device=device)), temperature)
        print(samples)
