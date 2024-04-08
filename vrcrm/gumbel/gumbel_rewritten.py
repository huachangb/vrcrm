from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape: Tuple[int], device: torch.cuda.device, eps:float = 1e-20) -> torch.Tensor:
    U = torch.rand(shape)
    samples = -torch.log(-torch.log( U + eps) + eps).to(device)
    return samples


def gumbel_softmax_sample(logits: torch.Tensor, device: torch.cuda.device, temperature: float) -> torch.Tensor:
    # logits : [bs, 2, n_labels]
    samples = Variable(sample_gumbel(shape=logits.size(), device=device))
    y = logits + samples
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits: torch.Tensor, temperature: float, device: torch.cuda.device, hard: bool = False):
    # y: [bs, 2, n_labels]
    y = gumbel_softmax_sample(logits=logits, device=device, temperature=temperature)

    if not hard:
        return y

    max_val, _ = torch.max(y, y.dim()-2,keepdim=True)
    y_hard = (y == max_val).expand(y.size())
    y_hard = y_hard.type(torch.FloatTensor).to(device)
    y_new = (y_hard - y).detach() + y
    return y_new
