import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from ..models import Policy
from ..gumbel.gumbel_multilabel import gumbel_softmax

def expected_loss(model: Policy, n_samples: int, X, labels):
    probs = model(X) # [n, labels]
    mask = torch.rand(n_samples, *probs.shape) # [n_samples, n, labels]
    samples = mask <= probs # [n_samples, n, labels]
    samples = samples.int()
    correct = (samples != labels).sum()
    return correct / (X.shape[0] * n_samples)


def MAP(model: Policy, X, labels):
    probs = model(X) # [n, labels]
    map_predictions = (probs >= 0.5).int()
    avg_hamming_loss = (map_predictions != labels).sum(dim=1).float().mean()
    return avg_hamming_loss


# def expected_loss(model: Policy, n_samples: int, X, labels):
#     probs = gumbel_softmax(model(X), 1)
#     cat_distr = Categorical(probs)
#     sampled_actions = cat_distr.sample([n_samples]) # [n_samples, n]
#     sampled_actions = F.one_hot(sampled_actions) # [n_samples, n, labels]
#     hamming_loss = (sampled_actions != labels).sum()
#     return hamming_loss / (X.shape[0] * n_samples)


# def MAP(model: Policy, X, labels):
#     probs = gumbel_softmax(model(X), 1)
#     actions = torch.argmax(probs, dim=1)
#     actions = F.one_hot(actions)
#     hamming_loss = (actions != labels).sum()
#     return hamming_loss / X.shape[0]



