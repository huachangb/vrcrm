import torch

from ..models import Policy


def expected_loss(model: Policy, n_samples: int, X, labels):
    probs = model(X) # [n, labels]
    mask = torch.rand(n_samples, *probs.shape) # [n_samples, n, labels]
    samples = mask <= probs # [n_samples, n, labels]
    samples = samples.int()
    hamming_loss = (samples != labels).sum(dim=2) # [n_samples, n]
    avg_hamming_loss = hamming_loss.float().mean(dim=0) # [n]
    return avg_hamming_loss.mean()


def MAP_loss(model: Policy, X, labels):
    probs = model(X) # [n, labels]
    map_predictions = (probs >= 0.5).int()
    avg_hamming_loss = (map_predictions != labels).sum() / X.shape[0]
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



