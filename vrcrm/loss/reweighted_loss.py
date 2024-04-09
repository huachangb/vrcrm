import torch

from vrcrm.network import Policy


def reweighted_loss(
        policy: Policy,
        X: torch.Tensor,
        y: torch.Tensor,
        log_prop: torch.Tensor,
        loss: torch.Tensor
    ) -> torch.tensor:
    """
    Computes reweighted loss. Assumes labels are drawn
    i.i.d., so that p(y_1, ..., y_n | x) = P(y_1 | x) * ... * p(y_n | x)

    Parameters
    :policy: policy to evaluate
    :X: context matrix of shape [batch size, features]
    :y: label matrix, where each row is a bitvector
    :log_prop: log of propensity scores
    :loss: feedback of actions

    Returns
    :loss: reweighted loss
    """
    probs = policy(X)
    probs = (probs * y).prod(dim=1)
    loss = torch.mean((probs / log_prop) * loss)
    return loss
