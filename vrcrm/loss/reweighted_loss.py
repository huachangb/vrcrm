from typing import Any, List

import torch

from vrcrm.network import Policy


def reweighted_loss(
        policy: Policy,
        X: torch.Tensor,
        y: torch.Tensor,
        log_prop: torch.Tensor,
        loss: torch.Tensor,
        labels: List[Any]
    ) -> torch.tensor:
    """
    Computes reweighted loss

    Parameters
    -----------------------------------
    :policy:
    :X:
    :y:
    :log_prop:
    :loss:
    :labels:

    Returns
    -----------------------------------
    :loss: reweighted loss
    """
    return 0
