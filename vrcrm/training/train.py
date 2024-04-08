import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vrcrm.loss import reweighted_loss
from vrcrm.network import Policy, T


def train(
        policy: Policy,
        t: T,
        bandit_dataloader: DataLoader,
        fgan_dataloader: DataLoader,
        max_epochs: int,
        max_it: int,
        reweighting_lr: float,
        divergence_min_lr: float,
        rho: float,
        device: torch.cuda.device
    ) -> None:
    """
    Implements algorithm 2 in http://proceedings.mlr.press/v80/wu18g/wu18g-supp.pdf

    Arguments
    :bandit_dataloader: dataloader for bandit dataset.
    :fgan_dataloader: dataloader for fgan dataset
    :max_epochs: max epochs for reweighting
    :max_iterations: max iterations for divergence minimization
    :reweighting_lr: learning rate for reweighting
    :divergence_min_lr: learning rate for divergence minimization
    :rho: threshold for divergence minimization
    :device: device
    """
    policy_reweighting_optimizer = torch.optim.Adam(params=policy.parameters(), lr=reweighting_lr)
    policy_div_min_optimizer = torch.optim.Adam(params=policy.parameters(), lr=divergence_min_lr)
    t_div_min_optimizer = torch.optim.Adam(params=t.parameters(), lr=divergence_min_lr)


    for i in range(max_epochs):
        for X, y, log_prop, loss, labels in bandit_dataloader:
            # estimate reweighted loss
            policy_reweighting_optimizer.zero_grad()
            loss = reweighted_loss()
            loss.backward()
            policy_reweighting_optimizer.step()

            # call algorithm 1
            policy_reweighting_optimizer.zero_grad()
            variational_minimizing_Df()


def variational_minimizing_Df(
        data, D_0: float, max_it: int, policy: Policy,
        discriminator: T, policy_optimizer: Optimizer,
        discriminator_optimizer: Optimizer, is_gumbel_hard: bool,
        device: torch.cuda.device
    ) -> None:
    """
    Implements algorithm 1 in http://proceedings.mlr.press/v80/wu18g/wu18g-supp.pdf.

    Arguments
    :data:
    :D_0:
    :max_it: max iterations
    :policy: Policy to learn. This is not the logging policy.
    :discriminator: discriminator network
    :is_gumbel_hard:
    :device: device
    """

    for i in range(max_it):
        policy_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()


