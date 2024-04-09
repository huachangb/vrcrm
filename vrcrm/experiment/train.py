import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vrcrm.loss import reweighted_loss
from vrcrm.models import Policy, T


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
        is_gumbel_hard: bool,
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


    for _ in range(max_epochs):
        for X, y, log_prop, loss in bandit_dataloader:
            # throw data on device
            X = X.to(device)
            y = y.to(device)
            log_prop = log_prop.to(device)
            loss = loss.to(device)

            # optimize reweighted loss
            policy_reweighting_optimizer.zero_grad()
            policy_loss = reweighted_loss(policy, X, y, log_prop)
            policy_loss.backward()
            policy_reweighting_optimizer.step()

            # call algorithm 1
            policy_reweighting_optimizer.zero_grad()
            variational_minimizing_Df(
                fgan_dataloader=fgan_dataloader,
                D_0=rho,
                max_it=max_it,
                policy=policy,
                discriminator=t,
                policy_optimizer=policy_div_min_optimizer,
                discriminator_optimizer=t_div_min_optimizer,
                is_gumbel_hard=is_gumbel_hard,
                device=device
            )


def variational_minimizing_Df(
        fgan_dataloader: DataLoader, D_0: float, max_it: int, policy: Policy,
        discriminator: T, policy_optimizer: Optimizer,
        discriminator_optimizer: Optimizer, is_gumbel_hard: bool,
        device: torch.cuda.device
    ) -> None:
    """
    Implements algorithm 1 in http://proceedings.mlr.press/v80/wu18g/wu18g-supp.pdf.

    Arguments
    :fgan_dataloader:
    :D_0: threshold
    :max_it: max iterations
    :policy: Policy to learn. This is not the logging policy.
    :discriminator: discriminator network
    :policy_optimizer:
    :discriminator_optimizer:
    :is_gumbel_hard:
    :device: device
    """

    for _ in range(max_it):
        policy_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        # sample mini batch from dataloader
        X, y = None, None
        # construct fake samples from policy using Gumbel soft-max
        # update parameters

    policy_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()


