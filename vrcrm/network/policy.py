import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, n_in: int, n1: int, n2: int, n3: int, n_out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, n1),
            nn.BatchNorm1d(n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.BatchNorm1d(n2),
            nn.ReLU(),
            nn.Linear(n2, n3),
            nn.BatchNorm1d(n3),
            nn.ReLU(),
            nn.Linear(n3, n_out),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__=="__main__":
    policy = Policy(10, 3, 5, 6, 3)

