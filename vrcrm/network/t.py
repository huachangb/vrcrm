import torch
from torch import nn


class T(nn.Module):
    def __init__(self, d_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_dim, d_dim // 2),
            nn.BatchNorm1d(d_dim // 2),
            nn.ReLU(),
            nn.Linear(d_dim // 2, 1)
            #nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


