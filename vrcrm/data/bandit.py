from typing import Any, Tuple

import torch
from torch.utils.data import Dataset


class BanditDataset(Dataset):
    """ Implements bandit data set """
    def __init__(self) -> None:
        super().__init__()
        self.x = []
        self.y = []
        self.log_prop = []
        self.loss = []
        self.labels = []

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor, float, float, Tuple[Any]]:
        """
        Retrieves training instance

        Returns
        :x: context
        :y: sampled action
        :log_prop: log propensity
        :loss: hamming loss
        :labels: true labels
        """
        x = self.x[index]
        y = self.y[index]
        log_prop = self.log_prop[index]
        loss = self.loss[index]
        labels = self.labels[index]

        return x, y, log_prop, loss, labels
