from typing import Any, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset


class BanditDataset(Dataset):
    """ Implements bandit data set """
    def __init__(self, features, labels, log_propensity, loss) -> None:
        super().__init__()
        self.features = torch.from_numpy(features.astype(np.float64).toarray())
        self.labels = torch.from_numpy(labels.astype(np.float64).toarray())
        self.log_propensity = torch.from_numpy(log_propensity.astype(np.float64).toarray())
        self.loss = torch.from_numpy(loss.astype(np.float64).toarray())


    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor, float, float, Tuple[Any]]:
        """
        Retrieves training instance

        Returns
        :x: context
        :y: sampled action
        :log_prop: log propensity
        :loss: hamming loss
        """
        x = self.features[index]
        labels = self.labels[index]
        log_prop = self.log_propensity[index]
        loss = self.loss[index]

        return x, labels, log_prop, loss
