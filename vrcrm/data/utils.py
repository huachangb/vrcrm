from typing import List, Tuple

import torch
from sklearn import datasets

from vrcrm.network import Policy


def load_dataset(
        train_path: str,
        test_path: str,
        n_features: int,
        device: torch.cuda.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
    """
    Loads dataset as Tensors

    Arguments
    :train_path: path to training data set
    :test_path: path to testing data set
    :n_features: number of features of data
    :device: device to store data on
    """
    x_train, y_train = datasets.load_svmlight_file(train_path, n_features=n_features, multilabel=True)
    x_train = torch.from_numpy(x_train).to(device)
    y_train = [torch.tensor(labels).to(device) for labels in y_train]

    x_test, y_test = datasets.load_svmlight_file(test_path, n_features=n_features, multilabel=True)
    x_test = torch.from_numpy(x_test).to(device)
    y_test = [torch.tensor(labels).to(device) for labels in y_test]

    return x_train, y_train, x_test, y_test


def convert_supervised_to_bandit(
        x_train: torch.Tensor, y_train: List[torch.Tensor], initial_policy: Policy
    ) -> None:
    pass

