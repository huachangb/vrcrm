from typing import Any

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression


class CRF():
    def __init__(self, n_labels: int, C: float, verbose: bool) -> None:
        self.verbose = verbose
        self.n_labels = n_labels
        self.predictors = []

        # create models
        for _ in range(n_labels):
            predictor = LogisticRegression(solver="liblinear", C = C, penalty = 'l2', tol = 1e-5, dual = True, fit_intercept = False)
            self.predictors.append(predictor)

    def __call__(self, X, as_tensor: bool = True) -> Any:
        probs = self.predict_proba(X)

        if as_tensor:
            probs = torch.from_numpy(probs)

        return probs

    def fit(self, X, y):
        for i in range(self.n_labels):
            labels = y[:, i]
            self.predictors[i].fit(X, labels)


    def predict(self, X):
        predictedLabels = np.zeros((X.shape[0], self.n_labels), dtype = np.int16)

        for i in range(self.n_labels):
            predictedLabels[:,i] = self.predictors[i].predict(X)

        return predictedLabels

    def predict_proba(self, X):
        predictedLabels = np.zeros((X.shape[0], self.n_labels), dtype = np.float32)

        for i in range(self.n_labels):
            probs = self.predictors[i].predict_proba(X)
            predictedLabels[:,i] = probs[:, 1]

        return predictedLabels

if __name__=="__main__":
    X = np.random.rand(100, 30)
    y = (np.random.rand(100, 5) > 0.5).astype(int)
    model = CRF(5, 0.1, False)
    model.fit(X, y)
    print(model(X))
