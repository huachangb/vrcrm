from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


class CRF():
    def __init__(self, n_labels: int, loggerC: float, verbose: bool) -> None:
        self.verbose = verbose
        self.n_labels = n_labels
        self.predictors = []
        self.tol =  1e-5

        # create models
        for _ in range(n_labels):
            predictor = LogisticRegression(solver="liblinear", C = loggerC,
                    penalty = 'l2', tol = self.tol, dual = True, fit_intercept = False)
            self.predictors.append(predictor)

    def __call__(self, X) -> Any:
        return self.predict_proba(X)

    def fit(self, X, y):
        for i in range(self.n_labels):
            labels = y[:, i]
            self.predictors[i].fit(X, labels)


    def predict(self, X):
        predictedLabels = np.zeros((X.shape[0], self.n_labels), dtype = np.int16)

        for i in range(self.n_labels):
            if self.predictors[i] is not None:
                predictedLabels[:,i] = self.predictors[i].predict(X)

        return predictedLabels

    def predict_proba(self, X):
        predictedLabels = np.zeros((X.shape[0], self.n_labels), dtype = np.int16)

        for i in range(self.n_labels):
            if self.predictors[i] is not None:
                probs = self.predictors[i].predict_proba(X)
                predictedLabels[:,i] = probs[:, 1]

        return predictedLabels
