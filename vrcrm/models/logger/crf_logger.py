import sys

import numpy as np

from vrcrm.models.CRF import CRF
from vrcrm.models.logger.logging_policy import LoggingPolicy


class CRFLogger(LoggingPolicy, CRF):
    def __init__(self, n_labels, C, verbose) -> None:
        LoggingPolicy.__init__(self, verbose=verbose)
        CRF.__init__(self, n_labels=n_labels, C=C, verbose=verbose)

    def generateLog(self, X, labels):
        numSamples, _ = np.shape(X)
        numLabels = self.n_labels

        sampledLabels = np.zeros((numSamples, numLabels), dtype = np.int16)
        logpropensity = np.zeros(numSamples, dtype = np.longdouble)

        for i in range(numLabels):
            regressor = self.predictors[i]

            if regressor is not None:
                predictedProbabilities = regressor.predict_log_proba(X)
                randomThresholds = np.log(np.random.rand(numSamples).astype(np.longdouble))
                sampledLabel = randomThresholds > predictedProbabilities[:,0]
                sampledLabels[:, i] = sampledLabel.astype(int)

                probSampledLabel = np.zeros(numSamples, dtype=np.longdouble)
                probSampledLabel[sampledLabel] = predictedProbabilities[sampledLabel, 1]
                remainingLabel = np.logical_not(sampledLabel)
                probSampledLabel[remainingLabel] = predictedProbabilities[remainingLabel, 0]
                logpropensity = logpropensity + probSampledLabel

        diffLabels = sampledLabels != labels
        sampledLoss = diffLabels.sum(axis = 1, dtype = np.longdouble) - numLabels

        if self.verbose:
            averageSampledLoss = sampledLoss.mean(dtype = np.longdouble)
            print("Logger: [Message] Sampled historical logs. [Mean train loss, numSamples]:", averageSampledLoss, np.shape(sampledLabels)[0])
            print("Logger: [Message] [min, max, mean] inv propensity", logpropensity.min(), logpropensity.max(), logpropensity.mean())
            sys.stdout.flush()

        return sampledLabels, logpropensity, sampledLoss