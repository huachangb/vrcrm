import numpy as np
import sys

from vrcrm.poem import Skylines


class LoggingPolicy():
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.model = None

    def generateLog(self, dataset):
        numSamples, _ = np.shape(dataset.trainFeatures)
        numLabels = np.shape(dataset.trainLabels)[1]

        sampledLabels = np.zeros((numSamples, numLabels), dtype = np.int16)
        logpropensity = np.zeros(numSamples, dtype = np.float64)

        for i in range(numLabels):
            if self.crf.labeler[i] is not None:
                regressor = self.crf.labeler[i]
                predictedProbabilities = regressor.predict_log_proba(dataset.trainFeatures)

                randomThresholds = np.log(np.random.rand(numSamples).astype(np.float64))
                sampledLabel = randomThresholds > predictedProbabilities[:,0]
                sampledLabels[:, i] = sampledLabel.astype(int)

                probSampledLabel = np.zeros(numSamples, dtype=np.longdouble)
                probSampledLabel[sampledLabel] = predictedProbabilities[sampledLabel, 1]
                remainingLabel = np.logical_not(sampledLabel)
                probSampledLabel[remainingLabel] = predictedProbabilities[remainingLabel, 0]
                logpropensity = logpropensity + probSampledLabel

        diffLabels = sampledLabels != dataset.trainLabels
        sampledLoss = diffLabels.sum(axis = 1, dtype = np.longdouble) - numLabels

        if self.verbose:
            averageSampledLoss = sampledLoss.mean(dtype = np.longdouble)
            print("Logger: [Message] Sampled historical logs. [Mean train loss, numSamples]:", averageSampledLoss, np.shape(sampledLabels)[0])
            print("Logger: [Message] [min, max, mean] inv propensity", logpropensity.min(), logpropensity.max(), logpropensity.mean())
            sys.stdout.flush()

        return sampledLabels, logpropensity, sampledLoss



class CRFLogger():
    def __init__(self, dataset, loggerC, stochasticMultiplier, verbose) -> None:
        super().__init__(verbose=verbose)
        crf = Skylines.CRF(dataset = dataset, tol = 1e-5, minC = loggerC, maxC = loggerC, verbose = self.verbose, parallel = True)
        crf.Name = "LoggerCRF"
        crf.validate()

        if not(stochasticMultiplier == 1):
            for i in range(len(crf.labeler)):
                if crf.labeler[i] is not None:
                    crf.labeler[i].coef_ = stochasticMultiplier * crf.labeler[i].coef_

        self.crf = crf
        if self.verbose:
            print("Logger: [Message] Trained logger crf. Weight-scale: ", stochasticMultiplier)
            sys.stdout.flush()
