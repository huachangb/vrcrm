import DatasetReader
import math
import numpy
import numpy.random
import scipy.sparse
import Skylines
import sys

from vrcrm.data import DatasetBase


class LoggingPolicy():
    def __init__(self, dataset: DatasetBase, loggerC: int, stochasticMultiplier: float, verbose: bool) -> None:
        self.verbose = verbose
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

    def generateLog(self, dataset):
        numSamples, _ = numpy.shape(dataset.trainFeatures)
        numLabels = numpy.shape(dataset.trainLabels)[1]

        sampledLabels = numpy.zeros((numSamples, numLabels), dtype = numpy.int16)
        logpropensity = numpy.zeros(numSamples, dtype = numpy.float64)

        for i in range(numLabels):
            if self.crf.labeler[i] is not None:
                regressor = self.crf.labeler[i]
                predictedProbabilities = regressor.predict_log_proba(dataset.trainFeatures)

                randomThresholds = numpy.log(numpy.random.rand(numSamples).astype(numpy.float64))
                sampledLabel = randomThresholds > predictedProbabilities[:,0]
                sampledLabels[:, i] = sampledLabel.astype(int)

                probSampledLabel = numpy.zeros(numSamples, dtype=numpy.longdouble)
                probSampledLabel[sampledLabel] = predictedProbabilities[sampledLabel, 1]
                remainingLabel = numpy.logical_not(sampledLabel)
                probSampledLabel[remainingLabel] = predictedProbabilities[remainingLabel, 0]
                logpropensity = logpropensity + probSampledLabel

        diffLabels = sampledLabels != dataset.trainLabels
        sampledLoss = diffLabels.sum(axis = 1, dtype = numpy.longdouble) - numLabels

        if self.verbose:
            averageSampledLoss = sampledLoss.mean(dtype = numpy.longdouble)
            print("Logger: [Message] Sampled historical logs. [Mean train loss, numSamples]:", averageSampledLoss, numpy.shape(sampledLabels)[0])
            print("Logger: [Message] [min, max, mean] inv propensity", logpropensity.min(), logpropensity.max(), logpropensity.mean())
            sys.stdout.flush()

        return sampledLabels, logpropensity, sampledLoss

