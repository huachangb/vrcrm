import numpy
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import sklearn.svm

from .skylines import Skylines


class CRF(Skylines):
    def __init__(self, dataset, tol, minC, maxC, verbose, parallel):
        Skylines.__init__(self, dataset, tol, minC, maxC, verbose, parallel)
        self.Name = "CRF"

    def generateModel(self, param, reportValidationResult):
        regressors = []

        predictedLabels = None
        if reportValidationResult:
            predictedLabels = numpy.zeros(numpy.shape(self.dataset.validateLabels), dtype = numpy.int16)

        numLabels = numpy.shape(self.dataset.trainLabels)[1]
        for i in range(numLabels):
            currLabels = self.dataset.trainLabels[:, i]
            if currLabels.sum() > 0:        #Avoid training labels with no positive instances
                logitRegressor = sklearn.linear_model.LogisticRegression(solver="liblinear", C = param,
                    penalty = 'l2', tol = self.tol, dual = True, fit_intercept = False)
                logitRegressor.fit(self.dataset.trainFeatures, currLabels)
                regressors.append(logitRegressor)
                if reportValidationResult:
                    predictedLabels[:,i] = logitRegressor.predict(self.dataset.validateFeatures)
            else:
                regressors.append(None)

        predictionError = None
        if reportValidationResult:
            predictionError = sklearn.metrics.hamming_loss(self.dataset.validateLabels,
                predictedLabels)

        return predictionError, regressors

    def generatePredictions(self, classifiers):
        predictedLabels = numpy.zeros(numpy.shape(self.dataset.testLabels), dtype = numpy.int16)
        numLabels = numpy.shape(predictedLabels)[1]
        for i in range(numLabels):
            if classifiers[i] is not None:
                predictedLabels[:,i] = classifiers[i].predict(self.dataset.testFeatures)

        return predictedLabels
