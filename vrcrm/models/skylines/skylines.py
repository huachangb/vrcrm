import numpy
from . import PRM
import scipy.linalg
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import sklearn.svm
import sys
import time
from abc import abstractmethod, ABC


class Skylines(ABC):
    def __init__(self, dataset, tol, minC, maxC, verbose, parallel):
        paramArray = numpy.logspace(minC, maxC, num = (maxC - minC + 1), base = 10.0)
        self.params = paramArray.tolist()
        self.tol = tol
        self.verbose = verbose
        self.dataset = dataset
        self.labeler = None
        self.pool = parallel

    def freeAuxiliaryMatrices(self):
        if self.labeler is not None:
            del self.labeler
        del self.dataset
        if self.verbose:
            print("Skylines: [Message] Freed matrices")
            sys.stdout.flush()

    @abstractmethod
    def generateModel(self, param, reportValidationResult):
        return


    def parallelValidate(self, param, reportValidationResult):
        predictionError, classifier = self.generateModel(param, reportValidationResult)
        return {'param': param, 'classifier': classifier, 'perf': predictionError}

    def validate(self):
        numSamples, numLabels = numpy.shape(self.dataset.trainLabels)
        numFeatures = numpy.shape(self.dataset.trainFeatures)[1]

        results = None
        start_time = time.time()
        if (self.pool is not None) and (len(self.params) > 1):
            dummy = [True]*len(self.params)
            results = self.pool.map(self.parallelValidate, self.params, dummy)
            results = list(results)
        else:
            results = []
            for param in self.params:
                results.append(self.parallelValidate(param, len(self.params) > 1))
        end_time = time.time()
        avg_time = (end_time - start_time)

        bestPerformance = None
        bestClassifier = None
        bestParam = None

        for result in results:
            param = result['param']
            classifier = result['classifier']
            predictionError = result['perf']
            if len(self.params) > 1:
                predictionError = predictionError * numLabels
                if self.verbose:
                    print(self.Name, " Validation. Parameter = ", param, " Performance: ", predictionError)
                    sys.stdout.flush()

            if (bestPerformance is None) or (bestPerformance > predictionError):
                bestPerformance = predictionError
                bestClassifier = classifier
                bestParam = param

        if self.verbose:
            print(self.Name, " Best. Parameter = ", bestParam, "Time: ", avg_time, " Performance: ", bestPerformance)
            sys.stdout.flush()

        self.labeler = bestClassifier
        return avg_time

    def test(self):
        predictedLabels = self.generatePredictions(self.labeler)
        numLabels = numpy.shape(self.dataset.testLabels)[1]
        predictionError = sklearn.metrics.hamming_loss(self.dataset.testLabels,
            predictedLabels) * numLabels

        if self.verbose:
            print(self.Name," Test. Performance: ", predictionError)
            sys.stdout.flush()
        return predictionError

    def expectedTestLoss(self):
        predictionError = None
        numLabels = numpy.shape(self.dataset.testLabels)[1]
        if self.Name == "SVM":
            predictionError = self.test()
        elif 'CRF' in self.Name:
            numFeatures = numpy.shape(self.dataset.testFeatures)[1]

            predictor = PRM.VanillaISEstimator(n_iter = 0, tol = 0, l2reg = 0,
                varpenalty = 0, clip = 1, verbose = False)
            predictor.coef_ = numpy.zeros((numFeatures,numLabels), dtype = numpy.longdouble)
            for i in range(numLabels):
                if self.labeler[i] is not None:
                    predictor.coef_[:,i] = self.labeler[i].coef_

            if self.verbose:
                print("wNorm", scipy.linalg.norm(predictor.coef_))
                sys.stdout.flush()

            predictionError = predictor.computeExpectedLoss(self.dataset.testFeatures,
                self.dataset.testLabels) * numLabels
        else:
            if self.verbose:
                print("wNorm", scipy.linalg.norm(self.labeler.coef_))
                sys.stdout.flush()

            predictionError = self.labeler.computeExpectedLoss(self.dataset.testFeatures,
                self.dataset.testLabels) * numLabels

        if self.verbose:
            print(self.Name,"Test. Expected Loss: ", predictionError)
            sys.stdout.flush()
        return predictionError
