import sys
import numpy
import scipy.linalg
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import sys

from . import skylines as Skylines
from . import PRM
from . import MajorizePRM

class PRMWrapper(Skylines):
    def __init__(self, dataset, n_iter, tol, minC, maxC, minV, maxV,
                    minClip, maxClip, estimator_type, verbose, parallel, smartStart):
        Skylines.__init__(self, dataset, tol, 0, 0, verbose, parallel)
        self.Name = "PRM("+estimator_type
        if maxV < minV:
            self.Name = self.Name + "-ERM)"
        else:
            self.Name = self.Name + "-SVP)"

        self.params = {}
        numSamples = numpy.shape(self.dataset.trainFeatures)[0]

        if minC <= maxC:
            l2Array = numpy.logspace(minC, maxC, num = (maxC - minC + 1), base = 10.0)
            l2List = l2Array.tolist()
            self.params['l2reg'] = l2List
        else:
            self.params['l2reg'] = [0]

        if minV <= maxV:
            varArray = numpy.logspace(minV, maxV, num = (maxV - minV + 1), base = 10.0)
            varList = varArray.tolist()
            self.params['varpenalty'] = varList
        else:
            self.params['varpenalty'] = [0]

        if minClip <= maxClip:
            clipArray = numpy.logspace(minClip, maxClip, num = (maxClip - minClip + 1), base = 10.0)
            clipArray = numpy.log(clipArray)
            clipList = clipArray.tolist()
            self.params['clip'] = clipList
        else:
            self.params['clip'] = [0]


        self.estimator_type = estimator_type
        self.n_iter = n_iter
        self.smart_start = smartStart

    def calibrateHyperParams(self):
        l2Array = numpy.array(self.params['l2reg'])
        l2penalty = l2Array
        self.params['l2reg'] = l2penalty.tolist()

        numSamples, numLabels = numpy.shape(self.dataset.trainSampledLabels)

        percentileMinPropensity = numpy.percentile(self.dataset.trainSampledLogPropensity, 10, interpolation = 'higher')
        percentileMaxPropensity = numpy.percentile(self.dataset.trainSampledLogPropensity, 90, interpolation = 'lower')
        percentileClip = percentileMaxPropensity - percentileMinPropensity

        if percentileClip < 1:
            percentileClip = 1
        if self.verbose:
            print(f"Calibrating clip to ", percentileClip)
            sys.stdout.flush()

        clipArray = numpy.array(self.params['clip'])
        clip = clipArray + percentileClip
        self.params['clip'] = clip.tolist()

        meanLoss = self.dataset.trainSampledLoss.mean(dtype = numpy.longdouble)
        lossDelta = self.dataset.trainSampledLoss - meanLoss
        sqrtLossVar = scipy.linalg.norm(lossDelta) / numpy.sqrt(numSamples*(numSamples - 1))
        max_val = - meanLoss / sqrtLossVar

        if self.verbose:
            print("Calibrating variance regularizer to ", max_val)
            sys.stdout.flush()

        varArray = numpy.array(self.params['varpenalty'])
        varpenalty = varArray * max_val

        self.params['varpenalty'] = varpenalty.tolist()

        self.params = list(sklearn.grid_search.ParameterGrid(self.params))

    def generateModel(self, param, reportValidationResult):
        predictor = None
        if self.estimator_type == 'Majorize':
            predictor = MajorizePRM.MajorizeISEstimator(n_iter = self.n_iter, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = self.verbose)
        elif (self.estimator_type == 'Stochastic'):
            numSamples = numpy.shape(self.dataset.trainSampledLabels)[0]
            epoch_batches = int(numSamples / 100)
            predictor = MajorizePRM.MajorizeStochasticEstimator(n_iter = max(2 * epoch_batches, self.n_iter),
                            min_iter = epoch_batches, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = False)
        elif (self.estimator_type == 'SelfNormal'):
            predictor = PRM.SelfNormalEstimator(n_iter = self.n_iter, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = self.verbose)
        else:
            predictor = PRM.VanillaISEstimator(n_iter = self.n_iter, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = False)

        predictor.setAuxiliaryMatrices(self.dataset.trainFeatures, self.dataset.trainSampledLabels,
            self.dataset.trainSampledLogPropensity, self.dataset.trainSampledLoss)

        predictor.fit(self.smart_start)

        predictionError = None
        if reportValidationResult:
            predictionError = predictor.computeCfactHammingLoss(self.dataset.validateFeatures, self.dataset.validateSampledLabels,
                                self.dataset.validateSampledLoss, self.dataset.validateSampledLogPropensity)

        predictor.freeAuxiliaryMatrices()
        return predictionError, predictor

    def generatePredictions(self, classifier):
        return classifier.predict(self.dataset.testFeatures)

