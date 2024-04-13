from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import numpy as np

from vrcrm.poem import DatasetReader, Logger
from vrcrm.poem_bridge.bandit_dataset import BanditDataset
from vrcrm.poem_bridge.prmwwrapper import PRMWrapperBackwardSupport as PRMWrapper
from vrcrm.models import Policy, T
from vrcrm.models.CRF import CRF
from vrcrm.models.logger import CRFLogger
from vrcrm.inference.train import train
from vrcrm.loss import expected_loss, MAP_loss

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

name = "scene"


exp_scores = defaultdict(list)
map_scores = defaultdict(list)

EVAL_N_SAMPLES = 16
VERBOSE = False
SUPERVISED_VALIDATE_FRAC = 0.25
LOGGER_DATA_FRAC = 0.05



for i in range(1):
    ok = False

    while not ok:
        try:

            dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = VERBOSE)
            if name == 'rcv1_topics':
                dataset.loadDataset(corpusName = name, labelSubset = [33, 59, 70, 102])
            else:
                dataset.loadDataset(corpusName = name)

            n_features = dataset.trainFeatures.shape[1]
            n_labels = dataset.trainLabels.shape[1]


            supervised_dataset = DatasetReader.SupervisedDataset(dataset = dataset, verbose = VERBOSE)
            supervised_dataset.createTrainValidateSplit(validateFrac = SUPERVISED_VALIDATE_FRAC)

            streamer = Logger.DataStream(dataset = dataset, verbose = VERBOSE)
            features, labels = streamer.generateStream(subsampleFrac = LOGGER_DATA_FRAC, replayCount = 1)

            # create logger
            # loggerC is -1 in original code, but gets converted to 0.1
            logger = CRFLogger(n_labels=n_labels, C = 0.1, verbose = VERBOSE)
            # logger = SanityLogger(n_labels=n_labels,  predict="none")
            logger.fit(features, labels)

            subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)
            subsampled_dataset.trainFeatures = features
            subsampled_dataset.trainLabels = labels
            logger_og = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = False)

            # create bandit dataset
            replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = VERBOSE)
            features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 4)
            replayed_dataset.trainFeatures = features
            replayed_dataset.trainLabels = labels
            sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(features, labels)
            bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = VERBOSE)
            # print("Sampled labels")
            # print(sampledLabels)
            # print("Train labels")
            # print(replayed_dataset.trainLabels)
            replayed_dataset.freeAuxiliaryMatrices()
            del replayed_dataset

            bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
            bandit_dataset.createTrainValidateSplit(validateFrac = 0.25)

            nn_train_data, nn_val_train_data = BanditDataset.from_poem(bandit_dataset)
            bandit_train_loader = DataLoader(nn_train_data, shuffle=True, batch_size=64)
            fgan_loader = DataLoader(nn_train_data, shuffle=True, batch_size=64)

            ##################################################################################################
            #
            # Logging policy
            #
            ##################################################################################################
            exp_loss = expected_loss(logger, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
            maps = MAP_loss(logger, X=eval_features, labels=eval_labels)
            exp_scores["logging"].append(exp_loss)
            map_scores["logging"].append(maps)

            # OG logger
            exp_scores["logger-og"].append(logger_og.crf.expectedTestLoss())
            map_scores["logger-og"].append(logger_og.crf.test())


            ##################################################################################################
            #
            # NN-noreg, NN-soft, NN-hard
            #
            ##################################################################################################
            policy = Policy(n_in=n_features, n1=15, n2=30, n_out=n_labels).to(torch.float32)
            discr = T(n_features + 2 * n_labels).to(torch.float32)


            train(max_epoch=0, bandit_train_loader=bandit_train_loader, fgan_loader=fgan_loader, hnet=policy, Dnet_xy=discr, steps_fgan=10)

            # evaluate NN
            exp_loss = expected_loss(policy, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
            maps = MAP_loss(policy, X=eval_features, labels=eval_labels)
            exp_scores["nn-noreg"].append(exp_loss)
            map_scores["nn-noreg"].append(maps)


            ##################################################################################################
            #
            # Supervised
            #
            ##################################################################################################

            # train CRF
            crf = CRF(n_labels=n_labels, C=0.1, verbose=False)
            crf.fit(supervised_dataset.trainFeatures, supervised_dataset.trainLabels)

            eval_features = supervised_dataset.testFeatures.toarray().astype(np.float32)
            eval_features = torch.from_numpy(eval_features)
            eval_labels = supervised_dataset.testLabels
            eval_labels = torch.from_numpy(eval_labels)

            # evaluate CRF
            exp_loss = expected_loss(crf, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
            maps = MAP_loss(crf, X=eval_features, labels=eval_labels)
            exp_scores["crf"].append(exp_loss)
            map_scores["crf"].append(maps)

            ##################################################################################################
            #
            # POEM stuff
            #
            ##################################################################################################

            prm = PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = -6, maxV = 0,
                                                minClip = 0, maxClip = 0, estimator_type = 'Vanilla', verbose = VERBOSE,
                                                parallel = None, smartStart = None)
            prm.calibrateHyperParams()
            prm.validate()
            exp_loss = expected_loss(prm, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
            maps = MAP_loss(prm, X=eval_features, labels=eval_labels)
            exp_scores["prm"].append(exp_loss)
            map_scores["prm"].append(maps)
            exp_scores["prm-og"].append(prm.expectedTestLoss())
            map_scores["prm-og"].append(prm.test())
            prm.freeAuxiliaryMatrices()
            del prm

            erm = PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1,
                                        minClip = 0, maxClip = 0, estimator_type = 'Vanilla', verbose = VERBOSE,
                                        parallel = None, smartStart = None)
            erm.calibrateHyperParams()
            erm.validate()
            exp_loss = expected_loss(erm, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
            maps = MAP_loss(erm, X=eval_features, labels=eval_labels)
            exp_scores["erm"].append(exp_loss)
            map_scores["erm"].append(maps)
            exp_scores["erm-og"].append(erm.expectedTestLoss())
            map_scores["erm-og"].append(erm.test())

            erm.freeAuxiliaryMatrices()
            del erm

            maj = PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = -6, maxV = 0,
                                        minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = VERBOSE,
                                        parallel = None, smartStart = None)
            maj.calibrateHyperParams()
            maj.validate()
            exp_loss = expected_loss(maj, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
            maps = MAP_loss(maj, X=eval_features, labels=eval_labels)
            exp_scores["maj"].append(exp_loss)
            map_scores["maj"].append(maps)
            exp_scores["maj-og"].append(maj.expectedTestLoss())
            map_scores["maj-og"].append(maj.test())

            maj.freeAuxiliaryMatrices()
            del maj

            majerm = PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1,
                                        minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = VERBOSE,
                                        parallel = None, smartStart = None)
            majerm.calibrateHyperParams()
            majerm.validate()
            exp_loss = expected_loss(majerm, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
            maps = MAP_loss(majerm, X=eval_features, labels=eval_labels)
            exp_scores["majerm"].append(exp_loss)
            map_scores["majerm"].append(maps)
            exp_scores["majerm-og"].append(majerm.expectedTestLoss())
            map_scores["majerm-pg"].append(majerm.test())
            ok = True
        except ValueError as e:
            print("Invalid split, attempting again...")


print(f"Results for {name}")
for model in exp_scores.keys():
    print(f"{model}\n\tMAP: {np.mean(map_scores[model]):.3f}\n\tEXP: {np.mean(exp_scores[model]):.3f}")
