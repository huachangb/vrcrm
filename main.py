from collections import defaultdict

import pickle
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
from vrcrm.inference.train_2 import train2, algorithm_1
from vrcrm.loss import expected_loss, MAP_loss

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


exp_scores = defaultdict(list)
map_scores = defaultdict(list)

EVAL_N_SAMPLES = 16
VERBOSE = False
SUPERVISED_VALIDATE_FRAC = 0.25
LOGGER_DATA_FRAC = 0.05

USING_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USING_CUDA else "cpu")


for name in ["scene", "yeast"]:
    for i in range(10):
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
                # sampledLabels, sampledLogPropensity, sampledLoss = logger_og.generateLog(replayed_dataset)
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
                bandit_train_loader = DataLoader(nn_train_data, shuffle=True, batch_size=512)
                fgan_loader = DataLoader(nn_train_data, shuffle=True, batch_size=512)


                eval_features = supervised_dataset.testFeatures.toarray().astype(np.float32)
                eval_features = torch.from_numpy(eval_features)
                eval_labels = supervised_dataset.testLabels
                eval_labels = torch.from_numpy(eval_labels)

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
                # NN veriyf
                #
                ##################################################################################################
                # layers = dict(n_in=n_features, n1=15, n2=30, n_out=n_labels)
                # max_epochs = 50
                # steps_fgan = 10


                # # no reg
                # policy = Policy(**layers).to(torch.float32).to(device)
                # discr = T(n_features + 2 * n_labels).to(torch.float32).to(device)
                # opt_h = torch.optim.Adam(params=policy.parameters(), lr=0.001)
                # opt_h2 = torch.optim.Adam(params=policy.parameters(), lr=0.001)
                # opt_d = torch.optim.Adam(params=discr.parameters(), lr=0.01)

                # algorithm_1(steps_fgan=steps_fgan, fgan_loader=fgan_loader, device=device, hnet=Policy, is_gumbel_hard=False, is_cuda=device.type == "cuda", Dnet_xy=discr, opts=[opt_h, opt_h2, opt_d])

                # # algorithm_1(
                # #     steps
                # #     ,
                # #     max_epoch=max_epochs, bandit_train_loader=bandit_train_loader, fgan_loader=fgan_loader, hnet=policy,
                # #       Dnet_xy=discr, steps_fgan=0, is_gumbel_hard=False, device=device, opts=[opt_h, opt_h2, opt_d]
                # #       )

                # # evaluate NN
                # policy = policy.cpu()
                # discr = discr.cpu()
                # exp_loss = expected_loss(policy, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
                # maps = MAP_loss(policy, X=eval_features, labels=eval_labels)
                # exp_scores["nn-noreg"].append(exp_loss)
                # map_scores["nn-noreg"].append(maps)


                ##################################################################################################
                #
                # NN-noreg, NN-soft, NN-hard
                #
                ##################################################################################################
                from itertools import product

                node_count = [4, 8, 16, 64]

                max_epochs= 100
                steps_fgan = 10

                for n1, n2 in product(node_count, node_count):
                    layers = dict(n_in=n_features, n1=n1, n2=n2, n_out=n_labels)

                    # no reg
                    policy = Policy(**layers).to(torch.float32).to(device)
                    discr = T(n_features + 2 * n_labels).to(torch.float32).to(device)
                    opt_h = torch.optim.Adam(params=policy.parameters(), lr=0.001)
                    opt_h2 = torch.optim.Adam(params=policy.parameters(), lr=0.01)
                    opt_d = torch.optim.Adam(params=discr.parameters(), lr=0.01)


                    train(max_epoch=max_epochs, bandit_train_loader=bandit_train_loader, fgan_loader=fgan_loader, hnet=policy,
                        Dnet_xy=discr, steps_fgan=0, is_gumbel_hard=False, device=device, opts=[opt_h, opt_h2, opt_d])

                    # evaluate NN
                    policy = policy.cpu()
                    discr = discr.cpu()
                    exp_loss = expected_loss(policy, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
                    maps = MAP_loss(policy, X=eval_features, labels=eval_labels)
                    exp_scores[f"nn-noreg {n1}-{n2}"].append(exp_loss)
                    map_scores[f"nn-noreg {n1}-{n2}"].append(maps)


                    # soft
                    policy = Policy(**layers).to(torch.float32).to(device)
                    discr = T(n_features + 2 * n_labels).to(torch.float32).to(device)
                    opt_h = torch.optim.Adam(params=policy.parameters(), lr=0.001)
                    opt_h2 = torch.optim.Adam(params=policy.parameters(), lr=0.01)
                    opt_d = torch.optim.Adam(params=discr.parameters(), lr=0.01)


                    train(max_epoch=max_epochs, bandit_train_loader=bandit_train_loader, fgan_loader=fgan_loader, hnet=policy,
                        Dnet_xy=discr, steps_fgan=steps_fgan, is_gumbel_hard=False, device=device, opts=[opt_h, opt_h2, opt_d])

                    # evaluate NN
                    policy = policy.cpu()
                    discr = discr.cpu()
                    exp_loss = expected_loss(policy, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
                    maps = MAP_loss(policy, X=eval_features, labels=eval_labels)
                    exp_scores[f"nn-soft {n1}-{n2}"].append(exp_loss)
                    map_scores[f"nn-soft {n1}-{n2}"].append(maps)

                    # hard
                    policy = Policy(**layers).to(torch.float32).to(device)
                    discr = T(n_features + 2 * n_labels).to(torch.float32).to(device)
                    opt_h = torch.optim.Adam(params=policy.parameters(), lr=0.001)
                    opt_h2 = torch.optim.Adam(params=policy.parameters(), lr=0.01)
                    opt_d = torch.optim.Adam(params=discr.parameters(), lr=0.01)


                    train(max_epoch=max_epochs, bandit_train_loader=bandit_train_loader, fgan_loader=fgan_loader, hnet=policy,
                        Dnet_xy=discr, steps_fgan=steps_fgan, is_gumbel_hard=False, device=device, opts=[opt_h, opt_h2, opt_d])

                    # evaluate NN
                    policy = policy.cpu()
                    discr = discr.cpu()
                    exp_loss = expected_loss(policy, n_samples = EVAL_N_SAMPLES, X=eval_features, labels=eval_labels)
                    maps = MAP_loss(policy, X=eval_features, labels=eval_labels)
                    exp_scores[f"nn-hard {n1}-{n2}"].append(exp_loss)
                    map_scores[f"nn-hard {n1}-{n2}"].append(maps)


                ##################################################################################################
                #
                # Supervised
                #
                ##################################################################################################

                # train CRF
                crf = CRF(n_labels=n_labels, C=0.1, verbose=False)
                crf.fit(supervised_dataset.trainFeatures, supervised_dataset.trainLabels)

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



with open('map_scores.pickle', 'wb') as handle:
    pickle.dump(map_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('exp_scores.pickle', 'wb') as handle:
    pickle.dump(exp_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(f"Results for {name}")
for model in exp_scores.keys():
    print(f"{model}\n\tMAP: {np.mean(map_scores[model]):.3f}\n\tEXP: {np.mean(exp_scores[model]):.3f}")
