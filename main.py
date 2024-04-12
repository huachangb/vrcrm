import torch
from torch.utils.data import DataLoader

from vrcrm.poem import DatasetReader, Skylines, Logger
from vrcrm.poem_bridge.bandit_dataset import BanditDataset
from vrcrm.models import Policy, T
from vrcrm.inference.train import train
from vrcrm.inference.validate import expected_loss, MAP

name = "scene"


crf_scores = []
crf_expected_scores = []
logger_scores = []
logger_map_scores = []

crf_time = []



for i in range(1):
    dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = True)
    if name == 'rcv1_topics':
        dataset.loadDataset(corpusName = name, labelSubset = [33, 59, 70, 102])
    else:
        dataset.loadDataset(corpusName = name)

    n_features = dataset.trainFeatures.shape[1]
    n_labels = dataset.trainLabels.shape[1]


    supervised_dataset = DatasetReader.SupervisedDataset(dataset = dataset, verbose = True)
    supervised_dataset.createTrainValidateSplit(validateFrac = 0.25)

    # CRF
    # crf = Skylines.CRF(dataset = supervised_dataset, tol = 1e-6, minC = -2, maxC = 2, verbose = True, parallel = None)
    # crf_time.append(crf.validate())
    # crf_scores.append(crf.test())
    # crf_expected_scores.append(crf.expectedTestLoss())

    supervised_dataset.freeAuxiliaryMatrices()
    del supervised_dataset

    streamer = Logger.DataStream(dataset = dataset, verbose = True)
    features, labels = streamer.generateStream(subsampleFrac = 0.05, replayCount = 1)

    subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)
    subsampled_dataset.trainFeatures = features
    subsampled_dataset.trainLabels = labels
    logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = True)
    logger_map_scores.append(logger.crf.test())
    logger_scores.append(logger.crf.expectedTestLoss())

    replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)

    features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 4)
    replayed_dataset.trainFeatures = features
    replayed_dataset.trainLabels = labels

    sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(replayed_dataset)

    bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = True)

    replayed_dataset.freeAuxiliaryMatrices()
    del replayed_dataset

    bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
    bandit_dataset.createTrainValidateSplit(validateFrac = 0.25)

    nn_train_data, nn_val_train_data = BanditDataset.from_poem(bandit_dataset)
    bandit_train_loader = DataLoader(nn_train_data, shuffle=True, batch_size=64)
    fgan_loader = DataLoader(nn_train_data, shuffle=True, batch_size=64)

    # train NN
    policy = Policy(n_in=n_features, n1=15, n2=30, n_out=n_labels)
    discr = T(n_features + 2 * n_labels)

    train(max_epoch=1, bandit_train_loader=bandit_train_loader, fgan_loader=fgan_loader, hnet=policy, Dnet_xy=discr, steps_fgan=10)

    exp_loss = expected_loss(policy, n_samples=32, X=nn_val_train_data.features.float(), labels=nn_val_train_data.labels)
    maps = MAP(policy, X=nn_val_train_data.features.float(), labels=nn_val_train_data.labels)
    print("EXP ", exp_loss)
    print("MAP ", maps)
