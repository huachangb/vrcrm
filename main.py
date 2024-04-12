import torch
from torch.utils.data import DataLoader

from vrcrm.poem import DatasetReader, Skylines, Logger
from vrcrm.poem_bridge.bandit_dataset import BanditDataset
from vrcrm.models import Policy, T
from vrcrm.models.logger import CRFLogger
from vrcrm.inference.train import train
from vrcrm.inference.validate import expected_loss, MAP

name = "scene"


crf_scores = []
crf_expected_scores = []
logger_exp_scores = []
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

    streamer = Logger.DataStream(dataset = dataset, verbose = True)
    features, labels = streamer.generateStream(subsampleFrac = 0.05, replayCount = 1)

    # create logger
    logger = CRFLogger(n_labels=n_labels, loggerC = -1, verbose = True)
    logger.fit(features, labels)
    logger_map_scores.append(MAP(logger))
    logger_exp_scores.append(expected_loss(logger))
    print()

    # create bandit dataset
    replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)
    features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 4)
    replayed_dataset.trainFeatures = features
    replayed_dataset.trainLabels = labels
    sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(features, labels)
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

    eval_features = nn_val_train_data.features.float()
    eval_features = torch.from_numpy(eval_features)
    eval_labels = nn_val_train_data.labels.int()
    eval_labels = torch.from_numpy(eval_labels)

    # evaluate logging policy
    exp_loss = expected_loss(logger, n_samples=32, X=eval_features, labels=eval_labels)
    maps = MAP(logger, X=eval_features, labels=eval_labels)
    print("Logging policy")
    print("EXP ", exp_loss)
    print("MAP ", maps)

    # evaluate NN
    exp_loss = expected_loss(policy, n_samples=32, X=eval_features, labels=eval_labels)
    maps = MAP(policy, X=eval_features, labels=eval_labels)
    print("NN")
    print("EXP ", exp_loss)
    print("MAP ", maps)
