from collections import defaultdict
from copy import deepcopy

import numpy as np

from vrcrm.data import BanditDataset, DatasetBase
from vrcrm.models.skylines import CRF
from vrcrm.models.logging_policy import LoggingPolicy

name = "tmc2007"

val_scores = defaultdict(list)
test_scores = defaultdict(list)
expected_test_losses = defaultdict(list)

for i in range(10):
    # create data set
    supervised_dataset = DatasetBase(dataset_name=name, directory=f"./data/{name}", verbose=True)
    supervised_dataset.create_train_validate_split(validateFrac = 0.25)

    # base line in POEM paper
    model = "baseline"
    crf = CRF(dataset = supervised_dataset, tol = 1e-6, minC = -2, maxC = 2, verbose = True, parallel = None)
    val_scores[model].append(crf.validate())
    test_scores[model].append(crf.test())
    expected_test_losses[model].append(crf.expectedTestLoss())

    # train logging policy
    features, labels = DatasetBase.generate_stream(subsampleFrac = 0.05, replayCount = 1)
    logging_dataset = deepcopy(supervised_dataset)
    supervised_dataset.trainFeatures = features
    supervised_dataset.trainLabels = labels
    logger = LoggingPolicy(logging_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = True)
    model =




print("Val: ", np.mean(val_scores))
print("Test: ", np.mean(test_scores))
print("Expected test loss: ", np.mean(expected_test_losses))
