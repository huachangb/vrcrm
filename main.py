import numpy as np

from vrcrm.data import BanditDataset, DatasetBase
from vrcrm.models.skylines import CRF

name = "yeast"

val_scores = []
test_scores = []
expected_test_losses = []

for i in range(10):
    supervised_dataset = DatasetBase(dataset_name=name, directory=f"./data/{name}", verbose=True)
    supervised_dataset.create_train_validate_split(validateFrac = 0.25)

    crf = CRF(dataset = supervised_dataset, tol = 1e-6, minC = -2, maxC = 2, verbose = True, parallel = None)
    val_scores.append(crf.validate())
    test_scores.append(crf.test())
    expected_test_losses.append(crf.expectedTestLoss())

print("Val: ", np.mean(val_scores))
print("Test: ", np.mean(test_scores))
print("Expected test loss: ", np.mean(expected_test_losses))