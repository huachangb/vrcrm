"""
POEM Dataset code ported to modern Python

Original code can be found at https://www.cs.cornell.edu/~adith/POEM/
under ICML'15
"""

from typing import List
import sys
import numpy as np
import scipy.sparse
import sklearn.datasets
import sklearn.decomposition
import sklearn.model_selection
import sklearn.preprocessing
import math


class DatasetBase():
    def __init__(
            self,
            dataset_name: str,
            directory: str,
            label_subset: List[str] = None,
            reduce_dims: int = None,
            initialize: bool = True,
            verbose: bool = False
        ) -> None:
        super().__init__()

        self.verbose = verbose
        self.trainFeatures = None
        self.trainLabels = None
        self.validateFeatures = None
        self.validateLabels = None
        self.testFeatures = None
        self.testLabels = None
        self.samples = None

        if not initialize:
            return

        self._load_dataset(dataset_name, directory, label_subset)

        if reduce_dims is not None:
            self._reduce_dimensionality(reduce_dims)

    def _sanitizeLabels(self, label_list):
        return_list = []
        for tup in label_list:
            if -1 in tup:
                return_list.append(())
            else:
                return_list.append(tup)
        return return_list

    def _reduce_dimensionality(self, ndims: int) -> None:
        LSAdecomp = sklearn.decomposition.TruncatedSVD(n_components = ndims, algorithm = 'arpack')
        LSAdecomp.fit(self.trainFeatures)
        self.trainFeatures = LSAdecomp.transform(self.trainFeatures)
        self.testFeatures = LSAdecomp.transform(self.testFeatures)

        if self.verbose:
            print(f"DatasetReader: [Message] Features now have shape: Train: {np.shape(self.trainFeatures)} Test: {np.shape(self.testFeatures)}")

    def _load_dataset(self, dataset_name: str, directory: str, label_subset: List[str]) -> None:
        train_file_name = f"{directory}/{dataset_name}_train.svm"
        test_file_name = f"{directory}/{dataset_name}_test.svm"

        labelTransform = sklearn.preprocessing.MultiLabelBinarizer(sparse_output = False)

        train_features, train_labels = sklearn.datasets.load_svmlight_file(
            train_file_name, dtype = np.longdouble, multilabel = True
        )
        sanitized_train_labels = self._sanitizeLabels(train_labels)

        numSamples, numFeatures = np.shape(train_features)

        biasFeatures = scipy.sparse.csr_matrix(np.ones((numSamples, 1),
            dtype = np.longdouble), dtype = np.longdouble)

        self.trainFeatures = scipy.sparse.hstack([train_features, biasFeatures], dtype = np.longdouble)
        self.trainFeatures = self.trainFeatures.tocsr()

        test_features, test_labels = sklearn.datasets.load_svmlight_file(
            test_file_name, n_features = numFeatures, dtype = np.longdouble, multilabel = True
        )
        sanitized_test_labels = self._sanitizeLabels(test_labels)

        numSamples, numFeatures = np.shape(test_features)

        biasFeatures = scipy.sparse.csr_matrix(np.ones((numSamples, 1),
            dtype = np.longdouble), dtype = np.longdouble)
        self.testFeatures = scipy.sparse.hstack([test_features, biasFeatures], dtype = np.longdouble)
        self.testFeatures = self.testFeatures.tocsr()

        self.testLabels = labelTransform.fit_transform(sanitized_test_labels)
        if label_subset is not None:
            self.testLabels = self.testLabels[:, label_subset]

        self.trainLabels = labelTransform.transform(sanitized_train_labels)
        if label_subset is not None:
            self.trainLabels = self.trainLabels[:, label_subset]

        if self.verbose:
            print(f"DatasetReader: [Message] Loaded {dataset_name} [n features, n labels, n_train, n_test]:")
            print(f"\t[{np.shape(self.trainFeatures)[1]}, {np.shape(self.trainLabels)[1]}, {np.shape(self.trainLabels)[0]}, {np.shape(self.testFeatures)[0]}]")

    def create_train_validate_split(self, validateFrac):
        self.trainFeatures, self.validateFeatures, self.trainLabels, self.validateLabels = \
            sklearn.model_selection.train_test_split(self.trainFeatures, self.trainLabels,
                test_size = validateFrac)

        if self.verbose:
            print(f"SupervisedDataset: [Message] Created supervised split [n_train, n_validate]: [{np.shape(self.trainFeatures)[0]}, {np.shape(self.validateFeatures)[0]}]")
            sys.stdout.flush()

    def generate_stream(self, subsampleFrac: float, replayCount: int, keep_samples: bool = False):
        numSamples = np.shape(self.trainFeatures)[0]

        # regenerate samples
        if self.samples is None or not keep_samples:
            self.samples = np.random.permutation(numSamples)

        numSubsamples = int(math.ceil(subsampleFrac*numSamples))

        sample_features = self.trainFeatures[self.samples, :][0:numSubsamples,:]
        sample_labels = self.trainFeatures[self.samples, :][0:numSubsamples,:]

        if self.verbose:
            print("DataStream: [Message] Selected subsamples [n_subsamples, n_samples]: ", numSubsamples, numSamples)
            sys.stdout.flush()

        if replayCount <= 1:
            return sample_features, sample_labels

        replicator = np.ones((replayCount, 1))
        repeatedFeatures = scipy.sparse.kron(replicator, sample_features, format='csr')
        repeatedLabels = np.kron(replicator, sample_labels)

        if self.verbose:
            print("DataStream: [Message] Replay samples ", np.shape(repeatedFeatures)[0])
            sys.stdout.flush()

        return repeatedFeatures, repeatedLabels


