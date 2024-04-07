from sklearn import datasets


# stats from http://proceedings.mlr.press/v80/wu18g/wu18g-supp.pdf
DATASET_STATS = {
    "yeast": dict(n_features=103, labels=14, n_train=1500, n_test=917),
    "scene": dict(n_features=294, labels=6, n_train=1211, n_test=1196),
    "tmc": dict(n_features=30438, labels=22, n_train=21519, n_test=7077),
    "lyrl": dict(n_features=47236, labels=103, n_train=23149, n_test=781265) # labels corrected based on https://scikit-learn.org/0.18/datasets/rcv1.html
}


def main(dataset: str, train_path: str, test_path: str):
    # load datasets
    n_features = DATASET_STATS[dataset]["n_features"]
    data_train = datasets.load_svmlight_file(train_path, n_features=n_features, multilabel=True)
    data_test = datasets.load_svmlight_file(test_path, n_features=n_features, multilabel=True)

    # verify datasets
    match_stats(dataset, data_train, data_test)


def flatten_list(arr) -> set:
    return set([label for labels in arr for label in labels])


def match_stats(dataset, data_train, data_test) -> bool:
    x_train, y_train = data_train
    x_test, y_test = data_test
    stats = DATASET_STATS[dataset]

    # check sizes of train and test set
    assert x_train.shape == (stats["n_train"], stats["n_features"])
    assert len(y_train) == stats["n_train"]
    assert x_test.shape == (stats["n_test"], stats["n_features"])
    assert len(y_test) == stats["n_test"]

    # check number of labels
    n = len(flatten_list(y_train) | flatten_list(y_test))
    assert n == stats["labels"], f"{dataset} detected {n} labels, expected {stats['labels']}"


if __name__=="__main__":
    paths = {
        "yeast": dict(train_path="yeast/yeast_train.svm", test_path="yeast/yeast_test.svm"),
        "scene": dict(train_path="scene/scene_train.svm", test_path="scene/scene_test.svm"),
        "tmc": dict(train_path="tmc2007/tmc2007_train.svm", test_path="tmc2007/tmc2007_test.svm"),
        "lyrl": dict(train_path="lyrl/rcv1_topics_train.svm", test_path="lyrl/rcv1_topics_test.svm"),
    }

    for dataset, config in paths.items():
        main(dataset=dataset, **config)
        print(f"Dataset {dataset} seems OK")
