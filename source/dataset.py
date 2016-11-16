#!/usr/bin/env python

import pickle
import numpy as np
from numpy.random import *

class Datasets(object):
    def __init__(self):
        self.train = None
        self.test = None
        self.validation = None

    def load_data(self):
        return self.train.features, self.train.labels, self.test.features, self.test.labels

class Dataset(object):
    def __init__(self, features, labels):
        assert features.shape[0] == labels.shape[0], ("features.shape: %s labels.shape: %s" % (features.shape, labels.shape))
        self._num_examples = features.shape[0]

        features = features.astype(np.float32)
        features = np.multiply(features - 130.0, 1.0 / 70.0) # [130.0 - 200.0] -> [0 - 1]
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]


# sample from Gaussian distribution
def sampling(n, mu, sigma):
    height = np.reshape(normal(mu, sigma, n), (n, 1))
    for i, h in enumerate(height):
        height[i][0] = np.min([h[0], 200.0])
        height[i][0] = np.max([h[0], 120.0])
    return height


# shuffle data and its label in association
def corresponding_shuffle(data, target):
    random_indices = permutation(len(data))
    _data = np.zeros_like(data)
    _target = np.zeros_like(target)
    for i, j in enumerate(random_indices):
        _data[i] = data[j]
        _target[i] = target[j]
    return _data, _target

# save dataset as a pickle file
def save_as_pickle(filename, dataset):
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


# entry point
if __name__ == '__main__':
    datasets = Datasets()

    n = 500 # number of data
    x_male = sampling(n, 171.66, 5.60)
    x_female = sampling(n, 158.32, 5.52)
    y_male = np.zeros_like(x_male) # male:0
    y_female = np.ones_like(x_male) # female:1

    # concat
    data = np.r_[x_male, x_female]
    target = np.r_[y_male, y_female]

    # shuffle
    data, target = corresponding_shuffle(data, target)

    # split data
    N_train = np.floor(n * 2 * 0.8).astype(np.int32)
    N_validation = np.floor(N_train * 0.2).astype(np.int32)
    x_train, x_test = np.split(data, [N_train])
    y_train, y_test = np.split(target, [N_train])
    x_validation = x_train[:N_validation]
    y_validation = y_train[:N_validation]

    # create dataset
    datasets.train = Dataset(x_train, y_train)
    datasets.test = Dataset(x_test, y_test)
    datasets.validation = Dataset(x_validation, y_validation)

    # save as a pickle file
    save_as_pickle('height.pkl', datasets)
