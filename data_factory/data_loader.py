import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)
        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)
        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


class SKABSegLoader(object):
    """Loader for the SKAB dataset using anomaly-free training data.

    The dataset directory should contain an ``anomaly-free`` subfolder with
    ``anomaly-free.csv`` and additional folders with CSV files containing
    anomalies.  The training set is built from the anomaly-free file while the
    test set concatenates all remaining CSV files.
    """

    def __init__(
        self,
        data_path,
        win_size,
        step,
        mode="train",
        train_start=0.0,
        train_end=1.0,
        return_index: bool = False,
    ):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index

        train_file = os.path.join(data_path, "anomaly-free", "anomaly-free.csv")
        train_df = pd.read_csv(train_file, sep=";")
        train_values = train_df.drop(
            columns=["datetime", "anomaly", "changepoint"], errors="ignore"
        ).values
        train_values = np.nan_to_num(train_values)
        self.scaler.fit(train_values)
        train_values = self.scaler.transform(train_values)
        start = int(len(train_values) * train_start)
        end = int(len(train_values) * train_end)
        self.train = train_values[start:end]

        test_frames = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if not f.endswith(".csv"):
                    continue
                if os.path.join(root, f) == train_file:
                    continue
                test_frames.append(pd.read_csv(os.path.join(root, f), sep=";"))
        test_df = pd.concat(test_frames, ignore_index=True)
        test_values = test_df.drop(
            columns=["datetime", "anomaly", "changepoint"], errors="ignore"
        ).values
        test_values = np.nan_to_num(test_values)
        self.test = self.scaler.transform(test_values)
        # use test data for validation as well
        self.val = self.test
        self.test_labels = test_df["anomaly"].values

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start : start + self.win_size]
            label = np.zeros(self.win_size)
        elif self.mode == "val":
            window = self.val[start : start + self.win_size]
            label = np.zeros(self.win_size)
        elif self.mode == "test":
            window = self.test[start : start + self.win_size]
            label = self.test_labels[start : start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start : start + self.win_size]
            label = self.test_labels[start : start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', train_start=0.0, train_end=1.0, return_index: bool = False):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode, train_start, train_end, return_index=return_index)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode, train_start, train_end, return_index=return_index)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode, train_start, train_end, return_index=return_index)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode, train_start, train_end, return_index=return_index)
    elif (dataset == 'SKAB'):
        dataset = SKABSegLoader(data_path, win_size, step, mode, train_start, train_end, return_index=return_index)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
