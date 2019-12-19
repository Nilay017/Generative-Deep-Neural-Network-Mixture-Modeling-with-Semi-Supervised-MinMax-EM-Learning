from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomMNISTdigit(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_data, train_labels, transform=None):
        """
        Args:
            train_data : Images 64 X 64
            train_labels : Corresponding MNIST labels
            transform : transformation on data
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.transform = transform
       
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        transform2 = transforms.ToPILImage()
        if self.transform:
            return (self.transform(transform2(np.array(self.train_data[idx]))), self.train_labels[idx])

        return (self.train_data[idx], self.train_labels[idx])

class CustomMNISTdigit2(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_data, train_labels, train_data_supervision, fixed_posterior, transform=None):
        """
        Args:
            train_data : Images 64 X 64
            train_labels : Corresponding MNIST labels
            transform : transformation on data
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_data_supervision = train_data_supervision
        self.fixed_posterior  = fixed_posterior
        self.transform = transform
       
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        transform2 = transforms.ToPILImage()
        if self.transform:
            return (self.transform(transform2(self.train_data[idx])), self.train_labels[idx], self.train_data_supervision[idx], self.fixed_posterior[idx])

        return (self.train_data[idx], self.train_labels[idx], self.train_data_supervision[idx], self.fixed_posterior[idx])