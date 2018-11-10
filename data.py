import numpy as np
from torch.utils.data import Dataset, DataLoader
from test import DEVICE
import torch

class XrayDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self,i):
        return torch.from_numpy(self.inputs[i]).float(),self.labels[i]

    def __len__(self):
        return len(self.inputs)

def get_loaders(normal, abnormal, percentage_train, batch_size):
    inputs = np.vstack((normal, abnormal))
    normal_labels = np.full((normal.shape[0]), 1)
    abnormal_labels = np.full((abnormal.shape[0]), 0)
    labels = np.concatenate((normal_labels, abnormal_labels))
    cutoff_train_eval = int((percentage_train / 100) * inputs.shape[0])

    train_dataset = XrayDataset(inputs[:cutoff_train_eval],labels[:cutoff_train_eval])
    evalu_dataset = XrayDataset(inputs[cutoff_train_eval:],labels[cutoff_train_eval:])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dev_loader   = DataLoader(evalu_dataset, shuffle=True, batch_size=batch_size)
    return train_loader, dev_loader
