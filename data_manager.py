import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch
import random

def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    mean = np.mean(x, axis=1).reshape(-1,1)
    return x - mean

class XrayDataset(Dataset):
    def __init__(self, inputs, labels):
        assert len(inputs) == len(labels)
        self.inputs = torch.from_numpy(sample_zero_mean(inputs.reshape(inputs.shape[0], -1)).reshape(inputs.shape)).float()
        self.labels = torch.from_numpy(labels)

    def __getitem__(self,i):
        return self.inputs[i].to(DEVICE), self.labels[i].to(DEVICE).long()

    def __len__(self):
        return len(self.inputs)



def get_loaders(train_d, val_d, test_d, train_l, val_l, test_l, batch_size):
    train_dataset = XrayDataset(train_d, train_l)
    val_dataset  = XrayDataset(val_d, val_l)
    test_dataset   = XrayDataset(test_d, test_l)


    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dev_loader   = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader   = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, dev_loader, test_loader


# depracated:
def get_loaders_normal_abnormal(normal, abnormal, percentage_train, batch_size):
    inputs = np.vstack((normal, abnormal))
    normal_labels = np.full((normal.shape[0]), 1)
    abnormal_labels = np.full((abnormal.shape[0]), 0)
    labels = np.concatenate((normal_labels, abnormal_labels))
    cutoff_train_eval = int((percentage_train / 100) * inputs.shape[0])

    zip_inputs = list(zip(inputs,labels))
    random.shuffle(zip_inputs)
    inputs,labels = zip(*zip_inputs)
    inputs = np.array(inputs)
    labels = np.array(labels)

    train_dataset = XrayDataset(inputs[:cutoff_train_eval],labels[:cutoff_train_eval])
    evalu_dataset = XrayDataset(inputs[cutoff_train_eval:],labels[cutoff_train_eval:])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dev_loader   = DataLoader(evalu_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, dev_loader
