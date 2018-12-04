from model import XRayModel
import numpy as np
import pdb
import scipy.misc
from trainer import Trainer
from data_manager import get_loaders
import torch
import torch.nn as nn
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    '''
        Tunable Parameters
    '''
    model = XRayModel()
    percentage_train = 75
    batch_size = 16
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
    save_name = "preprocess.pt"
    criterion = nn.CrossEntropyLoss()
    percent_train = 65
    percent_val = 30
    percent_test = 5

    # data checks:
    assert (percent_train + percent_val + percent_test == 100)

    '''
        LOAD DATA
    '''
    start = time.time()
    xray_data = np.load("data/picture_data.npy")
    xray_data = np.transpose(xray_data, (2, 0, 1))
    xray_label = np.load("data/labels.npy")
    end = time.time()
    print("Time to Load = ", end - start)

    num_imgs = xray_data.shape[0]
    num_train_imgs = int(num_imgs * percent_train / 100)
    num_val_imgs = int(num_imgs * percent_val / 100)
    num_test_imgs = int(num_imgs * percent_test / 100)
    loaders = get_loaders(train_d=xray_data[:num_train_imgs],
                               val_d=xray_data[num_train_imgs: num_train_imgs + num_val_imgs],
                               test_d=xray_data[-num_test_imgs:],
                               train_l=xray_label[:num_train_imgs],
                               val_l=xray_label[num_train_imgs: num_train_imgs + num_val_imgs],
                               test_l=xray_label[:num_test_imgs],
                               batch_size=batch_size)
    end_setup = time.time()
    print("Time to Setpu Completely = ", end - start)
    '''
        TRAIN
    '''
    trainer = Trainer(model, optimizer, loaders, save_name, criterion)
    trainer.run(3)


if __name__ == '__main__':
    main()
