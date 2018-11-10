from model import XRayModel
import numpy as np
import pdb
import scipy.misc
import data
import torch
import torch.nn as nn
from visualize import Logger


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    '''
        Tunable Parameters
    '''
    model = XRayModel()
    percentage_train = 75
    batch_size = 16
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
    save_name = "test.pt"
    criterion = nn.CrossEntropyLoss()

    '''
        LOAD DATA
    '''
    normal = np.load("data/normal-1.npy").reshape(-1, 1024, 1024)
    abnormal = np.load("data/abnormal.npy").reshape(-1, 1024, 1024)
    loaders = data.get_loaders(normal, abnormal, percentage_train, batch_size)

    '''
        TRAIN
    '''
    trainer = Trainer(model, optimizer, loaders, save_name, criterion)
    trainer.run(3)

class Trainer():
    def __init__(self, model, optimizer, loaders, save_name,\
                 criterion, load_path=None, log_freq=23, save_epoch_freq=3):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader = loaders
        self.criterion = criterion
        self.model = model.to(DEVICE)
        self.criterion = criterion.to(DEVICE)

        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

        self.save_name = save_name
        self.save_epoch_freq = save_epoch_freq
        if load_path is not None:
            print("loading")
            self.load_checkpoint(load_path)

        self.log_freq = log_freq
        self.tLog, self.vLog = Logger("./logs/train_pytorch_" + self.save_name),\
            Logger("./logs/val_pytorch_" + self.save_name)

        self.train = False
        self.curr_epoch = -1 # will up-date to 0 when it starts training

    def run(self, epochs):
        print("Running Training")
        for e in range(epochs):
            print("Epoch ", e)
            self.next_epoch()
            self.train_model()
            self.evaluate()
        self.save_checkpoint()

    def next_epoch(self):
        self.train_loss.append([])
        self.train_accuracy.append([])
        self.val_loss.append([])
        self.val_accuracy.append([])
        self.curr_epoch += 1

        if self.curr_epoch % self.save_epoch_freq == 0 and self.curr_epoch > 0:
            self.save_checkpoint()

    def train_model(self):
        print("train")
        self.train = True
        self.model.train()

        for idx, (data, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = data.to(DEVICE)
            data = torch.unsqueeze(data, 1) # so its batch x 1(channels) x height x width
            output = self.model(data)
            loss = self.criterion(output, label)
            self.append_results(idx, loss.data.item(), output, label, len(data))
            loss.backward()
            self.optimizer.step()
            print(loss)

            if idx > 0 and idx % self.log_freq == 0:
                print("Logging: ", idx)
                self.log(idx)

    def evaluate(self):
        print("evaluate")
        self.train = False
        self.model.eval()

        with torch.no_grad():
            for idx, (data_batch, label_batch, in_len, label_len) in enumerate(self.val_loader):
                data = data.to(DEVICE)
                data = torch.unsqueeze(data, 1) # so its batch x 1(channels) x height x width
                output = self.model(data)
                loss = self.criterion(output, label)
                self.append_results(idx, loss.data.item(), output, label, len(data))
                print(loss)

                if idx > 0 and idx % self.log_freq == 0:
                    print("Logging: ", idx)
                    self.log(idx)


    '''
        SAVE MODEL
    '''
    def save_checkpoint(self):
        state = {
                'epoch': self.curr_epoch,
                'model_type': type(self.model),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}
        filename =  "./models/" + self.save_name + str(self.curr_epoch) + ".pth.tar"
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        if filename:
            if os.path.isfile(filename):
                print("loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                params = HyperParameters()
                self.model = checkpoint['model_type'](N_PHONEMES + 1, params.hidden_size,
                                params.nlayers, params.frequencies)
                self.curr_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epoch']))
            else:
                print("no checkpoint found at '{}'".format(filename))


    '''
        GETTING STATS of training
    '''
    def append_results(self, idx, loss, output, labels, batch_size):
        argmax = output.argmax(dim=1)
        argmax = argmax.to(DEVICE).view(-1, 1)
        labels = labels.to(DEVICE).detach()

        # TERRIBLE: FIX!!
        correct = 0
        for i in range(len(argmax)):
            if(argmax[i].item() == labels[i].item()):
                correct += 1

        accuracy = correct / batch_size
        if self.train:
            self.train_loss[self.curr_epoch].append(loss)
            self.train_accuracy[self.curr_epoch].append(accuracy)
        else:
            self.val_loss[self.curr_epoch].append(loss)
            self.val_accuracy[self.curr_epoch].append(accuracy)

    def log(self, idx):
        num_vals = self.log_freq
        if self.train:
            mean_loss = np.mean(
                self.train_loss[self.curr_epoch][max(idx - num_vals, 0): idx])
            mean_accuracy = np.mean(
                self.train_accuracy[self.curr_epoch][max(idx - num_vals, 0): idx])
            log = self.tLog
            idx = len(self.train_loss[0]) * (self.curr_epoch + 1) + idx
        else:
            mean_loss = np.mean(
                self.val_loss[self.curr_epoch][max(idx - num_vals, 0): idx])
            mean_accuracy = np.mean(
                self.val_accuracy[self.curr_epoch][max(idx - num_vals, 0): idx])
            log = self.vLog
            idx = len(self.val_loss[0]) * (self.curr_epoch + 1) + idx

        tr_info = {'loss': mean_loss, 'accuracy': mean_accuracy}

        for tag, value in tr_info.items():
            log.log_scalar(tag, value, idx)
        for tag, value in self.model.named_parameters():
            log.log_histogram(tag, value.data.cpu().numpy(), idx)
            log.log_histogram(
                tag + '/grad', value.grad.data.cpu().numpy(), idx)

if __name__ == '__main__':
    main()
