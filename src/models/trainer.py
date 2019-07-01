# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - trainer.py
# md
# --------------------------------------------------------------------------------------------------------
import json
from time import sleep
from typing import Collection

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.models.callbacks import Callback, CallbackContainer, TensorboardCB, PrintLogs
from src.models.metrics import MetricContainer, Accuracy, MetricCallback, PredictionEntropy

# device = th.device('cpu')
device = th.device('cuda:0') if th.cuda.is_available() else th.device('cpu')
th.backends.cudnn.benchmark = True  # should speed thing up ?


class Trainer:
    """
    Trainer class provides training loop.
    It needs training and validation dataset, model, criterion, optizer, callback container and metriccontainer.
    The params dict must have:
        'workers': 0,
        'bs': 64,
        'n_epochs': 1000,
        'lr': 1e-3,
        'momentum': 0.90
    """

    def __init__(self, train_ds: Dataset, valid_ds: Dataset, model: nn.Module, criterion, optimizer,
                 params: dict, callbacks: CallbackContainer, metrics: MetricContainer):
        self.params = params
        self.train_dl = self.make_data_loader(train_ds)
        self.valid_dl = self.make_data_loader(valid_ds)
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, threshold=0.02,
        #                                                       patience=50, cooldown=100, min_lr=1e-6, verbose=True)
        self.scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, cycle_momentum=False)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.train_dl), eta_min=0, last_epoch=-1)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=256*2, eta_min=0, last_epoch=-1)
        self.cbc = callbacks
        self.mc = metrics

    def train(self):
        dummy_cont, dummy_cat, dummy_y = next(iter(self.train_dl))  # Todo: redundant with graph in experiment
        dummy_cont, dummy_cat, dummy_y = dummy_cont.to(device), dummy_cat.to(device), dummy_y.to(device)
        # dummy_input = (dummy_cont, dummy_cat)
        dummy_input = dummy_cont

        logs = {'params': self.params, 'model': self.model, 'dummy_input': dummy_input}

        self.cbc.on_train_begin(logs=logs)
        for epoch in range(1, self.params['n_epochs'] + 1):
            self.cbc.on_epoch_begin(epoch=epoch, logs=logs)

            # self.scheduler.step()

            self.model.train()
            logs['train_loss'] = []
            for batch, data in enumerate(self.train_dl, 1):
                self.cbc.on_batch_begin(batch=batch, logs=logs)

                x_cont, x_cat, y_true = data
                x_cont, x_cat, y_true = x_cont.to(device), x_cat.to(device), y_true.long().to(device)

                self.optimizer.zero_grad()
                # y_pred = self.model(x_cont, x_cat)
                y_pred = self.model(x_cont)

                self.cbc.on_loss_begin()

                # y_true = y_true.squeeze()
                y_true = x_cont

                # print(y_true)
                # print(y_pred)
                # print(y_true.shape,y_pred.shape)
                # print(y_true.max(),y_pred.max())
                #

                loss = self.criterion(y_pred, y_true)
                logs['train_loss'] = np.append(logs['train_loss'], loss.item())
                self.cbc.on_loss_end()

                self.cbc.on_backward_begin()
                loss.backward()
                self.optimizer.step()

                self.cbc.on_backward_end()

                self.cbc.on_batch_end(batch, logs=logs)

            if epoch % 10 == 0:
                with th.no_grad():  # Todo: Is this necessary
                    # print(y_true[:10])
                    # print(y_pred[:10])
                    print('\t\ty_true: [{:+10.5f} {:+10.5f}], mean: {:+10.5f}, std: {:+10.5f}'.format(y_true.min().item(), y_true.max().item(), y_true.mean().item(),
                                                                                                      y_true.std().item()))
                    print('\t\ty_pred: [{:+10.5f} {:+10.5f}], mean: {:+10.5f}, std: {:+10.5f}'.format(y_pred.min().item(), y_pred.max().item(), y_pred.mean().item(),
                                                                                                      y_pred.std().item()))
                    for param_group in self.optimizer.param_groups:
                        print('\t\t{:.5e}'.format(param_group['lr']))

            # Validation
            self.model.eval()  # Impacts dropout and batchnorm
            with th.no_grad():  # doesn't calculate grads
                logs['valid_loss'] = logs['y_true'] = np.array([])
                # logs['y_pred'] = np.empty((0, 3))  # y_pred has the form of [a,b,....] with a+b+....=1
                logs['y_pred'] = np.empty((0, 116))  # y_pred has the form of [a,b,....] with a+b+....=1

                for valid_batch, data in enumerate(self.valid_dl, 1):
                    x_cont, x_cat, y_true = data
                    x_cont, x_cat, y_true = x_cont.to(device), x_cat.to(device), y_true.long().to(device)

                    # y_pred = self.model(x_cont, x_cat)  # Not probabilities !!! Needs softmax to get probabilities
                    y_pred = self.model(x_cont)  # Not probabilities !!! Needs softmax to get probabilities

                    # y_true = y_true.squeeze()
                    y_true = x_cont

                    loss = self.criterion(y_pred, y_true)  # nn.CrossEntropyLoss() applies softmax on y_pred, so don't apply softmax on y_pred !

                    # y_pred = F.softmax(y_pred, dim=1)  # Only now change y_pred to probabilities
                    logs['valid_loss'] = np.append(logs['valid_loss'], [loss.item()])
                    logs['y_pred'] = np.vstack((logs['y_pred'], y_pred.cpu().detach().numpy()))  # y_pred has the form [[a,b],[c,d],...]
                    logs['y_true'] = np.append(logs['y_true'], y_true.cpu())

            # self.scheduler.step(sum(logs['valid_loss']) / len(logs['valid_loss']))
            # self.scheduler.step(loss)
            self.scheduler.step()

            self.cbc.on_epoch_end(epoch=epoch, logs=logs)

            # if epoch % 50 == 0:
            #     print(y_true[:10])
            #     _, y_pred = y_pred.max(1)
            #     print(y_pred[:10])
            #     print((y_true[:10] == y_pred[:10]) * 1)
            # if epoch % 10 == 0:
            #     with th.no_grad():  # Todo: Is this necessary
            #         # print(y_true[:10])
            #         # print(y_pred[:10])
            #         print(y_true.min().item(), y_true.max().item())
            #         print(y_pred.min().item(), y_pred.max().item())

        self.cbc.on_train_end(logs=logs)
        return self.model

    def make_data_loader(self, dataset: Dataset) -> DataLoader:
        """ put a Dataset into a Dataloader"""
        dataloader = DataLoader(dataset, batch_size=self.params['bs'], shuffle=True, num_workers=self.params['workers'])
        return dataloader
