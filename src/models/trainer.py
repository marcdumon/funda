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

from src.data.data_helpers import split_train_valid
from src.data.synthetic_data import make_double_spiral, SyntheticDataset
from src.models.callbacks import Callback, CallbackContainer, TensorboardCB, PrintLogs
from src.models.metrics import MetricContainer, Accuracy, MetricCallback, PredictionEntropy
from src.models.model import Linear3Layers

# device = torch.device('cuda:0')
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
        self.model = model.to(device)
        self.params = params
        self.criterion = criterion
        self.optimizer = optimizer
        self.cbc = callbacks
        self.mc = metrics
        self.train_dl = self.make_data_loader(train_ds)
        self.valid_dl = self.make_data_loader(valid_ds)

    def train(self):
        dummy_input = self.train_dl.dataset[0][0].to(device)  # Need this in the TensorboardCB
        logs = {'params': self.params, 'model': self.model, 'dummy_input': dummy_input}

        self.cbc.on_train_begin(logs=logs)
        for epoch in range(1, self.params['n_epochs'] + 1):
            self.cbc.on_epoch_begin(epoch=epoch, logs=logs)
            logs['train_loss'] = []
            for batch, data in enumerate(self.train_dl, 1):
                self.cbc.on_batch_begin(batch=batch, logs=logs)
                x, y_true = data
                x, y_true = x.to(device), y_true.long().to(device)

                self.optimizer.zero_grad()
                y_pred = self.model(x)

                self.cbc.on_loss_begin()
                loss = self.criterion(y_pred, y_true)
                logs['train_loss'] = np.append(logs['train_loss'], loss.item())
                self.cbc.on_loss_end()

                self.cbc.on_backward_begin()
                loss.backward()
                self.optimizer.step()
                self.cbc.on_backward_end()

                self.cbc.on_batch_end(batch, logs=logs)

            # Validation
            self.model.eval()
            with th.no_grad():
                logs['valid_loss'] = logs['y_true'] = np.array([])
                logs['y_pred'] = np.empty((0, 2))  # y_pred has the form of [a,b] with a+b=1
                for valid_batch, data in enumerate(self.valid_dl, 1):
                    x, y_true = data
                    x, y_true = x.to(device), y_true.long().to(device)
                    y_pred = self.model(x)  # Not probabilities !!! Needs softmax to get probabilities
                    loss = self.criterion(y_pred, y_true)  # nn.CrossEntropyLoss() applies softmax on y_pred, so don't apply softmax on y_pred !
                    y_pred = F.softmax(y_pred, dim=1)  # Only now change y_pred to probabilities
                    logs['valid_loss'] = np.append(logs['valid_loss'], [loss.item()])
                    logs['y_pred'] = np.vstack((logs['y_pred'], y_pred.cpu().detach().numpy()))  # y_pred has the form [[a,b],[c,d],...]
                    logs['y_true'] = np.append(logs['y_true'], y_true.cpu())
            self.cbc.on_epoch_end(epoch=epoch, logs=logs)
        self.cbc.on_train_end(logs=logs)

    def make_data_loader(self, dataset: Dataset) -> DataLoader:
        """ put a Dataset into a Dataloader"""
        dataloader = DataLoader(dataset, batch_size=self.params['bs'], shuffle=True, num_workers=self.params['workers'])
        return dataloader
