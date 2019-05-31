# --------------------------------------------------------------------------------------------------------
# 2019/04/20
# 0_ml_project_template - experiment.py
# md
# --------------------------------------------------------------------------------------------------------

import torch as th
import torch.nn as nn
from torch.optim import SGD

from src.data.data_helpers import split_train_valid
from src.models.callbacks import CallbackContainer, PrintLogs, TensorboardCB
from src.models.metrics import Accuracy, PredictionEntropy, MetricContainer, MetricCallback
from src.models.model import Linear3Layers
from src.data.synthetic_data import make_double_spiral, SyntheticDataset
from src.models.trainer import Trainer

""" 
Here we run the experiment.
"""


# Parameters
params = {
    'experiment': 'test',
    'n_samples': 5000,
    'noise': 1.0,
    'train_pct': 0.80,
    'workers': 0,
    'bs': 64,
    'n_epochs': 1000,
    'lr': 1e-3,
    'momentum': 0.90,
}

# DATA
Xy = make_double_spiral(n_samples=params['n_samples'], shuffle=True, noise=params['noise'])
Xy_train, Xy_valid = split_train_valid(Xy, train_pct=params['train_pct'])
n_features = Xy.shape[1] - 1  # all columns exept label
n_classes = len(set(Xy['label']))
# Datasets
train_ds = SyntheticDataset(Xy_train)
valid_ds = SyntheticDataset(Xy_valid)

# NETWORK
net = Linear3Layers(n_features, 50, 30, n_classes, 'relu')
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=params['lr'], momentum=params['momentum'])
# Metrics and Callbacks
acc = Accuracy()
entr = PredictionEntropy()
mc = MetricContainer(metrics=[acc, entr])
cbc = CallbackContainer()
cbc.register(MetricCallback(mc))
cbc.register(PrintLogs(every_n_epoch=10))
cbc.register(TensorboardCB(every_n_epoch=10, experiment_name=params['experiment']))

trainer = Trainer(train_ds=train_ds, valid_ds=valid_ds, model=net, criterion=criterion,
                  optimizer=optimizer, params=params, callbacks=cbc, metrics=mc)
trainer.train()
