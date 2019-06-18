# --------------------------------------------------------------------------------------------------------
# 2019/04/20
# 0_ml_project_template - experiment.py
# md
# --------------------------------------------------------------------------------------------------------
import sys
from pprint import pprint
from time import gmtime, strftime

import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim import SGD, Adam
import hiddenlayer as hl
from my_toolbox import MyOsTools as mos

from datasets import TabularDataset
from predict_model import predict
from src.models.callbacks import CallbackContainer, PrintLogs, TensorboardCB
from src.models.metrics import Accuracy, PredictionEntropy, MetricContainer, MetricCallback, Precision, Recall
from src.models.models import FeedForwardNN
from src.models.trainer import Trainer
from src.visualization.visualize import plot_confusion_matrix

# device = th.device('cpu')
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())  # Session datetime

# Parameters
params = {
    'experiment': 'baseline-no_dropouts',
    'workers': 0,
    'bs': 1024*10,
    'n_epochs': 2000,
    'lr': 5e-3,
    'lin_layer_sizes': [20, 5],
    'emb_dropout': 0.,
    'lin_layer_dropouts': [0., 0.],
    # 'momentum': 0.90,
}


# Output console to file
mos.create_directory('/mnt/Development/My_Projects/fundamental_stock_analysis/reports/experiments/{}/'.format(params['experiment']))
mos.create_directory('/mnt/Development/My_Projects/fundamental_stock_analysis/reports/experiments/{}/{}/'.format(params['experiment'], date_time))
sys.stdout = open('/mnt/Development/My_Projects/fundamental_stock_analysis/reports/experiments/{}/{}/log.txt'.format(params['experiment'], date_time), 'w')


print('Date: {}\nLogfile: {}'.format(date_time,params['experiment']))
print('=' * 150)
print('device =', device)
print('-' * 150)
print('Parameters:')
pprint(params)
print('-' * 150)

# DATA
data_path = '/mnt/Development/My_Projects/fundamental_stock_analysis/data/processed/'
Xy_train = pd.read_csv(data_path + 'train_dataset.csv', index_col=0)
Xy_valid = pd.read_csv(data_path + 'valid_dataset.csv', index_col=0)
Xy_test = pd.read_csv(data_path + 'test_dataset.csv', index_col=0)

categorical_features = ['m_start', 'm_end', 'q_start', 'sector', 'industry']
# categorical_features = []
output_feature = 'label'

# Combine train, test, valid to be sure to have all categories
Xy_all = pd.concat([Xy_train, Xy_valid, Xy_test])
label_encoders = {}
for cat_col in categorical_features:
    label_encoders[cat_col] = LabelEncoder().fit(Xy_all[cat_col])
    Xy_train[cat_col] = label_encoders[cat_col].transform(Xy_train[cat_col])
    Xy_valid[cat_col] = label_encoders[cat_col].transform(Xy_valid[cat_col])
    Xy_test[cat_col] = label_encoders[cat_col].transform(Xy_test[cat_col])

# Datasets
train_ds = TabularDataset(data=Xy_train, cat_cols=categorical_features, output_col='label')
valid_ds = TabularDataset(data=Xy_valid, cat_cols=categorical_features, output_col='label')
cat_dims = [int(Xy_all[col].nunique()) for col in categorical_features]
emb_dims = [(x, min(50, (x + 1) // 3)) for x in cat_dims]

# NETWORK
net = FeedForwardNN(emb_dims, no_of_cont=142, lin_layer_sizes=params['lin_layer_sizes'],
                    output_size=3, emb_dropout=params['emb_dropout'],
                    lin_layer_dropouts=params['lin_layer_dropouts']).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = SGD(net.parameters(), lr=params['lr'], momentum=params['momentum'])
optimizer = Adam(net.parameters(), lr=0.015)

# Metrics and Callbacks
acc = Accuracy()
entr = PredictionEntropy()
precision = Precision()
recall = Recall()

mc = MetricContainer(metrics=[acc, precision, recall, entr])
cbc = CallbackContainer()
cbc.register(MetricCallback(mc))
cbc.register(PrintLogs(every_n_epoch=10))
cbc.register(TensorboardCB(every_n_epoch=20, experiment_name=params['experiment']))

trainer = Trainer(train_ds=train_ds, valid_ds=valid_ds, model=net, criterion=criterion,
                  optimizer=optimizer, params=params, callbacks=cbc, metrics=mc)

print('Network:')
print(net)
print('-' * 150)
print('\n\nTraining log:')
print('-' * 150)
train_dl = trainer.make_data_loader(train_ds)
dummy_cont, dummy_cat, dummy_y = next(iter(train_dl))
dummy_cont, dummy_cat, dummy_y = dummy_cont.to(device), dummy_cat.to(device), dummy_y.to(device)
dummy_input = (dummy_cont, dummy_cat)

transforms = [hl.transforms.Fold('Constant > Gather > Gather', 'Embedding'),
              hl.transforms.Fold('Unsqueeze > BatchNorm > Squeeze', 'BatchNormXXX'),
              hl.transforms.Fold('Linear > Relu', 'Linear')]

graph = hl.build_graph(net, dummy_input, transforms=transforms)
graph.save('../../reports/experiments/{}/{}/graph.png'.format(params['experiment'], date_time), format='png')

model = trainer.train()
th.save(model, '../../reports/experiments/{}/{}/model.pth'.format(params['experiment'], date_time))

# Make confusion matrix
model.eval()  # should alse disable dropout
cont_X = valid_ds.cont_X
cat_X = valid_ds.cat_X
y = valid_ds.y
y_pred = predict(model, cont_X, cat_X, labels=True)
# flatten because y=[[],[],...]
y = y.flatten()
y_pred = y_pred.flatten()
plot_confusion_matrix('../../reports/experiments/{}/{}/conf_matrix.png'.format(params['experiment'], date_time), y, y_pred, )